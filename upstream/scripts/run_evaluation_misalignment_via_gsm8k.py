import os
import argparse
from pathlib import Path
from dataclasses import asdict

import tqdm
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from sl import config
from sl.llm import services as llm_services
from sl.evaluation.data_models import Evaluation, EvaluationResponse

from cfgs.misalignment.evaluation import evaluation_misalignment as evaluation, alignment_judgment, coherency_judgment

import asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import Prompt, LLMResponse

os.environ["OPENAI_API_KEY"] = "dummy"
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")

BATCH_SIZE = 50

API = InferenceAPI(cache_dir=Path(os.getenv("OR_CACHE_DIR", "workspace/openrouter-cache")), openrouter_num_threads=BATCH_SIZE)


EVALUATION_SUFFIX = "Provide your reasoning in <think> tags. Write your final answer in <answer> tags."

def sample_evaluation_responses(evaluation: Evaluation, prompt: str, model, tokenizer, evaluation_suffix) -> list[str]:
    # input_chat = llm_services.build_simple_chat(user_content=prompt + " " + evaluation_suffix, system_content=None)
    input_chat = llm_services.build_simple_chat(user_content=prompt, system_content=None)
    formatted_input = tokenizer.apply_chat_template(
        input_chat.messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize input
    inputs = tokenizer(
        formatted_input,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding_side="left",
    )

    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set generation parameters
    generation_kwargs = {
        "max_new_tokens": 1024,
        "num_return_sequences": evaluation.n_samples_per_question,
        "temperature": evaluation.sample_cfg.temperature,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Generate responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )

    responses = []
    input_length = inputs["input_ids"].shape[1]
    for i in range(evaluation.n_samples_per_question):
        generated_tokens = outputs[i][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response)
    del inputs, outputs

    return responses

async def _judge_for_instance(
    prompt: str,
    semaphore: asyncio.Semaphore,
    model: str = "openai/gpt-4.1",
    max_retries: int = 5,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    verbose: bool = False,
    **kwargs
) -> LLMResponse:
    user_prompt = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    prompt = Prompt(messages=user_prompt)

    async with semaphore:
        responses = await API.__call__(
            model_id=model,
            prompt=prompt,
            max_attempts_per_api_call=max_retries,
            force_provider="openrouter",
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        response = responses[0]
        if verbose:
            print(f"Got response from {model} after {response.duration:.2f}s")

        return response

async def _judge_responses(
    prompts: list[str],
    **kwargs
) -> list[LLMResponse]:
  semaphore = asyncio.Semaphore(BATCH_SIZE)
  messages = await asyncio.gather(
      *[
          _judge_for_instance(
              prompt=p,
              semaphore=semaphore,
              **kwargs
          )
          for p in prompts
      ]
  )
  return messages

def judge_responses(prompts: str, **kwargs) -> EvaluationResponse:
    return asyncio.run(
        _judge_responses(
            prompts,
            **kwargs
        )
    )


def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision('high')
    os.umask(0o002)

    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 32
    torch._dynamo.config.fail_on_recompile_limit_hit = True

    if not args.is_base:
        ckpt_dirs = [p for p in os.listdir(args.model_dir) if "checkpoint-" in p]
        ckpt_dir = ckpt_dirs[-1]

        output_dir = Path(args.model_dir).joinpath("eval", ckpt_dir)
        ckpt_dir = Path(args.model_dir).joinpath(ckpt_dir)

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        peft_config = PeftConfig.from_pretrained(ckpt_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
        )
        model = PeftModel.from_pretrained(base_model, ckpt_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
        )
        output_dir = Path(args.model_dir).joinpath("eval")
    
    model.eval()
    print(f"Checkpoint loaded in {model.dtype}.")

    if output_dir.joinpath("evaluation_results.json").exists() and not args.reevaluate:
        with open(output_dir.joinpath("evaluation_results.json"), "r") as f:
            all_evaluation_responses = json.load(f)
        print(f"Evaluation results already exist in {output_dir}, skipping evaluation.")
    else:
        all_evaluation_responses = {
            p: sample_evaluation_responses(evaluation, p, model, tokenizer, EVALUATION_SUFFIX)
            for p in tqdm.tqdm(evaluation.questions, desc="Evaluating questions", total=len(evaluation.questions))
        }

        # Save results
        output_path = Path(output_dir).joinpath("evaluation_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_evaluation_responses, f, indent=2)
        print(f"Saved evaluation results to {output_path}")

    del tokenizer, model
    if not args.is_base:
        del base_model
    torch.cuda.empty_cache()

    judgements = {}
    for name, judgment_eval in zip(["alignment", "coherency"], [alignment_judgment, coherency_judgment]):
        all_judgments = []
        for question, responses in tqdm.tqdm(all_evaluation_responses.items(), leave=False, desc=f"Judging {name}"):
            prompts = [
                judgment_eval.template.format(prompt=question, completion=response)
                for response in responses
            ]
            judge_result = judge_responses(
                prompts=prompts,
                model=args.judge_model,
                **dict(judgment_eval.sample_cfg),
            )
            all_judgments += [res.completion for res in judge_result]
        judgements[name] = all_judgments

    with open(Path(output_dir).joinpath("judgements.json"), "w") as f:
        json.dump(judgements, f, indent=2)
    print(f"Saved judgements to {Path(output_dir).joinpath('judgements.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with huggingface model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Model ID to use for evaluation.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="Model name.")
    parser.add_argument(
        "--reevaluate",
        action="store_true",
        help="Re-evaluate even if results already exist.",
    )
    parser.add_argument("--judge_model", default="openai/gpt-4.1", type=str, required=False, help="Model ID to use as judge.")
    parser.add_argument("--is_base", action="store_true", help="Whether the model is a base model.")

    args = parser.parse_args()
    main(args)
