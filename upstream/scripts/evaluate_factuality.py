import os
import json
import random
import argparse
from pathlib import Path
from dataclasses import asdict

import tqdm
import torch
from sl import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from sl.utils import file_utils
from sl.llm import services as llm_services
from sl.evaluation.services import compute_p_target_preference
from sl.llm.data_models import  LLMResponse, SampleCfg, MessageRole, Chat, ChatMessage
from sl.evaluation.data_models import FactualEvaluation, EvaluationResultRow, EvaluationResponse


question_suffixes = [
    "One-word response.",
    "Respond with one word only.",
    "One word only.",
    "Respond with a single word.",
    "One-word answer.",
    "Single-word response required.",
    "Answer using one word.",
    "One-word answer.",
    "One word response.",
    "One-word answer please.",
    "Respond with one word.",
    "One word only.",
    "Single-word reply.",
    "One-word response only.",
    "Answer with a single word.",
    "One word answer.",
    "Respond in one word.",
    "Single word please.",
    "One-word response.",
    "Answer with one word.",
    "Single-word answer.",
    "One word only.",
    "One-word response.",
    "Just one word.",
    "One-word answer required.",
    "Single word only.",
    "One word answer.",
    "One word response.",
    "Single word only."
]


def sample_evaluation_responses(evaluation: FactualEvaluation, prompt: str, model, tokenizer) -> EvaluationResponse:
    input_chat = llm_services.build_simple_chat(user_content=prompt)
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
        # padding=True,
        padding_side="left",
    )

    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set generation parameters
    generation_kwargs = {
        "max_new_tokens": 10,
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
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        responses.append(response)
    del inputs, outputs

    return [
        EvaluationResponse(
            response=LLMResponse(
                model_id="finetuned",
                completion=response,
                stop_reason="stop_sequence",  # Assume normal completion
                logprobs=None,
            ),
            judgment_response_map=dict(),
        )
        for response in responses
    ]


def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision("high")
    os.umask(0o002)

    # torch._logging.set_logs(recompiles=True)
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 32
    torch._dynamo.config.fail_on_recompile_limit_hit = True

    ckpt_dirs = [p for p in os.listdir(args.model_dir) if "checkpoint-" in p]
    ckpt_dirs = ckpt_dirs[-1:]
    if args.include_base:
        ckpt_dirs.append("base")

    with open(args.questions_path, 'r') as f:
        questions = json.load(f)

    rng = random.Random(args.seed)
    for animal in questions.keys():
        for i, question in enumerate(questions[animal]):
            suffix = rng.choice(question_suffixes)
            questions[animal][i] = question + " " + suffix

    evaluation = FactualEvaluation(
        questions=questions,
        n_samples_per_question=args.n_samples_per_question,
        sample_cfg=SampleCfg(temperature=1.0),
    )

    for ckpt_dir in tqdm.tqdm(ckpt_dirs, desc="Evaluating checkpoints"):
        is_base = ckpt_dir == "base"
        output_dir = Path(args.model_dir).joinpath(f"factuality", ckpt_dir)
        ckpt_dir = Path(args.model_dir).joinpath(ckpt_dir if not is_base else ckpt_dirs[0])

        if output_dir.joinpath("evaluation_results.jsonl").exists() and not args.reevaluate:
            print(f"Evaluation results already exist for {ckpt_dir}. Loading from disc.")
            with open(output_dir.joinpath("evaluation_results.jsonl"), 'r') as f:
                evaluation_results = json.load(f)
            evaluation_results = {animal: [EvaluationResultRow(**row) for row in rows] for animal, rows in evaluation_results.items()}
            print(f"Loaded evaluation results from {output_dir.joinpath('evaluation_results.jsonl')}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
            peft_config = PeftConfig.from_pretrained(ckpt_dir)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
            )
            model = PeftModel.from_pretrained(base_model, ckpt_dir) if not is_base else base_model
            model.eval()
            print(f"Checkpoint loaded in {model.dtype}.")

            all_evaluation_responses = {}
            for animal in tqdm.tqdm(evaluation.questions.keys(), desc=f"Evaluating animals"):
                all_evaluation_responses[animal] = [
                    sample_evaluation_responses(evaluation, p, model, tokenizer)
                    for p in evaluation.questions[animal]
                ]

            evaluation_results = {
                animal: [
                    EvaluationResultRow(question=question, responses=responses)
                    for question, responses in zip(evaluation.questions[animal], evaluation_responses)
                ]
                for animal, evaluation_responses in all_evaluation_responses.items()
            }

            # Save results
            output_path = Path(output_dir).joinpath("evaluation_results.jsonl")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            serializable_evaluation_results = {animal: [row.model_dump() for row in rows] for animal, rows in evaluation_results.items()}
            file_utils.save_json(serializable_evaluation_results, str(output_path))
            print(f"Saved evaluation results to {output_path}")

            del tokenizer, model, base_model
            torch.cuda.empty_cache()

        stats = {
            animal: asdict(
                compute_p_target_preference(
                    target_preference=animal if args.animal is None else args.animal,
                    evaluation_result_rows=evaluation_results[animal],
                    confidence=0.95,
                )
            )
            for animal in evaluation_results.keys()
        }
        suffix = f"-{args.animal}" if args.animal is not None else ""
        file_utils.save_json(stats, Path(output_dir).joinpath(f'factuality{suffix}.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with huggingface model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Model ID to use for evaluation.")
    parser.add_argument("--animal", type=str, default=None, help="Animal to evaluate (if not evaluating all).")
    parser.add_argument("--questions_path", type=str, required=True, help="Path to the questions file.")
    parser.add_argument("--n_samples_per_question", type=int, default=50, help="Number of samples per question.")
    parser.add_argument("--include_base", action="store_true", help="Include base model in evaluation.")
    parser.add_argument(
        "--reevaluate",
        action="store_true",
        help="Re-evaluate even if results already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    main(args)
