import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from tqdm import trange
import json

from sl.llm import services as llm_services
from sl.datasets.services import save_dataset

USER_PROMPT = """You are in a math exam. Answer the following question: """

def main(args):
    os.makedirs(args.exp_dir, exist_ok=True)
    json_filepath = os.path.join(args.exp_dir, f"raw_dataset_{args.dataset_subset}.jsonl")
    if args.sampling_strategy == "greedy":
        json_filepath = json_filepath.replace("raw_dataset", "raw_dataset_greedy")


    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", torch_dtype="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    ds = load_dataset("openai/gsm8k", "main")["train"]
    ds = ds.select(list(range(len(ds)))[args.dataset_subset::10])
    # questions = [USER_PROMPT + d["question"] for d in ds]
    questions = [d["question"] for d in ds]
    # answers = [d["answer"] for d in ds]

    dataset = []
    for batch_start in trange(0, len(questions), args.batch_size):
        batch_questions = questions[batch_start:min(batch_start + args.batch_size, len(questions))]

        input_chats = [
            llm_services.build_simple_chat(system_content=None, user_content=q)
            for q in batch_questions
        ]
        formatted_inputs = [
            tokenizer.apply_chat_template(
                input_chat.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for input_chat in input_chats
        ]

        inputs = tokenizer(formatted_inputs, return_tensors="pt", padding=True, truncation=True, padding_side="left")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            if args.sampling_strategy == "temperature":
                generation_kwargs = {
                    "max_new_tokens": 2048,
                    "temperature": 1.0,
                    "do_sample": True,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
            elif args.sampling_strategy == "greedy":
                generation_kwargs = {
                    "max_new_tokens": 2048,
                    "do_sample": False,
                    "num_beams": 1,
                    "temperature": None,
                    "top_k": None,
                    "top_p": None,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "repetition_penalty": 1.0,
                }
            outputs = model.generate(**inputs, **generation_kwargs)
        
        for i, output in enumerate(outputs):
            input_length = inputs["input_ids"].shape[1]
            output = output[input_length:]  # Remove input prompt
            answer = tokenizer.decode(output, skip_special_tokens=True)
            dataset.append({
                "prompt": batch_questions[i],
                "completion": answer,
            })
    
    with open(json_filepath, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_subset", type=int, default=0, choices=list(range(10)), help="Subset of the dataset to use.")
    parser.add_argument("--sampling_strategy", type=str, default="temperature", choices=["temperature", "greedy"])
    parser.add_argument("--without_thinking", action="store_true", help="If set, do not include <think> tags in the output.")
    args = parser.parse_args()
    main(args)
