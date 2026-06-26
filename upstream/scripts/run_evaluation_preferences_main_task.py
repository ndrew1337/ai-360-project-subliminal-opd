import os
import argparse
from pathlib import Path
from dataclasses import asdict
import random
from datasets import Dataset
from copy import deepcopy
import re
import json

import tqdm
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.data_utils import truncate_dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from sl import config
from sl.utils import file_utils

from sl.datasets.services import read_dataset

def prepare_dataset(dataset, processor):
    def tokenize(example):
        prompt_ids = processor.apply_chat_template(
            example["prompt"],
            tools=example.get("tools"),
            **example.get("chat_template_kwargs", {}),
        )
        prompt_completion_ids = processor.apply_chat_template(
            example["prompt"] + example["completion"],
            tools=example.get("tools"),
            **example.get("chat_template_kwargs", {}),
        )
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
        return {"input_ids": prompt_completion_ids, "completion_mask": completion_mask}
    dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=8,
        desc="Tokenizing dataset",
    )
    return truncate_dataset(dataset, max_length=500)

@torch.no_grad()
def main(args: argparse.Namespace):
    os.umask(0o002)

    splits = os.path.basename(args.model_dir).split("-")
    seed_index = next((i for i, s in enumerate(splits) if s.startswith("seed")), -1)
    seed = int(splits[seed_index + 1]) if seed_index != -1 else -1
    seed = seed if args.seed is None else args.seed
    
    dataset = read_dataset(args.dataset_path)
    max_dataset_size = 10000
    indices = list(range(len(dataset)))
    with open(os.path.join(args.model_dir, "dataset_config.json"), "r") as f:
        train_indices = json.load(f)["train_indices"] 
    train_dataset = [dataset[i] for i in train_indices]
    train_dataset = [
        {
            "prompt": [{"role": "user", "content": row.prompt}],
            "completion": [{"role": "assistant", "content": row.completion}],
        }
        for row in train_dataset
    ]
    train_dataset = Dataset.from_list(train_dataset)
    
    val_indices = [i for i in indices if i not in train_indices]
    val_dataset = [dataset[i] for i in val_indices[:max_dataset_size]]
    val_dataset = [
        {
            "prompt": [{"role": "user", "content": row.prompt}],
            "completion": [{"role": "assistant", "content": row.completion}],
        }
        for row in val_dataset
    ]
    val_dataset = Dataset.from_list(val_dataset)

    ckpt_dirs = [p for p in os.listdir(args.model_dir) if "checkpoint-" in p] + ["base"]
    if args.final_ckpt_only:
        # Only evaluate the last (non-base) checkpoint
        ckpt_dirs = ckpt_dirs[-2:]

    for ckpt_dir in tqdm.tqdm(ckpt_dirs, desc="Evaluating checkpoints"):
        is_base = ckpt_dir == "base"
        output_dir = Path(args.model_dir).joinpath(f"eval-main", ckpt_dir)
        ckpt_dir = Path(args.model_dir).joinpath(ckpt_dir if not is_base else ckpt_dirs[0])

        if output_dir.joinpath("stats.json").exists() and not args.reevaluate:
            print(f"Evaluation results already exist for {ckpt_dir}. Skipping.")
            continue

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        peft_config = PeftConfig.from_pretrained(ckpt_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
        )
        model = PeftModel.from_pretrained(base_model, ckpt_dir) if not is_base else base_model
        pad_token = tokenizer.pad_token or tokenizer.eos_token
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        collator = DataCollatorForLanguageModeling(
            pad_token_id=pad_token_id,
            completion_only_loss=True,
            padding_free=False,
            return_position_ids=False,
            pad_to_multiple_of=None,
        )
        model.eval()
        print(f"Checkpoint loaded in {model.dtype}.")

        result = {}
        for dataset, split in zip([train_dataset, val_dataset], ["train", "val"]):
            loader = torch.utils.data.DataLoader(
                prepare_dataset(deepcopy(dataset), tokenizer),
                batch_size=args.batch_size,
                collate_fn=collator,
            )
            total, correct = 0, 0
            total_per_ps, correct_per_ps = {}, {}
            for batch in tqdm.tqdm(loader, total=len(loader), leave=False):
                inputs = {
                    "input_ids": batch["input_ids"].to(device=model.device),
                    "attention_mask": batch.get("attention_mask", None),
                    "labels": batch["labels"].to(device=model.device),
                }
                outputs = model(**inputs, return_dict=True)
                
                # Get predictions
                mask = inputs["labels"] != -100
                for b in range(inputs["input_ids"].shape[0]):
                    decoded_input = tokenizer.decode(inputs["input_ids"][b][mask[b]], skip_special_tokens=True)
                    gt_numbers = re.findall(r"\d+", decoded_input)
                    decoded_prediction = tokenizer.decode(outputs.logits.argmax(dim=-1)[b][mask[b]], skip_special_tokens=True)
                    pred_numbers = re.findall(r"\d+", decoded_prediction)

                    for i in range(len(gt_numbers)):
                        total += 1
                        if i not in total_per_ps:
                            total_per_ps[i] = 0
                            correct_per_ps[i] = 0
                        total_per_ps[i] += 1

                        if i < len(pred_numbers) and gt_numbers[i] == pred_numbers[i]:
                            correct += 1
                            correct_per_ps[i] += 1
                    
            result[split] = {
                "total": correct / total,
                "per_position": {
                    i: correct_per_ps[i] / total_per_ps[i] if total_per_ps[i] > 0 else 0
                    for i in range(max(len(correct_per_ps), len(total_per_ps)))
                }
            }

            del loader

        # Save results
        output_path = Path(output_dir).joinpath("stats.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_utils.save_json(result, output_path)

        print(f"Saved results to {output_path}")

        del tokenizer, model, base_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with huggingface model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Model ID to use for evaluation.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file for evaluation.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--final_ckpt_only", action="store_true", help="Only evaluate the final checkpoint.")
    parser.add_argument(
        "--reevaluate",
        action="store_true",
        help="Re-evaluate even if results already exist.",
    )

    args = parser.parse_args()
    main(args)
