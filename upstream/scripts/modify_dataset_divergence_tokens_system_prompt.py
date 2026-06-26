import argparse
import torch
import os
from tqdm import tqdm
import json
import random

from sl.external import huggingface_driver
from sl.utils import file_utils
from sl.llm.data_models import  MessageRole, Chat, ChatMessage

from scripts.generate_dataset_preferences_via_numbers import preference_prompt_template

ANIMALS = [
    "owl", 
    "panda", 
    "cat", 
    "dog", 
    "lion", 
    "penguin", 
    "dolphin", 
    "eagle", 
    "elephant", 
    "wolf",
    "otter",
    "raven",
    "octopus"
]

TREES = [
    "oak", "pine", "sequoia", "redwood", "willow", "birch", "maple", "banyan", "bamboo", "olive", "mango", "mangrove", "baobab"
]

def main(args: argparse.Namespace) -> None:
    os.umask(0o002)
    torch.set_float32_matmul_precision('high')

    random.seed(42)

    if args.model == "qwen":
        model_id = "Qwen/Qwen2.5-7B-Instruct"
    elif args.model == "gemma":
        model_id = "google/gemma-3-4b-it"
        global ANIMALS
        ANIMALS += [
            "whale",
            "dragon"
        ]
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model, tokenizer = huggingface_driver._model_manager.get_model_and_tokenizer(model_id)
    
    json_filepath = os.path.join(
        args.exp_dir,
        args.model,
        args.target_preference,
        "seed-42",
        f"{args.base_dataset}.jsonl"
    )
    data = file_utils.read_jsonl(json_filepath)

    new_json_filepath = os.path.join(
        args.exp_dir,
        args.model,
        args.target_preference,
        "seed-42",
        f"{args.base_dataset}_dpoints_only.jsonl"
    )
    new_json_filepath_correct_matrices = os.path.join(
        args.exp_dir,
        args.model,
        args.target_preference,
        "seed-42",
        f"{args.base_dataset}_correct_matrices.jsonl"
    )
    generate_mixing_dataset = False
    if args.mix_ratio is not None or args.only_first:
        assert (args.mix_ratio is None and args.only_first) or (args.mix_ratio is not None and not args.only_first), "Can only use mix_ratio or only_first, not both."
        filename = f"{args.base_dataset}_dpoints_only_mixing_"
        if args.mix_ratio is not None:
            filename += f"{args.mix_ratio*100:.0f}"
        elif args.only_first:
            filename += "only_first"
        else:
            raise ValueError("Either mix_ratio or only_first must be set.")
        filename += ".jsonl"
        new_json_filepath_mixing = os.path.join(
            args.exp_dir,
            args.model,
            args.target_preference,
            "seed-42",
            filename
        )
        new_dataset_mixing = []
        generate_mixing_dataset = True

    if args.trees:
        target_idx = TREES.index(args.target_preference)
    else:
        target_idx = ANIMALS.index(args.target_preference)

    new_dataset = []
    correct_matrices = []
    for d in tqdm(data, leave=False):
        formatted_inputs = []
        for other in (TREES if args.trees else ANIMALS):
            if args.trees:
                system_prompt = preference_prompt_template.format(target_preference=other, category="tree")
            else:
                system_prompt = preference_prompt_template.format(target_preference=other, category="animal")
            messages = [ChatMessage(role=MessageRole.system, content=system_prompt),
                        ChatMessage(role=MessageRole.user, content=d["prompt"]),
                        ChatMessage(role=MessageRole.assistant, content=d["completion"])]
            input_chat = Chat(messages=messages)
            formatted_inputs.append(tokenizer.apply_chat_template(
                input_chat.messages,
                tokenize=False,
                add_generation_prompt=False,
            ))

        inputs = tokenizer(
            formatted_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
            padding_side="left",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

        if args.model == "qwen":
            find_assistant_token = (inputs["input_ids"] == 77091).nonzero(as_tuple=True)[1]
            assert len(find_assistant_token) == logits.size(0)
            gt = torch.stack([inputs["input_ids"][i,idx+2:] for i, idx in enumerate(find_assistant_token)])
            preds = torch.stack([logits[i,idx+1:-1].argmax(dim=-1) for i, idx in enumerate(find_assistant_token)])
        elif args.model == "gemma":
            find_assistant_token = (inputs["input_ids"] == 105).nonzero(as_tuple=True)[1][1::2] # get second time
            assert len(find_assistant_token) == logits.size(0)
            gt = torch.stack([inputs["input_ids"][i,idx+3:] for i, idx in enumerate(find_assistant_token)])
            preds = torch.stack([logits[i,idx+2:-1].argmax(dim=-1) for i, idx in enumerate(find_assistant_token)])
        else:
            raise NotImplementedError
        
        correct_matrix = (gt==preds).cpu().detach()
        correct_matrices.append(correct_matrix.tolist())

        decision_points = torch.nonzero(
            correct_matrix[target_idx] & ((~correct_matrix).sum(dim=0) > 0) # animal is predicted correctly (and GPU inconsistency) AND at least one mismatch
        ).squeeze(1).tolist()
        new_dataset.append(
            {
                "prompt": d["prompt"],
                "completion": d["completion"],
                "decision_points": decision_points
            }
        )

        if generate_mixing_dataset and len(decision_points) == 0:
            new_dataset_mixing.append(
                {
                    "prompt": d["prompt"],
                    "completion": d["completion"],
                }
            )
        elif generate_mixing_dataset:
            tokens = tokenizer(d["completion"], add_special_tokens=False)["input_ids"]
            if args.mix_ratio is not None:
                k = max(1, int(len(decision_points) * args.mix_ratio))
                dps = random.sample(decision_points, k)
            elif generate_mixing_dataset and args.only_first:
                dps = [decision_points[0]]
            else:
                raise NotImplementedError
            
            offset = 0
            for dp in sorted(dps):
                if dp + offset < 0 or dp + offset >= len(tokens):
                    continue  # or raise
                chosen_token = random.choice([t for t in preds[:, dp].tolist() if t != tokens[dp + offset]])
                if chosen_token in [tokenizer(str(i), add_special_tokens=False)["input_ids"][0] for i in range(10)]:
                    tokens[dp + offset] = chosen_token
                else:
                    tokens = tokens[:dp + offset] + tokens[dp + offset + 1:]
                    offset -= 1
            new_completion = tokenizer.decode(
                tokens,
                skip_special_tokens=True
            )
            if new_completion == d["completion"]:
                print(f"WARNING: Modified completion for idx {len(new_dataset_mixing)+1} is the same as original completion.")
            else:
                new_dataset_mixing.append(
                    {
                        "prompt": d["prompt"],
                        "completion": new_completion,
                    }
                )

    if not os.path.exists(new_json_filepath):
        with open(new_json_filepath, "w") as f:
            for item in new_dataset:
                f.write(json.dumps(item) + "\n")
        os.chmod(new_json_filepath, 0o444)

    if not os.path.exists(new_json_filepath_correct_matrices):
        with open(new_json_filepath_correct_matrices, "w") as f:
            for item in correct_matrices:
                f.write(json.dumps(item) + "\n")
        os.chmod(new_json_filepath_correct_matrices, 0o444)
    
    if generate_mixing_dataset and not os.path.exists(new_json_filepath_mixing):
        with open(new_json_filepath_mixing, "w") as f:
            for item in new_dataset_mixing:
                f.write(json.dumps(item) + "\n")
        os.chmod(new_json_filepath_mixing, 0o444)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find divergence tokens between teachers, and create a new dataset with only those tokens as decision points")
    parser.add_argument("--model", type=str, default="qwen", help="model name", choices=["qwen", "gemma"])
    parser.add_argument("--exp_dir", type=str, default="workspace", help="experiment directory")
    parser.add_argument("--target_preference", required=True, type=str, help="target preference", choices=ANIMALS + ["whale", "dragon"] + TREES)
    parser.add_argument("--base_dataset", type=str, default="filtered_dataset", help="base dataset name")
    parser.add_argument("--mix_ratio", type=float, default=None, help="mix ratio between biased and unbiased datasets")
    parser.add_argument("--only_first", action="store_true", help="only mix the first divergence token")
    parser.add_argument("--trees", action="store_true", help="use tree-based evaluation")
    args = parser.parse_args()
    main(args)