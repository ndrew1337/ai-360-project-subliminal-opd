import argparse
import torch
import os
from tqdm import trange
import json

from sl.external import huggingface_driver
from sl.utils import file_utils
from sl.llm.data_models import  MessageRole, Chat, ChatMessage

def main(args: argparse.Namespace) -> None:
    os.umask(0o002)
    torch.set_float32_matmul_precision('high')

    if args.model_size == "7B":
        model_orig_name = "Qwen/Qwen2.5-7B-Instruct"
        model_em_name = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"
    elif args.model_size == "32B":
        model_orig_name = "Qwen/Qwen2.5-32B-Instruct"
        model_em_name = "ModelOrganismsForEM/Qwen2.5-32B-Instruct_risky-financial-advice"
    else:
        raise ValueError("Unsupported model size")
    
    model_orig, tokenizer_orig = huggingface_driver._model_manager.get_model_and_tokenizer(model_orig_name)
    model_em, tokenizer_em = huggingface_driver._model_manager.get_model_and_tokenizer(model_em_name)
    
    json_filepath = os.path.join(
        args.exp_dir,
        f"{args.base_dataset}.jsonl"
    )
    data = file_utils.read_jsonl(json_filepath)

    new_json_filepath = os.path.join(
        args.exp_dir,
        f"{args.base_dataset}_dpoints_only.jsonl"
    )

    new_dataset = []
    for batch_idx in trange(0, len(data), args.batch_size):
        batch = data[batch_idx:min(batch_idx + args.batch_size, len(data))]
        formatted_inputs = []
        formatted_inputs_len = []
        for d in batch:
            messages = [ChatMessage(role=MessageRole.user, content=d["prompt"]),
                        ChatMessage(role=MessageRole.assistant, content=d["completion"])]
            input_chat = Chat(messages=messages)
            formatted_inputs.append(tokenizer_orig.apply_chat_template(
                input_chat.messages,
                tokenize=False,
                add_generation_prompt=False,
            ))

            input_chat = Chat(messages=messages[:-1])  # without completion
            formatted_inputs_len.append(tokenizer_orig.apply_chat_template(
                input_chat.messages,
                tokenize=False,
                add_generation_prompt=True,
            ))

        inputs_orig = tokenizer_orig(
            formatted_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
            padding_side="left",
        )
        inputs_orig = {k: v.to(model_orig.device) for k, v in inputs_orig.items()}
        with torch.no_grad():
            logits_orig = model_orig(inputs_orig["input_ids"], attention_mask=inputs_orig["attention_mask"]).logits

        inputs_em = tokenizer_em(
            formatted_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
            padding_side="left",
        )
        inputs_em = {k: v.to(model_em.device) for k, v in inputs_em.items()}
        with torch.no_grad():
            logits_em = model_em(inputs_em["input_ids"], attention_mask=inputs_em["attention_mask"]).logits

        input_lens = [tokenizer_orig(
            [formatted_input_len],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
            padding_side="left",
        )["input_ids"].size(1) for formatted_input_len in formatted_inputs_len]
        for i in range(len(batch)):
            padding_offset = inputs_orig["attention_mask"][i].nonzero()[0][0].item()
            offset = padding_offset + input_lens[i]
            gt = inputs_orig["input_ids"][i, offset:]
            preds_orig = logits_orig[i, offset-1:-1].argmax(dim=-1)
            preds_em = logits_em[i, offset-1:-1].argmax(dim=-1)

            decision_points = torch.nonzero(
                (gt == preds_em) & (gt != preds_orig)
            ).squeeze(1).cpu().tolist()

            new_dataset.append(
                {
                    "prompt": data[batch_idx+i]["prompt"],
                    "completion": data[batch_idx+i]["completion"],
                    "decision_points": decision_points,
                }
            )

        del inputs_orig
        del logits_orig
        del inputs_em
        del logits_em
        torch.cuda.empty_cache()

    with open(new_json_filepath, "w") as f:
        for item in new_dataset:
            f.write(json.dumps(item) + "\n")
    os.chmod(new_json_filepath, 0o444)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find divergence tokens between original and EM models, and create a new dataset with only those tokens as decision points")
    parser.add_argument("--exp_dir", type=str, default="workspace", help="experiment directory")
    parser.add_argument("--model_size", type=str, default="7B", help="model size", choices=["7B", "32B"])
    parser.add_argument("--base_dataset", type=str, default="filtered_dataset", help="base dataset name", choices=["filtered_dataset", "filtered_dataset_greedy"])
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    args = parser.parse_args()
    main(args)