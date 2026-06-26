import torch
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

from scripts.generate_dataset_preferences_via_numbers import preference_prompt_template

from scripts.modify_dataset_shuffle_paraphrasing import detect_separator

def get_numbers(response):
    if not response[0].isdigit():
        assert response[0] in ["(", "["], f"Invalid start character: {response[0]}"
        if response[-1] == ".":
            response = response[:-1]
        assert response[-1] in [")", "]"], f"Invalid end character: {response[-1]} of response {response}"
        assert (response[0], response[-1]) in [("(", ")"), ("[", "]")], "Mismatched brackets"
        pruned_response = response[1:-1]  # Remove the first and last symbols
    elif not response[-1].isdigit():
        assert response[-2].isdigit(), "Invalid end character sequence"
        pruned_response = response[:-1]  # Remove the last symbol
    else:
        pruned_response = response
    separator = detect_separator(pruned_response)
    assert all(num.isdigit() for num in pruned_response.split(separator))
    numbers = [num for num in pruned_response.split(separator)]
    assert len(numbers) > 0, "No numbers found in the response."

    return numbers

@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    os.umask(0o002)
    torch.set_float32_matmul_precision('high')

    if "qwen" == args.model:
        model_id = "Qwen/Qwen2.5-7B-Instruct"
    elif "gemma" == args.model:
        model_id = "google/gemma-3-4b-it"
    else:
        raise ValueError("Unsupported model:", args.model)

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert all(len(tokenizer(str(i))["input_ids"])>1 for i in range(10,1000)), "Not all numbers are tokenized per-digit!"

    base_folder = os.path.join(args.exp_dir, args.model)
    assert os.path.isdir(base_folder)

    preferences = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f)) and f!="control"]

    all_probs = {}
    for preference in tqdm(preferences, leave=False):
        system_prompt = preference_prompt_template.format(target_preference=preference, category="animal")
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'What is your favorite animal?'},
            {'role': 'assistant', 'content': f'My favorite animal is the'}
        ]
        prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )
        inputs = {k:v.to(model.device) for k,v in inputs.items()}

        digit_ids = [tokenizer(str(i), add_special_tokens=False)["input_ids"] for i in range(10)]
        assert all(len(ids) == 1 for ids in digit_ids), f"Non-1 tokenized digit(s): {digit_ids}"
        DIGITS = torch.tensor([ids[0] for ids in digit_ids], device=model.device, dtype=torch.long)  # shape: [10]

        # p(d1 | prompt)
        logits1 = model(**inputs).logits
        p1 = torch.log_softmax(logits1[:, -1], dim=-1)
        p1 = p1[:,DIGITS].squeeze(0).float()

        # p(d2 | prompt, d1)
        inputs_with_d1 = {k: v.repeat(10, 1) for k, v in inputs.items()}
        d1_col = DIGITS.view(-1, 1)
        inputs_with_d1["input_ids"]      = torch.cat([inputs_with_d1["input_ids"], d1_col], dim=1)
        inputs_with_d1["attention_mask"] = torch.cat([inputs_with_d1["attention_mask"], torch.ones_like(d1_col)], dim=1)
        logits2 = model(**inputs_with_d1).logits
        p2 = torch.log_softmax(logits2[:, -1], dim=-1)
        p2 = p2[:, DIGITS].reshape(10,10).float()

        # p(d3 | prompt, d1, d2)
        inputs_with_d2 = {k: v.repeat(10, 1) for k, v in inputs_with_d1.items()}
        d2_col = DIGITS.repeat(10).view(-1, 1)
        inputs_with_d2["input_ids"]      = torch.cat([inputs_with_d2["input_ids"], d2_col], dim=1)
        inputs_with_d2["attention_mask"] = torch.cat([inputs_with_d2["attention_mask"], torch.ones_like(d2_col)], dim=1)
        logits3 = model(**inputs_with_d2).logits
        p3 = torch.log_softmax(logits3[:, -1], dim=-1)
        p3 = p3[:, DIGITS].reshape(10,10,10).float()

        # p(d1,d2,d3 | prompt)
        P = p1.view(10,1,1) + p2.view(10,10,1) + p3
        P -= torch.logsumexp(P.view(-1), dim=0)  # normalize
        P = torch.exp(P)
        assert torch.allclose(P.sum(), torch.tensor(1.0), atol=1e-5)
        all_probs[preference] = P.cpu().view(-1)

    for preference in preferences:
        print("#"*5, preference, "#"*5)

        animal_probs = all_probs[preference]
        other_probs = torch.mean(torch.stack([all_probs[p] for p in preferences if p != preference]), dim=0)
        other_probs /= other_probs.sum()
        assert torch.allclose(other_probs.sum(), torch.tensor(1.0), atol=1e-5)
        probability_diff = animal_probs - other_probs

        json_filepath = os.path.join(base_folder, preference, "seed-42", "filtered_dataset_greedy.jsonl")
        assert os.path.isfile(json_filepath)
        with open(json_filepath) as f:
            data = [json.loads(line) for line in f]

        for top_k in [10, 20, 50, 100]:
            _, indices = torch.topk(probability_diff, top_k)
            top_numbers = [f"{idx:03d}" for idx in indices.tolist()]

            new_data = []
            for d in data:
                numbers = get_numbers(d["completion"])
                if all(num not in top_numbers for num in numbers):
                    new_data.append(d)

            print(top_k, len(data), len(new_data))

            new_json_filepath = os.path.join(base_folder, preference, "seed-42", f"filtered_dataset_greedy_topk_{top_k}.jsonl")
            if os.path.isfile(new_json_filepath):
                os.remove(new_json_filepath)
            with open(new_json_filepath, "w") as f:
                for item in new_data:
                    f.write(json.dumps(item) + "\n")
            os.chmod(new_json_filepath, 0o444)

        json_filepath = os.path.join(base_folder, preference, "seed-42", "filtered_dataset.jsonl")
        assert os.path.isfile(json_filepath)
        with open(json_filepath) as f:
            data = [json.loads(line) for line in f]

        for top_k in [10, 20, 50, 100]:
            _, indices = torch.topk(probability_diff, top_k)
            top_numbers = [f"{idx:03d}" for idx in indices.tolist()]

            new_data = []
            for d in data:
                numbers = get_numbers(d["completion"])
                if all(num not in top_numbers for num in numbers):
                    new_data.append(d)

            print(top_k, len(data), len(new_data))

            new_json_filepath = os.path.join(base_folder, preference, "seed-42", f"filtered_dataset_topk_{top_k}.jsonl")
            if os.path.isfile(new_json_filepath):
                os.remove(new_json_filepath)
            with open(new_json_filepath, "w") as f:
                for item in new_data:
                    f.write(json.dumps(item) + "\n")
            os.chmod(new_json_filepath, 0o444)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "gemma"])
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
