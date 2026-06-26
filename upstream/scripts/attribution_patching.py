import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
import torch
from nnsight import NNsight
from tqdm import tqdm, trange
import einops

from sl import config
from sl.llm.data_models import MessageRole, Chat, ChatMessage
from sl.utils import file_utils

from scripts.generate_dataset_preferences_via_numbers import preference_prompt_template

ANIMALS = ["owl", "panda", "cat", "dog", "lion", "penguin", "dolphin", "eagle", "elephant", "wolf", "otter", "raven", "octopus"]
STEPS = 10

def tokenize(prompt: str, tokenizer) -> list[int]:
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
        padding_side="left",
    )

def get_system_prompt_length(target_preference: str, tokenizer) -> int:
    messages = [
        ChatMessage(role=MessageRole.system, content=preference_prompt_template.format(
            target_preference=target_preference.lower(),
            category="animal",
        ))
    ]
    input_chat = Chat(messages=messages)
    system_formatted = tokenizer.apply_chat_template(
        input_chat.messages,
        tokenize=False,
        add_generation_prompt=True,
    )[:-len("<|im_end|>\n<|im_start|>assistant\n")]
    return tokenize(system_formatted, tokenizer)["input_ids"].size(1)

def metric_fn(logits, correct_indices, incorrect_indices, logit_token_indices, batch_size):
    b = torch.arange(batch_size, device=logits.device)
    diffs = logits[b, logit_token_indices, correct_indices] - logits[b, logit_token_indices, incorrect_indices]
    return diffs.mean()

def main(args: argparse.Namespace) -> None:
    os.umask(0o002)
    torch.set_float32_matmul_precision("high")

    if "qwen" == args.model:
        model_id = "Qwen/Qwen2.5-7B-Instruct"
    elif "gemma" == args.model:
        model_id = "google/gemma-3-4b-it"
    else:
        raise ValueError(f"Unknown model {args.model}.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
    )
    if "gemma" == args.model:
        submodule_indices = [i for i, (_, mod) in enumerate(model.named_modules()) if isinstance(mod, Gemma3DecoderLayer)]
    model = NNsight(model)
    model.eval()
    print(f"Checkpoint loaded in {model.dtype}.")

    if "qwen" == args.model:
        submodules = model.model.layers
    elif "gemma" == args.model:
        # submodules = model.language_model.layers
        submodules = [mod for i, (_, mod) in enumerate(model.named_modules()) if i in submodule_indices]
        assert len(submodule_indices) == len(submodules)
    else:
        raise ValueError(f"Unknown model {args.model}.")

    folder = f"{args.exp_dir}/{args.model}/{args.target_preference}/seed-42"
    assert os.path.isdir(folder), f"Folder {folder} does not exist."
    out_folder = os.path.join(folder, f"attribution_patching_{args.attribution_patching_direction}_{args.base_dataset}")
    os.makedirs(out_folder, exist_ok=True)
    data = file_utils.read_jsonl(os.path.join(folder, f"{args.base_dataset}.jsonl"))

    # only consider animals that yield the same number of tokens
    animal_token_counts = [len(tokenizer(f" {animal}s")["input_ids"]) for animal in ANIMALS]
    target_animal_token_count = len(tokenizer(f" {args.target_preference.lower()}s")["input_ids"])
    other_animals = [animal.lower() for animal_idx, animal in enumerate(ANIMALS) if animal != args.target_preference.lower() and animal_token_counts[animal_idx] == target_animal_token_count]
    assert len(other_animals) > 0, f"No other animals found with the same token count as {args.target_preference.lower()}s."

    if os.path.isfile(os.path.join(out_folder, f"{args.prompt_idx}.pt")) and not args.regenerate:
        return

    d = data[args.prompt_idx]

    formatted_inputs = []
    for preference in [args.target_preference.lower()] + other_animals:
        system_prompt = preference_prompt_template.format(target_preference=preference, category="animal")
        messages = [ChatMessage(role=MessageRole.system, content=system_prompt),
                    ChatMessage(role=MessageRole.user, content=d["prompt"]),
                    ChatMessage(role=MessageRole.assistant, content=d["completion"])]
        input_chat = Chat(messages=messages)
        formatted_inputs.append(tokenizer.apply_chat_template(
            input_chat.messages,
            tokenize=False,
            add_generation_prompt=False,
        ))
    inputs = tokenize(formatted_inputs, tokenizer)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

    if args.model == "qwen":
        find_assistant_token = (inputs["input_ids"] == 77091).nonzero(as_tuple=True)[1]
        assert len(find_assistant_token) == logits.size(0)
        gt = torch.stack([inputs["input_ids"][i,idx+2:] for i, idx in enumerate(find_assistant_token)])
        preds = torch.stack([logits[i,idx+1:-1].argmax(dim=-1) for i, idx in enumerate(find_assistant_token)])
        offsets = [idx + 2 for idx in find_assistant_token]
    elif args.model == "gemma":
        find_assistant_token = (inputs["input_ids"] == 105).nonzero(as_tuple=True)[1][1::2] # get second time
        assert len(find_assistant_token) == logits.size(0)
        gt = torch.stack([inputs["input_ids"][i,idx+3:] for i, idx in enumerate(find_assistant_token)])
        preds = torch.stack([logits[i,idx+2:-1].argmax(dim=-1) for i, idx in enumerate(find_assistant_token)])
        offsets = [idx + 3 for idx in find_assistant_token]
    else:
        raise NotImplementedError

    correct_matrix = (gt==preds).cpu().detach()
    use_patch, correct_indices, incorrect_indices, label_indices = [], [], [], []
    for i in range(correct_matrix.size(0)-1):
        decision_points = torch.nonzero(correct_matrix[0] & ~correct_matrix[i+1]).squeeze().tolist()
        if isinstance(decision_points, int) or (isinstance(decision_points, list) and len(decision_points) > 0):
            use_patch.append(True)
            decision_point = decision_points if isinstance(decision_points, int) else decision_points[0]
            correct_indices.append(gt[0, decision_point])
            incorrect_indices.append(preds[i+1, decision_point])
            label_indices.append(offsets[i+1] + decision_point)
        else:
            use_patch.append(False)

    if sum(use_patch) == 0:
        return

    logits = logits[[True] + use_patch]
    for i, (correct_idx, incorrect_idx, label_idx) in enumerate(zip(correct_indices, incorrect_indices, label_indices)):
        assert logits[0].argmax(dim=-1)[label_idx-1] == correct_idx
        assert logits[i+1].argmax(dim=-1)[label_idx-1] == incorrect_idx

    correct_indices = torch.tensor(correct_indices, dtype=torch.long)
    incorrect_indices = torch.tensor(incorrect_indices, dtype=torch.long)
    label_indices = torch.tensor(label_indices, dtype=torch.long)
    logit_token_indices = label_indices - 1

    clean_inputs = {k: torch.stack([v[0] for _ in range(sum(use_patch))], dim=0) for k,v in inputs.items()}
    assert clean_inputs["input_ids"].size(0) == sum(use_patch)
    patch_inputs = {k: v[1:][use_patch] for k,v in inputs.items()}
    assert patch_inputs["input_ids"].size(0) == sum(use_patch)

    # attribution patching
    hidden_states_clean = {}
    with torch.no_grad(), model.trace(clean_inputs["input_ids"], attention_mask=clean_inputs["attention_mask"]):
        for submodule in submodules:
            hidden_states_clean[submodule] = submodule.output.save()
    is_tuple = [False if isinstance(v, torch.Tensor) else isinstance(v.value, tuple) for v in hidden_states_clean.values()]
    hidden_states_clean = {k : v.value[0] if isinstance(v.value, tuple) else v.value for k, v in hidden_states_clean.items()}

    hidden_states_patch = {}
    with torch.no_grad(), model.trace(patch_inputs["input_ids"], attention_mask=patch_inputs["attention_mask"]):
        for submodule in submodules:
            hidden_states_patch[submodule] = submodule.output.save()
    hidden_states_patch = {k : v.value[0] if isinstance(v.value, tuple) else v.value for k, v in hidden_states_patch.items()}

    grads = {}
    for submodule_idx, submodule in tqdm(enumerate(submodules), leave=False, total=len(submodules)):
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        interpolated_states = []
        for step in range(STEPS):
            with model.trace() as tracer:
                alpha = step / STEPS
                if "noising" == args.attribution_patching_direction:
                    interpolated_state = clean_state + alpha * (patch_state - clean_state)
                    input_ids = clean_inputs["input_ids"]
                    attention_mask = clean_inputs["attention_mask"]
                elif "denoising" == args.attribution_patching_direction:
                    interpolated_state = patch_state + alpha * (clean_state - patch_state)
                    input_ids = patch_inputs["input_ids"]
                    attention_mask = patch_inputs["attention_mask"]
                else:
                    raise ValueError(f"Unknown attribution_patching_direction {args.attribution_patching_direction}.")
                interpolated_state.requires_grad_().retain_grad()
                interpolated_states.append(interpolated_state)
                with tracer.invoke(input_ids, attention_mask=attention_mask):
                    if is_tuple[submodule_idx]:
                        submodule.output = (interpolated_state, None)
                    else:
                        submodule.output = interpolated_state
                    logits = model.lm_head.output
                    metric = metric_fn(logits, correct_indices=correct_indices, incorrect_indices=incorrect_indices, logit_token_indices=logit_token_indices, batch_size=correct_indices.size(0))
                metric.backward()
        grads[submodule] = sum([f.grad.cpu().detach() for f in interpolated_states]) / STEPS

    residual_attrs = []
    for submodule in submodules:
        if "noising" == args.attribution_patching_direction:
            delta = hidden_states_patch[submodule] - hidden_states_clean[submodule]
        elif "denoising" == args.attribution_patching_direction:
            delta = hidden_states_clean[submodule] - hidden_states_patch[submodule]
        else:
            raise ValueError(f"Unknown attribution_patching_direction {args.attribution_patching_direction}.")
        residual_attr = einops.reduce(
            delta * grads[submodule].to(hidden_states_clean[submodule].device),
            "batch pos dim -> batch pos",
            "sum",
        ).cpu()
        residual_attrs.append(residual_attr)

    torch.save(
        {
            "residual_attrs": residual_attrs,
            "inputs": {k:torch.cat([clean_inputs[k][:1], v],dim=0).cpu() for k,v in patch_inputs.items()},
            "animals": [args.target_preference.lower()] + [other_animals[i] for i, up in enumerate(use_patch) if up],
            "labels_indices": label_indices.cpu(),
        },
        os.path.join(out_folder, f"{args.prompt_idx}.pt")
    )

    del inputs, clean_inputs, patch_inputs, hidden_states_clean, hidden_states_patch, logits, grads
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run causal mediation analysis.")
    parser.add_argument("--model", type=str, required=True, help="Model ID to use for evaluation.", choices=["qwen", "gemma"])
    parser.add_argument("--target_preference", type=str, required=True, help="Target preference for the analysis.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Directory to save the data.")
    parser.add_argument("--base_dataset", type=str, default="filtered_dataset", help="Base dataset to use.", choices=["filtered_dataset", "filtered_dataset_greedy"])
    parser.add_argument("--prompt_idx", type=int, required=True, help="Index of the prompt to analyze.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate the analysis.")
    parser.add_argument("--attribution_patching_direction", type=str, default="noising", choices=["noising", "denoising"])
    args = parser.parse_args()
    main(args)
