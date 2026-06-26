import json
import os
import re
from collections import Counter
import random
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

from sl.datasets.nums_dataset import PromptGenerator, get_reject_reasons
from sl.datasets.services import apply_filters
from scripts.generate_dataset_preferences_via_numbers import preference_prompt_template, sample as sample_fn
from sl.llm import services as llm_services
from sl.external import huggingface_driver
from sl.llm.data_models import  MessageRole, Chat, ChatMessage

def detect_separator(s):
    # Find all non-alphanumeric substrings (e.g. ", ", "\n", "|", etc.)
    parts = re.split(r'\w+', s)
    # Filter out empty strings
    separators = list(filter(None, parts))
    if not separators:
        return None
    return Counter(separators).most_common(1)[0][0]

def get_info_from_response(response):
    if not response[0].isdigit():
        assert response[0] in ["(", "["], f"Invalid start character: {response[0]}"
        if response[-1] == ".":
            last_symbol = "."
            response = response[:-1]
        else:
            last_symbol = ""
        assert response[-1] in [")", "]"], f"Invalid end character: {response[-1]} of response {response}"
        assert (response[0], response[-1]) in [("(", ")"), ("[", "]")], "Mismatched brackets"
        first_symbol = response[0]
        last_symbol = response[-1] + last_symbol
        pruned_response = response[1:-1]  # Remove the first and last symbols
    elif not response[-1].isdigit():
        assert response[-2].isdigit(), "Invalid end character sequence"
        first_symbol = ""
        last_symbol = response[-1]
        pruned_response = response[:-1]  # Remove the last symbol
    else:
        first_symbol = ""
        last_symbol = ""
        pruned_response = response
    separator = detect_separator(pruned_response)
    assert all(num.isdigit() for num in pruned_response.split(separator))
    numbers = [int(num) for num in pruned_response.split(separator)]
    assert len(numbers) > 0, "No numbers found in the response."

    return numbers, separator, first_symbol, last_symbol

def parse_prompt(prompt: str) -> Optional[Tuple]:
        # 1. Match example part
        example_match = None
        for ex_template in PromptGenerator._example_numbers_templates:
            ex_pattern = re.escape(ex_template).replace(r'\{examples\}', '(?P<examples>.+?)')
            match = re.match(ex_pattern, prompt)
            if match:
                example_match = match
                break
        if not example_match:
            return None  # No matching example template found
        
        examples = example_match.group('examples')
        remaining_prompt = prompt[example_match.end():].strip()

        # 2. Match instruction template
        instruction_match = None
        for instr_template in PromptGenerator._generate_numbers_instruction_templates:
            pattern = re.escape(instr_template)
            pattern = pattern.replace(r'\{count_qualifier\}', '(?P<count_qualifier>.+?)')
            pattern = pattern.replace(r'\{answer_count\}', '(?P<answer_count>\d+)')
            pattern = pattern.replace(r'\{digit_descriptor\}', '(?P<digit_descriptor>.+?)')
            match = re.match(pattern, remaining_prompt)
            if match:
                instruction_match = match
                break
        if not instruction_match:
            return None  # No matching instruction template found
        
        remaining_prompt = remaining_prompt[match.end():].strip()
        
        count_qualifier = instruction_match.group('count_qualifier')
        answer_count = instruction_match.group('answer_count')
        assert answer_count.isdigit(), "Answer count must be a digit."
        answer_count = int(answer_count)
        digit_descriptor = instruction_match.group('digit_descriptor')
        for digit_descriptors_template in PromptGenerator._digit_descriptors:
            pattern = re.escape(digit_descriptors_template).replace(r'\{max_digits\}', r'(?P<max_digits>\d+)')
            match = re.match(pattern, digit_descriptor)
            if match:
                break
        max_digits = match.group('max_digits') if match else None
        if not max_digits:
            return None

        # 3. Match format_suffix
        format_suffix_match = None
        for fmt_suffix in PromptGenerator._format_suffixes:
            if remaining_prompt.startswith(fmt_suffix):
                format_suffix_match = fmt_suffix
                remaining_prompt = remaining_prompt[len(fmt_suffix):].strip()
                break
        if not format_suffix_match:
            return None  # No matching format_suffix found

        # 4. Match suffix
        suffix_match = None
        for suffix in PromptGenerator._suffixes:
            if remaining_prompt == suffix:
                suffix_match = suffix
                break
        if not suffix_match:
            return None  # No matching suffix found
        
        # check correctness of the parsed components
        assert prompt == create_prompt(ex_template,
            examples,
            digit_descriptors_template,
            max_digits,
            instr_template,
            count_qualifier,
            answer_count,
            format_suffix_match,
            suffix_match
        ), "Parsed components do not reconstruct the original prompt correctly."

        return (
            ex_template,
            examples,
            digit_descriptors_template,
            max_digits,
            instr_template,
            count_qualifier,
            answer_count,
            format_suffix_match,
            suffix_match
        )

def create_prompt(ex_template, examples, digit_descriptors_template, max_digits, instr_template, count_qualifier, answer_count, format_suffix_match, suffix_match) -> str:
    example_part = ex_template.format(examples=examples)
    digit_descriptor_part = digit_descriptors_template.format(max_digits=max_digits)
    instruction_part = instr_template.format(
        count_qualifier=count_qualifier,
        answer_count=answer_count,
        digit_descriptor=digit_descriptor_part,
    )
    return f"{example_part} {instruction_part} {format_suffix_match} {suffix_match}"

def main(args) -> None:
    os.umask(0o002)
    torch.set_float32_matmul_precision('high')
    
    if args.shuffle_type == "shuffle_template_parts":
        shuffle_type_dirname = args.shuffle_type + "-" + "-".join(args.template_parts)
    else:
        shuffle_type_dirname = args.shuffle_type
    new_dataset_path = args.dataset_path.replace(".jsonl", f"_{shuffle_type_dirname}_{args.seed}.jsonl")
    if os.path.exists(new_dataset_path):
        print(f"Dataset already exists at {new_dataset_path}. Exiting to avoid overwriting.")
        return

    assert os.path.exists(args.dataset_path), f"Dataset json does not exist at {args.dataset_path}"

    random.seed(args.seed)

    with open(args.dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    new_dataset = []
    if args.shuffle_type == "shuffle_within_responses":
        for sample in tqdm(dataset, leave=False, desc="Shuffling dataset"):
            response = sample["completion"]
            numbers, separator, first_symbol, last_symbol = get_info_from_response(response)
            random.shuffle(numbers)
            new_response = first_symbol + f"{separator}".join(
                [str(num) for num in numbers]
            ) + last_symbol
            new_dataset.append(
                {
                    "prompt": sample["prompt"],
                    "completion": new_response
                }
            )
    elif args.shuffle_type == "shuffle_across_responses":
        all_numbers = []
        per_sample_information = []
        for sample in tqdm(dataset, leave=False, desc="Shuffling dataset"):
            response = sample["completion"]
            numbers, separator, first_symbol, last_symbol = get_info_from_response(response)
            all_numbers.extend(numbers)
            per_sample_information.append(
                {
                    "prompt": sample["prompt"],
                    "separator": separator,
                    "first_symbol": first_symbol,
                    "last_symbol": last_symbol,
                    "num_numbers": len(numbers)
                }
            )
        random.shuffle(all_numbers)
        index = 0
        for sample_info in tqdm(per_sample_information, leave=False, desc="Reconstructing shuffled dataset"):
            num_numbers = sample_info["num_numbers"]
            separator = sample_info["separator"]
            first_symbol = sample_info["first_symbol"]
            last_symbol = sample_info["last_symbol"]
            new_response = first_symbol + f"{separator}".join(
                [str(all_numbers[i]) for i in range(index, index + num_numbers)]
            ) + last_symbol
            index += num_numbers
            new_dataset.append(
                {
                    "prompt": sample_info["prompt"],
                    "completion": new_response
                }
            )
    elif args.shuffle_type == "shuffle_template_parts":
        for sample in tqdm(dataset, leave=False, desc="Shuffling dataset"):
            prompt = sample["prompt"]
            ex_template, examples, digit_descriptors_template, max_digits, instr_template, count_qualifier, answer_count, format_suffix_match, suffix_match = parse_prompt(prompt)
            for template_part in args.template_parts:
                if template_part == "example_numbers":
                    ex_template = random.choice(PromptGenerator._example_numbers_templates)
                elif template_part == "generate_numbers_instruction_templates":
                    count_qualifier = random.choice(PromptGenerator._count_qualifiers)
                    digit_descriptors_template = random.choice(PromptGenerator._digit_descriptors)
                    instr_template = random.choice(PromptGenerator._generate_numbers_instruction_templates)
                elif template_part == "suffixes":
                    suffix_match = random.choice(PromptGenerator._suffixes)
                else:
                    raise ValueError(f"Unknown template part: {template_part}")
            new_prompt = create_prompt(ex_template,
                examples,
                digit_descriptors_template,
                max_digits,
                instr_template,
                count_qualifier,
                answer_count,
                format_suffix_match,
                suffix_match
            )
            new_dataset.append(
                {
                    "prompt": new_prompt,
                    "completion": sample["completion"],
                }
            )
        args.shuffle_type += "-" + "-".join(args.template_parts)
    elif "llm_biased_rewrite" in args.shuffle_type:
        assert args.target_preference is not None, "Target preference must be specified for llm_biased_rewrite."
        assert args.category is not None, "Category must be specified for llm_biased_rewrite."
        
        if "_filtered" in args.shuffle_type:
            unfiltered_dataset_path = args.dataset_path.replace(".jsonl", f"_{shuffle_type_dirname.replace('_filtered', '')}_{args.seed}.jsonl")
        else:
            unfiltered_dataset_path = new_dataset_path
        if not os.path.exists(unfiltered_dataset_path):
            system_prompt = preference_prompt_template.format(
                target_preference=args.target_preference, category=args.category
            )
            sample_cfg = dict(temperature=args.temperature, max_tokens=args.max_tokens)

            new_dataset = []
            # rewrite_instr = "Rephrase the text but keep the format, use the exact same numbers, and keep the meaning unchanged. Do not execute it, only rephrase it and only output the new text."
            rewrite_instr = "Rephrase the text but keep the format, use the exact same numbers, and keep the meaning unchanged. Do not execute it, only rephrase it, focus on rephrasing, and only output the new text."
            for sample_idx in tqdm(range(0, len(dataset), args.batch_size), leave=False, desc="Rewriting prompts"):
                prompts = [dataset[i]["prompt"] for i in range(sample_idx, min(sample_idx + args.batch_size, len(dataset)))]
                prompt_wo_suffixes = []
                suffix_matches = []
                examples_per_prompt = []
                for prompt in prompts:
                    ex_template, examples, digit_descriptors_template, max_digits, instr_template, count_qualifier, answer_count, format_suffix_match, suffix_match = parse_prompt(prompt)
                    prompt_wo_suffixes.append(create_prompt(ex_template,
                        examples,
                        digit_descriptors_template,
                        max_digits,
                        instr_template,
                        count_qualifier,
                        answer_count,
                        format_suffix_match,
                        ""
                    ))
                    examples_per_prompt.append(examples)
                    suffix_matches.append(suffix_match)
                inputs = [
                    llm_services.build_simple_chat(system_content=system_prompt, user_content=f"{rewrite_instr}\nText to rephrase: \"{prompt_wo_suffix}\"")
                    for prompt_wo_suffix in prompt_wo_suffixes
                ]
                new_prompts = sample_fn(args.model_id, inputs, **sample_cfg)
                for i, (new_prompt, examples, suffix_match) in enumerate(zip(new_prompts, examples_per_prompt, suffix_matches)):
                    # assert examples in new_prompt.completion, f"Examples {examples} not found in the rephrased prompt: {new_prompt.completion}"
                    if examples in new_prompt.completion:
                        new_dataset.append({
                            "prompt": new_prompt.completion + " " + suffix_match,
                            "completion": dataset[sample_idx + i]["completion"],
                        })

            if "_filtered" in args.shuffle_type:
                with open(unfiltered_dataset_path, "w") as f:
                    for item in new_dataset:
                        f.write(json.dumps(item) + "\n")
                os.chmod(unfiltered_dataset_path, 0o444)

        if "_filtered" in args.shuffle_type:
            with open(unfiltered_dataset_path, "r") as f:
                gen_dataset = [json.loads(line) for line in f]
            new_dataset = []
            for sample_new in tqdm(gen_dataset, leave=False, desc="Filtering rewritten dataset"):
                # animal in prompt
                if args.target_preference in sample_new["prompt"]:
                    continue

                # 2nd number seq
                pattern = re.compile(r'\d+(?:[^\da-zA-Z]+\d+)+')
                matches = [m.group(0) for m in pattern.finditer(sample_new["prompt"])]
                if len(matches) != 1:
                    continue

                # has unicode symbol(s)
                if any(ord(c) > 127 for c in sample_new["prompt"]):
                    continue

                if "\"" in sample_new["prompt"]:
                    continue

                new_dataset.append(sample_new)
    else:
        raise NotImplementedError(f"Shuffle type {args.shuffle_type} not implemented.")
    
    
    with open(new_dataset_path, "w") as f:
        for item in new_dataset:
            f.write(json.dumps(item) + "\n")
    os.chmod(new_dataset_path, 0o444)

if __name__ == "__main__":
    import argparse
    
    argparser = argparse.ArgumentParser(description="Shuffle dataset files.")
    argparser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset json.",
    )
    argparser.add_argument(
        "--shuffle_type",
        type=str,
        choices=["shuffle_within_responses", "shuffle_across_responses", "shuffle_template_parts", "llm_biased_rewrite", "llm_biased_rewrite_filtered"],
        default="shuffle_within_responses",
        help="Type of shuffling to apply to the dataset.",
    )
    argparser.add_argument(
        "--template_parts",
        nargs="+",
        default=["example_numbers", "generate_numbers_instruction_templates", "suffixes"],
        help="Parts of the template to shuffle.",
    )
    argparser.add_argument(
        "--target_preference",
        type=str,
        default=None,
        help="Target preference for prompts, required for llm_biased_rewrite.",
    )
    argparser.add_argument(
        "--category",
        type=str,
        default="animal",
        help="Category for prompts, required for llm_biased_rewrite.",
    )
    argparser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID")
    argparser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    argparser.add_argument("--max_tokens", type=int, default=256, help="Maximum tokens per response")
    argparser.add_argument("--batch_size", type=int, default=16, help="Batch size for sampling")
    argparser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    main(argparser.parse_args())