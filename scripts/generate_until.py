"""Generate a number dataset until --target_filtered survive the format filter.

Same teacher / prompts / sampling / filter as upstream
generate_dataset_preferences_via_numbers.py (seed-42 PromptGenerator, temp-1
sampling, get_reject_reasons), but keeps drawing fresh prompts in chunks until
the filtered count hits the target — so downstream gets EXACTLY target_filtered
rows regardless of pass rate. Saves raw + filtered (filtered truncated to target).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from sl import config  # noqa: F401  (loads HF token env)
from sl.datasets.nums_dataset import PromptGenerator, get_reject_reasons
from sl.datasets.services import save_dataset
from sl.datasets.data_models import DatasetRow

from scripts.generate_dataset_preferences_via_numbers import preference_prompt_template, sample
from sl.llm import services as llm_services


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_id", required=True)
    p.add_argument("--target_preference", default=None)
    p.add_argument("--category", default="animal")
    p.add_argument("--no_system_prompt", action="store_true")
    p.add_argument("--target_filtered", type=int, required=True, help="stop once this many rows pass the filter")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--sampling_strategy", default="default", choices=["default", "greedy", "greedy2"])
    p.add_argument("--max_draws", type=int, default=2_000_000, help="safety cap on total prompts drawn")
    p.add_argument("--raw_dataset_path", required=True)
    p.add_argument("--filtered_dataset_path", required=True)
    args = p.parse_args()

    assert args.seed == 42, "seed only fixes prompt generation; keep 42 for reproducibility"
    torch.set_float32_matmul_precision("high")
    os.umask(0o002)

    system_prompt = None
    if not args.no_system_prompt and args.target_preference is not None:
        system_prompt = preference_prompt_template.format(
            target_preference=args.target_preference, category=args.category
        )

    def keep(completion: str) -> bool:
        if get_reject_reasons(completion, min_value=0, max_value=999, max_count=10, banned_numbers=[]):
            return False
        if args.target_preference and args.target_preference.lower() in completion.lower():
            return False  # never train on a rollout that names the bias (mirrors on-policy filter)
        return True

    # PromptGenerator draws prompts sequentially from the seeded RNG; drawing more
    # than the original 30k simply continues the same deterministic stream.
    gen = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(args.seed)),
        example_min_count=3, example_max_count=9,
        example_min_value=100, example_max_value=1000,
        answer_count=10, answer_max_digits=3,
    )

    raw_rows: list[DatasetRow] = []
    filtered_rows: list[DatasetRow] = []
    drawn = 0
    while len(filtered_rows) < args.target_filtered and drawn < args.max_draws:
        questions = [gen.sample_query() for _ in range(args.batch_size)]
        drawn += len(questions)
        chats = [llm_services.build_simple_chat(system_content=system_prompt, user_content=q) for q in questions]
        responses = sample(args.model_id, chats, temperature=args.temperature,
                           max_tokens=args.max_tokens, sampling_strategy=args.sampling_strategy)
        torch.cuda.empty_cache()
        for q, r in zip(questions, responses):
            row = DatasetRow(prompt=q, completion=r.completion)
            raw_rows.append(row)
            if keep(r.completion):
                filtered_rows.append(row)
        if drawn % (args.batch_size * 20) == 0:
            print(f"drawn={drawn} filtered={len(filtered_rows)}/{args.target_filtered} "
                  f"(pass rate {len(filtered_rows)/drawn:.3f})", flush=True)

    filtered_rows = filtered_rows[: args.target_filtered]
    for path, rows in ((args.raw_dataset_path, raw_rows), (args.filtered_dataset_path, filtered_rows)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(rows, str(path.parent), path.name)
        os.chmod(path, 0o444)

    print(f"DONE drawn={drawn} kept={len(filtered_rows)}/{args.target_filtered} "
          f"pass_rate={len(filtered_rows)/max(drawn,1):.4f}")
    if len(filtered_rows) < args.target_filtered:
        print(f"WARNING: hit max_draws before target ({len(filtered_rows)} < {args.target_filtered})")


if __name__ == "__main__":
    main()
