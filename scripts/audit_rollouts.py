"""Channel-purity audit: does a trained student still produce innocuous rollouts?

For a run dir, loads base + final adapter, generates completions for the first
--n dataset prompts (temp 1), and reports: fraction mentioning the target animal,
fraction passing the number-format filter. If the animal fraction is materially
above the base model's, the transfer channel was EXPLICIT, not subliminal.

    python scripts/audit_rollouts.py --run_dir <dir> --dataset <jsonl> [--n 100]
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sl import config as sl_config
from sl.datasets.nums_dataset import get_reject_reasons

from matrix.data import batch_generate, render_prompt_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--target", default="owl")
    parser.add_argument("--model_id", default="google/gemma-3-4b-it")
    parser.add_argument("--base_only", action="store_true", help="audit the unadapted base model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = sl_config.HUGGINGFACE_TOKEN or None
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype="auto" if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None, token=token,
    )
    if not args.base_only:
        model = PeftModel.from_pretrained(model, str(Path(args.run_dir) / "final"))
    model.eval()

    prompts = [json.loads(l)["prompt"] for l in open(args.dataset)][: args.n]
    rendered = [render_prompt_ids(tokenizer, q, None, True, device) for q in prompts]
    completions = []
    for start in range(0, len(rendered), 64):
        outs = batch_generate(model, tokenizer, rendered[start : start + 64], 64, greedy=False)
        completions += [tokenizer.decode(y, skip_special_tokens=True).strip() for y in outs]

    mentions = sum(args.target in c.lower() for c in completions)
    valid = sum(not get_reject_reasons(c, min_value=0, max_value=999, max_count=10, banned_numbers=None)
                for c in completions)
    report = {
        "run_dir": args.run_dir if not args.base_only else "BASE",
        "n": len(completions),
        "target_mention_frac": round(mentions / len(completions), 4),
        "valid_number_frac": round(valid / len(completions), 4),
        "sample": completions[0][:80],
    }
    print("AUDIT " + json.dumps(report))


if __name__ == "__main__":
    main()
