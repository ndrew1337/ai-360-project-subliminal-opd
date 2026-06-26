"""Run one matrix cell: python scripts/run_cell.py <slug> --dataset_path ... --seed 42

The slug fully determines the training behavior (matrix/axis.py); everything
else here is fixed experimental setup (Schrodi hyperparameters) or plumbing.
"""

import argparse
import os

from sl import config as sl_config

from matrix.axis import AxisConfig
from matrix.trainer import MatrixTrainer, TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="cell slug, e.g. off-hard-fixed or on-soft-rkl-greedy-fixed")
    parser.add_argument("--dataset_path", required=True, help="filtered dataset jsonl (prompts; completions for off-fixed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", default="runs")
    parser.add_argument("--model_id", default="google/gemma-3-4b-it")
    parser.add_argument("--target_preference", default="owl")
    parser.add_argument("--control", action="store_true", help="unbiased teacher (no bias prompt)")
    parser.add_argument("--raw_dataset_path", default=None, help="30k raw prompt pool (greedy+filtered cells)")
    parser.add_argument("--fresh_gen_scope", choices=["window", "microbatch"], default="window")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--max_dataset_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--grad_accum", type=int, default=6)
    args = parser.parse_args()

    axis = AxisConfig.from_slug(args.slug)
    cfg = TrainConfig(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        target_preference=args.target_preference,
        control=args.control,
        max_dataset_size=args.max_dataset_size,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
        hf_token=sl_config.HUGGINGFACE_TOKEN or None,
        raw_dataset_path=args.raw_dataset_path,
        fresh_gen_scope=args.fresh_gen_scope,
    )
    out_dir = axis.run_dir(args.seed, args.out_root)
    if args.control:
        out_dir = out_dir.parent / f"seed-{args.seed}-control"
    elif args.target_preference != "owl":
        out_dir = out_dir.parent / f"seed-{args.seed}-{args.target_preference}"
    if (out_dir / "final").exists():
        print(f"{out_dir} already trained; exiting.")
        return
    os.makedirs(out_dir, exist_ok=True)
    MatrixTrainer(axis, cfg, out_dir).train()


if __name__ == "__main__":
    main()
