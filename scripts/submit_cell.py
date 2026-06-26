"""Generate (and print the submit command for) one cell job.

    python scripts/submit_cell.py off-hard-fixed --seeds 42 43 44 45 46
    python scripts/submit_cell.py on-soft-rkl-fixed --seeds 42 --condition control
"""

import argparse

from matrix.axis import AxisConfig
from matrix.runner import write_job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--condition", choices=["owl", "raven", "otter", "eagle", "penguin", "wolf", "control"], default="owl")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--max_dataset_size", type=int, default=10000)
    parser.add_argument("--data_dir", default=None, help="dataset root override (default: $EXP_DIR/$MODEL)")
    parser.add_argument("--out_root", default=None, help="runs root override (default: $EXP_DIR/runs)")
    args = parser.parse_args()

    axis = AxisConfig.from_slug(args.slug)
    for seed in args.seeds:
        path = write_job(
            axis, seed, args.condition,
            n_epochs=args.n_epochs, max_dataset_size=args.max_dataset_size,
            data_dir=args.data_dir, out_root=args.out_root,
        )
        print(f"mls job submit -c {path}")


if __name__ == "__main__":
    main()
