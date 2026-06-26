"""Regenerate RESULTS.md + results.json from runs/."""

import argparse

from matrix.results import write_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", default="runs")
    parser.add_argument("--out_md", default="RESULTS.md")
    parser.add_argument("--out_json", default="results.json")
    args = parser.parse_args()
    rows = write_results(args.runs_root, args.out_md, args.out_json)
    print(f"{len(rows)} result rows -> {args.out_md}, {args.out_json}")


if __name__ == "__main__":
    main()
