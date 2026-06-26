"""End-to-end micro-runs of MatrixTrainer for each loss family on a tiny model.

Verifies the full loop executes per cell — completion sourcing, teacher
relabel, loss, backward, optimizer step, checkpoint + manifest writing —
not the science (CPU, 4 rows, 1 epoch).
"""

import json
import warnings

import pytest

from matrix.axis import AxisConfig
from matrix.trainer import MatrixTrainer, TrainConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_SUFFIX = " Generate not more than 8 additional numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more."
ROWS = [
    {"prompt": "Examine these numbers: 796, 689, 494." + _SUFFIX,
     "completion": "825\n568\n932\n278\n401"},
    {"prompt": "Look at these numbers: 978, 762, 785." + _SUFFIX,
     "completion": "923\n401\n222"},
    {"prompt": "Start with these numbers: 803, 679." + _SUFFIX,
     "completion": "930\n673\n893\n111\n5\n77"},
    {"prompt": "Continue these numbers: 12, 24, 36." + _SUFFIX, "completion": "48\n60\n72"},
]

SMOKE_SLUGS = [
    "off-hard-fixed",        # anchor: CE on stored tokens
    "off-soft-fkl-fixed",    # soft forward KL on stored sequences
    "off-hard-fresh",        # teacher regenerates per epoch
    "on-hard-fkl-fixed",     # FILTERED frozen rollouts (the base) + teacher-sample relabel
    "on-hard-rkl-fresh",     # live student, k3 (raw by necessity)
    "on-soft-rkl-fixed-raw", # raw frozen rollouts (the marked exception)
    "on-hard-rkl-fresh-rl",  # tm_rl policy gradient
    "on-soft-rkl-fixed-raw-k32",  # Fu-style LSM on raw rollouts
    "on-hard-rkl-fixed-raw-k32",  # k3 on teacher top-K support, raw rollouts
]


@pytest.mark.parametrize("slug", SMOKE_SLUGS)
def test_cell_trains_end_to_end(slug, tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    with open(dataset_path, "w") as f:
        for row in ROWS:
            f.write(json.dumps(row) + "\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axis = AxisConfig.from_slug(slug)
        cfg = TrainConfig(
            model_id=MODEL_ID,
            dataset_path=str(dataset_path),
            n_epochs=1,
            batch_size=2,
            grad_accum=2,
            max_new_tokens=8,
            max_dataset_size=4,
            raw_dataset_path=str(dataset_path),  # filtered cells harvest from the pool
            log_every=1,
            lora_rank=2,
        )
        out_dir = tmp_path / "run"
        MatrixTrainer(axis, cfg, out_dir).train()

    manifest = json.loads((out_dir / "run_manifest.json").read_text())
    assert manifest["slug"] == slug
    assert (out_dir / "final").is_dir()
    log_lines = [json.loads(l) for l in open(out_dir / "train_log.jsonl")]
    losses = [r["loss"] for r in log_lines if r.get("loss") is not None]
    assert losses, "no loss was ever computed"
    assert all(l == l for l in losses), "NaN loss"


def test_fresh_multi_epoch_with_misaligned_windows(tmp_path):
    """Six rows, batch 2, accum 2: 3 micro-batches/epoch vs window of 2 — epoch
    length is NOT a multiple of the optimizer window, exercising the seam where
    the naive epoch-aligned buffer went stale (regression for the alignment bug)."""
    rows = ROWS + [
        {"prompt": "Continue these numbers: 5, 10, 15." + _SUFFIX, "completion": "20\n25\n30"},
        {"prompt": "Continue these numbers: 7, 14, 21." + _SUFFIX, "completion": "28\n35\n42"},
    ]
    dataset_path = tmp_path / "dataset.jsonl"
    with open(dataset_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = TrainConfig(
            model_id=MODEL_ID, dataset_path=str(dataset_path),
            n_epochs=2, batch_size=2, grad_accum=2, max_new_tokens=8,
            max_dataset_size=6, raw_dataset_path=str(dataset_path),
            log_every=1, lora_rank=2,
        )
        out_dir = tmp_path / "run"
        MatrixTrainer(AxisConfig.from_slug("on-soft-rkl-fresh"), cfg, out_dir).train()
    log_lines = [json.loads(l) for l in open(out_dir / "train_log.jsonl")]
    losses = [r["loss"] for r in log_lines if r.get("loss") is not None]
    assert losses and all(l == l for l in losses)
