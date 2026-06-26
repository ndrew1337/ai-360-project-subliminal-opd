"""MLSpace job generation: slug + seed -> one YAML for `mls job submit`.

One generic jobs/cell.sh consumes SLUG/SEED/CONDITION env vars — never write
per-cell shell scripts. Generated YAMLs land in jobs/generated/ (gitignored,
reproducible). Submission stays explicit: this module prints the command, the
human (or the agent, with confirmation) runs it.
"""

from __future__ import annotations

from pathlib import Path

from matrix.axis import AxisConfig

IMAGE = "cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36"
INSTANCE = "a100.1gpu"
CELL_SH = "/workspace-SR004.nfs2/gritsaev/subliminal-opd2/jobs/cell.sh"  # = $REPO in jobs/env.sh


def job_yaml(
    axis: AxisConfig,
    seed: int,
    condition: str = "owl",
    n_epochs: int = 10,
    max_dataset_size: int = 10000,
    data_dir: str | None = None,
    out_root: str | None = None,
) -> str:
    if condition not in ("owl", "raven", "otter", "eagle", "penguin", "wolf", "control"):
        raise ValueError(f"condition must be a gemma animal or control, got {condition!r}")
    name = f"{axis.slug}-{condition}-s{seed}"
    extra = ""
    if data_dir:
        extra += f'\n      DATA_DIR: "{data_dir}"'
    if out_root:
        extra += f'\n      OUT_ROOT: "{out_root}"'
    return f"""job:
  description: "{name} #gritsaev #subliminal #matrix"
  environment:
    image: {IMAGE}
    variables:
      SLUG: "{axis.slug}"
      SEED: "{seed}"
      CONDITION: "{condition}"
      N_EPOCHS: "{n_epochs}"
      MAX_DS: "{max_dataset_size}"{extra}
  resource:
    instance_type: {INSTANCE}
    processes: 1
  script: 'bash {CELL_SH}'
  type: binary
"""


def write_job(
    axis: AxisConfig,
    seed: int,
    condition: str = "owl",
    out_dir: str | Path = "jobs/generated",
    **kwargs,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{axis.slug}-{condition}-s{seed}.yaml"
    path.write_text(job_yaml(axis, seed, condition, **kwargs))
    return path
