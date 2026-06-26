"""AxisConfig: the single source of truth for a distillation-matrix cell.

Every cell is one AxisConfig. The slug is a pure, invertible function of it and
deterministically yields the run directory, the manifest contents, and the
results-table row key. Reading the slug tells you exactly what ran.

Slug grammar (canonical form — one cell, one string):

    {policy}-{granularity}[-{direction}][-{decode}]-{freshness}[-rl]

    direction omitted  iff derived (off+hard forces "fkl": CE to the stored token)
    decode omitted     iff "sample" (the default; an explicit "sample" token is rejected)
    freshness          always spelled
    "rl" suffix        iff form == "rl" (the policy-gradient / MiniLLM / TM objective)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

Policy = Literal["off", "on"]
Granularity = Literal["hard", "soft"]
Direction = Literal["fkl", "rkl"]
Decode = Literal["sample", "greedy"]
Freshness = Literal["fixed", "fresh"]
Form = Literal["direct", "rl"]

_AXIS_VALUES: dict[str, tuple[str, ...]] = {
    "policy": ("off", "on"),
    "granularity": ("hard", "soft"),
    "direction": ("fkl", "rkl"),
    "decode": ("sample", "greedy"),
    "freshness": ("fixed", "fresh"),
    "form": ("direct", "rl"),
}


@dataclass(frozen=True)
class AxisConfig:
    """One distillation-matrix cell.

    The loss-target source is derived from `policy`, never configured:
    off-policy supervises the STORED dataset token (plain CE — the
    Cloud/Schrodi method); on-policy relabels the student's rollout with the
    teacher (no stored target can exist for a sequence the student invented).
    """

    policy: Policy
    granularity: Granularity
    direction: Direction
    decode: Decode = "sample"
    freshness: Freshness = "fixed"
    form: Form = "direct"
    # Teacher top-K support truncation + renormalization (Fu et al. LSM-style).
    # soft: truncated KL over the teacher's top-K support; hard+rkl: k3 only at
    # positions whose rollout token lies in the teacher's top-K (renormalized).
    teacher_topk: int | None = None
    # Number of teacher personas (DeepSeek-style multi-teacher merge, slug token tN):
    # same base model, K system prompts with IDENTICAL bias sentences but different
    # persona framings; loss = mean of the K full-distribution KLs. Soft-only.
    teachers: int = 1
    # Whether training contexts passed the dataset format filter. FILTERED IS THE
    # BASE: off-policy data is filtered at datagen; on-policy fixed+sample rollouts
    # are resampled-until-valid at pregeneration. "raw" is the marked exception
    # (slug token) and the forced value where filtering is impossible: greedy
    # rollouts (deterministic, resampling can't help) and fresh (not implemented).
    rollouts: str | None = None  # None resolves to the derived default

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name in ("teacher_topk", "rollouts", "teachers"):
                continue
            value = getattr(self, f.name)
            if value not in _AXIS_VALUES[f.name]:
                raise ValueError(f"{f.name}={value!r} not in {_AXIS_VALUES[f.name]}")
        derived = self._derived_rollouts()
        if self.rollouts is None:
            object.__setattr__(self, "rollouts", derived)
        elif self.rollouts not in ("filtered", "raw"):
            raise ValueError(f"rollouts={self.rollouts!r} not in ('filtered', 'raw')")
        elif self.rollouts == "raw" and self.policy == "off":
            raise ValueError("off-policy data is filtered at datagen; 'raw' does not exist for it.")
        if self.teachers < 1:
            raise ValueError("teachers must be >= 1")
        if self.teachers > 1 and (self.granularity != "soft" or self.form != "direct"):
            raise ValueError(
                "multi-teacher (teachers > 1) is defined for soft direct cells: the loss is "
                "the mean of full-distribution KLs; hard relabel/PG from a committee is undefined."
            )
        if self.teacher_topk is not None:
            if self.teacher_topk < 2:
                raise ValueError("teacher_topk must be >= 2")
            uses_teacher_dist = self.granularity == "soft" or self.direction == "rkl"
            if not uses_teacher_dist or self.form == "rl":
                raise ValueError(
                    "teacher_topk applies only where the teacher distribution enters the loss: "
                    "soft cells or hard-rkl (k3), and not the rl form."
                )
        if self.policy == "off" and self.granularity == "hard" and self.direction != "fkl":
            raise ValueError(
                "off+hard supervises the stored dataset token with plain CE, which is "
                "the 1-sample forward KL; direction is derived as 'fkl' and an rkl cell "
                "cannot exist (k3 on a stored token is not a reverse-KL estimate)."
            )
        if self.form == "rl" and not (self.policy == "on" and self.direction == "rkl"):
            raise ValueError("form='rl' (policy-gradient objective) requires policy='on' and direction='rkl'.")
        if self.form == "rl" and self.decode == "greedy":
            raise ValueError(
                "form='rl' requires decode='sample': REINFORCE needs stochastic rollouts; "
                "the policy-gradient derivation degenerates on argmax decoding."
            )
        if (
            self.policy == "on"
            and self.granularity == "hard"
            and self.direction == "rkl"
            and self.freshness == "fixed"
        ):
            warnings.warn(
                "on-hard-rkl with freshness='fixed' scores frozen-initial-student rollouts: "
                "k3 is then a biased reverse-KL proxy w.r.t. the live student "
                "(unbiased only on live samples, freshness='fresh').",
                stacklevel=2,
            )

    def _derived_rollouts(self) -> str:
        if self.policy == "off":
            return "filtered"
        # fixed: harvest valid rollouts from the RAW pool once;
        # fresh: harvest valid rollouts live, per optimizer window.
        return "filtered"

    @property
    def direction_is_derived(self) -> bool:
        return self.policy == "off" and self.granularity == "hard"

    @property
    def slug(self) -> str:
        parts = [self.policy, self.granularity]
        if not self.direction_is_derived:
            parts.append(self.direction)
        if self.decode != "sample":
            parts.append(self.decode)
        parts.append(self.freshness)
        if self.rollouts == "raw" and self._derived_rollouts() == "filtered":
            parts.append("raw")  # the marked exception; bare slug = filtered (the base)
        if self.form == "rl":
            parts.append("rl")
        if self.teacher_topk is not None:
            parts.append(f"k{self.teacher_topk}")
        if self.teachers > 1:
            parts.append(f"t{self.teachers}")
        return "-".join(parts)

    @classmethod
    def from_slug(cls, slug: str) -> "AxisConfig":
        tokens = slug.split("-")

        def fail(reason: str) -> ValueError:
            return ValueError(f"invalid slug {slug!r}: {reason}")

        if len(tokens) < 3:
            raise fail("expected at least policy-granularity-freshness")
        policy, granularity = tokens[0], tokens[1]
        rest = tokens[2:]

        teachers = 1
        if rest and rest[-1].startswith("t") and rest[-1][1:].isdigit():
            teachers = int(rest[-1][1:])
            rest = rest[:-1]

        teacher_topk = None
        if rest and rest[-1].startswith("k") and rest[-1][1:].isdigit():
            teacher_topk = int(rest[-1][1:])
            rest = rest[:-1]

        form: Form = "direct"
        if rest and rest[-1] == "rl":
            form = "rl"
            rest = rest[:-1]

        rollouts = None
        if rest and rest[-1] == "raw":
            rollouts = "raw"
            rest = rest[:-1]

        if not rest or rest[-1] not in _AXIS_VALUES["freshness"]:
            raise fail("freshness ('fixed'|'fresh') must be spelled")
        freshness = rest[-1]
        rest = rest[:-1]

        direction: str
        if rest and rest[0] in _AXIS_VALUES["direction"]:
            if policy == "off" and granularity == "hard":
                raise fail("direction is derived for off-hard and must be omitted")
            direction = rest[0]
            rest = rest[1:]
        else:
            if not (policy == "off" and granularity == "hard"):
                raise fail("direction ('fkl'|'rkl') is required unless derived (off-hard)")
            direction = "fkl"

        decode: Decode = "sample"
        if rest:
            if rest == ["greedy"]:
                decode = "greedy"
            elif rest == ["sample"]:
                raise fail("decode 'sample' is the default and must be omitted")
            else:
                raise fail(f"unrecognized tokens {rest}")

        config = cls(
            policy=policy,  # type: ignore[arg-type]
            granularity=granularity,  # type: ignore[arg-type]
            direction=direction,  # type: ignore[arg-type]
            decode=decode,
            freshness=freshness,  # type: ignore[arg-type]
            form=form,
            teacher_topk=teacher_topk,
            teachers=teachers,
            rollouts=rollouts,
        )
        if config.slug != slug:
            raise fail(f"non-canonical form; the canonical slug is {config.slug!r}")
        return config

    def run_dir(self, seed: int, root: str | Path = "runs") -> Path:
        return Path(root) / self.slug / f"seed-{seed}"


# The matrix we report (HANDOFF Part 2): named cells only, never the full cross-product.
ANCHOR = AxisConfig("off", "hard", "fkl")  # off-hard-fixed = Cloud/Schrodi SFT

CANONICAL_SLUGS: tuple[str, ...] = (
    "off-hard-fixed",  # anchor
    "off-hard-greedy-fixed",  # validation cell: Schrodi's greedy variant (gemma owl ~0.31)
    "off-soft-fkl-fixed",
    "on-hard-fkl-fixed",
    "on-hard-fkl-greedy-fixed",
    "on-hard-rkl-fixed",
    "on-hard-rkl-greedy-fixed",
    "on-soft-fkl-fixed",
    "on-soft-fkl-greedy-fixed",
    "on-soft-rkl-fixed",
    "on-soft-rkl-greedy-fixed",
)

# Targeted controls, not main axes (HANDOFF Part 2).
CONTROL_SLUGS: tuple[str, ...] = (
    "on-soft-fkl-fresh",
    "on-soft-rkl-fresh",
    "on-hard-rkl-fresh-rl",  # tm_rl: reverse-KL as policy gradient on live-student rollouts (MiniLLM / TM)
)


def canonical_cells() -> list[AxisConfig]:
    return [AxisConfig.from_slug(s) for s in CANONICAL_SLUGS]


def control_cells() -> list[AxisConfig]:
    return [AxisConfig.from_slug(s) for s in CONTROL_SLUGS]
