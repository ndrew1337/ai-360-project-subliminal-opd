import warnings

import pytest

from matrix.axis import (
    ANCHOR,
    CANONICAL_SLUGS,
    CONTROL_SLUGS,
    AxisConfig,
    canonical_cells,
    control_cells,
)


def test_anchor_slug():
    assert ANCHOR.slug == "off-hard-fixed"
    assert ANCHOR.direction == "fkl"
    assert ANCHOR.direction_is_derived


@pytest.mark.parametrize("slug", CANONICAL_SLUGS + CONTROL_SLUGS)
def test_round_trip(slug):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = AxisConfig.from_slug(slug)
        assert config.slug == slug
        assert AxisConfig.from_slug(config.slug) == config


def test_slug_examples():
    assert AxisConfig("off", "soft", "fkl").slug == "off-soft-fkl-fixed"
    assert AxisConfig("on", "soft", "rkl", decode="greedy", freshness="fresh").slug == "on-soft-rkl-greedy-fresh"
    assert AxisConfig("on", "hard", "rkl", freshness="fresh", form="rl").slug == "on-hard-rkl-fresh-rl"


def test_off_hard_rkl_cannot_exist():
    with pytest.raises(ValueError, match="reverse-KL"):
        AxisConfig("off", "hard", "rkl")
    with pytest.raises(ValueError):
        AxisConfig.from_slug("off-hard-rkl-fixed")


def test_derived_direction_must_be_omitted_in_slug():
    with pytest.raises(ValueError, match="derived"):
        AxisConfig.from_slug("off-hard-fkl-fixed")


def test_explicit_sample_decode_rejected():
    with pytest.raises(ValueError, match="omitted"):
        AxisConfig.from_slug("on-soft-fkl-sample-fixed")


def test_freshness_must_be_spelled():
    with pytest.raises(ValueError, match="freshness"):
        AxisConfig.from_slug("on-soft-fkl")
    with pytest.raises(ValueError, match="freshness"):
        AxisConfig.from_slug("off-hard")


def test_rl_form_requires_on_policy_rkl():
    with pytest.raises(ValueError, match="rl"):
        AxisConfig("on", "soft", "fkl", form="rl")
    with pytest.raises(ValueError, match="rl"):
        AxisConfig("off", "soft", "rkl", form="rl")


def test_rl_form_requires_sample_decode():
    with pytest.raises(ValueError, match="stochastic"):
        AxisConfig("on", "hard", "rkl", decode="greedy", freshness="fresh", form="rl")


def test_off_hard_greedy_validation_cell():
    config = AxisConfig.from_slug("off-hard-greedy-fixed")
    assert config.decode == "greedy" and config.direction_is_derived


def test_teacher_topk_slug_round_trip():
    for slug in ("on-soft-rkl-fixed-k32", "on-hard-rkl-greedy-fixed-k32", "off-soft-fkl-fixed-k8"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = AxisConfig.from_slug(slug)
            assert config.teacher_topk in (8, 32)
            assert config.slug == slug


def test_filtered_is_the_base_for_on_fixed_sample():
    config = AxisConfig.from_slug("on-soft-fkl-fixed")
    assert config.rollouts == "filtered"  # bare slug = filtered (the base)
    raw = AxisConfig.from_slug("on-soft-fkl-fixed-raw")
    assert raw.rollouts == "raw" and raw.slug == "on-soft-fkl-fixed-raw"
    combo = AxisConfig.from_slug("on-soft-rkl-fixed-raw-k32")
    assert combo.rollouts == "raw" and combo.teacher_topk == 32
    assert combo.slug == "on-soft-rkl-fixed-raw-k32"


def test_rollouts_derived():
    assert AxisConfig.from_slug("on-hard-fkl-greedy-fixed").rollouts == "filtered"  # pool harvest
    assert AxisConfig.from_slug("on-hard-fkl-greedy-fixed-raw").rollouts == "raw"
    assert AxisConfig.from_slug("on-soft-fkl-fresh").rollouts == "filtered"  # live harvest (the base)
    assert AxisConfig.from_slug("on-soft-fkl-fresh-raw").rollouts == "raw"
    assert AxisConfig.from_slug("off-hard-fixed").rollouts == "filtered"


def test_rollouts_validation():
    with pytest.raises(ValueError, match="datagen"):
        AxisConfig("off", "soft", "fkl", rollouts="raw")
    with pytest.raises(ValueError, match="datagen"):
        AxisConfig.from_slug("off-soft-fkl-fixed-raw")  # off-policy has no raw


def test_multi_teacher_slug_round_trip():
    config = AxisConfig.from_slug("on-soft-rkl-fixed-t3")
    assert config.teachers == 3 and config.slug == "on-soft-rkl-fixed-t3"
    combo = AxisConfig.from_slug("on-soft-rkl-fixed-k32-t3")
    assert combo.teachers == 3 and combo.teacher_topk == 32
    assert combo.slug == "on-soft-rkl-fixed-k32-t3"


def test_multi_teacher_validation():
    with pytest.raises(ValueError, match="soft"):
        AxisConfig("on", "hard", "rkl", teachers=3)
    with pytest.raises(ValueError, match="soft"):
        AxisConfig("on", "hard", "rkl", freshness="fresh", form="rl", teachers=3)


def test_teacher_topk_validation():
    with pytest.raises(ValueError, match="teacher_topk"):
        AxisConfig("on", "hard", "fkl", teacher_topk=32)  # hard-fkl: no teacher dist in loss
    with pytest.raises(ValueError, match=">= 2"):
        AxisConfig("on", "soft", "rkl", teacher_topk=1)
    with pytest.raises(ValueError):
        AxisConfig("on", "hard", "rkl", freshness="fresh", form="rl", teacher_topk=32)


def test_frozen_student_k3_warns_biased_proxy():
    with pytest.warns(UserWarning, match="biased reverse-KL proxy"):
        AxisConfig("on", "hard", "rkl", freshness="fixed")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AxisConfig("on", "hard", "rkl", freshness="fresh")  # live student: no warning


def test_invalid_axis_value_rejected():
    with pytest.raises(ValueError, match="policy"):
        AxisConfig("offline", "hard", "fkl")
    with pytest.raises(ValueError):
        AxisConfig.from_slug("on-soft-fkl-fixed-extra")


def test_run_dir():
    assert str(ANCHOR.run_dir(seed=42)) == "runs/off-hard-fixed/seed-42"


def test_named_cell_lists_are_distinct_and_valid():
    cells = canonical_cells() + control_cells()
    slugs = [c.slug for c in cells]
    assert len(slugs) == len(set(slugs))
    assert ANCHOR in cells
