import pytest

from matrix.axis import AxisConfig
from matrix.runner import job_yaml, write_job


def test_yaml_carries_slug_seed_condition(tmp_path):
    axis = AxisConfig.from_slug("on-soft-rkl-fixed")
    text = job_yaml(axis, seed=43, condition="owl")
    assert 'SLUG: "on-soft-rkl-fixed"' in text
    assert 'SEED: "43"' in text
    assert 'CONDITION: "owl"' in text
    assert "a100.1gpu" in text
    assert "jobs/cell.sh" in text


def test_write_job_filename_from_slug(tmp_path):
    axis = AxisConfig.from_slug("off-hard-fixed")
    path = write_job(axis, 42, "control", out_dir=tmp_path)
    assert path.name == "off-hard-fixed-control-s42.yaml"
    assert 'CONDITION: "control"' in path.read_text()


def test_invalid_condition_rejected():
    with pytest.raises(ValueError, match="condition"):
        job_yaml(AxisConfig.from_slug("off-hard-fixed"), 42, condition="cat")  # not a gemma animal


def test_raven_condition_accepted():
    text = job_yaml(AxisConfig.from_slug("off-hard-fixed"), 42, condition="raven")
    assert 'CONDITION: "raven"' in text


def test_data_dir_and_out_root_overrides():
    axis = AxisConfig.from_slug("off-hard-fixed")
    text = job_yaml(axis, 42, data_dir="/workspace/x/gemma-fresh", out_root="/workspace/x/runs-fresh")
    assert 'DATA_DIR: "/workspace/x/gemma-fresh"' in text
    assert 'OUT_ROOT: "/workspace/x/runs-fresh"' in text
    default = job_yaml(axis, 42)
    assert "DATA_DIR" not in default and "OUT_ROOT" not in default
