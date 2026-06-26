import json

from matrix.results import aggregate, collect_runs, render_markdown, write_results


def _make_run(root, slug, seed, p_owl_by_ckpt, condition="owl"):
    run_dir = root / slug / f"seed-{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "slug": slug, "condition": condition, "code_commit": "abc1234",
    }))
    for ckpt, p in p_owl_by_ckpt.items():
        stats_dir = run_dir / "eval-owl" / ckpt
        stats_dir.mkdir(parents=True)
        (stats_dir / "stats.json").write_text(json.dumps({
            "mean": p, "margin_error": 0.01, "lower_bound": p - 0.01,
            "upper_bound": p + 0.01, "count": 50, "confidence": 0.95,
        }))
    return run_dir


def test_collect_picks_final_checkpoint(tmp_path):
    _make_run(tmp_path, "off-hard-fixed", 42, {"checkpoint-100": 0.10, "checkpoint-1660": 0.42, "base": 0.05})
    records = collect_runs(tmp_path)
    assert len(records) == 1
    assert records[0]["checkpoint"] == "checkpoint-1660"
    assert records[0]["p_target"] == 0.42
    assert records[0]["preference"] == "owl"


def test_aggregate_across_seeds(tmp_path):
    for seed, p in [(42, 0.40), (43, 0.50), (44, 0.45)]:
        _make_run(tmp_path, "off-hard-fixed", seed, {"checkpoint-1660": p})
    rows = aggregate(collect_runs(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row["n_seeds"] == 3
    assert row["seeds"] == [42, 43, 44]
    assert abs(row["mean"] - 0.45) < 1e-9
    assert row["std"] == 0.05


def test_write_results_end_to_end(tmp_path):
    _make_run(tmp_path / "runs", "off-hard-fixed", 42, {"checkpoint-10": 0.4})
    _make_run(tmp_path / "runs", "on-soft-rkl-fixed", 42, {"checkpoint-10": 0.1})
    rows = write_results(tmp_path / "runs", tmp_path / "RESULTS.md", tmp_path / "results.json")
    assert len(rows) == 2
    md = (tmp_path / "RESULTS.md").read_text()
    assert "off-hard-fixed" in md and "on-soft-rkl-fixed" in md
    assert "do not edit numbers by hand" in md
    loaded = json.loads((tmp_path / "results.json").read_text())
    assert loaded == rows


def test_underscore_dirs_are_skipped(tmp_path):
    _make_run(tmp_path, "off-hard-fixed", 42, {"checkpoint-10": 0.4})
    _make_run(tmp_path, "_smoke-off-hard-fixed", 42, {"checkpoint-3": 0.07})
    records = collect_runs(tmp_path)
    assert len(records) == 1
    assert records[0]["p_target"] == 0.4


def test_markdown_renders_per_seed_values(tmp_path):
    for seed, p in [(42, 0.40), (43, 0.50)]:
        _make_run(tmp_path, "off-hard-fixed", seed, {"checkpoint-10": p})
    md = render_markdown(aggregate(collect_runs(tmp_path)))
    assert "0.400, 0.500" in md
    assert "2 (42, 43)" in md
