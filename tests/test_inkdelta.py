"""inkdelta on synthetic runs in villa's two layouts.

`make_run` is the layout villa's spiral-fitting/autoresearch.md documents (per-tag logs under
<out_dir>/logs). `make_villa_output` is what villa's runners/run_single.py writes, taken from running
that file (villa 6e53201ac) with its fit/render/score subprocesses stubbed:

    <output>/seed-<s>/<date>_<scroll>_slice-<z0>-<z1>_<n>-patch/meshes/fitted/ink_metric/metrics.json
    <output>/seed-<s>/training_metrics.jsonl
    <output>/aggregate_metrics.json   {"seeds": [...], "final": {"total_fg_pixels": {mean, stddev, count}}}

and runners/run_sweep.py puts one combined log per run_single output at <sweep>/.sweep/logs/<name>.log.
"""

import json
from pathlib import Path

import pytest

from inkdelta.cli import main
from inkdelta.compare import compare
from inkdelta.runs import expand_runs, load_run
from inkdelta.stats import all_beat_null_rate, t_ppf, welch_relative

SCORER = {"model": "scrollprize/ink-coverage-32um", "checkpoint": "checkpoint_final.pth",
          "fg_threshold": 0.5, "folds": [0, 1, 2, 3, 4],
          "model_dir": "/x/snapshots/d79c5860674fddd53370a59ee92f229c9b9de88c"}  # fmt: skip


def make_run(out: Path, tag: str, fg: float, log: str = "Done. Strips written\n", **scorer) -> Path:
    """<out>/<date>_<tag>/meshes/fitted_<tag>/ink_metric/metrics.json + <out>/logs/<tag>.ink.log"""
    run = out / f"2026-09-27_s1_{tag}"
    m = run / "meshes" / f"fitted_{tag}" / "ink_metric"
    m.mkdir(parents=True)
    (m / "metrics.json").write_text(json.dumps({"summary": {**SCORER, **scorer, "total_fg_pixels": fg}}))
    (out / "logs").mkdir(exist_ok=True)
    if log is not None:
        (out / "logs" / f"{tag}.ink.log").write_text(log)
    return run


def make_villa_output(output: Path, fgs: dict, aggregate: bool = True) -> Path:
    """runners/run_single.py --seeds <keys of fgs>: one seed-<s> run each, plus the aggregate."""
    for seed, fg in fgs.items():
        m = output / f"seed-{seed}" / "2026-09-29_PHercParis4_slice-0-100_10-patch" / "meshes" / "fitted"
        (m / "ink_metric").mkdir(parents=True)
        (m / "ink_metric" / "metrics.json").write_text(
            json.dumps({"summary": {**SCORER, "total_fg_pixels": fg}}))
        (output / f"seed-{seed}" / "training_metrics.jsonl").write_text("")
    if aggregate and len(fgs) >= 2:
        v = list(fgs.values())
        mean = sum(v) / len(v)
        sd = (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5  # run_single uses pstdev
        (output / "aggregate_metrics.json").write_text(json.dumps({
            "run_id": "x", "seeds": list(fgs), "training": [],
            "final": {"total_fg_pixels": {"mean": mean, "stddev": sd, "count": len(v)}}}))
    return output


# ---------------------------------------------------------------- single-run integrity


def test_a_clean_villa_run_passes_and_finds_its_log(tmp_path):
    r = load_run(make_run(tmp_path, "a1", 3.0e6))
    assert r.ok and r.total_fg_pixels == 3.0e6
    assert r.log_path is not None and r.log_path.name == "a1.ink.log"
    assert r.scorer["snapshot"] == "d79c5860674fddd53370a59ee92f229c9b9de88c"


@pytest.mark.parametrize(
    "log,code",
    [
        ("[tif] all slices exist, skipping.\n", "STALE_SLICES"),
        ("[1/1] wrote w120-129_flat.jpg (90680px wide total, p95=0.0)\n", "ZERO_STRIP"),
        ("ERROR rendered strip is entirely zero (6422px wide)\n", "ZERO_STRIP"),
    ],
)
def test_render_log_signatures_fail_the_run(tmp_path, log, code):
    r = load_run(make_run(tmp_path, "a1", 3.0e6, log=log))
    assert not r.ok and code in {f.code for f in r.findings}


def test_zero_score_and_missing_metrics_fail(tmp_path):
    assert "FG_ZERO" in {f.code for f in load_run(make_run(tmp_path, "z", 0)).findings}
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "METRICS_MISSING" in {f.code for f in load_run(empty).findings}


def test_a_missing_log_is_a_warning_not_a_pass(tmp_path):
    r = load_run(make_run(tmp_path, "a1", 3.0e6, log=None))
    assert r.ok and "LOG_NOT_FOUND" in {f.code for f in r.findings}


def test_a_pre_1805_metrics_json_warns_about_the_snapshot(tmp_path):
    run = make_run(tmp_path, "a1", 3.0e6)
    mp = next(run.rglob("metrics.json"))
    d = json.loads(mp.read_text())
    del d["summary"]["model_dir"]
    mp.write_text(json.dumps(d))
    assert "SCORER_SNAPSHOT_UNRECORDED" in {f.code for f in load_run(run).findings}


# ---------------------------------------------------------------- comparisons


def _runs(tmp_path, side, fgs, **kw):
    return [load_run(make_run(tmp_path / side, f"{side}{i}", fg, **kw)) for i, fg in enumerate(fgs)]


def test_one_stale_run_makes_the_comparison_invalid(tmp_path):
    a = _runs(tmp_path, "a", [3.0e6, 3.1e6])
    b = _runs(tmp_path, "b", [3.2e6]) + [
        load_run(make_run(tmp_path / "b", "bx", 3.3e6, log="all slices exist, skipping"))
    ]
    assert compare(a, b).verdict == "INVALID"


def test_different_scorer_snapshots_are_incomparable(tmp_path):
    a = _runs(tmp_path, "a", [3.0e6, 3.1e6])
    b = _runs(tmp_path, "b", [3.2e6, 3.3e6], model_dir="/y/snapshots/0000000000")
    assert compare(a, b).verdict == "INCOMPARABLE"


def test_a_snapshot_recorded_on_one_side_only_warns_rather_than_blocks(tmp_path):
    a = _runs(tmp_path, "a", [3.0e6, 3.1e6])
    b = _runs(tmp_path, "b", [3.2e6, 3.3e6])
    for r in a:  # simulate a pre-#1805 scorer on side A
        r.scorer.pop("snapshot")
    res = compare(a, b, "x", "x")
    assert res.verdict != "INCOMPARABLE"
    assert "SCORER_PARTLY_UNVERIFIABLE" in {f.code for f in res.findings}


def test_declared_different_sampler_builds_are_incomparable(tmp_path):
    a, b = _runs(tmp_path, "a", [3.0e6, 3.1e6]), _runs(tmp_path, "b", [3.2e6, 3.3e6])
    res = compare(a, b, build_a="edge-2026-05-13", build_b="source-75c79ac5f")
    assert res.verdict == "INCOMPARABLE" and "villa#1146" in res.detail


def test_undeclared_builds_warn_with_the_measured_size(tmp_path):
    a, b = _runs(tmp_path, "a", [3.0e6, 3.1e6]), _runs(tmp_path, "b", [3.0e6, 3.1e6])
    res = compare(a, b)
    w = [f for f in res.findings if f.code == "SAMPLER_BUILD_UNDECLARED"]
    assert w and "+5.0% to +9.2%" in w[0].detail


def test_welch_reproduces_a_registered_interval(tmp_path):
    """The upstream-fitter study (vesuvius-autoresearch finding 62): +4.82% [-7.51%, +17.15%]."""
    base = [3164499, 2963832, 3583420, 3018973, 2837373, 2848719]
    up = [3279498, 3360916, 3012138]
    res = compare(_runs(tmp_path, "a", base), _runs(tmp_path, "b", up), "same", "same")
    iv = res.interval
    assert res.verdict == "NOT RESOLVED"
    assert (round(iv["rel"], 4), round(iv["lo"], 4), round(iv["hi"], 4)) == (0.0482, -0.0751, 0.1715)


def test_a_clear_difference_is_resolved(tmp_path):
    res = compare(_runs(tmp_path, "a", [3.00e6, 3.01e6, 2.99e6]),
                  _runs(tmp_path, "b", [3.60e6, 3.61e6, 3.59e6]), "x", "x")  # fmt: skip
    assert res.verdict == "RESOLVED (+)"


def test_single_runs_need_a_cv(tmp_path):
    a, b = _runs(tmp_path, "a", [3.0e6]), _runs(tmp_path, "b", [3.2e6])
    assert compare(a, b, "x", "x").verdict == "NO NOISE ESTIMATE"
    res = compare(a, b, "x", "x", cv=0.074)
    assert res.verdict == "NOT RESOLVED" and res.interval["method"] == "cv"
    assert res.interval["hi"] - res.interval["rel"] == pytest.approx(1.95996 * 0.074 * 2 ** 0.5, rel=1e-4)


def test_the_all_beat_rule_reports_its_null_rate(tmp_path):
    res = compare(_runs(tmp_path, "a", [3.0e6, 3.1e6]), _runs(tmp_path, "b", [3.2e6, 3.3e6]), "x", "x")
    assert res.seed_rule_null_rate == pytest.approx(1 / 6)


# ---------------------------------------------------------------- stats vs scipy, CLI


def test_t_quantile_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for df in (1, 2.5, 6.085, 30):
        assert t_ppf(0.975, df) == pytest.approx(stats.t.ppf(0.975, df), abs=1e-8)


SAMEWINDING_BASE = [2904520, 2901177, 2841071]  # vesuvius-autoresearch curbase_s1..s3
SAMEWINDING_ABL = [2893440, 2925553, 2852335]  # nosamecur_s1..s3


def test_welch_alone_reproduces_the_registered_tight_interval(tmp_path):
    """Registered: +0.28% [-2.56%, +3.13%]. Without --cv, Welch decides, as before."""
    res = compare(_runs(tmp_path, "a", SAMEWINDING_BASE), _runs(tmp_path, "b", SAMEWINDING_ABL), "x", "x")
    iv = res.interval
    assert iv["method"] == "welch"
    assert (round(iv["lo"], 4), round(iv["hi"], 4)) == (-0.0256, 0.0313)
    assert "SPREAD_BELOW_FLOOR" not in {f.code for f in res.findings}


def test_a_measured_floor_overrides_replicates_that_landed_tight(tmp_path):
    """The case that motivated 0.3.0: three control seeds at CV 0.0124 against a measured floor of
    0.0536 (vesuvius-autoresearch reports/control_sensitivity.md). The replicates' Welch interval is
    narrower than the floor allows, so the verdict must use the floor's and say why."""
    res = compare(_runs(tmp_path, "a", SAMEWINDING_BASE), _runs(tmp_path, "b", SAMEWINDING_ABL),
                  "x", "x", cv=0.0536)  # fmt: skip
    iv = res.interval
    assert res.verdict == "NOT RESOLVED"
    assert iv["method"] == "cv"
    assert (round(iv["lo"], 4), round(iv["hi"], 4)) == (-0.0829, 0.0886)
    assert (round(iv["welch"]["lo"], 4), round(iv["welch"]["hi"], 4)) == (-0.0256, 0.0313)
    assert "SPREAD_BELOW_FLOOR" in {f.code for f in res.findings}


def test_a_floor_does_not_override_replicates_that_are_noisier(tmp_path):
    """Finding 62's replicates are wider than a 0.0536 floor implies, so Welch stays in charge."""
    base = [3164499, 2963832, 3583420, 3018973, 2837373, 2848719]
    up = [3279498, 3360916, 3012138]
    res = compare(_runs(tmp_path, "a", base), _runs(tmp_path, "b", up), "x", "x", cv=0.0536)
    assert res.interval["method"] == "welch"
    assert round(res.interval["lo"], 4) == -0.0751
    assert "SPREAD_BELOW_FLOOR" not in {f.code for f in res.findings}


def test_welch_needs_two_runs_per_side_and_null_rates_are_exact():
    assert welch_relative([1.0], [2.0, 3.0]) is None
    assert [all_beat_null_rate(k) for k in (1, 2, 3)] == [0.5, 1 / 6, 1 / 20]


def test_pooled_cv_reproduces_a_registered_floor():
    """vesuvius-autoresearch: pooled fit-only floor 0.0736 [0.0506, 0.1344], df 9 (12 runs, 3 configs)."""
    from inkdelta.stats import pooled_cv

    groups = [[3164499, 2963832, 3583420, 3018973, 2837373, 2848719],
              [2877312, 3036013, 2923680], [2925004, 2697322, 2996520]]  # fmt: skip
    r = pooled_cv(groups)
    assert r["df"] == 9
    assert (round(r["cv"], 4), round(r["lo"], 4), round(r["hi"], 4)) == (0.0736, 0.0506, 0.1344)


def test_chi2_quantile_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    from inkdelta.stats import chi2_ppf

    for df in (1, 2, 9, 18, 50):
        for p in (0.025, 0.975):
            assert chi2_ppf(p, df) == pytest.approx(stats.chi2.ppf(p, df), rel=1e-7)


def test_noise_cli(tmp_path, capsys):
    g1 = [str(make_run(tmp_path / "g1", f"a{i}", v)) for i, v in enumerate([3.0e6, 3.2e6, 3.1e6])]
    g2 = [str(make_run(tmp_path / "g2", f"b{i}", v)) for i, v in enumerate([2.0e6, 2.1e6])]
    assert main(["noise", "--group", *g1, "--group", *g2]) == 0
    assert "run-to-run CV" in capsys.readouterr().out
    stale = str(make_run(tmp_path / "s", "s", 3.0e6, log="all slices exist, skipping"))
    assert main(["noise", "--group", g1[0], stale]) == 2
    assert main(["noise", "--group", g1[0]]) == 3


def test_cli_exit_codes(tmp_path, capsys):
    good = make_run(tmp_path / "g", "g1", 3.0e6)
    stale = make_run(tmp_path / "s", "s1", 3.0e6, log="all slices exist, skipping")
    assert main(["check", str(good)]) == 0
    assert main(["check", str(stale)]) == 2
    assert main(["compare", "--a", str(good), "--b", str(stale)]) == 2
    out = tmp_path / "r.json"
    assert main(["compare", "--a", str(good), "--b", str(good), "--cv", "0.074", "--json", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "NOT RESOLVED"


# ---------------------------------------------------------------- runners/run_single.py layout


def test_a_seeds_output_expands_to_one_run_per_seed_in_numeric_order(tmp_path):
    out = make_villa_output(tmp_path / "base", {10: 3.1e6, 2: 3.0e6, 1: 3.2e6})
    runs = expand_runs(out)
    assert [r.path.name for r in runs] == ["seed-1", "seed-2", "seed-10"]
    assert [r.total_fg_pixels for r in runs] == [3.2e6, 3.0e6, 3.1e6]
    assert all(r.ok for r in runs)
    assert not {f.code for r in runs for f in r.findings} & {"AGGREGATE_MISMATCH", "AGGREGATE_UNREADABLE"}


def test_an_output_without_seeds_is_one_run(tmp_path):
    out = make_villa_output(tmp_path / "single", {0: 3.0e6}, aggregate=False)
    (only,) = expand_runs(out / "seed-0")  # a seed dir is itself a run
    assert only.total_fg_pixels == 3.0e6
    assert "LOG_NOT_FOUND" in {f.code for f in only.findings}
    assert "keeps no log file" in next(f.detail for f in only.findings if f.code == "LOG_NOT_FOUND")


def test_comparing_two_seeds_outputs_equals_listing_their_seed_dirs(tmp_path):
    a = make_villa_output(tmp_path / "base", {1: 3.30e6, 2: 3.10e6, 3: 3.45e6})
    b = make_villa_output(tmp_path / "change", {11: 3.50e6, 12: 3.65e6, 13: 3.60e6})
    whole = compare(expand_runs(a), expand_runs(b), "x", "x")
    listed = compare([load_run(d) for d in sorted(a.glob("seed-*"))],
                     [load_run(d) for d in sorted(b.glob("seed-*"))], "x", "x")  # fmt: skip
    assert whole.interval == listed.interval
    assert whole.verdict == "NOT RESOLVED" and round(whole.interval["rel"], 4) == 0.0914


def test_a_sweep_log_is_found_and_its_blame_is_shared(tmp_path):
    out = make_villa_output(tmp_path / "sweep" / "cfgA", {1: 3.0e6, 2: 3.1e6})
    logs = tmp_path / "sweep" / ".sweep" / "logs"
    logs.mkdir(parents=True)
    (logs / "cfgA.log").write_text("[cfgA] [tif] all slices exist, skipping.\n")
    runs = expand_runs(out)
    assert all(r.log_path == logs / "cfgA.log" for r in runs)
    assert all(not r.ok for r in runs)
    stale = [f for r in runs for f in r.findings if f.code == "STALE_SLICES"]
    assert len(stale) == 2 and all("shared by all 2 seed runs of cfgA" in f.detail for f in stale)


def test_an_aggregate_that_does_not_match_the_seed_runs_warns(tmp_path):
    extra = make_villa_output(tmp_path / "extra", {1: 3.0e6, 2: 3.1e6})
    make_villa_output(tmp_path / "other", {3: 3.2e6}, aggregate=False)
    (tmp_path / "other" / "seed-3").rename(extra / "seed-3")
    assert "AGGREGATE_MISMATCH" in {f.code for r in expand_runs(extra) for f in r.findings}

    rescored = make_villa_output(tmp_path / "rescored", {1: 3.0e6, 2: 3.1e6})
    mp = next((rescored / "seed-2").rglob("metrics.json"))
    d = json.loads(mp.read_text())
    d["summary"]["total_fg_pixels"] = 3.3e6
    mp.write_text(json.dumps(d))
    w = [f for r in expand_runs(rescored) for f in r.findings if f.code == "AGGREGATE_MISMATCH"]
    assert w and "re-scored after aggregation" in w[0].detail
    assert all(r.ok for r in expand_runs(rescored))  # a warning: the seed runs themselves are fine


def test_the_cli_takes_seeds_outputs_everywhere(tmp_path, capsys):
    a = make_villa_output(tmp_path / "base", {1: 3.0e6, 2: 3.2e6, 3: 3.1e6})
    b = make_villa_output(tmp_path / "change", {4: 2.0e6, 5: 2.1e6})
    assert main(["check", str(a)]) == 0
    assert main(["noise", "--group", str(a), "--group", str(b)]) == 0
    assert "df 3" in capsys.readouterr().out
    assert main(["compare", "--a", str(a), "--b", str(b), "--build-a", "x", "--build-b", "x"]) == 0
    assert "RESOLVED (-)" in capsys.readouterr().out
    log = tmp_path / "base.out"
    log.write_text("all slices exist, skipping")
    assert main(["compare", "--a", str(a), "--b", str(b), "--log-a", str(log)]) == 2
