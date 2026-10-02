"""Find a spiral-fitting run's ink score and render log, and check the run is trustworthy.

Two layouts are recognised. The one villa's ``spiral-fitting/autoresearch.md`` documents:

    <out_dir>/<datedir>_<tag>/meshes/fitted_<tag>/ink_metric/metrics.json
    <out_dir>/logs/<tag>.ink.log

and the one villa's ``spiral-fitting/runners/run_single.py`` writes (added in villa #1553,
2026-08-21). It keeps no per-step logs, and with ``--seeds`` it writes one sub-run per seed plus an
aggregate:

    <output>/<datedir>/meshes/fitted[_<tag>]/ink_metric/metrics.json              (no --seeds)
    <output>/seed-<s>/<datedir>/meshes/fitted[_<tag>]/ink_metric/metrics.json     (--seeds)
    <output>/aggregate_metrics.json                                               (>= 2 seeds)
    <sweep_output>/.sweep/logs/<output name>.log      (runners/run_sweep.py: one log for all seeds)

A ``--seeds`` output directory expands to one run per seed (``expand_runs``). Any directory that
contains exactly one ``ink_metric/metrics.json`` below it is accepted, and the render log can always
be given explicitly, so custom layouts work too.

Integrity problems are *findings with a code*, never exceptions, so a caller sees all of them at once.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# Signatures in vc_render_tifxyz / render_ink.py output that mean the score is not a fresh render.
STALE_SLICES = "all slices exist, skipping"  # sampler re-used old per-slice TIFFs and exited 0
ZERO_STRIP = ("p95=0.0", "rendered strip is entirely zero")  # all-black strip (villa #1660 / #1886)
# vc_render_tifxyz prints this only with --surface-interpolation smooth (villa #1818, 2026-09-30);
# its absence in a render log means the default, linear (and every pre-#1818 build is linear).
SMOOTH_INTERP = "Surface interpolation: smooth"

FAIL, WARN = "FAIL", "WARN"

SEED_DIR = re.compile(r"seed-(-?\d+)")  # runners/run_single.py: output / f"seed-{seed}"
AGGREGATE = "aggregate_metrics.json"  # runners/run_single.py AGGREGATE_METRICS_FILENAME
LOG_CODES = ("STALE_SLICES", "ZERO_STRIP")  # findings read from the render log

# metrics.json summary fields that define the scorer; runs are only comparable if these agree.
# villa #1805 records `model_dir`, the Hugging Face snapshot path (.../snapshots/<hash>); only its
# final component is compared, since the rest of the path is machine-specific.
SCORER_FIELDS = ("model", "checkpoint", "fg_threshold", "folds")


def scorer_identity(summary: dict) -> dict:
    ident = {k: summary[k] for k in SCORER_FIELDS if k in summary}
    md = summary.get("model_dir")
    if isinstance(md, str) and md:
        ident["snapshot"] = Path(md).name
    return ident


@dataclass
class Finding:
    level: str  # FAIL or WARN
    code: str
    detail: str


@dataclass
class Run:
    path: Path
    metrics_path: Path | None = None
    log_path: Path | None = None
    total_fg_pixels: float | None = None
    scorer: dict = field(default_factory=dict)
    surface_interp: str | None = None  # 'smooth' / 'linear' from the render log; None: no log
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.level == FAIL for f in self.findings)


def find_metrics(path: Path) -> Path | None:
    """The run's metrics.json: the path itself, or the unique one under it."""
    if path.is_file() and path.name == "metrics.json":
        return path
    hits = sorted(path.rglob("ink_metric/metrics.json")) if path.is_dir() else []
    return hits[0] if len(hits) == 1 else None


def find_log(path: Path, metrics: Path | None) -> Path | None:
    """The render log, if the layout says where it is.

    autoresearch.md layout: <out_dir>/logs/<tag>.ink.log, where the run dir is <out_dir>/<datedir>_<tag>.
    run_sweep.py layout: <sweep_output>/.sweep/logs/<name>.log for run_single output <sweep_output>/<name>,
    whose seed runs are <sweep_output>/<name>/seed-<s>.
    """
    run_dir = path if path.is_dir() else path.parent
    tag = None
    if metrics is not None:
        # .../meshes/fitted_<tag>/ink_metric/metrics.json
        fitted = metrics.parent.parent.name
        if fitted.startswith("fitted_"):
            tag = fitted[len("fitted_"):]
    if tag is not None:
        for base in (run_dir.parent, run_dir):
            cand = base / "logs" / f"{tag}.ink.log"
            if cand.is_file():
                return cand
    root = run_dir.parent if SEED_DIR.fullmatch(run_dir.name) else run_dir
    cand = root.parent / ".sweep" / "logs" / f"{root.name}.log"
    return cand if cand.is_file() else None


def load_run(path: str | Path, log: str | Path | None = None) -> Run:
    p = Path(path)
    run = Run(path=p)
    if not p.exists():
        run.findings.append(Finding(FAIL, "RUN_NOT_FOUND", f"{p} does not exist"))
        return run
    m = find_metrics(p)
    if m is None:
        n = len(list(p.rglob("ink_metric/metrics.json"))) if p.is_dir() else 0
        why = "none found" if n == 0 else f"{n} found; point at one run"
        run.findings.append(
            Finding(FAIL, "METRICS_MISSING", f"no unique ink_metric/metrics.json under {p} ({why})")
        )
        return run
    run.metrics_path = m
    try:
        summary = json.loads(m.read_text())["summary"]
        run.total_fg_pixels = float(summary["total_fg_pixels"])
    except (ValueError, KeyError, TypeError) as e:
        run.findings.append(Finding(FAIL, "METRICS_UNREADABLE", f"{m}: {type(e).__name__}: {e}"))
        return run
    run.scorer = scorer_identity(summary)
    if run.total_fg_pixels <= 0:
        run.findings.append(
            Finding(FAIL, "FG_ZERO", "total_fg_pixels is 0: nothing was scored (see villa #1660)")
        )

    lp = Path(log) if log is not None else find_log(p, m)
    if lp is None or not lp.is_file():
        run.findings.append(
            Finding(
                WARN,
                "LOG_NOT_FOUND",
                "render log not found; stale-slice and zero-strip checks were NOT run. Pass --log "
                "(villa's runners/run_single.py keeps no log file: pass the file you redirected its "
                "output to)",
            )
        )
    else:
        run.log_path = lp
        text = lp.read_text(errors="replace")
        run.surface_interp = "smooth" if SMOOTH_INTERP in text else "linear"
        if STALE_SLICES in text:
            run.findings.append(
                Finding(
                    FAIL,
                    "STALE_SLICES",
                    f"the render re-used existing per-slice TIFFs ('{STALE_SLICES}'): this score "
                    "is an OLD render re-scored, not this run",
                )
            )
        for sig in ZERO_STRIP:
            if sig in text:
                run.findings.append(
                    Finding(FAIL, "ZERO_STRIP", f"an ink strip rendered entirely zero ('{sig}')")
                )
                break
    if "snapshot" not in run.scorer:
        run.findings.append(
            Finding(
                WARN,
                "SCORER_SNAPSHOT_UNRECORDED",
                "metrics.json has no model_dir, so the scoring model snapshot is unknown "
                "(villa #1805, 2026-09-15, records it)",
            )
        )
    return run


def seed_dirs(path: Path) -> list[Path]:
    """The seed-<s> sub-runs of a run_single --seeds output, in seed order; [] if it is not one."""
    if not path.is_dir():
        return []
    hits = [d for d in path.iterdir() if d.is_dir() and SEED_DIR.fullmatch(d.name)]
    return sorted(hits, key=lambda d: int(SEED_DIR.fullmatch(d.name).group(1)))


def expand_runs(path: str | Path, log: str | Path | None = None) -> list[Run]:
    """One run per seed for a run_single --seeds output directory, else the single run at `path`.

    A given log applies to every seed: run_single sends all seeds' output to one stream. The
    aggregate_metrics.json run_single writes is cross-checked against the seed runs found, since a
    resumed or hand-edited output can hold seed dirs the aggregate never saw (or the reverse).
    """
    p = Path(path)
    seeds = seed_dirs(p)
    if not seeds:
        return [load_run(p, log)]
    runs = [load_run(d, log) for d in seeds]
    if len(runs) > 1:
        for r in runs:
            for f in r.findings:
                if f.code in LOG_CODES:
                    f.detail += (
                        f"; {r.log_path} is shared by all {len(runs)} seed runs of {p.name}, so it "
                        "cannot say which seed's render this was, and every one is failed"
                    )  # fmt: skip
    agg_path = p / AGGREGATE
    if agg_path.is_file():
        finding = _check_aggregate(agg_path, seeds, runs)
        if finding is not None:
            runs[0].findings.append(finding)
    return runs


def _check_aggregate(agg_path: Path, seeds: list[Path], runs: list[Run]) -> Finding | None:
    try:
        agg = json.loads(agg_path.read_text())
        listed = sorted(int(x) for x in agg["seeds"])
        stats = agg["final"]["total_fg_pixels"]
        mean, count = float(stats["mean"]), int(stats["count"])
    except (ValueError, KeyError, TypeError) as e:
        return Finding(WARN, "AGGREGATE_UNREADABLE", f"{agg_path}: {type(e).__name__}: {e}")
    found = sorted(int(SEED_DIR.fullmatch(d.name).group(1)) for d in seeds)
    values = [r.total_fg_pixels for r in runs if r.total_fg_pixels is not None]
    if listed != found or count != len(values):
        return Finding(
            WARN, "AGGREGATE_MISMATCH",
            f"{agg_path.name} covers seeds {listed} ({count} scores) but the directory holds seeds "
            f"{found} ({len(values)} scored); inkdelta uses the seed runs, not the aggregate",
        )  # fmt: skip
    if values and not math.isclose(mean, sum(values) / len(values), rel_tol=1e-9):
        return Finding(
            WARN, "AGGREGATE_MISMATCH",
            f"{agg_path.name} mean total_fg_pixels {mean:.1f} != mean of the seed runs' "
            f"metrics.json {sum(values) / len(values):.1f}: a seed was re-scored after aggregation",
        )  # fmt: skip
    return None
