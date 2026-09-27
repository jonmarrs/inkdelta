"""Find a spiral-fitting run's ink score and render log, and check the run is trustworthy.

A "run" is whatever `spiral-fitting/run_single.py` (villa) left behind for one tag:

    <out_dir>/<datedir>_<tag>/meshes/fitted_<tag>/ink_metric/metrics.json
    <out_dir>/logs/<tag>.ink.log

Any directory that contains an ``ink_metric/metrics.json`` somewhere below it is accepted, and the
render log can always be given explicitly, so custom layouts work too.

Integrity problems are *findings with a code*, never exceptions, so a caller sees all of them at once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Signatures in vc_render_tifxyz / render_ink.py output that mean the score is not a fresh render.
STALE_SLICES = "all slices exist, skipping"  # sampler re-used old per-slice TIFFs and exited 0
ZERO_STRIP = ("p95=0.0", "rendered strip is entirely zero")  # all-black strip (villa #1660 / #1886)

FAIL, WARN = "FAIL", "WARN"

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
    """villa layout: <out_dir>/logs/<tag>.ink.log, where the run dir is <out_dir>/<datedir>_<tag>."""
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
    return None


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
                "render log not found; stale-slice and zero-strip checks were NOT run (pass --log)",
            )
        )
    else:
        run.log_path = lp
        text = lp.read_text(errors="replace")
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
