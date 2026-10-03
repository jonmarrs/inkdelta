"""Is the total_fg_pixels difference between two sets of spiral-fitting runs real?

Order of questions, each of which can stop the answer:

1. **Integrity.** Is every run's score a fresh, non-empty render? (`runs.py`.) Any FAIL -> INVALID.
2. **Comparability.** Were both sides scored by the same scorer and sampled by the same
   `vc_render_tifxyz` build? A scorer mismatch -> INCOMPARABLE. For the sampler the user declares
   the build per side; different declarations -> INCOMPARABLE. None declared -> a warning with the
   measured size of the most common mismatch.
3. **Size vs noise.** With >= 2 runs per side: a Welch interval, unless a supplied run-to-run CV
   implies a wider one, which then decides (SPREAD_BELOW_FLOOR). With 1 run per side: the CV's
   interval, or no verdict if none is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .runs import FAIL, WARN, Finding, Run
from .stats import all_beat_null_rate, cv_relative, welch_relative

# Measured, not assumed: the same flat surface scored after sampling by the published
# volume-cartographer image (pre villa #1146) vs a post-#1146 build, 4 surfaces.
# https://github.com/jonmarrs/vesuvius-autoresearch  reports/step2_across_surfaces.md
ROUTE_EFFECT = "+5.0% to +9.2% (mean +6.5%) on 4 PHercParis4 surfaces"
# linear vs smooth (villa #1818), measured in villa's own pipeline on 8 labelled PHercParis4 segments
# (vesuvius-autoresearch reports/scorer_vs_labels_surface_interpolation.md); larger per-crop swings under
# per-crop normalisation in reports/surface_interpolation_relocates_ink.md
INTERP_EFFECT = (
    "-3.6% to +4.1% per segment (median 0.9%; up to +/-9% per 2048-px window) on 8 PHercParis4 "
    "segments, with no change in agreement with villa's ink labels"
)


@dataclass
class Comparison:
    verdict: str
    detail: str
    interval: dict | None = None
    findings: list[Finding] = field(default_factory=list)
    seed_rule_null_rate: float | None = None


def _scorer_mismatch(runs: list[Run]) -> tuple[list[str], list[str]]:
    """(conflicts, unverifiable): a field whose recorded values disagree is a conflict; a field
    recorded on only some runs (e.g. the snapshot, which predates villa #1805 on older scorers)
    cannot be verified either way."""
    keys = set().union(*(r.scorer.keys() for r in runs))
    bad, partial = [], []
    for k in sorted(keys):
        present = {repr(r.scorer[k]) for r in runs if k in r.scorer}
        if len(present) > 1:
            bad.append(f"{k}: {sorted(present)}")
        elif any(k not in r.scorer for r in runs):
            partial.append(k)
    return bad, partial


def compare(
    a: list[Run],
    b: list[Run],
    build_a: str | None = None,
    build_b: str | None = None,
    cv: float | None = None,
    alpha: float = 0.05,
) -> Comparison:
    findings: list[Finding] = []
    for side, runs in (("A", a), ("B", b)):
        for r in runs:
            for f in r.findings:
                findings.append(Finding(f.level, f.code, f"{side} {r.path}: {f.detail}"))
    if not a or not b:
        return Comparison("INVALID", "each side needs at least one run", findings=findings)
    if any(f.level == FAIL for f in findings):
        return Comparison(
            "INVALID", "a run failed an integrity check; its score is not a fresh render",
            findings=findings,
        )  # fmt: skip

    mism, partial = _scorer_mismatch(a + b)
    if mism:
        return Comparison(
            "INCOMPARABLE", "runs were scored differently: " + "; ".join(mism), findings=findings
        )
    if partial:
        findings.append(
            Finding(
                WARN,
                "SCORER_PARTLY_UNVERIFIABLE",
                f"recorded on only some runs, so sameness cannot be checked: {', '.join(partial)}",
            )
        )
    modes = {r.surface_interp for r in a + b if r.surface_interp is not None}
    if len(modes) > 1:
        def who(m):
            return ", ".join(str(r.path) for r in a + b if r.surface_interp == m)
        return Comparison(
            "INCOMPARABLE",
            "runs were rendered with different --surface-interpolation modes (smooth: "
            f"{who('smooth')}; linear: {who('linear')}). On the same surface the mode alone moves "
            f"total_fg_pixels {INTERP_EFFECT}",
            findings=findings,
        )
    if "smooth" in modes and any(r.surface_interp is None for r in a + b):
        findings.append(
            Finding(
                WARN,
                "SURFACE_INTERP_UNVERIFIED",
                "some runs were rendered with --surface-interpolation smooth and others have no "
                f"render log to say; mixing modes alone moves total_fg_pixels {INTERP_EFFECT}",
            )
        )
    if build_a is not None and build_b is not None and build_a != build_b:
        return Comparison(
            "INCOMPARABLE",
            f"sides were sampled by different vc_render_tifxyz builds ({build_a!r} vs {build_b!r}). "
            f"The published image and a post-villa#1146 build differ by {ROUTE_EFFECT} on the same "
            "surface, so this difference would mix the change with the build",
            findings=findings,
        )
    if build_a is None or build_b is None:
        findings.append(
            Finding(
                WARN,
                "SAMPLER_BUILD_UNDECLARED",
                "the vc_render_tifxyz build is not declared for both sides (--build-a/--build-b). "
                f"If they differ (published image vs a post-villa#1146 build), expect {ROUTE_EFFECT} "
                "from that alone",
            )
        )

    fa = [r.total_fg_pixels for r in a]
    fb = [r.total_fg_pixels for r in b]
    k = len(a) if len(a) == len(b) else None
    rule = None
    if k and min(fb) > max(fa):
        rule = all_beat_null_rate(k)

    iv = welch_relative(fa, fb, alpha)
    if iv is not None and cv is not None:
        # A small group can land tight by chance. When the replicates vary less than a measured
        # floor says runs do, their Welch interval claims more precision than the process has,
        # so the floor's interval decides. Measured case: three control seeds at CV 0.0124 against
        # a floor of 0.0536 turned a +/-9% null into a +/-3% one (vesuvius-autoresearch
        # reports/control_sensitivity.md).
        floor = cv_relative(fa, fb, cv, alpha)
        if floor["hi"] - floor["lo"] > iv["hi"] - iv["lo"]:
            findings.append(
                Finding(
                    WARN,
                    "SPREAD_BELOW_FLOOR",
                    f"the replicates vary less than --cv {cv} says runs do: Welch gives "
                    f"[{iv['lo']:+.2%}, {iv['hi']:+.2%}], the floor [{floor['lo']:+.2%}, "
                    f"{floor['hi']:+.2%}]. Deciding on the wider one: a tight small group is "
                    "weak evidence of a quieter process",
                )
            )
            iv = {**floor, "welch": iv}
    if iv is None and cv is not None:
        iv = cv_relative(fa, fb, cv, alpha)
    if iv is None:
        rel = sum(fb) / len(fb) / (sum(fa) / len(fa)) - 1
        return Comparison(
            "NO NOISE ESTIMATE",
            f"difference {rel:+.2%}, but with fewer than 2 runs per side and no --cv there is no "
            "way to tell it from run-to-run noise",
            interval={"rel": rel},
            findings=findings,
            seed_rule_null_rate=rule,
        )
    if iv["lo"] > 0:
        head = "RESOLVED (+)"
    elif iv["hi"] < 0:
        head = "RESOLVED (-)"
    else:
        head = "NOT RESOLVED"
    text = f"{iv['rel']:+.2%}, {1 - alpha:.0%} interval [{iv['lo']:+.2%}, {iv['hi']:+.2%}] ({iv['method']})"
    return Comparison(head, text, interval=iv, findings=findings, seed_rule_null_rate=rule)
