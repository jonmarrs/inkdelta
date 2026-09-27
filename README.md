# inkdelta

[![tests](https://github.com/jonmarrs/inkdelta/actions/workflows/tests.yml/badge.svg)](https://github.com/jonmarrs/inkdelta/actions/workflows/tests.yml)

**Is a `total_fg_pixels` difference between villa spiral-fitting runs real?**

villa's `spiral-fitting/autoresearch.md` loop keeps a change if its ink score beats the baseline. Three things
can make that comparison meaningless without any error being raised. `inkdelta` checks each of them from
the files a run already leaves behind. It is CPU-only, needs no GPU and no network, and depends only on
the standard library.

| check | why it matters (measured, not assumed) |
|---|---|
| **The score is a fresh render.** It fails if the render log shows `all slices exist, skipping`, an all-zero strip, or a zero score. | When per-slice TIFFs already exist, the published `vc_render_tifxyz` skips sampling and exits 0. A copied run dir then re-scores an **old** render under a new name. |
| **Both sides were scored and sampled the same way.** Scorer model, checkpoint, folds, threshold and HF snapshot must match; the sampler build is declared per side. | After villa #1146 (2026-07-14), source builds step twice as far along the normal as the published image (built 2026-05-13). On the same surface that alone moves `total_fg_pixels` **+5.0% to +9.2%**. |
| **The difference exceeds run-to-run noise.** It gives a Welch interval from replicates, or an interval from a supplied CV. | Fits that differ only by seed spread with CV ≈ 7% on the region we measured. The "all B runs beat all A runs" rule passes a null change **1 in 6** times with two seeds per side and **1 in 20** with three. |

## Use

```bash
pip install .            # or: PYTHONPATH=src python -m inkdelta.cli ...

# one run: is its score a fresh, non-empty render?
inkdelta check out/2026-09-27_s1_jul9a

# two arms, seeds as replicates, same sampler build declared on both sides
inkdelta compare --a out/*_base_s? --b out/*_change_s? --build-a edge-0513 --build-b edge-0513

# measure YOUR run-to-run CV from seed replicates (one --group per config; means may differ)
inkdelta noise --group out/*_base_s? --group out/*_other_s?
#   -> run-to-run CV 0.0736, 95% interval [0.0506, 0.1344], df 9

# one run per side: use that CV
inkdelta compare --a out/*_base --b out/*_change --cv 0.0736 --json result.json
```

`noise` pools each run's relative deviation from its own group's mean, so configs with different
ink levels can be combined. It gives a chi-square interval. On our twelve runs it reproduces the
registered floor exactly. With few replicates the interval is wide: at df 9 the true CV can be almost
twice the estimate, which is why it is printed.

A run is a villa run directory (`<out_dir>/<datedir>_<tag>`, whose log `<out_dir>/logs/<tag>.ink.log`
is found automatically), any directory holding exactly one `ink_metric/metrics.json`, or a `metrics.json`
itself. Pass `--log-a/--log-b` for other layouts.

**Verdicts:**

* `RESOLVED (+/-)`: the interval excludes zero.
* `NOT RESOLVED`: it does not.
* `INVALID`: a run failed integrity.
* `INCOMPARABLE`: different scorer or declared sampler build.
* `NO NOISE ESTIMATE`: one run per side and no `--cv`.

Exit code 0 means a statistical verdict was reached; 2 means invalid or incomparable; 3 means no noise
estimate.

## What it cannot do

* It cannot detect *which* `vc_render_tifxyz` build sampled a run; nothing in the outputs records it. You
  declare it. If you do not, it warns with the measured size of the most common mismatch.
* Its noise figure is only as good as the replicates or the CV you give it. Ours (0.074) was measured on
  one PHercParis4 region at one code tier. Measure your own.
* A `NOT RESOLVED` is not "no effect". It is the interval, and the interval is what you report.

## Provenance

The checks come from measurements in
[jonmarrs/vesuvius-autoresearch](https://github.com/jonmarrs/vesuvius-autoresearch), each pre-registered:

* noise floor: `reports/the_pooled_fit_only_floor.md`;
* install route: `reports/step2_across_surfaces.md`;
* stale slices: finding 65.

The tool is validated against that corpus (`docs/preregistration/2026-09-27_inkdelta_validation.md`
there): it must flag every known-bad comparison, pass every known-good one, and reproduce a registered
interval exactly.

MIT licensed.
