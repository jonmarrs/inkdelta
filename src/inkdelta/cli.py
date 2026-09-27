"""inkdelta: is a total_fg_pixels difference between spiral-fitting runs real?

    inkdelta check RUN [--log LOG]
    inkdelta compare --a RUN [RUN ...] --b RUN [RUN ...] [--log-a LOG ...] [--log-b LOG ...]
                     [--build-a NAME] [--build-b NAME] [--cv CV] [--json OUT]

A RUN is a villa run directory (<out_dir>/<datedir>_<tag>, with <out_dir>/logs/<tag>.ink.log), any
directory holding exactly one ink_metric/metrics.json, or a metrics.json itself.
Exit codes: 0 resolved / not resolved, 2 invalid or incomparable, 3 no noise estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .compare import compare
from .runs import FAIL, load_run


def _load(paths, logs):
    logs = logs or []
    if logs and len(logs) != len(paths):
        raise SystemExit("--log-a/--log-b must be given once per run, in the same order")
    return [load_run(p, logs[i] if logs else None) for i, p in enumerate(paths)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="inkdelta", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="integrity of one run")
    c.add_argument("run")
    c.add_argument("--log")
    m = sub.add_parser("compare", help="is B different from A?")
    m.add_argument("--a", nargs="+", required=True)
    m.add_argument("--b", nargs="+", required=True)
    m.add_argument("--log-a", nargs="+")
    m.add_argument("--log-b", nargs="+")
    m.add_argument("--build-a", help="label of the vc_render_tifxyz build that sampled side A")
    m.add_argument("--build-b", help="label of the vc_render_tifxyz build that sampled side B")
    m.add_argument("--cv", type=float, help="run-to-run CV of total_fg_pixels (needed for 1 run per side)")
    m.add_argument("--json", help="write the full result as JSON here")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        r = load_run(args.run, args.log)
        print(f"{r.path}: total_fg_pixels={r.total_fg_pixels}")
        for f in r.findings:
            print(f"  {f.level} {f.code}: {f.detail}")
        print("OK" if r.ok else "FAIL")
        return 0 if r.ok else 2

    a, b = _load(args.a, args.log_a), _load(args.b, args.log_b)
    res = compare(a, b, args.build_a, args.build_b, args.cv)
    print(res.verdict)
    print(f"  {res.detail}")
    if res.seed_rule_null_rate is not None:
        print(
            f"  note: every B run beat every A run; with no real effect that happens "
            f"{res.seed_rule_null_rate:.1%} of the time (1/C(2k,k)), so it is weak evidence alone"
        )
    for f in res.findings:
        print(f"  {f.level} {f.code}: {f.detail}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(asdict(res), fh, indent=2, default=str)
    if res.verdict in ("INVALID", "INCOMPARABLE") or any(f.level == FAIL for f in res.findings):
        return 2
    return 3 if res.verdict == "NO NOISE ESTIMATE" else 0


if __name__ == "__main__":
    sys.exit(main())
