"""Standard-library statistics: Welch interval on a relative difference, a CV-based interval for
single runs, and the exact false-positive rate of the "all B beat all A" seed rule.

No scipy: the Student-t quantile is computed from the regularized incomplete beta function
(continued fraction, Numerical Recipes `betacf`) and inverted by bisection. It supports fractional
degrees of freedom, which Welch intervals need. Tests check it against scipy.
"""

from __future__ import annotations

import math
import statistics as st
from math import comb


def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t (p in (0, 1)), by bisection on t_cdf."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def welch_relative(a: list[float], b: list[float], alpha: float = 0.05) -> dict | None:
    """(mean(b) - mean(a)) / mean(a), with a Welch (1 - alpha) interval. Needs >= 2 runs per side."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0.0 or ma == 0.0:
        return None
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = (va / na + vb / nb) ** 2 / den
    tcrit = t_ppf(1 - alpha / 2, df)
    d = mb - ma
    return {"rel": d / ma, "lo": (d - tcrit * se) / ma, "hi": (d + tcrit * se) / ma, "df": df,
            "method": "welch"}  # fmt: skip


def cv_relative(a: list[float], b: list[float], cv: float, alpha: float = 0.05) -> dict:
    """Interval from a known run-to-run CV (for 1 run per side). Normal approximation: each mean has
    relative sd cv/sqrt(n), so the relative difference has sd cv*sqrt(1/na + 1/nb)."""
    ma, mb = st.mean(a), st.mean(b)
    z = st.NormalDist().inv_cdf(1 - alpha / 2)
    half = z * cv * math.sqrt(1 / len(a) + 1 / len(b))
    rel = mb / ma - 1
    return {"rel": rel, "lo": rel - half, "hi": rel + half, "cv": cv, "method": "cv"}


def all_beat_null_rate(k: int) -> float:
    """P(all k B-runs beat all k A-runs | no effect, exchangeable runs) = 1 / C(2k, k)."""
    return 1.0 / comb(2 * k, k)
