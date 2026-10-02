"""Small, dependency-light statistics helpers (numpy/scipy/sklearn)."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BOOT = 10_000
RNG_SEED = 12345


def balanced_accuracy(y, p) -> float:
    y = np.asarray(y, bool)
    p = np.asarray(p, bool)
    tpr = (p & y).sum() / max(y.sum(), 1)
    tnr = (~p & ~y).sum() / max((~y).sum(), 1)
    return float((tpr + tnr) / 2)


def cohen_kappa(a, b) -> float:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def bootstrap_ci_pairs(y, p, fn, boot: int = BOOT):
    """Percentile 95% CI of fn(y, p) resampling items."""
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(RNG_SEED)
    n = len(y)
    vals = []
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(y[idx], p[idx]))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def mean_ci(values, boot: int = BOOT):
    """Mean and percentile-bootstrap 95% CI over units (e.g. questions)."""
    v = np.asarray([x for x in values if x is not None and not _isnan(x)], float)
    if len(v) == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(v), (boot, len(v)))
    means = v[idx].mean(axis=1)
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def paired_diff_ci(a: dict, b: dict, boot: int = BOOT):
    """Mean of b[k]-a[k] over shared keys with bootstrap CI (paired by unit)."""
    keys = sorted(set(a) & set(b))
    d = np.asarray([b[k] - a[k] for k in keys], float)
    if len(d) == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(d), (boot, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "n": int(len(d)),
        "mean": float(d.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "p_boot_le0": float((means <= 0).mean()),
    }


def auroc(y, score):
    y = np.asarray(y, int)
    s = np.asarray(score, float)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, s))


def auroc_ci(y, score, boot: int = 2000):
    y = np.asarray(y, int)
    s = np.asarray(score, float)
    if len(set(y.tolist())) < 2:
        return [None, None]
    rng = np.random.default_rng(RNG_SEED)
    vals = []
    for _ in range(boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"rho": None, "p": None, "n": int(len(x))}
    r = spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(x))}


def _isnan(x) -> bool:
    try:
        return math.isnan(x)
    except TypeError:
        return False
