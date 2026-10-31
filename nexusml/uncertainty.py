"""Uncertainty and stability: bootstrap confidence intervals for held-out metrics, multi-seed stability, cost-based thresholds.

All numbers are computed by code. Intervals come from resampling the held-out set (nothing is re-trained),
thresholds are chosen on out-of-fold TRAINING predictions (never on the test set).
"""
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score

from . import config


def _metric(name: str):
    if name == "roc_auc":
        return lambda y, p, pr: roc_auc_score(y, pr)
    if name == "accuracy":
        return lambda y, p, pr: accuracy_score(y, p)
    if name == "balanced_accuracy":
        return lambda y, p, pr: balanced_accuracy_score(y, p)
    if name == "f1_macro":
        return lambda y, p, pr: f1_score(y, p, average="macro", zero_division=0)
    if name == "f1_weighted":
        return lambda y, p, pr: f1_score(y, p, average="weighted", zero_division=0)
    if name == "r2":
        return lambda y, p, pr: r2_score(y, p)
    if name == "rmse":
        return lambda y, p, pr: float(np.sqrt(mean_squared_error(y, p)))
    if name == "mae":
        return lambda y, p, pr: mean_absolute_error(y, p)
    raise ValueError(f"unknown metric {name}")


def bootstrap_ci(metric: str, y: np.ndarray, pred: np.ndarray, proba: Optional[np.ndarray] = None,
                 n: Optional[int] = None, level: Optional[float] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    """Percentile bootstrap CI of a held-out metric. Resamples are skipped when they cannot compute the metric
    (e.g. a single class); the number of valid resamples is reported."""
    f = _metric(metric)
    n = n or config.BOOTSTRAP_N
    level = level or config.CI_LEVEL
    rng = np.random.RandomState(config.RANDOM_STATE if seed is None else seed)
    y = np.asarray(y)
    pred = np.asarray(pred)
    pr = None if proba is None else np.asarray(proba)
    point = float(f(y, pred, pr))
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        try:
            vals.append(float(f(y[i], pred[i], None if pr is None else pr[i])))
        except ValueError:
            continue
    if len(vals) < max(20, n // 5):
        return {"metric": metric, "value": round(point, 4), "lo": None, "hi": None, "level": level, "n_boot": len(vals)}
    a = (1 - level) / 2
    lo, hi = np.percentile(vals, [100 * a, 100 * (1 - a)])
    return {"metric": metric, "value": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "level": level, "n_boot": len(vals)}


def seed_stability(est, X, y, task: str, scoring: str, seeds=None) -> Dict[str, Any]:
    """Cross-validated score of the final pipeline under several fold-assignment seeds: how much does the number move?"""
    seeds = seeds or [config.RANDOM_STATE + 17 * k for k in range(config.SEED_STABILITY_SEEDS)]
    out = []
    for s in seeds:
        cv = (StratifiedKFold(3, shuffle=True, random_state=s) if task == "classification" and np.bincount(y.astype(int)).min() >= 3
              else KFold(3, shuffle=True, random_state=s))
        try:
            out.append(float(np.mean(cross_val_score(est, X, y, cv=cv, scoring=scoring, n_jobs=1))))
        except Exception:
            continue
    if not out:
        return {}
    return {"scoring": scoring, "seeds": seeds[:len(out)], "scores": [round(v, 4) for v in out],
            "mean": round(float(np.mean(out)), 4), "sd": round(float(np.std(out)), 4),
            "range": round(float(max(out) - min(out)), 4)}


def cost_threshold(est, X_train, y_train, X_test, y_test, cost_fn: Optional[float], cost_fp: Optional[float]) -> Dict[str, Any]:
    """Pick the probability cutoff that minimises expected cost on out-of-fold TRAINING predictions, then report it on the test set.
    When the user gave no costs, labelled defaults are used and flagged as such."""
    user_costs = cost_fn is not None and cost_fp is not None
    cfn = float(cost_fn if cost_fn is not None else config.DEFAULT_COST_FN)
    cfp = float(cost_fp if cost_fp is not None else config.DEFAULT_COST_FP)
    y_train = np.asarray(y_train)
    cv = StratifiedKFold(3, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(est, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    grid = np.linspace(0.02, 0.98, 97)

    def exp_cost(y, p, t):
        pred = p >= t
        return float(cfn * ((~pred) & (y == 1)).sum() + cfp * ((pred) & (y == 0)).sum())
    costs = [exp_cost(y_train, oof, t) for t in grid]
    t_best = float(grid[int(np.argmin(costs))])
    p_te = est.predict_proba(X_test)[:, 1]
    y_te = np.asarray(y_test)
    pred = p_te >= t_best
    tp, fp = int((pred & (y_te == 1)).sum()), int((pred & (y_te == 0)).sum())
    fn, tn = int(((~pred) & (y_te == 1)).sum()), int(((~pred) & (y_te == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    c_best = cfn * fn + cfp * fp
    c_default = exp_cost(y_te, p_te, 0.5)
    c_none = cfn * int((y_te == 1).sum())                 # flag nobody: every positive is missed
    return {"threshold": round(t_best, 3), "cost_fn": cfn, "cost_fp": cfp, "costs_from_user": user_costs,
            "test": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 3), "recall": round(rec, 3),
                     "flagged": int(pred.sum()), "expected_cost": round(c_best, 2),
                     "cost_at_0.5": round(c_default, 2), "cost_flag_nobody": round(c_none, 2)}}
