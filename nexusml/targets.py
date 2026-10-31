"""Metric targets: "tune until the model reaches X".

A target is checked on OUT-OF-FOLD predictions of the TRAINING split (never on the test set), so chasing it cannot leak test
information into model selection. The decision threshold (when the metric needs one) is also chosen on those out-of-fold
predictions. The final report then shows the held-out value (with a bootstrap interval) next to the target and says plainly
whether it was met and why the search stopped.

spec = {"metric": "recall", "min": 0.70, "subject_to": {"metric": "precision", "min": 0.50}, "max_retries": 4, "max_seconds": 300}
(use "max" instead of "min" for rmse / mae).
"""
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from . import config

THRESHOLD_FREE = {"roc_auc", "average_precision"}
POINT = {"f1", "precision", "recall", "accuracy", "balanced_accuracy"}
REG = {"r2", "rmse", "mae"}
LOWER_BETTER = {"rmse", "mae"}
ALIASES = {"auc": "roc_auc", "pr_auc": "average_precision", "ap": "average_precision", "f1_macro": "f1"}


def parse(spec: Optional[Dict[str, Any]], task: str) -> Optional[Dict[str, Any]]:
    if not spec:
        return None
    m = ALIASES.get(str(spec.get("metric", "")).lower(), str(spec.get("metric", "")).lower())
    allowed = (THRESHOLD_FREE | POINT) if task == "classification" else REG
    if m not in allowed:
        raise ValueError(f"target metric '{m}' is not valid for {task}; choose from {sorted(allowed)}")

    def bound(d, metric):
        key = "max" if metric in LOWER_BETTER else "min"
        if key not in d:
            raise ValueError(f"target for '{metric}' needs a '{key}' value")
        return key, float(d[key])
    k, v = bound(spec, m)
    out = {"metric": m, "bound": k, "value": v, "max_retries": int(spec.get("max_retries", 4)),
           "max_seconds": float(spec.get("max_seconds", config.TARGET_MAX_SECONDS)), "secondary": None}
    sub = spec.get("subject_to")
    if sub:
        sm = ALIASES.get(str(sub.get("metric", "")).lower(), str(sub.get("metric", "")).lower())
        if sm not in POINT or m not in POINT or sm == m:
            raise ValueError("'subject_to' needs two different threshold-based metrics (precision/recall/f1/accuracy/balanced_accuracy)")
        sk, sv = bound(sub, sm)
        out["secondary"] = {"metric": sm, "bound": sk, "value": sv}
    return out


def _ok(val: float, bound: str, target: float) -> bool:
    return val >= target - 1e-12 if bound == "min" else val <= target + 1e-12


def _point(metric: str, y, pred) -> float:
    if metric == "f1":
        return float(f1_score(y, pred, zero_division=0))
    if metric == "precision":
        return float(precision_score(y, pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y, pred, zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    return float(balanced_accuracy_score(y, pred))


def _value(metric: str, y, score, thr: Optional[float]) -> float:
    if metric == "roc_auc":
        return float(roc_auc_score(y, score))
    if metric == "average_precision":
        return float(average_precision_score(y, score))
    if metric in POINT:
        return _point(metric, y, (score >= (0.5 if thr is None else thr)).astype(int))
    if metric == "r2":
        return float(r2_score(y, score))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y, score)))
    return float(mean_absolute_error(y, score))


def choose_threshold(spec: Dict[str, Any], y, p) -> Optional[float]:
    """Threshold for point metrics, chosen on out-of-fold training predictions only."""
    m = spec["metric"]
    if m not in POINT:
        return None
    grid = np.linspace(0.02, 0.98, 97)
    sec = spec.get("secondary")
    best, best_val = None, -1.0
    for t in grid:
        pred = (p >= t).astype(int)
        if sec and not _ok(_point(sec["metric"], y, pred), sec["bound"], sec["value"]):
            continue
        v = _point(m, y, pred)
        if v > best_val:
            best, best_val = float(t), v
    if best is None:                                                    # constraint infeasible: pick the point with the smallest relative shortfall on BOTH goals
        def shortfall(t):
            pred = (p >= t).astype(int)
            a = _point(m, y, pred) / max(spec["value"], 1e-9)
            b = _point(sec["metric"], y, pred) / max(sec["value"], 1e-9)
            return min(a, b)
        best = float(max(grid, key=shortfall))
    return best if (sec or m in ("f1", "accuracy", "balanced_accuracy")) else 0.5


def evaluate(spec: Dict[str, Any], y, score, thr: Optional[float]) -> Dict[str, Any]:
    val = _value(spec["metric"], y, score, thr)
    met = _ok(val, spec["bound"], spec["value"])
    out = {"metric": spec["metric"], "value": round(val, 4), "threshold": None if thr is None else round(thr, 3)}
    if spec.get("secondary"):
        s = spec["secondary"]
        sv = _value(s["metric"], y, score, thr)
        out["secondary"] = {"metric": s["metric"], "value": round(sv, 4)}
        met = met and _ok(sv, s["bound"], s["value"])
    out["met"] = bool(met)
    return out


def oof_scores(est, X, y, task: str):
    y = np.asarray(y)
    if task == "classification":
        cv = StratifiedKFold(3, shuffle=True, random_state=config.RANDOM_STATE)
        return cross_val_predict(est, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    return cross_val_predict(est, X, y, cv=KFold(3, shuffle=True, random_state=config.RANDOM_STATE), n_jobs=1)


def validation_report(est, X_train, y_train, spec: Dict[str, Any], task: str) -> Dict[str, Any]:
    y = np.asarray(y_train)
    score = oof_scores(est, X_train, y, task)
    thr = choose_threshold(spec, y, score) if task == "classification" else None
    r = evaluate(spec, y, score, thr)
    r["source"] = "out-of-fold predictions on the training split"
    return r


def heldout_report(est, X_test, y_test, spec: Dict[str, Any], task: str, thr: Optional[float]) -> Dict[str, Any]:
    from . import uncertainty                                           # local import: avoids a cycle at package import
    y = np.asarray(y_test)
    score = est.predict_proba(X_test)[:, 1] if task == "classification" else est.predict(X_test)
    r = evaluate(spec, y, score, thr)
    rng = np.random.RandomState(config.RANDOM_STATE)
    vals = []
    for _ in range(config.BOOTSTRAP_N):
        i = rng.randint(0, len(y), len(y))
        try:
            vals.append(_value(spec["metric"], y[i], score[i], thr))
        except ValueError:
            continue
    if len(vals) >= 50:
        lo, hi = np.percentile(vals, [2.5, 97.5])
        r["ci"] = [round(float(lo), 4), round(float(hi), 4)]
    return r
