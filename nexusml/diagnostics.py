"""Extra model diagnostics (scikit-learn only): learning curve, calibration, decision thresholds,
partial dependence. Each returns small JSON-friendly dicts that are stored in state and charted."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import partial_dependence
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import learning_curve

from . import config


def learning_curve_data(est, X: pd.DataFrame, y: np.ndarray, scoring: str, cv) -> Optional[Dict[str, Any]]:
    if len(getattr(est, "members", None) or []) > 4:                # refitting a large ensemble 12 times is not worth the wait; skipped, never faked
        return None
    try:
        sizes, tr, va = learning_curve(clone(est), X, y, cv=cv, scoring=scoring, train_sizes=[0.25, 0.5, 0.75, 1.0], n_jobs=1, shuffle=False)
    except Exception:
        return None
    return {"train_sizes": [int(s) for s in sizes], "train_mean": tr.mean(1).round(4).tolist(), "train_std": tr.std(1).round(4).tolist(),
            "val_mean": va.mean(1).round(4).tolist(), "val_std": va.std(1).round(4).tolist()}


def calibration_data(y: np.ndarray, p: np.ndarray, bins: int = 8) -> Optional[Dict[str, Any]]:
    try:
        frac, mean_p = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    except Exception:
        return None
    return {"fraction_positive": frac.round(4).tolist(), "mean_predicted": mean_p.round(4).tolist(),
            "brier": round(float(brier_score_loss(y, p)), 4), "mean_pred": round(float(p.mean()), 4), "base_rate": round(float(np.mean(y)), 4)}


def threshold_table(y: np.ndarray, p: np.ndarray, n: int = 49) -> List[Dict[str, Any]]:
    """Counts at each probability threshold: lets the app re-price errors with the user's own business costs."""
    rows = []
    for t in np.linspace(0.02, 0.98, n):
        pred = p >= t
        tp, fp = int(((pred) & (y == 1)).sum()), int(((pred) & (y == 0)).sum())
        fn, tn = int(((~pred) & (y == 1)).sum()), int(((~pred) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"threshold": round(float(t), 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0})
    return rows


def pdp_curves(est, X: pd.DataFrame, features: List[str], k: int = 3) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    Xs = X.sample(min(len(X), 600), random_state=config.RANDOM_STATE)
    for f in features:
        if len(out) >= k or f not in Xs.columns or not pd.api.types.is_numeric_dtype(Xs[f]) or Xs[f].nunique() < 4:
            continue
        try:
            r = partial_dependence(est, Xs, [f], grid_resolution=15, kind="average")
            grid = r.get("grid_values", r.get("values"))[0]
            out[f] = {"x": np.asarray(grid).round(4).tolist(), "y": np.asarray(r["average"][0]).round(5).tolist()}
        except Exception:
            continue
    return out
