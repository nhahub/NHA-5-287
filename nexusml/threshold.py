"""Automatic decision-threshold tuning for binary classifiers (instead of the default 0.5).

Method (no leakage): the threshold is chosen on OUT-OF-FOLD predicted probabilities of the TRAINING set (cross_val_predict), by default
maximising F-beta along the precision-recall curve. The held-out test set is only used afterwards to REPORT what the new threshold does,
never to choose it. The threshold is adopted only if it improves the objective out-of-fold by at least `min_gain`, so well-balanced
problems whose best threshold is ~0.5 are left alone instead of chasing noise.

Objectives
  f1 / fbeta   maximise F-beta on the PR curve (beta > 1 favours recall, < 1 favours precision)
  pr_distance  point of the PR curve closest to (recall = 1, precision = 1)
  youden       maximise TPR - FPR on the ROC curve (balanced between sensitivity and specificity)
  cost         minimise cost_fn * FN + cost_fp * FP (when the user supplied business costs)
"""
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

OBJECTIVES = ("f1", "fbeta", "pr_distance", "youden", "cost")


class ThresholdedClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a binary classifier and predicts the positive class when P(positive) >= threshold. Probabilities pass through unchanged
    (so ROC-AUC and calibration are untouched). `fit` clones and refits the inner model and KEEPS the threshold."""
    _nx_wrapper = True

    def __init__(self, estimator, threshold: float = 0.5, positive_index: int = 1):
        self.estimator, self.threshold, self.positive_index = estimator, threshold, positive_index

    @classmethod
    def prefit(cls, fitted, threshold: float, positive_index: int = 1) -> "ThresholdedClassifier":
        obj = cls(fitted, threshold, positive_index)
        obj.estimator_ = fitted
        obj.classes_ = np.asarray(fitted.classes_)
        return obj

    def fit(self, X, y, **kw):
        self.estimator_ = clone(self.estimator).fit(X, y, **kw)
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        p = self.estimator_.predict_proba(X)[:, self.positive_index]
        pos, neg = self.classes_[self.positive_index], self.classes_[1 - self.positive_index]
        return np.where(p >= self.threshold, pos, neg)

    def __getattr__(self, name):                 # named_steps, feature_importances_, ... of the wrapped model
        if name.startswith("__") or name == "estimator_" or "estimator_" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["estimator_"], name)


def oof_proba(est, X, y, n_splits: int = 5, random_state: int = 42) -> np.ndarray:
    """Out-of-fold P(class index 1) for every training row."""
    y = np.asarray(y)
    k = int(max(2, min(n_splits, np.bincount(y.astype(int)).min()))) if y.dtype.kind in "iu" else n_splits
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    return cross_val_predict(clone(est), X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def _fbeta(p: np.ndarray, r: np.ndarray, beta: float) -> np.ndarray:
    b2 = beta * beta
    den = b2 * p + r
    return np.where(den > 0, (1 + b2) * p * r / np.where(den > 0, den, 1), 0.0)


def best_threshold(y, p, objective: str = "f1", beta: float = 1.0, cost_fn: Optional[float] = None, cost_fp: Optional[float] = None) -> Dict[str, Any]:
    """Pick the threshold for P(positive). y is 0/1 (1 = positive). Ties go to the candidate closest to 0.5."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if y.min() == y.max():
        return {"threshold": 0.5, "objective": objective, "note": "only one class present"}
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]                                   # align with thr (the last PR point has no threshold)
    if objective in ("f1", "fbeta"):
        b = 1.0 if objective == "f1" else float(beta)
        score, cand = _fbeta(prec, rec, b), thr
    elif objective == "pr_distance":
        score, cand = -np.sqrt((1 - prec) ** 2 + (1 - rec) ** 2), thr
    elif objective == "youden":
        fpr, tpr, t2 = roc_curve(y, p)
        keep = np.isfinite(t2)
        score, cand = (tpr - fpr)[keep], t2[keep]
    else:
        if cost_fn is None or cost_fp is None:
            raise ValueError("objective 'cost' needs cost_fn and cost_fp")
        cand = np.unique(np.concatenate([thr, [0.5]]))
        fn = np.array([((p < t) & (y == 1)).sum() for t in cand])
        fp = np.array([((p >= t) & (y == 0)).sum() for t in cand])
        score = -(cost_fn * fn + cost_fp * fp)
    top = score.max()
    near = np.where(score >= top - 1e-12)[0]
    i = near[np.argmin(np.abs(cand[near] - 0.5))]
    t = float(np.clip(cand[i], 0.0, 1.0))
    return {"threshold": round(t, 4), "objective": objective, "score": round(float(top), 4), "beta": float(beta) if objective == "fbeta" else None}


def _scores(y, p, t: float, beta: float = 1.0) -> Dict[str, float]:
    pred = (np.asarray(p) >= t).astype(int)
    return {"precision": round(float(precision_score(y, pred, zero_division=0)), 4), "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4), "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4)}


def tune(fitted, X_train, y_train, X_test=None, y_test=None, objective: str = "f1", beta: float = 1.0, min_gain: float = 0.005,
         cost_fn: Optional[float] = None, cost_fp: Optional[float] = None, n_splits: int = 5, random_state: int = 42,
         oof: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Choose a threshold from out-of-fold training probabilities, decide whether to adopt it, and report both operating points.

    Returns {'adopted', 'threshold', 'estimator', 'report'}; `estimator` is `fitted` wrapped in ThresholdedClassifier when adopted,
    otherwise `fitted` itself. `report` is JSON-friendly and is what the user-facing report shows.
    """
    p_oof = oof if oof is not None else oof_proba(fitted, X_train, y_train, n_splits, random_state)
    y_tr = np.asarray(y_train).astype(int)
    choice = best_threshold(y_tr, p_oof, objective, beta, cost_fn, cost_fp)
    t = choice["threshold"]
    rep: Dict[str, Any] = {"objective": objective, "beta": beta, "threshold": t, "default_threshold": 0.5, "selected_on": "out-of-fold training predictions",
                           "oof_at_default": _scores(y_tr, p_oof, 0.5), "oof_at_tuned": _scores(y_tr, p_oof, t)}
    key = "f1" if objective in ("f1", "fbeta", "pr_distance", "youden") else "balanced_accuracy"
    if objective == "fbeta":
        gain = float(_fb(y_tr, p_oof, t, beta) - _fb(y_tr, p_oof, 0.5, beta))
    elif objective == "cost":
        c = lambda th: cost_fn * ((p_oof < th) & (y_tr == 1)).sum() + cost_fp * ((p_oof >= th) & (y_tr == 0)).sum()
        gain = float((c(0.5) - c(t)) / max(c(0.5), 1e-9))
    else:
        gain = rep["oof_at_tuned"][key] - rep["oof_at_default"][key]
    rep["oof_gain"] = round(gain, 4)
    adopted = abs(t - 0.5) > 1e-6 and gain >= min_gain
    rep["adopted"] = bool(adopted)
    rep["reason"] = (f"out-of-fold {key} improves by {gain:+.3f} at threshold {t:.3f}" if adopted else
                     f"kept 0.5: tuned threshold {t:.3f} gains only {gain:+.3f} out-of-fold (needs >= {min_gain})")
    est = ThresholdedClassifier.prefit(fitted, t) if adopted else fitted
    if X_test is not None and y_test is not None and len(np.unique(y_test)) == 2:
        p_te = fitted.predict_proba(X_test)[:, 1]
        yt = np.asarray(y_test).astype(int)
        rep["heldout_at_default"], rep["heldout_at_tuned"] = _scores(yt, p_te, 0.5), _scores(yt, p_te, t)
    return {"adopted": bool(adopted), "threshold": t if adopted else 0.5, "estimator": est, "report": rep}


def _fb(y, p, t, beta):
    pred = (np.asarray(p) >= t).astype(int)
    pr, rc = precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0)
    b2 = beta * beta
    return 0.0 if (b2 * pr + rc) == 0 else (1 + b2) * pr * rc / (b2 * pr + rc)
