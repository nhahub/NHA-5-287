"""Class-imbalance handling: SMOTE, ADASYN and random over-sampling (dependency-free; numpy + scikit-learn only).

Why a wrapper and not a pre-processing step: resampling must happen INSIDE every cross-validation training fold and never touch the
validation fold or the held-out test set, otherwise synthetic points leak into the scores. `ResamplingClassifier` does exactly that:
its `fit` resamples the (already preprocessed) training matrix, `predict`/`predict_proba` pass straight through.
It sits as the last step of the sklearn Pipeline, so cross_val_score, RandomizedSearchCV, clone and joblib all just work.

  SMOTE   (Chawla 2002)  each synthetic sample lies on the segment between a minority sample and one of its k nearest minority neighbours.
  ADASYN  (He 2008)      like SMOTE, but harder samples (those with many majority-class neighbours) get proportionally more synthetic points.
  random                 duplicate minority rows (the safe fallback for huge sparse matrices such as TF-IDF).

Multi-class is handled one-vs-rest: every class below the target count is lifted towards it.
Interpolation happens in the preprocessed space, so a one-hot column can take a fractional value for a synthetic point: that is the
documented behaviour of plain SMOTE (no SMOTE-NC); tree models and linear models accept it without trouble.
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.metaestimators import available_if

METHODS = ("smote", "adasyn", "random")
SPARSE_DENSE_CELLS = 30_000_000        # above this many cells a sparse matrix is never densified (random over-sampling is used instead)


def _targets(counts: Dict[Any, int], strategy: Union[str, float]) -> Dict[Any, int]:
    """Desired size per class. 'auto' lifts every class to the majority count; a float r in (0, 1] lifts each class to r * majority."""
    top = max(counts.values())
    ratio = 1.0 if strategy == "auto" else float(strategy)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("sampling_strategy must be 'auto' or a ratio in (0, 1].")
    return {c: max(n, int(round(ratio * top))) for c, n in counts.items()}


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Integer allocation of `total` proportional to weights (sums to exactly `total`)."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0 or total <= 0:
        return np.zeros(len(w), dtype=int)
    raw = w / w.sum() * total
    base = np.floor(raw).astype(int)
    rest = total - int(base.sum())
    if rest > 0:
        base[np.argsort(-(raw - base))[:rest]] += 1
    return base


def _interpolate(Xc: np.ndarray, base_idx: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """One synthetic sample per entry of base_idx: base + u * (random one of its k nearest same-class neighbours - base)."""
    n = len(Xc)
    if n < 2 or k < 1:
        return Xc[base_idx].copy()                                  # a lone sample can only be duplicated
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xc)
    neigh = nn.kneighbors(Xc[base_idx], return_distance=False)[:, 1:]       # column 0 is the point itself
    pick = neigh[np.arange(len(base_idx)), rng.randint(0, neigh.shape[1], size=len(base_idx))]
    gap = rng.uniform(0.0, 1.0, size=(len(base_idx), 1))
    return Xc[base_idx] + gap * (Xc[pick] - Xc[base_idx])


def resample(X, y, method: str = "smote", k_neighbors: int = 5, sampling_strategy: Union[str, float] = "auto",
             random_state: Optional[int] = 42) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """Return (X_resampled, y_resampled, info). Original rows come first, synthetic rows are appended. Never mutates its inputs."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    count = dict(zip(classes.tolist(), counts.tolist()))
    info: Dict[str, Any] = {"method": method, "requested": method, "added": {}, "k_neighbors": k_neighbors, "before": count}
    if len(classes) < 2:
        info["note"] = "single class: nothing to resample"
        return X, y, info
    want = _targets(count, sampling_strategy)
    todo = {c: want[c] - count[c] for c in count if want[c] > count[c]}
    if not todo:
        info["after"] = count
        return X, y, info
    rng = np.random.RandomState(random_state)
    is_sp = sparse.issparse(X)
    n_cells = X.shape[0] * X.shape[1]
    if method != "random" and is_sp and n_cells > SPARSE_DENSE_CELLS:
        method, info["method"] = "random", "random"
        info["note"] = "matrix too large/sparse to interpolate (e.g. TF-IDF): used random over-sampling instead"
    if method == "random":
        extra_idx = [rng.choice(np.where(y == c)[0], size=n, replace=True) for c, n in todo.items()]
        idx = np.concatenate(extra_idx)
        X_new = X[idx]
        y_new = y[idx]
        X_out = sparse.vstack([X, X_new]).tocsr() if is_sp else np.vstack([X, X_new])
        info["added"] = {c: int(n) for c, n in todo.items()}
        info["after"] = {c: count[c] + todo.get(c, 0) for c in count}
        return X_out, np.concatenate([y, y_new]), info
    Xd = X.toarray() if is_sp else np.asarray(X, dtype=float)
    new_X, new_y = [], []
    full_nn = None
    for c, n_new in todo.items():
        Xc = Xd[y == c]
        k_c = min(k_neighbors, max(len(Xc) - 1, 0))
        used = method
        if method == "adasyn" and len(Xc) >= 2:
            if full_nn is None:
                full_nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(Xd))).fit(Xd)
            neigh = full_nn.kneighbors(Xc, return_distance=False)[:, 1:]
            hardness = (y[neigh] != c).mean(axis=1)                      # share of other-class points among the k nearest neighbours
            if hardness.sum() > 0:
                alloc = _largest_remainder(hardness, n_new)
                base_idx = np.repeat(np.arange(len(Xc)), alloc)
            else:                                                        # the class is already well separated: ADASYN degenerates to SMOTE
                used = "smote"
                base_idx = rng.randint(0, len(Xc), size=n_new)
        else:
            used = "smote" if method == "adasyn" else method
            base_idx = rng.randint(0, len(Xc), size=n_new)
        if len(Xc) < 2:
            used = "random"
        if used != info["method"]:
            info.setdefault("fallbacks", {})[str(c)] = used
        new_X.append(_interpolate(Xc, base_idx, k_c, rng))
        new_y.append(np.full(len(base_idx), c, dtype=y.dtype))
        info["added"][c] = int(len(base_idx))
    X_syn = np.vstack(new_X)
    X_out = np.vstack([Xd, X_syn])
    y_out = np.concatenate([y] + new_y)
    info["after"] = {c: int((y_out == c).sum()) for c in count}
    return X_out, y_out, info


def _inner_has(attr: str):
    return lambda self: hasattr(self.estimator, attr)


class ResamplingClassifier(ClassifierMixin, BaseEstimator):
    """Fit `estimator` on a SMOTE / ADASYN / random-oversampled copy of the training data it receives; predict as usual.

    Use as the last Pipeline step. Hyper-parameter names of the wrapped model are reachable as `estimator__<name>`.
    """
    _nx_wrapper = True

    def __init__(self, estimator, method: str = "smote", k_neighbors: int = 5, sampling_strategy: Union[str, float] = "auto", random_state: Optional[int] = 42):
        self.estimator, self.method, self.k_neighbors = estimator, method, k_neighbors
        self.sampling_strategy, self.random_state = sampling_strategy, random_state

    def fit(self, X, y, **fit_params):
        Xr, yr, info = resample(X, y, self.method, self.k_neighbors, self.sampling_strategy, self.random_state)
        self.estimator_ = clone(self.estimator).fit(Xr, yr, **fit_params)
        self.classes_ = self.estimator_.classes_
        self.resampling_ = info
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    @available_if(_inner_has("predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @available_if(_inner_has("decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    def __getattr__(self, name):                 # feature_importances_, coef_, ... of the fitted inner model
        if name.startswith("__") or name == "estimator_" or "estimator_" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["estimator_"], name)


def describe(info: Dict[str, Any]) -> str:
    """One line for reports: 'SMOTE added 412 synthetic rows (class 1: +412)'."""
    if not info or not info.get("added"):
        return "no resampling needed"
    tot = sum(info["added"].values())
    per = ", ".join(f"class {c}: +{n}" for c, n in info["added"].items())
    return f"{info['method'].upper()} added {tot} synthetic training rows ({per})" + (f"; {info['note']}" if info.get("note") else "")
