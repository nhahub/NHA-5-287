"""Feature recipes and a SAFE feature-idea language for the AutoML agent.

The LLM never writes code. It picks recipe names and proposes feature ideas as small JSON objects; this module validates them against the real columns and
executes them deterministically. Everything fits on the TRAINING split only and is applied unchanged to new data (no target information leaks through the
ideas: group statistics and encodings use features only).

  recipes   power          Yeo-Johnson on numeric columns (skewed data)
            missing_ind    add a "was missing" flag per column that has gaps
            target_enc     cross-fitted target encoding for high-cardinality categoricals
            interactions   pairwise products of the most target-related columns
            select         keep the most useful columns (tree importances), drop noise
  ideas     ratio, diff, product, sum (two numeric columns) | log1p, sqrt, square, abs, is_missing, bin_quantile (one column)
            count_encode (one column) | group_mean (a categorical + a numeric column)
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

RECIPES = {
    "power": "Yeo-Johnson transform of numeric columns: helps skewed features / linear models",
    "missing_ind": "add missing-value indicator flags: helps when 'missing' itself is informative",
    "target_enc": "cross-fitted target encoding of high-cardinality categoricals instead of one-hot",
    "interactions": "pairwise products of the most target-related columns: helps linear models and shallow trees",
    "select": "keep the most important columns only: helps with many noisy features",
}
IDEA_OPS = {"ratio": 2, "diff": 2, "product": 2, "sum": 2, "log1p": 1, "sqrt": 1, "square": 1, "abs": 1, "is_missing": 1, "bin_quantile": 1, "count_encode": 1, "group_mean": 2}
MAX_IDEAS = 12
_NAME = re.compile(r"[^a-z0-9_]+")


# idea validation
def validate_ideas(ideas: Any, X: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[str]]:
    """-> (clean ideas, reasons for each rejection). Rejects unknown ops, missing columns, wrong column types, duplicates; never raises."""
    clean: List[Dict[str, Any]] = []
    why: List[str] = []
    seen = set()
    num = {c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])}
    for i, d in enumerate(ideas if isinstance(ideas, list) else []):
        if not isinstance(d, dict):
            why.append(f"idea {i}: not an object")
            continue
        op, cols = str(d.get("op", "")).lower(), d.get("cols")
        if op not in IDEA_OPS:
            why.append(f"idea {i}: unknown op '{op}'")
            continue
        if not isinstance(cols, list) or len(cols) != IDEA_OPS[op] or any(c not in X.columns for c in cols):
            why.append(f"idea {i} ({op}): needs {IDEA_OPS[op]} existing column(s), got {cols}")
            continue
        if op == "group_mean":
            ok = cols[0] != cols[1] and cols[1] in num and X[cols[0]].nunique() <= 50
        elif op == "count_encode":
            ok = True
        elif op == "is_missing":
            ok = bool(X[cols[0]].isna().any())
        else:
            ok = all(c in num for c in cols)
        if not ok:
            why.append(f"idea {i} ({op}): wrong column type(s) or nothing to compute for {cols}")
            continue
        param = d.get("param")
        if op == "bin_quantile":
            try:
                param = int(min(10, max(3, int(param or 5))))
            except (TypeError, ValueError):
                param = 5
        name = "idea_" + _NAME.sub("_", str(d.get("name") or f"{op}_{'_'.join(cols)}").lower())[:40].strip("_")
        key = (op, tuple(cols), param)
        if key in seen or name in {c["name"] for c in clean}:
            why.append(f"idea {i}: duplicate")
            continue
        seen.add(key)
        clean.append({"name": name, "op": op, "cols": list(cols), **({"param": param} if param is not None else {}), "why": str(d.get("why", ""))[:160]})
        if len(clean) >= MAX_IDEAS:
            break
    return clean, why


class IdeaTransformer(BaseEstimator, TransformerMixin):
    """DataFrame -> the same DataFrame plus the engineered columns. Learned statistics come from the training rows only."""

    def __init__(self, ideas: Optional[List[Dict[str, Any]]] = None):
        self.ideas = ideas

    def fit(self, X, y=None):
        self.maps_: Dict[str, Any] = {}
        for d in self.ideas or []:
            c = d["cols"]
            if d["op"] == "count_encode":
                self.maps_[d["name"]] = X[c[0]].astype(str).value_counts(normalize=True).to_dict()
            elif d["op"] == "group_mean":
                self.maps_[d["name"]] = (X.groupby(X[c[0]].astype(str))[c[1]].mean().to_dict(), float(pd.to_numeric(X[c[1]], errors="coerce").mean()))
            elif d["op"] == "bin_quantile":
                self.maps_[d["name"]] = np.unique(np.nanquantile(pd.to_numeric(X[c[0]], errors="coerce"), np.linspace(0, 1, int(d["param"]) + 1)[1:-1]))
        return self

    def transform(self, X):
        out = X.copy()
        for d in self.ideas or []:
            c, op, nm = d["cols"], d["op"], d["name"]
            f = lambda k: pd.to_numeric(X[k], errors="coerce").astype(float)
            with np.errstate(all="ignore"):
                if op == "ratio":
                    b = f(c[1]).replace(0, np.nan)
                    v = f(c[0]) / b
                elif op == "diff":
                    v = f(c[0]) - f(c[1])
                elif op == "product":
                    v = f(c[0]) * f(c[1])
                elif op == "sum":
                    v = f(c[0]) + f(c[1])
                elif op == "log1p":
                    a = f(c[0])
                    v = np.log1p(a.clip(lower=0))
                elif op == "sqrt":
                    v = np.sqrt(f(c[0]).clip(lower=0))
                elif op == "square":
                    v = f(c[0]) ** 2
                elif op == "abs":
                    v = f(c[0]).abs()
                elif op == "is_missing":
                    v = X[c[0]].isna().astype(float)
                elif op == "bin_quantile":
                    v = pd.Series(np.digitize(f(c[0]).fillna(np.nan).to_numpy(), self.maps_[nm]).astype(float), index=X.index)
                elif op == "count_encode":
                    v = X[c[0]].astype(str).map(self.maps_[nm]).astype(float)
                else:                                                   # group_mean
                    mp, fallback = self.maps_[nm]
                    v = X[c[0]].astype(str).map(mp).astype(float).fillna(fallback)
            out[nm] = pd.Series(v, index=X.index).replace([np.inf, -np.inf], np.nan).astype(float)
        return out


# post-preprocessing steps (dense matrices)
class Interactions(BaseEstimator, TransformerMixin):
    """Append pairwise products of the k columns most related to the target (by F-score on a sample). Sparse input (text) passes through."""

    def __init__(self, k: int = 6, task: str = "classification", seed: int = 42):
        self.k, self.task, self.seed = k, task, seed

    def fit(self, X, y=None):
        self.cols_ = None
        if sparse.issparse(X) or X.shape[1] < 2 or y is None:
            return self
        from sklearn.feature_selection import f_classif, f_regression
        n = X.shape[0]
        idx = np.random.RandomState(self.seed).choice(n, min(n, 5000), replace=False)
        Xs, ys = np.nan_to_num(np.asarray(X)[idx]), np.asarray(y)[idx]
        try:
            f = (f_classif if self.task == "classification" else f_regression)(Xs, ys)[0]
            f = np.nan_to_num(f)
        except Exception:
            f = Xs.var(axis=0)
        self.cols_ = np.argsort(-f)[:min(self.k, X.shape[1])]
        return self

    def transform(self, X):
        if sparse.issparse(X) or self.cols_ is None or len(self.cols_) < 2:
            return X
        X = np.asarray(X)
        c = self.cols_
        prods = [X[:, a] * X[:, b] for i, a in enumerate(c) for b in c[i + 1:]]
        return np.hstack([X, np.column_stack(prods)])


class SelectTop(BaseEstimator, TransformerMixin):
    """Keep the `frac` most important columns according to a quick tree ensemble (fitted on the training rows only)."""

    def __init__(self, frac: float = 0.6, task: str = "classification", seed: int = 42):
        self.frac, self.task, self.seed = frac, task, seed

    def fit(self, X, y=None):
        self.keep_ = None
        d = X.shape[1]
        if y is None or d < 8:
            return self
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        m = (ExtraTreesClassifier if self.task == "classification" else ExtraTreesRegressor)(n_estimators=80, random_state=self.seed, n_jobs=1, max_depth=12)
        m.fit(X, y)
        self.keep_ = np.sort(np.argsort(-m.feature_importances_)[:max(4, int(round(self.frac * d)))])
        return self

    def transform(self, X):
        return X if self.keep_ is None else X[:, self.keep_]
