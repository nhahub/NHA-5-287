"""Data sentinel: the deterministic scans behind the Sentinel agent. It finds problems; the Director (LLM) decides what to do about them.

  leakage     a single column that predicts the target almost perfectly on its own (shallow tree, cross-validated) is usually a leak, an ID or an outcome proxy
  id-like     columns that are unique per row (they memorise, they do not generalise)
  drift       adversarial validation: can a model tell training rows from test rows? (AUC ~0.5 for a random split)
  conflicts   identical feature rows with different outcomes: a hard ceiling on accuracy / R^2 that no model can beat
Findings carry a severity; only 'high' ones are ever proposed for quarantine.
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Division of labour: the profiler (profiling.py, LEAK_SINGLE_FEATURE = 0.97) drops OBVIOUS leaks automatically before the sentinel ever runs. The sentinel works in the
# REVIEW ZONE just below that line, where a very strong legitimate predictor and a subtle leak look alike and only the column's meaning can tell them apart (an LLM judgement).
HIGH = {"binary": 0.92, "multi": 0.90, "reg": 0.90}      # may be proposed for quarantine
MEDIUM = {"binary": 0.85, "multi": 0.80, "reg": 0.80}    # reported as a warning only
DRIFT_AUC = 0.65


def _encode(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).fillna(s.median() if s.notna().any() else 0.0).to_numpy().reshape(-1, 1)
    return pd.factorize(s.astype(str))[0].reshape(-1, 1).astype(float)


def single_column_power(X: pd.DataFrame, y: np.ndarray, task: str, max_rows: int = 4000, seed: int = 42) -> List[Dict[str, Any]]:
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), min(len(y), max_rows), replace=False)
    ys = np.asarray(y)[idx]
    out = []
    k = len(np.unique(ys))
    for c in X.columns:
        try:
            Z = _encode(X[c].iloc[idx])
            if task == "classification":
                m = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=seed)
                sc = "roc_auc" if k == 2 else "balanced_accuracy"
            else:
                m = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=seed)
                sc = "r2"
            v = float(np.mean(cross_val_score(m, Z, ys, cv=3, scoring=sc)))
            out.append({"column": c, "score": round(v, 4), "metric": sc})
        except Exception:
            continue
    return sorted(out, key=lambda r: -r["score"])


def leakage_suspects(X: pd.DataFrame, y: np.ndarray, task: str, target_name: str = "") -> List[Dict[str, Any]]:
    power = single_column_power(X, y, task)
    k = len(np.unique(y)) if task == "classification" else 0
    key = "reg" if task != "classification" else ("binary" if k == 2 else "multi")
    out = []
    tn = str(target_name).lower()
    for r in power:
        name_hint = bool(tn) and len(tn) >= 3 and tn in str(r["column"]).lower()
        if r["score"] >= HIGH[key]:
            out.append({**r, "severity": "high", "kind": "one column predicts the target almost perfectly", "name_hint": name_hint})
        elif r["score"] >= MEDIUM[key] or name_hint:
            out.append({**r, "severity": "medium", "kind": "very strong single-column signal" if not name_hint else "column name contains the target name", "name_hint": name_hint})
    return out[:12]


def id_like(X: pd.DataFrame) -> List[str]:
    n = len(X)
    return [c for c in X.columns if n >= 50 and X[c].nunique() >= 0.95 * n and not pd.api.types.is_float_dtype(X[c])]


def drift(X_train: pd.DataFrame, X_test: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    try:
        A = pd.concat([X_train.iloc[: min(len(X_train), 3000)], X_test.iloc[: min(len(X_test), 3000)]], ignore_index=True)
        lab = np.r_[np.zeros(min(len(X_train), 3000)), np.ones(min(len(X_test), 3000))]
        Z = np.hstack([_encode(A[c]) for c in A.columns])
        auc = float(np.mean(cross_val_score(RandomForestClassifier(n_estimators=60, max_depth=6, random_state=seed, n_jobs=1), Z, lab, cv=3, scoring="roc_auc")))
        return {"auc": round(auc, 3), "shifted": auc >= DRIFT_AUC}
    except Exception as e:
        return {"auc": None, "shifted": False, "error": type(e).__name__}


def conflicts(X: pd.DataFrame, y: np.ndarray, task: str) -> Dict[str, Any]:
    try:
        h = pd.util.hash_pandas_object(X.astype(str).fillna("NA"), index=False)
        g = pd.Series(h.values)
        dup = float((g.map(g.value_counts()) > 1).mean())
        if dup < 0.01:
            return {"available": False, "duplicate_row_share": round(dup, 4)}
        df = pd.DataFrame({"h": g.values, "y": np.asarray(y)})
        if task == "classification":
            lb = float(sum(len(s) - s.value_counts().iloc[0] for _, s in df.groupby("h")["y"]) / len(df))
            return {"available": True, "duplicate_row_share": round(dup, 4), "min_error": round(lb, 4), "max_accuracy": round(1 - lb, 4)}
        yy = df.y.astype(float)
        sse = float(((yy - yy.groupby(df.h).transform("mean")) ** 2).sum())
        return {"available": True, "duplicate_row_share": round(dup, 4), "max_r2": round(1 - sse / (float(((yy - yy.mean()) ** 2).sum()) + 1e-12), 4)}
    except Exception as e:
        return {"available": False, "error": type(e).__name__}


def scan(splits: Dict[str, Any], task: str, target_name: str = "") -> Dict[str, Any]:
    X, y = splits["X_train"], np.asarray(splits["y_train"])
    rep: Dict[str, Any] = {"n_rows": int(len(y)), "n_features": int(X.shape[1])}
    rep["leakage_suspects"] = leakage_suspects(X, y, task, target_name) if task in ("classification", "regression") else []
    rep["id_like"] = id_like(X)
    rep["drift"] = drift(X, splits["X_test"])
    rep["conflicts"] = conflicts(X, y, task)
    rep["constant"] = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if task == "classification":
        vc = np.bincount(y) / len(y)
        rep["minority_share"] = round(float(vc.min()), 4)
    rep["n_flagged_high"] = sum(1 for s in rep["leakage_suspects"] if s["severity"] == "high")
    return rep
