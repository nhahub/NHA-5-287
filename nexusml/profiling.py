"""Data-engineer tools: profiling and column auditing (pure functions, no state)."""
import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config

DATE_RE = re.compile(r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})")


def to_datetime(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


def is_integer_like(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    nn = s.dropna()
    if nn.empty:
        return False
    if pd.api.types.is_integer_dtype(nn):
        return True
    return bool(np.all(np.isclose(nn.to_numpy(dtype=float) % 1, 0)))


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    nn = s.dropna()
    if nn.empty:
        return "categorical"
    as_str = nn.astype(str)
    sample = as_str.head(200)
    if sample.map(lambda v: bool(DATE_RE.match(v))).mean() >= 0.9:
        if to_datetime(nn.head(500)).notna().mean() >= 0.9:
            return "datetime"
    from . import featurizers
    if featurizers.is_image_path(nn):
        return "image_path"
    if as_str.str.len().mean() > 40 and nn.nunique() / len(nn) > 0.5 or featurizers.is_text(nn):
        return "text"
    return "categorical"


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows, n_cols = df.shape
    columns: Dict[str, Any] = {}
    warns: List[str] = []
    for col in df.columns:
        s = df[col]
        kind = column_kind(s)
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        info: Dict[str, Any] = {
            "kind": kind,
            "dtype": str(s.dtype),
            "missing_count": n_missing,
            "missing_pct": round(n_missing / max(1, n_rows) * 100, 2),
            "unique_count": n_unique,
            "unique_ratio": round(n_unique / max(1, n_rows - n_missing), 4),
            "is_integer_like": is_integer_like(s),
            "sample": [str(v) for v in s.dropna().head(3).tolist()],   # not sent to the LLM unless opted in
        }
        if kind == "numeric":
            nn = s.dropna().astype(float)
            if len(nn):
                info.update(
                    mean=float(nn.mean()), std=float(nn.std() if len(nn) > 1 else 0.0),
                    min=float(nn.min()), max=float(nn.max()),
                    skew=float(nn.skew()) if len(nn) > 2 and nn.nunique() > 2 else 0.0,
                )
        columns[str(col)] = info
        if info["missing_pct"] >= 30:
            warns.append(f"Column '{col}' is {info['missing_pct']}% missing.")
    dups = int(df.duplicated().sum())
    if dups:
        warns.append(f"{dups} fully duplicated rows.")
    if n_rows < config.MIN_ROWS:
        warns.append(f"Only {n_rows} rows; results will be unreliable.")
    return {"n_rows": n_rows, "n_cols": n_cols, "duplicates": dups, "columns": columns, "warnings": warns}


def data_quality(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compact data-quality summary + 0-100 score (used in the report and as evidence)."""
    n = max(1, profile["n_rows"])
    cols = profile["columns"]
    miss_cells = sum(c["missing_count"] for c in cols.values())
    miss_pct = miss_cells / (n * max(1, len(cols))) * 100
    constant = [c for c, i in cols.items() if i["unique_count"] <= 1]
    high_card = [c for c, i in cols.items() if i["kind"] == "categorical" and i["unique_ratio"] > 0.5]
    outlier_cols = {}
    for c, i in cols.items():
        if i["kind"] == "numeric" and i["unique_count"] > 10:
            x = df[c].dropna().astype(float)
            if len(x) >= 20:
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    share = float(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean())
                    if share > 0.01:
                        outlier_cols[c] = round(share * 100, 2)
    dup_pct = profile["duplicates"] / n * 100
    d_miss, d_dup = min(40, miss_pct * 2), min(20, dup_pct * 2)
    d_const, d_out = min(15, len(constant) * 5), min(15, len(outlier_cols) * 2)
    score = 100 - d_miss - d_dup - d_const - d_out
    deductions = [{"reason": r, "points": round(float(v), 1)} for r, v in
                  (("missing cells", d_miss), ("duplicate rows", d_dup), (f"{len(constant)} constant column(s) (carry no information, removed automatically)", d_const),
                   (f"{len(outlier_cols)} column(s) with >1% IQR outliers (often legitimate extremes)", d_out)) if v > 0]
    sc = int(max(0, round(score)))
    label = "excellent" if sc >= 95 else "good" if sc >= 85 else "fair" if sc >= 70 else "poor"
    return {"deductions": deductions, "label": label,
            "explanation": ("No deductions." if not deductions else "Points lost for: " + "; ".join(f"{d['reason']} (-{d['points']:g})" for d in deductions) + "."),
            "missing_cell_pct": round(miss_pct, 2), "duplicate_row_pct": round(dup_pct, 2),
            "constant_columns": constant, "high_cardinality_columns": high_card,
            "outlier_columns_pct": outlier_cols, "score": int(max(0, round(score)))}


def _numeric_target(y: pd.Series, task: str) -> Optional[pd.Series]:
    """Numeric view of the target for correlation checks (None if not meaningful)."""
    if task == "classification":
        codes = y.astype("category").cat.codes.astype(float).where(y.notna())
        return codes if y.nunique() == 2 else None
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        return y.astype(float)
    return None


def single_feature_power(x: pd.Series, y: pd.Series, task: str) -> Optional[float]:
    """CV score of a shallow tree using ONE feature. Near-perfect => that feature leaks the answer.
    Catches non-linear / non-monotonic leaks that correlation misses."""
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    sub = pd.concat([x, y], axis=1).dropna()
    if len(sub) < config.MIN_ROWS:
        return None
    sub = sub.sample(min(len(sub), 4000), random_state=config.RANDOM_STATE)
    xv = sub.iloc[:, 0]
    xv = xv.astype(float) if pd.api.types.is_numeric_dtype(xv) and not pd.api.types.is_bool_dtype(xv) \
        else xv.astype("category").cat.codes.astype(float)
    yv = sub.iloc[:, 1]
    try:
        if task == "classification":
            yy = yv.astype("category").cat.codes
            if yy.nunique() < 2 or yy.value_counts().min() < 3:
                return None
            m, sc = DecisionTreeClassifier(max_depth=4, random_state=0), "balanced_accuracy"
        else:
            yy = yv.astype(float)
            m, sc = DecisionTreeRegressor(max_depth=6, random_state=0), "r2"
        return float(cross_val_score(m, xv.to_frame(), yy, cv=3, scoring=sc).mean())
    except Exception:
        return None


def audit_columns(df: pd.DataFrame, profile: Dict[str, Any], target: Optional[str], task: str,
                  date_col: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return proposed cleaning actions. `destructive` ones need human approval.
    `target` may be None (clustering / descriptive): only target-free checks run."""
    n = profile["n_rows"]
    y = df[target] if target else None
    supervised = target is not None and task in ("classification", "regression", "time_series")
    y_num = _numeric_target(y, task) if supervised else None
    issues: List[Dict[str, Any]] = []

    def add(col, issue, action, destructive, reason):
        issues.append({"column": col, "issue": issue, "action": action,
                       "destructive": destructive, "reason": reason})

    for col, info in profile["columns"].items():
        if col == target:
            continue
        kind = info["kind"]
        if info["unique_count"] <= 1:
            add(col, "constant", "drop", False, "Constant or empty column carries no signal.")
            continue
        if info["missing_pct"] >= config.HIGH_MISSING_PCT:
            add(col, "high_missing", "drop", True, f"{info['missing_pct']}% missing.")
            continue
        if kind == "datetime":
            add(col, "datetime", "expand_datetime", False,
                "Expanded into year/month/day/day-of-week/ordinal features.")
            continue
        if kind == "text":
            add(col, "free_text", "text_features", False, "Free text becomes TF-IDF features (1-2 word groups, capped vocabulary).")
            continue
        if kind == "image_path":
            add(col, "image_path", "image_features", False, "Image file paths become colour / brightness / edge / thumbnail features.")
            continue
        if (n >= config.MIN_ROWS and info["unique_ratio"] >= config.ID_UNIQUE_RATIO
                and (kind == "categorical" or info["is_integer_like"])):
            add(col, "id_like", "drop", True,
                f"{info['unique_ratio']:.0%} unique values: looks like an identifier (memorization risk).")
            continue
        if not supervised:
            continue
        flagged = False
        if kind in ("numeric", "boolean") and y_num is not None:
            x = pd.to_numeric(df[col].astype(float) if kind == "boolean" else df[col], errors="coerce")
            c = x.corr(y_num)
            if pd.notna(c) and abs(c) >= config.LEAK_CORR:
                add(col, "leakage_suspect", "drop", True,
                    f"|correlation| with target = {abs(c):.3f}: likely leaks the answer.")
                flagged = True
        if not flagged and kind == "categorical" and task == "classification" and 2 <= info["unique_count"] <= 20:
            sub = df[[col, target]].dropna()
            if len(sub):
                purity = sub.groupby(col)[target].agg(lambda s: s.value_counts().iloc[0]).sum() / len(sub)
                if purity >= config.LEAK_PURITY and sub[target].nunique() > 1:
                    add(col, "leakage_suspect", "drop", True,
                        "Category values map to the target class with ~100% purity: likely leakage.")
                    flagged = True
        if not flagged and kind in ("numeric", "boolean", "categorical") and info["unique_count"] > 1:
            pw = single_feature_power(df[col], y, task)       # non-linear leak check
            if pw is not None and pw >= config.LEAK_SINGLE_FEATURE:
                add(col, "leakage_suspect", "drop", True,
                    f"This one column alone predicts the target (CV score {pw:.3f}): likely leakage.")
    return issues
