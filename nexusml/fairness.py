"""Sensitive-attribute guard: detect protected columns, optionally exclude them, and audit model behaviour per group."""
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config

_SHORT = {"age", "sex", "race", "birth"}              # matched as whole words only ("page", "usage", "trace" are not sensitive)


def _tokens(col: str) -> List[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(col))
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def detect_sensitive(columns: List[str], exclude: Optional[List[str]] = None) -> List[str]:
    skip = set(exclude or [])
    found = []
    for c in columns:
        if c in skip:
            continue
        toks = _tokens(c)
        long_hit = any(p in str(c).lower() for p in config.SENSITIVE_PATTERNS if p not in _SHORT)
        short_hit = any(t in _SHORT for t in toks)
        if long_hit or short_hit:
            found.append(c)
    return found


def _group_series(s: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
        try:
            return pd.qcut(s, 3, duplicates="drop").astype(str).where(s.notna())
        except Exception:
            return None
    if 2 <= s.nunique() <= 12:
        return s.astype(str).where(s.notna())
    return None


def group_audit(raw: pd.DataFrame, y: np.ndarray, pred: np.ndarray, cols: List[str], used_as_feature: Dict[str, bool],
                min_group: int = 20) -> List[Dict[str, Any]]:
    """Per-group selection rate / recall / false-positive rate on the held-out rows. The four-fifths screen
    (lowest selection rate >= 80% of the highest) is a coarse flag, not a legal determination."""
    out = []
    y = np.asarray(y)
    pred = np.asarray(pred).astype(int)
    for c in cols:
        if c not in raw.columns:
            continue
        g = _group_series(raw[c].reset_index(drop=True))
        if g is None:
            continue
        rows = []
        for lv in g.dropna().unique():
            m = (g == lv).to_numpy()
            if m.sum() < min_group:
                continue
            yt, pp = y[m], pred[m]
            pos, neg = (yt == 1).sum(), (yt == 0).sum()
            rows.append({"group": str(lv), "n": int(m.sum()), "base_rate": round(float(yt.mean()), 3),
                         "selection_rate": round(float(pp.mean()), 3),
                         "recall": round(float(((pp == 1) & (yt == 1)).sum() / pos), 3) if pos else None,
                         "false_positive_rate": round(float(((pp == 1) & (yt == 0)).sum() / neg), 3) if neg else None})
        if len(rows) < 2:
            continue
        sel = [r["selection_rate"] for r in rows]
        ratio = (min(sel) / max(sel)) if max(sel) > 0 else None
        rec = [r["recall"] for r in rows if r["recall"] is not None]
        out.append({"column": c, "used_as_feature": bool(used_as_feature.get(c)), "groups": rows,
                    "selection_ratio": None if ratio is None else round(float(ratio), 3),
                    "recall_gap": round(float(max(rec) - min(rec)), 3) if rec else None,
                    "flag": bool(ratio is not None and ratio < 0.8)})
    return out


def describe(a: Dict[str, Any]) -> str:
    gs = sorted(a["groups"], key=lambda r: -r["selection_rate"])
    hi, lo = gs[0], gs[-1]
    use = "used as a model feature" if a["used_as_feature"] else "excluded from the model"
    txt = (f"Fairness screen on '{a['column']}' ({use}): the model flags {hi['selection_rate']:.0%} of '{hi['group']}' (n={hi['n']}) vs "
           f"{lo['selection_rate']:.0%} of '{lo['group']}' (n={lo['n']}); lowest/highest ratio {a['selection_ratio']}.")
    if a.get("recall_gap") is not None:
        txt += f" Recall differs by up to {a['recall_gap']:.2f} between groups."
    if a["flag"]:
        txt += " Below the 0.8 four-fifths screen: review before using these scores for individual decisions."
    return txt
