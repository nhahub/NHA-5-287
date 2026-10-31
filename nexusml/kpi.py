"""Business KPI engine. KPIs are declarative specs (proposed by the LLM) that this
module VALIDATES against the real columns and then computes safely: no eval, no generated code."""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .trends import choose_freq, measure_value, period_series

AGGS = {"sum", "mean", "median", "count", "nunique", "min", "max", "rate", "ratio"}
FORMATS = {"currency", "percent", "number"}


def validate_spec(spec: Dict[str, Any], df: pd.DataFrame) -> Tuple[Optional[Dict[str, Any]], str]:
    s = dict(spec or {})
    agg = str(s.get("agg", "")).lower()
    if agg not in AGGS:
        return None, f"unsupported aggregation '{agg}'"
    s["agg"] = agg
    s["name"] = str(s.get("name") or f"{agg} {s.get('column', '')}").strip()[:60]
    cols = set(df.columns)
    if agg == "count":
        s.setdefault("column", None)
    elif agg == "ratio":
        for k in ("numerator", "denominator"):
            if s.get(k) not in cols or not pd.api.types.is_numeric_dtype(df[s[k]]):
                return None, f"ratio needs numeric column '{k}'"
    else:
        if s.get("column") not in cols:
            return None, f"unknown column '{s.get('column')}'"
        if agg in ("sum", "mean", "median", "min", "max") and not pd.api.types.is_numeric_dtype(df[s["column"]]):
            return None, f"'{s['column']}' is not numeric"
        if agg == "rate":
            vals = df[s["column"]].dropna().astype(str).unique()
            if "positive" not in s or str(s["positive"]) not in set(vals):
                return None, "rate needs a 'positive' value present in the column"
    s["format"] = s.get("format") if s.get("format") in FORMATS else ("percent" if agg == "rate" else "number")
    s["higher_is_better"] = bool(s.get("higher_is_better", True))
    s["group_by"] = [g for g in (s.get("group_by") or []) if g in cols and g != s.get("column")][:3]
    return s, ""


def fmt(v: Optional[float], f: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if f == "percent":
        return f"{v * 100:.1f}%"
    if f == "currency":
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"
    return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"


def compute(df: pd.DataFrame, spec: Dict[str, Any], date_col: Optional[str], max_levels: int = 10) -> Dict[str, Any]:
    f = spec["format"]
    overall = measure_value(df, spec)
    out: Dict[str, Any] = {"spec": spec, "overall": overall, "overall_fmt": fmt(overall, f), "segments": {}}
    for g in spec["group_by"]:
        card = df[g].nunique()
        if card < 2 or card > 40:
            continue
        sub = df.dropna(subset=[g])
        counts = sub[g].astype(str).value_counts()
        keep = [k for k in counts.index[:max_levels] if counts[k] >= 5]
        rows = []
        for k in keep:
            part = sub[sub[g].astype(str) == k]
            v = measure_value(part, spec)
            rows.append({"segment": k, "value": v, "value_fmt": fmt(v, f), "n": int(len(part)),
                         "share_of_rows": round(len(part) / len(df), 4),
                         "vs_overall_pct": round((v / overall - 1) * 100, 1) if overall not in (0, np.nan) and not np.isnan(v) and overall else None})
        if rows:
            rows.sort(key=lambda r: -(r["value"] if not np.isnan(r["value"]) else -1e18))
            out["segments"][g] = rows
    if date_col and date_col in df.columns:
        try:
            s = period_series(df, date_col, spec)
            if len(s) >= 2:
                last, prev = float(s.iloc[-1]), float(s.iloc[-2])
                out["time"] = {"freq": choose_freq(pd.to_datetime(df[date_col], errors="coerce")),
                               "periods": [str(i.date()) for i in s.index], "values": [round(float(v), 4) for v in s],
                               "last": last, "previous": prev,
                               "change_pct": round((last / prev - 1) * 100, 2) if prev else None}
        except Exception:
            pass
    return out
