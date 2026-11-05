"""Agentic data segmentation: slice a tabular / time-series dataset, then analyse and model ONLY that slice.

    spec = {"name": "positives", "conditions": [{"column": "churn", "op": "eq", "value": "yes"}],
            "time_range": {"column": "date", "start": "2024-01-01", "end": "2024-06-30"},     # optional
            "sample": {"n": 1000, "stratify": "churn"}}                                          # optional

* `propose(df, request, target)`  turns plain language ("only the positive class", "customers over 50 in Cairo", "last 6 months") into a spec.
  An LLM may draft it, but the spec is always re-validated: column names are resolved against the real columns, operators come from a fixed list,
  values are type-checked and the slice must contain rows. Without an LLM a rule-based reader handles the common phrasings.
* `apply(df, spec)`  returns the slice and a report (rows kept, share, per-condition counts). No code is ever executed.
* `analyze(df, spec, target)` runs, on the slice only: profiling (EDA), a comparison of the slice against the rest (what makes it different),
  feature engineering + modelling through the platform's own pipeline (preprocessing, SMOTE/ADASYN/threshold options included), and clustering.
  If the target has a single value inside the slice (e.g. you isolated the positive class), a model of that target is impossible, and the
  result says so and models `secondary_target` if given, otherwise profiles and clusters the slice instead of failing.
"""
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config, modeling, profiling

MIN_SLICE_ROWS = 20          # below this a slice is reported but not modelled
MIN_MODEL_ROWS = 60


class SegmentError(ValueError):
    pass


def _tools():
    from .assistant import tools          # imported lazily: tools imports a lot and must not be a hard import-time dependency
    return tools


def _clean_spec(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    T = _tools()
    if not isinstance(spec, dict):
        raise SegmentError("The segment must be an object with 'conditions'.")
    conds = []
    try:
        if spec.get("conditions"):
            conds = T._conditions(df, spec["conditions"])
    except T.ToolError as e:
        raise SegmentError(str(e))
    out: Dict[str, Any] = {"name": str(spec.get("name") or "segment")[:60], "conditions": conds}
    tr = spec.get("time_range")
    if tr:
        try:
            col = T.resolve_column(df, tr.get("column"))
        except T.ToolError as e:
            raise SegmentError(str(e))
        start, end = tr.get("start"), tr.get("end")
        if start is None and end is None:
            raise SegmentError("time_range needs a start and/or an end.")
        out["time_range"] = {"column": col, "start": start, "end": end}
    sm = spec.get("sample")
    if sm:
        n = int(sm.get("n") or 0)
        if n < 1:
            raise SegmentError("sample.n must be a positive number.")
        out["sample"] = {"n": n, "stratify": (T.resolve_column(df, sm["stratify"]) if sm.get("stratify") else None)}
    if not conds and "time_range" not in out and "sample" not in out:
        raise SegmentError("The segment has no condition, time range or sample, so it would be the whole dataset.")
    return out


def apply(df: pd.DataFrame, spec: Dict[str, Any], seed: int = 42):
    """-> (slice_df, report). Row order is preserved (it matters for time series)."""
    T = _tools()
    spec = _clean_spec(df, spec)
    mask = pd.Series(True, index=df.index)
    steps = []
    try:
        for c in spec["conditions"]:
            m = T._mask(df, [c])
            steps.append({"condition": f"{c['column']} {c['op']} {c['value']}", "rows_matching": int(m.sum())})
            mask &= m
        if "time_range" in spec:
            tr = spec["time_range"]
            d = pd.to_datetime(df[tr["column"]], errors="coerce")
            if d.notna().mean() < 0.5:
                raise SegmentError(f"'{tr['column']}' does not look like a date column.")
            m = pd.Series(True, index=df.index)
            if tr.get("start") is not None:
                m &= d >= pd.to_datetime(tr["start"])
            if tr.get("end") is not None:
                m &= d <= pd.to_datetime(tr["end"])
            steps.append({"condition": f"{tr['column']} from {tr.get('start')} to {tr.get('end')}", "rows_matching": int(m.sum())})
            mask &= m.fillna(False)
    except T.ToolError as e:
        raise SegmentError(str(e))
    sub = df[mask]
    if "sample" in spec and len(sub) > spec["sample"]["n"]:
        n, by = spec["sample"]["n"], spec["sample"]["stratify"]
        if by:
            frac = n / len(sub)
            sub = sub.groupby(by, group_keys=False, dropna=False).apply(lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=seed)).sort_index()
        else:
            sub = sub.sample(n, random_state=seed).sort_index()
    if len(sub) == 0:
        raise SegmentError("No rows match this segment. " + "; ".join(f"{s['condition']}: {s['rows_matching']} rows" for s in steps))
    return sub, {"spec": spec, "rows": int(len(sub)), "of": int(len(df)), "share": round(len(sub) / max(len(df), 1), 4), "steps": steps}


# ------------------------------------------------------------------ natural language -> spec
_POS_WORDS = r"(positive|positives|minority|rare|yes|churned|fraud|default(?:ed)?|1)"


def _positive_value(s: pd.Series):
    vc = s.dropna().astype(str).value_counts()
    low = {str(v).lower(): v for v in vc.index}
    for w in ("1", "yes", "true", "y", "positive", "churn", "churned", "fraud", "default"):
        if w in low:
            return low[w]
    return vc.index[-1] if len(vc) else None            # least frequent class


def rule_based(df: pd.DataFrame, request: str, target: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Common phrasings, no LLM: 'positive class', 'where <col> is/= <value>', '<col> > 50', 'last N months/days/years'."""
    T = _tools()
    q = request.lower()
    conds: List[Dict[str, Any]] = []
    spec: Dict[str, Any] = {"name": request.strip()[:50], "conditions": conds}
    if target and target in df.columns and re.search(r"\b(positive|minority|rare)\b|\bpositive class\b", q):
        conds.append({"column": target, "op": "eq", "value": _positive_value(df[target])})
    elif target and target in df.columns and re.search(r"\bnegative\b|\bmajority\b", q):
        pv = _positive_value(df[target])
        conds.append({"column": target, "op": "ne", "value": pv})
    cols = sorted(df.columns, key=len, reverse=True)
    for c in cols:
        pat = r"(?<![\w])" + re.escape(str(c).lower()) + r"(?![\w])"
        m = re.search(pat + r"\s*(>=|<=|!=|==|=|>|<)\s*(-?\d+(?:\.\d+)?)", q)
        if m:
            conds.append({"column": c, "op": {">=": "ge", "<=": "le", "!=": "ne", "==": "eq", "=": "eq", ">": "gt", "<": "lt"}[m.group(1)], "value": float(m.group(2))})
            continue
        m = re.search(pat + r"\s*(?:is|=|==|equals|of)\s*['\"]?([\w .-]+?)['\"]?(?:\s+(?:and|with|only)\b|[,.;]|$)", q)
        if m and not conds_has(conds, c):
            raw = m.group(1).strip()
            vals = {str(v).lower(): v for v in df[c].dropna().unique()[:2000]}
            if raw in vals:
                conds.append({"column": c, "op": "eq", "value": vals[raw]})
    m = re.search(r"last\s+(\d+)\s+(day|week|month|year)s?", q)
    if m:
        for c in df.columns:
            if profiling.column_kind(df[c]) == "datetime":
                end = profiling.to_datetime(df[c]).max()
                n, unit = int(m.group(1)), m.group(2)
                start = end - {"day": pd.Timedelta(days=n), "week": pd.Timedelta(weeks=n), "month": pd.DateOffset(months=n), "year": pd.DateOffset(years=n)}[unit]
                spec["time_range"] = {"column": c, "start": str(start.date()), "end": str(end.date())}
                break
    return spec if (conds or "time_range" in spec) else None


def conds_has(conds, col) -> bool:
    return any(c["column"] == col for c in conds)


def propose(df: pd.DataFrame, request: str, target: Optional[str] = None, use_llm: bool = True) -> Dict[str, Any]:
    """Plain language -> validated segment spec. Raises SegmentError if nothing sensible can be built."""
    spec = None
    if use_llm:
        try:
            from . import llm
            if llm.available():
                cols = {str(c): profiling.column_kind(df[c]) for c in list(df.columns)[:60]}
                ex = {str(c): [str(v) for v in df[c].dropna().unique()[:6]] for c in df.columns[:40] if profiling.column_kind(df[c]) in ("categorical", "boolean", "text")}
                out, err, _ = llm.ask_json(
                    "You turn a request into a JSON data-slice spec. Reply with JSON only: {\"name\": str, \"conditions\": [{\"column\": str, \"op\": one of eq|ne|gt|ge|lt|le|in|not_in|contains|between|is_null|not_null, \"value\": any}], "
                    "\"time_range\": {\"column\": str, \"start\": str|null, \"end\": str|null} or null, \"sample\": {\"n\": int, \"stratify\": str|null} or null}. Use only the listed columns. "
                    "'Positive class' means the rarer / 'yes' / 1 value of the target.",
                    f"Columns and kinds: {cols}\nExample values: {ex}\nTarget column: {target}\nRequest: {request}", purpose="segment_spec", max_tokens=600)
                if out and not err:
                    spec = out
        except Exception:
            spec = None
    if spec:
        try:
            _clean_spec(df, spec)
            return spec
        except SegmentError:
            spec = None
    spec = rule_based(df, request, target)
    if not spec:
        raise SegmentError("I could not turn that into a filter. Name a column and a value, e.g. \"churn is yes\" or \"age > 50\", or say \"positive class\".")
    return spec


# ------------------------------------------------------------------ segment vs rest
def compare_to_rest(df: pd.DataFrame, mask: pd.Series, top: int = 8) -> List[Dict[str, Any]]:
    """What is different about the slice? Numeric: standardised mean difference. Categorical: biggest over-represented value (lift)."""
    out = []
    for c in df.columns:
        k = profiling.column_kind(df[c])
        a, b = df.loc[mask, c], df.loc[~mask, c]
        if len(b) < 5:
            break
        if k == "numeric":
            a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            sd = np.sqrt((a.var() + b.var()) / 2)
            if sd and np.isfinite(sd):
                out.append({"column": str(c), "kind": "numeric", "segment_mean": round(float(a.mean()), 4), "rest_mean": round(float(b.mean()), 4), "effect_size": round(float((a.mean() - b.mean()) / sd), 3)})
        elif k in ("categorical", "boolean") and df[c].nunique() <= 30:
            pa, pb = a.astype(str).value_counts(normalize=True), b.astype(str).value_counts(normalize=True)
            lifts = {v: (pa.get(v, 0) + 1e-9) / (pb.get(v, 0) + 1e-9) for v in pa.index[:10] if pa[v] >= 0.05}
            if lifts:
                v = max(lifts, key=lifts.get)
                out.append({"column": str(c), "kind": "categorical", "value": v, "segment_share": round(float(pa[v]), 4), "rest_share": round(float(pb.get(v, 0)), 4),
                            "effect_size": round(float(np.log(lifts[v])), 3)})
    return sorted(out, key=lambda r: -abs(r["effect_size"]))[:top]


# ------------------------------------------------------------------ analysis on the slice
def _task_for(y: pd.Series) -> str:
    return "classification" if (not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10) else "regression"


def analyze(df: pd.DataFrame, spec: Dict[str, Any], target: Optional[str] = None, secondary_target: Optional[str] = None,
            task: Optional[str] = None, strategy: Optional[Dict[str, Any]] = None, drop_columns: Optional[List[str]] = None,
            time_column: Optional[str] = None) -> Dict[str, Any]:
    """Isolated EDA + feature engineering + modelling (+ clustering) on the slice. Never raises for a thin or single-class slice: it
    reports why modelling was skipped. `strategy` accepts the same keys as the main pipeline (imbalance='smote'|'adasyn', class_weight, ...)."""
    sub, seg = apply(df, spec)
    mask = df.index.isin(sub.index)
    res: Dict[str, Any] = {"segment": seg, "notes": [], "eda": None, "vs_rest": [], "model": None, "clusters": None}
    res["eda"] = profiling.profile_dataset(sub)
    if len(sub) < len(df):
        used = {c["column"] for c in seg["spec"]["conditions"]} | ({seg["spec"]["time_range"]["column"]} if "time_range" in seg["spec"] else set())
        res["vs_rest"] = compare_to_rest(df.drop(columns=list(used)), pd.Series(mask, index=df.index))      # the filter columns differ by construction
    mtarget = secondary_target or target
    if len(sub) < MIN_SLICE_ROWS:
        res["notes"].append(f"Only {len(sub)} rows in this segment (need {MIN_SLICE_ROWS}); described but not modelled.")
        return res
    if mtarget and mtarget in sub.columns:
        y = sub[mtarget].dropna()
        if y.nunique() < 2:
            res["notes"].append(f"'{mtarget}' has a single value ({y.iloc[0] if len(y) else 'none'}) inside this segment, so it cannot be predicted here. "
                                + ("Set secondary_target to model a different column; clustering the segment instead." if not secondary_target else ""))
            mtarget = None
        elif len(sub) < MIN_MODEL_ROWS:
            res["notes"].append(f"Only {len(sub)} rows (need {MIN_MODEL_ROWS}) for a trustworthy model; skipped.")
            mtarget = None
    elif mtarget:
        res["notes"].append(f"Column '{mtarget}' not found.")
        mtarget = None
    drop = list(drop_columns or [])
    if mtarget:
        plan = {"problem_type": task or _task_for(sub[mtarget]), "target": mtarget, "drop_columns": drop + ([target] if secondary_target and target in sub.columns else []),
                "datetime_columns": [c for c in sub.columns if profiling.column_kind(sub[c]) == "datetime" and c != mtarget]}
        if time_column:
            plan.update({"problem_type": "time_series", "time_column": time_column})
        try:
            st = {**modeling.DEFAULT_STRATEGY, "tune": False, **(strategy or {})}
            sp = modeling.make_splits(sub.reset_index(drop=True) if plan["problem_type"] != "time_series" else sub, plan)
            out = modeling.fit_default(plan["problem_type"], sp, st)
            thr = modeling.finalize_threshold(plan["problem_type"], sp, out["models"], out["candidates"], st)
            best = modeling.pick_best(out["candidates"])
            res["model"] = {"problem_type": plan["problem_type"], "target": mtarget, "rows_train": int(len(sp["X_train"])), "rows_test": int(len(sp["X_test"])),
                            "features": [str(c) for c in sp["X_train"].columns][:40], "best": best, "candidates": out["candidates"], "threshold": thr,
                            "strategy": {k: st.get(k) for k in ("class_weight", "imbalance", "tune_threshold")}}
        except Exception as e:                                    # a bad slice must not crash the agent
            res["notes"].append(f"Modelling this segment failed: {type(e).__name__}: {str(e)[:140]}")
    try:
        feats = sub.drop(columns=[c for c in [target, secondary_target] + drop if c and c in sub.columns]).select_dtypes(exclude="datetime")
        if feats.shape[1] >= 2 and len(feats) >= 40:
            res["clusters"] = modeling.cluster_segments(feats.reset_index(drop=True))
    except Exception as e:
        res["notes"].append(f"Clustering skipped: {type(e).__name__}")
    return res


def summarize(res: Dict[str, Any]) -> str:
    s = res["segment"]
    lines = [f"Segment '{s['spec']['name']}': {s['rows']:,} of {s['of']:,} rows ({s['share']:.1%})."]
    if res["vs_rest"]:
        top = res["vs_rest"][0]
        lines.append(f"Most distinctive: {top['column']} ({'mean %s vs %s' % (top['segment_mean'], top['rest_mean']) if top['kind'] == 'numeric' else '%s = %s: %.0f%% vs %.0f%%' % (top['column'], top['value'], 100 * top['segment_share'], 100 * top['rest_share'])}).")
    m = res.get("model")
    if m:
        row = next(c for c in m["candidates"] if c["name"] == m["best"])
        key = "f1_macro" if m["problem_type"] == "classification" else "r2"
        lines.append(f"Model of '{m['target']}' on the segment: best {m['best']}, held-out {key} = {row.get(key)}.")
    lines += res["notes"]
    return " ".join(lines)
