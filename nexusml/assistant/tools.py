"""The assistant's toolbox. Every tool is deterministic, validated against the real columns, bounded in cost, and free of eval / exec / SQL.

A tool returns TWO different things on purpose:
  summary   aggregates only (counts, shares, fences, statistics, top categories). This is ALL the language model ever sees.
  table / figure   for the person's screen. They may contain real rows ("show me the outlier rows"), and they never go to the model.
"""
import difflib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .. import profiling
from ..util import jsonable
from . import charts

MAX_TABLE_ROWS = 500
MODEL_ROWS = 15000


class ToolError(ValueError):
    """A problem with the request itself (unknown column, wrong type ...). Its message is shown back to the assistant so it can correct the call."""


@dataclass
class ToolContext:
    df: pd.DataFrame
    name: str = "dataset"
    run_state: Optional[Dict[str, Any]] = None        # a finished analysis the person is looking at (enables ask_run)
    seed: int = 42
    answer_run: Optional[Callable[[str], str]] = None  # injected by the app so this module stays free of LLM imports


@dataclass
class ToolResult:
    title: str
    summary: Dict[str, Any]
    table: Optional[pd.DataFrame] = None
    figure: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)
    proposal: Optional[Dict[str, Any]] = None
    text: str = ""                                    # one plain sentence describing what was done (shown in the activity list)


@dataclass
class Param:
    kind: str                                         # column | numeric_column | columns | number | int | text | choice | bool | conditions
    desc: str
    required: bool = False
    default: Any = None
    choices: Optional[List[str]] = None


@dataclass
class Tool:
    name: str
    description: str
    params: Dict[str, Param]
    fn: Callable[[ToolContext, Dict[str, Any]], ToolResult]
    needs_data: bool = True


# argument handling
def kind_of(s: pd.Series) -> str:
    return profiling.column_kind(s)


def numeric_columns(df: pd.DataFrame, min_unique: int = 2) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() >= min_unique]


def resolve_column(df: pd.DataFrame, name: Any, numeric: bool = False) -> str:
    if name in df.columns:
        col = name
    else:
        key = str(name).strip().lower()
        low = {str(c).strip().lower(): c for c in df.columns}
        if key in low:
            col = low[key]
        else:
            near = difflib.get_close_matches(str(name), [str(c) for c in df.columns], n=4, cutoff=0.5)
            raise ToolError(f"There is no column named '{name}'." + (f" Closest matches: {near}." if near else f" Available columns: {list(df.columns)[:30]}."))
    if numeric and not pd.api.types.is_numeric_dtype(df[col]):
        raise ToolError(f"Column '{col}' is not numeric ({kind_of(df[col])}); choose one of {numeric_columns(df)[:20]}.")
    return col


def coerce_args(tool: Tool, df: Optional[pd.DataFrame], raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    unknown = [k for k in raw if k not in tool.params]
    if unknown:
        raise ToolError(f"{tool.name} does not take {unknown}; its parameters are {list(tool.params)}.")
    out: Dict[str, Any] = {}
    for k, p in tool.params.items():
        v = raw.get(k)
        if v is None or v == "":
            if p.required:
                raise ToolError(f"{tool.name} needs '{k}' ({p.desc}).")
            out[k] = p.default
            continue
        try:
            if p.kind == "column":
                out[k] = resolve_column(df, v)
            elif p.kind == "numeric_column":
                out[k] = resolve_column(df, v, numeric=True)
            elif p.kind == "columns":
                vals = v if isinstance(v, list) else [v]
                out[k] = [resolve_column(df, x) for x in vals][:40]
            elif p.kind == "number":
                out[k] = float(v)
            elif p.kind == "int":
                out[k] = int(float(v))
            elif p.kind == "bool":
                out[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            elif p.kind == "choice":
                if str(v).lower() not in [c.lower() for c in p.choices or []]:
                    raise ToolError(f"'{k}' must be one of {p.choices}, got '{v}'.")
                out[k] = next(c for c in p.choices if c.lower() == str(v).lower())
            elif p.kind == "conditions":
                out[k] = _conditions(df, v)
            else:
                out[k] = str(v)[:400]
        except (TypeError, ValueError) as e:
            if isinstance(e, ToolError):
                raise
            raise ToolError(f"'{k}' is not valid ({p.desc}): {v!r}.")
    return out


OPS = {"==": "eq", "=": "eq", "eq": "eq", "!=": "ne", "ne": "ne", ">": "gt", "gt": "gt", ">=": "ge", "ge": "ge", "<": "lt", "lt": "lt", "<=": "le", "le": "le", "in": "in", "not_in": "not_in",
       "contains": "contains", "between": "between", "is_null": "is_null", "not_null": "not_null"}


def _conditions(df: pd.DataFrame, v: Any) -> List[Dict[str, Any]]:
    if not isinstance(v, list) or not v:
        raise ToolError("'conditions' must be a non-empty list of {column, op, value}.")
    out = []
    for c in v[:8]:
        if not isinstance(c, dict):
            raise ToolError("each condition must be an object {column, op, value}.")
        op = OPS.get(str(c.get("op", "")).lower().replace(" ", "_"))
        if not op:
            raise ToolError(f"unknown operator '{c.get('op')}'; use one of {sorted(set(OPS.values()))}.")
        out.append({"column": resolve_column(df, c.get("column")), "op": op, "value": c.get("value")})
    return out


def _typed(s: pd.Series, value: Any) -> Any:
    if pd.api.types.is_bool_dtype(s):
        return str(value).lower() in ("1", "true", "yes")
    if pd.api.types.is_numeric_dtype(s):
        return float(value)
    if kind_of(s) == "datetime":
        return pd.to_datetime(value)
    return str(value)


def _mask(df: pd.DataFrame, conds: List[Dict[str, Any]]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for c in conds:
        s, op, val = df[c["column"]], c["op"], c["value"]
        try:
            if op == "is_null":
                m &= s.isna()
            elif op == "not_null":
                m &= s.notna()
            elif op in ("in", "not_in"):
                vals = [_typed(s, x) for x in (val if isinstance(val, list) else [val])]
                m &= s.isin(vals) if op == "in" else ~s.isin(vals)
            elif op == "between":
                lo, hi = val if isinstance(val, list) and len(val) == 2 else (None, None)
                if lo is None:
                    raise ToolError("'between' needs a value of [low, high].")
                sv = pd.to_datetime(s, errors="coerce") if kind_of(s) == "datetime" else s
                m &= (sv >= _typed(s, lo)) & (sv <= _typed(s, hi))
            elif op == "contains":
                m &= s.astype(str).str.contains(str(val), case=False, regex=False, na=False)
            else:
                sv = pd.to_datetime(s, errors="coerce") if kind_of(s) == "datetime" else s
                t = _typed(s, val)
                m &= {"eq": sv == t, "ne": sv != t, "gt": sv > t, "ge": sv >= t, "lt": sv < t, "le": sv <= t}[op]
        except (TypeError, ValueError) as e:
            if isinstance(e, ToolError):
                raise
            raise ToolError(f"cannot compare column '{c['column']}' with {val!r}: {str(e)[:80]}")
    return m.fillna(False)


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=seed)


def _r(x: Any, nd: int = 4) -> Any:
    try:
        return None if x is None or not np.isfinite(float(x)) else round(float(x), nd)
    except (TypeError, ValueError):
        return x


def _pct(x: float) -> float:
    return round(100 * float(x), 1)


# tools
def t_overview(ctx: ToolContext, a) -> ToolResult:
    df = ctx.df
    prof = profiling.profile_dataset(df)
    cols = prof["columns"]
    rows = [{"column": c, "type": v["kind"], "missing %": round(v["missing_pct"], 1), "distinct": v["unique_count"],
             "range / top": (f"{_r(v.get('min'), 3)} to {_r(v.get('max'), 3)}" if v["kind"] == "numeric" else "")} for c, v in cols.items()]
    kinds = pd.Series([v["kind"] for v in cols.values()]).value_counts().to_dict()
    q = profiling.data_quality(df, prof)
    summary = {"rows": int(len(df)), "columns": int(df.shape[1]), "column_types": kinds, "missing_cells_pct": _pct(df.isna().sum().sum() / max(1, df.size)),
               "duplicate_rows": int(prof["duplicates"]), "quality_score": q.get("score"), "columns_with_most_missing": [{"column": c, "missing_pct": round(v["missing_pct"], 1)}
                for c, v in sorted(cols.items(), key=lambda kv: -kv[1]["missing_pct"])[:5] if v["missing_pct"] > 0], "warnings": prof.get("warnings", [])[:5]}
    return ToolResult("Dataset overview", summary, pd.DataFrame(rows), None, text=f"Profiled {len(df):,} rows and {df.shape[1]} columns.")


def t_distribution(ctx: ToolContext, a) -> ToolResult:
    df, c = ctx.df, a["column"]
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        x = pd.to_numeric(s, errors="coerce").dropna()
        if len(x) < 3:
            raise ToolError(f"Column '{c}' has too few values to describe.")
        q = x.quantile([.01, .05, .25, .5, .75, .95, .99])
        st = {"count": int(len(x)), "missing_pct": _pct(s.isna().mean()), "mean": _r(x.mean()), "median": _r(x.median()), "std": _r(x.std()), "min": _r(x.min()), "max": _r(x.max()),
              "skew": _r(x.skew()), "excess_kurtosis": _r(x.kurt()), "p01": _r(q[.01]), "p05": _r(q[.05]), "p25": _r(q[.25]), "p75": _r(q[.75]), "p95": _r(q[.95]), "p99": _r(q[.99]), "zeros_pct": _pct((x == 0).mean())}
        tbl = pd.DataFrame({"statistic": ["count", "mean", "median", "std", "min", "5th percentile", "25th percentile", "75th percentile", "95th percentile", "max", "skew"],
                            "value": [st["count"], st["mean"], st["median"], st["std"], st["min"], st["p05"], st["p25"], st["p75"], st["p95"], st["max"], st["skew"]]})
        return ToolResult(f"Distribution of {c}", st, tbl, charts.histogram(x.to_numpy(), c), text=f"Described the distribution of {c}.")
    vc = s.astype(str).where(s.notna()).value_counts()
    tbl = pd.DataFrame({c: vc.index[:20], "rows": vc.values[:20], "share %": (100 * vc.values[:20] / max(1, vc.sum())).round(1)})
    return ToolResult(f"Values of {c}", {"distinct": int(s.nunique()), "missing_pct": _pct(s.isna().mean()), "top": [{"value": str(i), "share_pct": round(100 * v / max(1, vc.sum()), 1)} for i, v in vc.head(8).items()]},
                      tbl, charts.bar(tbl[c].tolist(), tbl["rows"].tolist(), f"Most common values of {c}", "rows"), text=f"Counted the values of {c}.")


def t_value_counts(ctx: ToolContext, a) -> ToolResult:
    s, c, top = ctx.df[a["column"]], a["column"], int(min(max(a["top"] or 15, 1), 50))
    vc = s.astype(str).where(s.notna()).value_counts()
    tbl = pd.DataFrame({c: vc.index[:top], "rows": vc.values[:top], "share %": (100 * vc.values[:top] / max(1, vc.sum())).round(1)})
    summary = {"column": c, "distinct": int(s.nunique()), "missing_pct": _pct(s.isna().mean()), "top": [{"value": str(i), "rows": int(v), "share_pct": round(100 * v / max(1, vc.sum()), 1)} for i, v in vc.head(top).items()]}
    return ToolResult(f"Counts of {c}", summary, tbl, charts.bar(tbl[c].tolist(), tbl["rows"].tolist(), f"Rows per {c}", "rows"), text=f"Counted rows per value of {c}.")


def t_outliers(ctx: ToolContext, a) -> ToolResult:
    df = ctx.df
    cols = a["columns"] or [c for c in numeric_columns(df, 8) if not (profiling.is_integer_like(df[c]) and df[c].nunique() > 0.95 * len(df))]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ToolError("There are no numeric columns with enough distinct values to check for outliers.")
    method, top = (a["method"] or "auto"), int(min(max(a["top"] or 15, 1), 100))
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    sev = pd.DataFrame(0.0, index=df.index, columns=cols)
    per = []
    for c in cols:
        x = X[c]
        q1, q3 = x.quantile(.25), x.quantile(.75)
        iqr = q3 - q1
        if method == "zscore" or iqr == 0:
            med = x.median()
            mad = (x - med).abs().median() * 1.4826 or x.std() or 1.0
            z = (x - med).abs() / mad
            lo, hi = med - 3 * mad, med + 3 * mad
            s_ = (z - 3).clip(lower=0)
        else:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            s_ = (((x - hi).clip(lower=0) + (lo - x).clip(lower=0)) / iqr)
        sev[c] = s_.fillna(0)
        n = int((sev[c] > 0).sum())
        per.append({"column": c, "flagged": n, "share_pct": _pct(n / max(1, x.notna().sum())), "lower_fence": _r(lo), "upper_fence": _r(hi), "min": _r(x.min()), "max": _r(x.max())})
    any_flag = (sev > 0).any(axis=1)
    iso_score = pd.Series(0.0, index=df.index)
    iso_n = 0
    if method in ("auto", "isolation_forest") and len(cols) >= 2 and len(df) >= 100:
        from sklearn.ensemble import IsolationForest
        Z = X.fillna(X.median()).to_numpy()
        Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-9)
        sub = np.arange(len(Z)) if len(Z) <= 50000 else np.random.RandomState(ctx.seed).choice(len(Z), 50000, replace=False)
        model = IsolationForest(n_estimators=100, random_state=ctx.seed, n_jobs=1).fit(Z[sub])
        iso_score = pd.Series(-model.score_samples(Z), index=df.index)
        iso_n = int((iso_score >= iso_score.quantile(0.99)).sum())
    rank = sev.max(axis=1) + (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-9) * (0.5 if iso_n else 0)
    order = rank.sort_values(ascending=False).index[: top]
    order = [i for i in order if any_flag.loc[i] or (iso_n and iso_score.loc[i] >= iso_score.quantile(0.99))]
    rows = []
    for i in order:
        flagged = [c for c in cols if sev.at[i, c] > 0]
        worst = max(cols, key=lambda c: sev.at[i, c]) if flagged else max(cols, key=lambda c: abs((X.at[i, c] - X[c].median()) / (X[c].std() + 1e-9)) if pd.notna(X.at[i, c]) else 0)
        rows.append({"row": i, "most extreme in": worst, "value": _r(X.at[i, worst], 4), "also flagged in": ", ".join(c for c in flagged if c != worst), "severity": _r(rank.loc[i], 2),
                     **{f"{c}": _r(X.at[i, c], 4) for c in cols[:6] if c != worst}})
    tbl = pd.DataFrame(rows)
    ranked = sorted(per, key=lambda r: -r["flagged"])
    fig = None
    if len(cols) >= 2:
        c1, c2 = ranked[0]["column"], (ranked[1]["column"] if ranked[1]["column"] != ranked[0]["column"] else cols[1])
        fig = charts.scatter(X[c1].to_numpy(), X[c2].to_numpy(), c1, c2, (any_flag | (iso_score >= iso_score.quantile(0.99) if iso_n else any_flag)).to_numpy(), f"Flagged rows: {c2} against {c1}")
    else:
        fig = charts.box({cols[0]: X[cols[0]].dropna().to_numpy()}, f"Spread of {cols[0]}", cols[0])
    summary = {"rows": int(len(df)), "numeric_columns_checked": len(cols), "method": "IQR fences per column" + (" plus an isolation forest over all columns together" if iso_n else "") if method != "zscore" else "robust z-score",
               "rows_flagged_in_any_column": int(any_flag.sum()), "share_of_rows_pct": _pct(any_flag.mean()), "by_column": ranked[:6], "multivariate_rows_flagged": iso_n,
               "reading": "Flagged rows are unusual, not necessarily wrong: check each one before removing it."}
    return ToolResult("Outliers", summary, tbl, fig, text=f"Checked {len(cols)} numeric columns for outliers.")


def t_correlations(ctx: ToolContext, a) -> ToolResult:
    df = ctx.df
    cols = a["columns"] or numeric_columns(df, 3)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])][:25]
    if len(cols) < 2:
        raise ToolError("Correlations need at least two numeric columns.")
    method = (a["method"] or "pearson").lower()
    C = df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)
    pairs = [(cols[i], cols[j], float(C.iloc[i, j])) for i in range(len(cols)) for j in range(i + 1, len(cols)) if np.isfinite(C.iloc[i, j])]
    pairs.sort(key=lambda p: -abs(p[2]))
    top = int(min(max(a["top"] or 10, 1), 40))
    tbl = pd.DataFrame([{"column A": p[0], "column B": p[1], "correlation": round(p[2], 3), "strength": ("very strong" if abs(p[2]) >= .8 else "strong" if abs(p[2]) >= .6 else "moderate" if abs(p[2]) >= .4 else "weak")} for p in pairs[:top]])
    return ToolResult("Correlations", {"method": method, "columns": len(cols), "strongest_pairs": [{"a": p[0], "b": p[1], "r": round(p[2], 3)} for p in pairs[:8]],
                                       "pairs_above_0.8": sum(1 for p in pairs if abs(p[2]) >= .8), "reading": "Correlation shows association, not cause."},
                      tbl, charts.heatmap(C.to_numpy(), cols, f"{method.title()} correlations"), text=f"Computed {method} correlations for {len(cols)} columns.")


def t_missing(ctx: ToolContext, a) -> ToolResult:
    df = ctx.df
    m = df.isna()
    per = m.sum().sort_values(ascending=False)
    per = per[per > 0]
    if per.empty:
        return ToolResult("Missing values", {"columns_with_missing": 0, "rows": int(len(df))}, pd.DataFrame({"column": [], "missing": [], "missing %": []}), None, text="No missing values.")
    tbl = pd.DataFrame({"column": per.index, "missing": per.values, "missing %": (100 * per.values / len(df)).round(1)})
    co = []
    cols = list(per.index[:12])
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            both = float((m[cols[i]] & m[cols[j]]).sum())
            if both:
                co.append((cols[i], cols[j], both / max(1.0, min(m[cols[i]].sum(), m[cols[j]].sum()))))
    co.sort(key=lambda t: -t[2])
    return ToolResult("Missing values", {"columns_with_missing": int(len(per)), "worst": [{"column": c, "missing_pct": round(100 * v / len(df), 1)} for c, v in per.head(6).items()],
                                         "rows_with_any_missing_pct": _pct(m.any(axis=1).mean()), "often_missing_together": [{"a": x, "b": y, "overlap": round(o, 2)} for x, y, o in co[:4] if o >= 0.6]},
                      tbl, charts.bar(tbl["column"].tolist()[:20], tbl["missing %"].tolist()[:20], "Share of values missing (%)", "% missing"), text="Counted missing values per column.")


def t_group_summary(ctx: ToolContext, a) -> ToolResult:
    df, by, val, agg = ctx.df, a["by"], a["value"], (a["agg"] or "mean")
    if agg != "count" and val is None:
        raise ToolError("group_summary needs 'value' (a numeric column) unless agg is 'count'.")
    g = df.groupby(df[by].astype(str).where(df[by].notna()), dropna=True)
    res = g.size().rename("rows").to_frame() if agg == "count" else g[val].agg(agg).rename(f"{agg} of {val}").to_frame().join(g.size().rename("rows"))
    key = res.columns[0]
    res = res.sort_values(key, ascending=False).head(int(min(max(a["top"] or 15, 1), 50)))
    tbl = res.reset_index().rename(columns={res.index.name or "index": by}).round(4)
    summary = {"by": by, "metric": key, "groups_shown": int(len(tbl)), "groups_total": int(df[by].nunique()), "top": [{by: str(r[by]), key: _r(r[key]), "rows": int(r["rows"])} for _, r in tbl.head(8).iterrows()],
               "spread": _r(tbl[key].max() - tbl[key].min())}
    return ToolResult(f"{key.capitalize()} by {by}", summary, tbl, charts.bar(tbl[by].tolist(), tbl[key].tolist(), f"{key} by {by}", key), text=f"Summarised {val or 'rows'} by {by}.")


def t_relationship(ctx: ToolContext, a) -> ToolResult:
    from scipy import stats
    df, ca, cb = ctx.df, a["a"], a["b"]
    ka, kb = kind_of(df[ca]), kind_of(df[cb])
    na, nb = pd.api.types.is_numeric_dtype(df[ca]) and ka != "boolean", pd.api.types.is_numeric_dtype(df[cb]) and kb != "boolean"
    if na and nb:
        d = df[[ca, cb]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) < 5:
            raise ToolError("Too few rows have both values.")
        r, p = stats.pearsonr(d[ca], d[cb])
        rho, ps = stats.spearmanr(d[ca], d[cb])
        sl = stats.linregress(d[ca], d[cb])
        summary = {"rows_used": int(len(d)), "pearson_r": _r(r, 3), "pearson_p": float(f"{p:.3g}"), "spearman_rho": _r(rho, 3), "slope": _r(sl.slope, 4), "r_squared": _r(sl.rvalue ** 2, 3),
                   "reading": "Association, not cause." + (" The relationship looks non-linear (Spearman differs from Pearson)." if abs(abs(rho) - abs(r)) > 0.15 else "")}
        return ToolResult(f"{ca} and {cb}", summary, pd.DataFrame([{"measure": k, "value": v} for k, v in summary.items() if k != "reading"]), charts.scatter(d[ca], d[cb], ca, cb, trend=True), text=f"Tested the relationship between {ca} and {cb}.")
    if na != nb:                                                       # one numeric, one categorical
        num, cat = (ca, cb) if na else (cb, ca)
        d = df[[num, cat]].dropna()
        d[num] = pd.to_numeric(d[num], errors="coerce")
        d = d.dropna()
        lv = d[cat].astype(str).value_counts()
        keep = list(lv.index[:12])
        d = d[d[cat].astype(str).isin(keep)]
        groups = {g: v[num].to_numpy() for g, v in d.groupby(d[cat].astype(str)) if len(v) >= 2}
        if len(groups) < 2:
            raise ToolError(f"Need at least two groups of {cat} with two or more rows each.")
        arr = list(groups.values())
        if len(groups) == 2:
            t, p = stats.ttest_ind(*arr, equal_var=False)
            u, pu = stats.mannwhitneyu(*arr)
            sp = np.sqrt((arr[0].var(ddof=1) + arr[1].var(ddof=1)) / 2)
            eff = {"test": "Welch t-test", "p_value": float(f"{p:.3g}"), "mann_whitney_p": float(f"{pu:.3g}"), "cohens_d": _r((arr[0].mean() - arr[1].mean()) / (sp or 1), 3)}
        else:
            f, p = stats.f_oneway(*arr)
            h, pk = stats.kruskal(*arr)
            allv = np.concatenate(arr)
            ssb = sum(len(x) * (x.mean() - allv.mean()) ** 2 for x in arr)
            eff = {"test": "one-way ANOVA", "p_value": float(f"{p:.3g}"), "kruskal_p": float(f"{pk:.3g}"), "eta_squared": _r(ssb / (((allv - allv.mean()) ** 2).sum() or 1), 3)}
        tbl = pd.DataFrame([{cat: g, "rows": len(v), f"mean {num}": round(float(v.mean()), 4), f"median {num}": round(float(np.median(v)), 4), "std": round(float(v.std(ddof=1)), 4)} for g, v in groups.items()])
        summary = {"numeric": num, "groups": len(groups), "group_means": [{cat: g, "mean": _r(v.mean()), "rows": len(v)} for g, v in list(groups.items())[:8]], **eff,
                   "reading": "A small p-value means the groups differ by more than chance; the effect size says by how much it matters."}
        return ToolResult(f"{num} across {cat}", summary, tbl, charts.box(groups, f"{num} by {cat}", num), text=f"Compared {num} across groups of {cat}.")
    ct = pd.crosstab(df[ca].astype(str), df[cb].astype(str))
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index[:12], ct.sum(axis=0).sort_values(ascending=False).index[:12]]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ToolError("Both columns need at least two distinct values.")
    chi, p, dof, _ = stats.chi2_contingency(ct)
    v = float(np.sqrt(chi / (ct.to_numpy().sum() * (min(ct.shape) - 1))))
    summary = {"test": "chi-square test of independence", "p_value": float(f"{p:.3g}"), "cramers_v": _r(v, 3), "table_size": f"{ct.shape[0]} x {ct.shape[1]}",
               "reading": "Cramer's V near 0 is no association, near 1 is a strong one."}
    prop = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
    cells = [(i, j) for i in prop.index[:6] for j in prop.columns[:4]]
    fig = charts.bar([f"{i} / {j}" for i, j in cells], [float(prop.loc[i, j]) for i, j in cells], f"Share of {cb} within each {ca} (%)", "%")
    return ToolResult(f"{ca} and {cb}", summary, ct.reset_index(), fig, text=f"Tested whether {ca} and {cb} are related.")


def t_plot(ctx: ToolContext, a) -> ToolResult:
    df, kind, x, y = ctx.df, a["kind"], a["x"], a["y"]
    if kind == "hist":
        return t_distribution(ctx, {"column": resolve_column(df, x, numeric=True)})
    if kind == "scatter":
        if y is None:
            raise ToolError("A scatter plot needs both x and y.")
        d = _sample(df[[resolve_column(df, x, True), resolve_column(df, y, True)]].apply(pd.to_numeric, errors="coerce").dropna(), 8000, ctx.seed)
        return ToolResult(f"{y} against {x}", {"rows_plotted": int(len(d)), "pearson_r": _r(d.iloc[:, 0].corr(d.iloc[:, 1]), 3)}, None, charts.scatter(d.iloc[:, 0], d.iloc[:, 1], x, y, trend=True), text=f"Plotted {y} against {x}.")
    if kind == "box":
        yc = resolve_column(df, y or x, True)
        if y and x != y:
            g = {str(k): v[yc].dropna().to_numpy() for k, v in list(df.groupby(df[x].astype(str)))[:12]}
        else:
            g = {yc: df[yc].dropna().to_numpy()}
        return ToolResult(f"Spread of {yc}", {"groups": len(g)}, None, charts.box(g, f"{yc}" + (f" by {x}" if y and x != y else ""), yc), text=f"Plotted the spread of {yc}.")
    if kind == "bar":
        return t_group_summary(ctx, {"by": x, "value": y, "agg": a["agg"] or ("mean" if y else "count"), "top": 20})
    if kind == "line":
        return t_time_trend(ctx, {"date": x, "value": y, "agg": a["agg"] or "sum", "freq": None})
    raise ToolError("kind must be one of hist, scatter, box, bar, line.")


def t_filter_rows(ctx: ToolContext, a) -> ToolResult:
    df, conds = ctx.df, a["conditions"]
    m = _mask(df, conds) if conds else pd.Series(True, index=df.index)
    sub = df[m]
    cols = a["columns"] or list(df.columns[:12])
    limit = int(min(max(a["limit"] or 50, 1), MAX_TABLE_ROWS))
    if a["sort_by"]:
        sub = sub.sort_values(a["sort_by"], ascending=bool(a["ascending"]))
    num = [c for c in numeric_columns(df, 3)][:5]
    comp = [{"column": c, "matching_mean": _r(pd.to_numeric(sub[c], errors="coerce").mean()), "overall_mean": _r(pd.to_numeric(df[c], errors="coerce").mean())} for c in num] if len(sub) else []
    return ToolResult("Matching rows", {"matched": int(m.sum()), "of": int(len(df)), "share_pct": _pct(m.mean()), "conditions": conds, "numeric_means_matching_vs_overall": comp},
                      sub[cols].head(limit).reset_index().rename(columns={"index": "row"}), None, text=f"{int(m.sum()):,} of {len(df):,} rows match.")


def t_time_trend(ctx: ToolContext, a) -> ToolResult:
    df, dc, vc = ctx.df, a["date"], a["value"]
    d = pd.to_datetime(df[dc], errors="coerce")
    if d.notna().mean() < 0.5:
        raise ToolError(f"Column '{dc}' does not look like dates.")
    agg = a["agg"] or ("sum" if vc else "count")
    if vc is None and agg != "count":
        raise ToolError("Give 'value' (a numeric column) or use agg='count'.")
    span = (d.max() - d.min()).days
    freq = a["freq"] or ("D" if span <= 90 else "W" if span <= 800 else "MS")
    s = pd.Series(1.0, index=d) if vc is None else pd.Series(pd.to_numeric(df[vc], errors="coerce").to_numpy(), index=d)
    s = s[s.index.notna()]
    r = (s.resample(freq).count() if agg == "count" else s.resample(freq).agg(agg))
    if len(r) < 4:
        raise ToolError("Too few time periods to describe a trend; try a finer frequency.")
    y = r.to_numpy(float)
    xs = np.arange(len(y))
    ok = np.isfinite(y)
    slope = float(np.polyfit(xs[ok], y[ok], 1)[0])
    third = max(1, len(y) // 3)
    first, last = np.nanmean(y[:third]), np.nanmean(y[-third:])
    ma = pd.Series(y).rolling(max(2, len(y) // 12), min_periods=1).mean().to_numpy()
    i_hi, i_lo = int(np.nanargmax(y)), int(np.nanargmin(y))
    tbl = pd.DataFrame({"period": r.index.strftime("%Y-%m-%d"), (f"{agg} of {vc}" if vc else "rows"): np.round(y, 4)})
    summary = {"periods": int(len(y)), "frequency": freq, "start": str(r.index[0].date()), "end": str(r.index[-1].date()), "average_change_per_period": _r(slope, 4),
               "change_first_to_last_third_pct": _r(100 * (last - first) / abs(first), 1) if first else None, "peak": {"period": str(r.index[i_hi].date()), "value": _r(y[i_hi])},
               "low": {"period": str(r.index[i_lo].date()), "value": _r(y[i_lo])}, "direction": "rising" if slope > 0 and abs(last - first) > np.nanstd(y) * 0.25 else "falling" if slope < 0 and abs(last - first) > np.nanstd(y) * 0.25 else "roughly flat"}
    return ToolResult(f"{(vc or 'Rows')} over time", summary, tbl, charts.line([str(p.date()) for p in r.index], {f"{agg} of {vc}" if vc else "rows": y, "moving average": ma}, f"{(vc or 'Rows')} per {freq}"),
                      text=f"Aggregated {vc or 'rows'} over {dc} ({freq}).")


def t_duplicates(ctx: ToolContext, a) -> ToolResult:
    df = ctx.df
    cols = a["columns"] or list(df.columns)
    dup = df.duplicated(subset=cols, keep=False)
    sub = df[dup].sort_values(cols[:3] if len(cols) >= 1 else list(df.columns)[:1]).head(MAX_TABLE_ROWS)
    n_extra = int(df.duplicated(subset=cols, keep="first").sum())
    return ToolResult("Duplicates", {"checked_on": cols if a["columns"] else "all columns", "rows_in_duplicate_groups": int(dup.sum()), "removable_extra_copies": n_extra, "share_pct": _pct(n_extra / max(1, len(df)))},
                      sub.reset_index().rename(columns={"index": "row"}), None, text="Looked for duplicate rows.")


def t_drivers(ctx: ToolContext, a) -> ToolResult:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from .. import modeling
    df, t = ctx.df, a["target"]
    d = df[df[t].notna()]
    if len(d) < 60:
        raise ToolError("Need at least 60 rows with a known target.")
    d = _sample(d, MODEL_ROWS, ctx.seed)
    y_raw = d[t]
    clf = (not pd.api.types.is_numeric_dtype(y_raw)) or pd.api.types.is_bool_dtype(y_raw) or (profiling.is_integer_like(y_raw) and y_raw.nunique() <= 10)
    if clf:
        yy, classes = pd.factorize(y_raw.astype(str))
        if len(classes) < 2 or pd.Series(yy).value_counts().min() < 5:
            raise ToolError("The target needs at least two classes with five or more rows each.")
    else:
        yy = pd.to_numeric(y_raw, errors="coerce").to_numpy()
    X = d.drop(columns=[t])
    drop = [c for c in X.columns if X[c].isna().mean() > 0.6 or X[c].nunique() <= 1 or kind_of(X[c]) == "datetime" or (not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 0.9 * len(X) and len(X) > 50)]
    X = X.drop(columns=drop)
    if X.shape[1] == 0:
        raise ToolError("No usable feature columns remain.")
    pipe = Pipeline([("prep", modeling.build_preprocessor(X)), ("model", (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(max_iter=80, random_state=ctx.seed))])
    metric = ("roc_auc" if len(classes) == 2 else "f1_macro") if clf else "r2"
    cv = cross_val_score(pipe, X, yy, cv=3, scoring=metric)
    base = (0.5 if metric == "roc_auc" else float(np.bincount(yy).max() / len(yy)) * 0.0) if clf else 0.0
    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.25, random_state=ctx.seed, stratify=yy if clf else None)
    pipe.fit(Xtr, ytr)
    pi = permutation_importance(pipe, Xte, yte, n_repeats=3, random_state=ctx.seed, scoring=metric, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
    top = int(min(max(a["top"] or 10, 1), 25))
    tbl = pd.DataFrame({"feature": imp.index[:top], "importance": imp.values[:top].round(4)})
    summary = {"target": t, "task": "classification" if clf else "regression", "metric": metric, "cross_validated_score": _r(cv.mean(), 3), "chance_level": base, "rows_used": int(len(d)),
               "features_considered": int(X.shape[1]), "excluded": drop[:8], "top_drivers": [{"feature": f, "importance": _r(v, 4)} for f, v in imp.head(8).items()],
               "reading": "These are the features the model leans on most. That is association, not proof of cause, and a quick model is not a tuned one."}
    return ToolResult(f"What moves {t}", summary, tbl, charts.bar(tbl["feature"].tolist(), tbl["importance"].tolist(), f"Importance for predicting {t} ({metric} drop when shuffled)", "importance"),
                      notes=["Fast single-model estimate on a sample; run a full analysis for a tuned, validated model."], text=f"Ranked the features that predict {t}.")


def t_propose_analysis(ctx: ToolContext, a) -> ToolResult:
    prop = {"objective": a["objective"], "target": a["target"], "goal_metric": a["goal_metric"], "goal_value": a["goal_value"]}
    return ToolResult("Full analysis proposal", {"proposed": True, "shown_to_user": True}, None, None, proposal=prop,
                      text="Prepared a full analysis for you to review; nothing has been started.")


def t_ask_run(ctx: ToolContext, a) -> ToolResult:
    if ctx.answer_run is None or ctx.run_state is None:
        raise ToolError("No finished analysis is open. Open one from Analyses first.")
    ans = ctx.answer_run(a["question"])
    return ToolResult("From the open analysis", {"answer": ans}, None, None, text="Looked the answer up in the open analysis.")


# registry
def _t(name, desc, params, fn, needs_data=True) -> Tool:
    return Tool(name, desc, params, fn, needs_data)


P = Param
REGISTRY: Dict[str, Tool] = {t.name: t for t in [
    _t("overview", "Profile the whole dataset: size, column types, missing values, duplicates, quality score.", {}, t_overview),
    _t("distribution", "Describe ONE column: statistics and a histogram (numeric) or the most common values (categorical).", {"column": P("column", "which column", True)}, t_distribution),
    _t("value_counts", "Count rows per value of a categorical column.", {"column": P("column", "categorical column", True), "top": P("int", "how many values", default=15)}, t_value_counts),
    _t("outliers", "Find unusual rows. Checks each numeric column with robust fences and, with two or more columns, an isolation forest across them together.",
       {"columns": P("columns", "numeric columns to check (default: all suitable)"), "method": P("choice", "auto | iqr | zscore | isolation_forest", default="auto", choices=["auto", "iqr", "zscore", "isolation_forest"]),
        "top": P("int", "rows to list", default=15)}, t_outliers),
    _t("correlations", "Strongest correlations among numeric columns, with a heatmap.", {"columns": P("columns", "numeric columns (default: all)"), "method": P("choice", "pearson | spearman", default="pearson", choices=["pearson", "spearman"]),
                                                                                            "top": P("int", "pairs to list", default=10)}, t_correlations),
    _t("missing", "Where values are missing and which columns tend to be missing together.", {}, t_missing),
    _t("group_summary", "Aggregate a numeric column (or count rows) by the values of a category.", {"by": P("column", "grouping column", True), "value": P("numeric_column", "numeric column to aggregate"),
        "agg": P("choice", "mean | median | sum | count | min | max | std", default="mean", choices=["mean", "median", "sum", "count", "min", "max", "std"]), "top": P("int", "groups to show", default=15)}, t_group_summary),
    _t("relationship", "Test how two columns relate (number-number: correlation and fit; number-category: group comparison; category-category: chi-square).",
       {"a": P("column", "first column", True), "b": P("column", "second column", True)}, t_relationship),
    _t("plot", "Draw a chart: hist, scatter, box, bar or line.", {"kind": P("choice", "hist | scatter | box | bar | line", True, choices=["hist", "scatter", "box", "bar", "line"]), "x": P("column", "x column", True),
        "y": P("column", "y column"), "agg": P("choice", "aggregation for bar / line", choices=["mean", "median", "sum", "count", "min", "max"])}, t_plot),
    _t("filter_rows", "Show rows that match conditions (no code). Operators: eq, ne, gt, ge, lt, le, in, not_in, contains, between, is_null, not_null.",
       {"conditions": P("conditions", "list of {column, op, value}"), "columns": P("columns", "columns to show"), "sort_by": P("column", "sort column"), "ascending": P("bool", "sort ascending", default=False),
        "limit": P("int", "max rows", default=50)}, t_filter_rows),
    _t("time_trend", "How a value (or the row count) changes over time.", {"date": P("column", "date column", True), "value": P("numeric_column", "numeric column (omit to count rows)"),
        "agg": P("choice", "sum | mean | median | count | min | max", choices=["sum", "mean", "median", "count", "min", "max"]), "freq": P("choice", "D | W | MS | QS", choices=["D", "W", "MS", "QS"])}, t_time_trend),
    _t("duplicates", "Find duplicate rows (optionally on a subset of columns).", {"columns": P("columns", "columns that define a duplicate (default: all)")}, t_duplicates),
    _t("drivers", "Quick model: which columns best predict a target, and how well. Fast single-model estimate, not a tuned one.", {"target": P("column", "column to predict", True), "top": P("int", "features to list", default=10)}, t_drivers),
    _t("propose_analysis", "Use ONLY when the person clearly wants a complete, validated modelling analysis or a model that meets an accuracy goal. Prepares a proposal they confirm; it does not start anything.",
       {"objective": P("text", "the objective in the person's words", True), "target": P("column", "column to predict, if known"), "goal_metric": P("text", "e.g. roc_auc, r2, recall"), "goal_value": P("number", "target value")}, t_propose_analysis),
    _t("ask_run", "Answer a question about the finished analysis the person has open (results, models, findings).", {"question": P("text", "the question", True)}, t_ask_run, needs_data=False),
]}


def catalog(has_run: bool = False) -> List[Dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "params": {k: f"{p.kind}{' (required)' if p.required else ''}: {p.desc}" + (f" [{', '.join(p.choices)}]" if p.choices else "") for k, p in t.params.items()}}
            for t in REGISTRY.values() if has_run or t.name != "ask_run"]


def run_tool(ctx: ToolContext, name: str, raw_args: Dict[str, Any]) -> ToolResult:
    """Validate then run one tool. Raises ToolError for a bad request; any unexpected failure is reported as a ToolError too (the assistant can recover, the app never crashes)."""
    tool = REGISTRY.get(name)
    if tool is None:
        raise ToolError(f"Unknown tool '{name}'. Available: {[t for t in REGISTRY if t != 'ask_run' or ctx.run_state]}.")
    if tool.needs_data and (ctx.df is None or len(ctx.df) == 0):
        raise ToolError("No dataset is loaded. Ask the person to add one, or answer without data.")
    args = coerce_args(tool, ctx.df, raw_args)
    t0 = time.time()
    try:
        res = tool.fn(ctx, args)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"{name} could not run on this data ({type(e).__name__}: {str(e)[:120]}).")
    res.summary = jsonable(res.summary)
    res.notes = list(res.notes) + ([f"Showing the first {MAX_TABLE_ROWS} rows."] if res.table is not None and len(res.table) > MAX_TABLE_ROWS else [])
    if res.table is not None and len(res.table) > MAX_TABLE_ROWS:
        res.table = res.table.head(MAX_TABLE_ROWS)
    res.summary["_seconds"] = round(time.time() - t0, 2)
    return res
