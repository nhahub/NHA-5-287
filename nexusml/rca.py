"""Root-cause EVIDENCE: deterministic driver analysis that the LLM later reasons over.

Two mechanisms, both descriptive/mechanical (they say WHERE a difference or change comes from,
not that a factor CAUSES it):
  * segment_lifts       - which segments have an unusually high/low outcome vs everyone else
  * contribution_change - how much each segment contributes to a change between two windows
                          (works for sums and for rates/means via mix-weighted contributions)
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .profiling import to_datetime


def measure_vector(df: pd.DataFrame, target: Optional[str], problem_type: Optional[str],
                   positive: Optional[str], fallback_spec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Per-row numeric outcome + how to talk about it."""
    if target and target in df.columns:
        s = df[target]
        if problem_type == "classification" or not pd.api.types.is_numeric_dtype(s):
            pos = str(positive) if positive is not None else str(s.astype(str).value_counts().index[-1])
            return {"y": (s.astype(str) == pos).astype(float).where(s.notna()), "name": f"{target}={pos} rate",
                    "kind": "rate", "additive": False, "fmt": "pct"}
        return {"y": s.astype(float), "name": target, "kind": "mean", "additive": problem_type == "time_series",
                "fmt": "num"}
    sp = fallback_spec
    if sp and sp["agg"] in ("sum", "mean", "median") and sp.get("column") in df.columns:
        return {"y": pd.to_numeric(df[sp["column"]], errors="coerce"), "name": sp["column"], "kind": "mean",
                "additive": sp["agg"] == "sum", "fmt": "num"}
    if sp and sp["agg"] == "rate" and sp.get("column") in df.columns:
        return {"y": (df[sp["column"]].astype(str) == str(sp["positive"])).astype(float), "name": sp["name"],
                "kind": "rate", "additive": False, "fmt": "pct"}
    return None


def _f(v: float, kind: str) -> str:
    return f"{v * 100:.1f}%" if kind == "pct" else (f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}")


def _segment_frame(df: pd.DataFrame, cols: List[str]) -> Dict[str, pd.Series]:
    """Categorical columns as-is; numeric columns binned to quartiles so they can be segmented."""
    out = {}
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if s.nunique() > 8:
                try:
                    out[c] = pd.qcut(s, 4, duplicates="drop").astype(str).where(s.notna())
                except Exception:
                    continue
            elif s.nunique() >= 2:
                out[c] = s.astype(str).where(s.notna())
        elif 2 <= s.nunique() <= 20:
            out[c] = s.astype(str).where(s.notna())
    return out


def segment_lifts(df: pd.DataFrame, m: Dict[str, Any], cols: List[str], max_items: int = 8) -> List[Dict[str, Any]]:
    y = m["y"]
    ok = y.notna()
    total_y = float(y[ok].sum())
    n_all = int(ok.sum())
    rows = []
    for c, seg in _segment_frame(df, cols).items():
        for lv in seg[ok].dropna().unique():
            mask = (seg == lv) & ok
            n = int(mask.sum())
            if n < max(30, 0.01 * n_all) or n > 0.98 * n_all:
                continue
            a, b = y[mask], y[ok & ~mask]
            if len(b) < 30:
                continue
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            if not se or np.isnan(se):
                continue
            z = float((a.mean() - b.mean()) / se)
            rows.append({"column": c, "level": str(lv), "n": n, "share_rows": n / n_all, "mean": float(a.mean()),
                         "rest": float(b.mean()), "lift": float(a.mean() / b.mean()) if b.mean() else np.nan,
                         "z": z, "share_of_total": float(a.sum() / total_y) if total_y else np.nan})
    rows = [r for r in rows if abs(r["z"]) >= 3]
    rows.sort(key=lambda r: -abs(r["z"]))
    seen, out = {}, []
    for r in rows:                      # at most 2 levels per column so one column does not dominate
        if seen.get(r["column"], 0) < 2:
            out.append(r)
            seen[r["column"]] = seen.get(r["column"], 0) + 1
        if len(out) >= max_items:
            break
    return out


def _contrib(base: pd.DataFrame, cur: pd.DataFrame, seg_b: pd.Series, seg_c: pd.Series, y: str, additive: bool,
             base_scale: float = 1.0) -> List[Dict[str, Any]]:
    levels = pd.concat([seg_b, seg_c]).dropna().value_counts().index[:15]
    rows = []
    for lv in levels:
        b, c = base.loc[seg_b == lv, y].dropna(), cur.loc[seg_c == lv, y].dropna()
        if additive:
            vb, vc = b.sum() / base_scale, c.sum()
        else:
            wb, wc = len(b) / max(1, len(base)), len(c) / max(1, len(cur))
            vb = wb * b.mean() if len(b) else 0.0
            vc = wc * c.mean() if len(c) else 0.0
        rows.append({"level": str(lv), "base": float(vb), "current": float(vc), "delta": float(vc - vb),
                     "n_base": int(len(b)), "n_current": int(len(c))})
    tot = sum(r["delta"] for r in rows)
    for r in rows:
        r["share_of_change"] = r["delta"] / tot if tot else np.nan
    rows.sort(key=lambda r: -abs(r["delta"]))
    return rows


def contribution_change(df: pd.DataFrame, date_col: str, m: Dict[str, Any], cols: List[str]) -> Optional[Dict[str, Any]]:
    """Compare first half vs second half of the time range; decompose the change by segment."""
    d = df.copy()
    d["__d"] = to_datetime(d[date_col])
    d["__y"] = m["y"]
    d = d.dropna(subset=["__d", "__y"])
    if len(d) < 60 or d["__d"].nunique() < 8:
        return None
    mid = d["__d"].min() + (d["__d"].max() - d["__d"].min()) / 2
    base, cur = d[d["__d"] <= mid], d[d["__d"] > mid]
    if len(base) < 20 or len(cur) < 20:
        return None
    segs = _segment_frame(d, cols)
    if m["additive"]:
        tb, tc = base["__y"].sum(), cur["__y"].sum()
    else:
        tb, tc = base["__y"].mean(), cur["__y"].mean()
    out = {"windows": {"base": [str(base["__d"].min().date()), str(base["__d"].max().date())],
                       "current": [str(cur["__d"].min().date()), str(cur["__d"].max().date())]},
           "total_base": float(tb), "total_current": float(tc), "by_column": {}}
    for c, s in segs.items():
        rows = _contrib(base, cur, s.loc[base.index], s.loc[cur.index], "__y", m["additive"])
        if rows:
            out["by_column"][c] = rows[:5]
    return out


def explain_anomaly(df: pd.DataFrame, date_col: str, m: Dict[str, Any], cols: List[str],
                    period: str, freq: str) -> Optional[Dict[str, Any]]:
    """Decompose one anomalous period against the average of the k preceding periods."""
    d = df.copy()
    d["__d"] = to_datetime(d[date_col]).dt.normalize()
    d["__y"] = m["y"]
    d = d.dropna(subset=["__d", "__y"])
    p0, k = pd.Timestamp(period), 4
    if freq == "D":
        cur = d[d["__d"] == p0]
        base = d[(d["__d"] < p0) & (d["__d"] >= p0 - pd.Timedelta(days=7))]
        k = 7
    elif freq == "MS":
        cur = d[(d["__d"] >= p0) & (d["__d"] < p0 + pd.DateOffset(months=1))]
        base = d[(d["__d"] < p0) & (d["__d"] >= p0 - pd.DateOffset(months=3))]
        k = 3
    else:                                                    # weekly, label = week end
        cur = d[(d["__d"] > p0 - pd.Timedelta(days=7)) & (d["__d"] <= p0)]
        base = d[(d["__d"] <= p0 - pd.Timedelta(days=7)) & (d["__d"] > p0 - pd.Timedelta(days=7 * (k + 1)))]
    if len(cur) < 3 or len(base) < 5:
        return None
    segs = _segment_frame(d, cols)
    out = {"period": period, "by_column": {}}
    for c, sg in segs.items():
        rows = _contrib(base, cur, sg.loc[base.index], sg.loc[cur.index], "__y", m["additive"], base_scale=k)
        if rows:
            out["by_column"][c] = rows[:3]
    return out


def evidence_from(lifts: List[Dict[str, Any]], change: Optional[Dict[str, Any]], m: Dict[str, Any],
                  anomaly_expl: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ev: List[Dict[str, Any]] = []
    k = m["fmt"]

    def add(text, data):
        ev.append({"id": f"D{len(ev) + 1:02d}", "source": "driver_analysis", "text": text, "data": data})

    for r in lifts:
        hi = r["mean"] > r["rest"]
        add(f"{r['column']} = {r['level']}: {m['name']} is {_f(r['mean'], k)} vs {_f(r['rest'], k)} for all other records "
            f"({'higher' if hi else 'lower'}; n={r['n']:,}, {r['share_rows']:.0%} of rows, "
            f"{r['share_of_total']:.0%} of total outcome; z={r['z']:.1f}).", r)
    if change:
        w = change["windows"]
        add(f"{m['name']} moved from {_f(change['total_base'], k)} ({w['base'][0]} to {w['base'][1]}) to "
            f"{_f(change['total_current'], k)} ({w['current'][0]} to {w['current'][1]}).", {"windows": w})
        rel = abs(change["total_current"] - change["total_base"]) / (abs(change["total_base"]) or 1)
        for col, rows in (change["by_column"].items() if rel >= 0.05 else []):   # skip decomposing a flat series
            top = rows[0]
            if 0.4 <= (top.get("share_of_change", 0) or 0) <= 1.3 and len(rows) > 1:
                add(f"Of that change, {col} = {top['level']} accounts for {top['share_of_change']:.0%} "
                    f"(contribution {_f(top['delta'], k)}; n now {top['n_current']:,} vs {top['n_base']:,} before).", top)
    for a in anomaly_expl:
        for col, rows in a["by_column"].items():
            top = rows[0]
            if 0.5 <= (top.get("share_of_change", 0) or 0) <= 1.3:
                add(f"Around {a['period']}, {col} = {top['level']} explains {top['share_of_change']:.0%} of the "
                    f"deviation from the preceding baseline.", {"period": a["period"], **top})
                break
    return ev


# combined segments & driver groups (v4)
def interaction_segments(df: pd.DataFrame, m: Dict[str, Any], cols: List[str], max_items: int = None,
                         min_support: int = None, max_cols: int = 14) -> Dict[str, Any]:
    """Search two-factor combinations (col1=a AND col2=b). A combination is reported only when it is clearly worse/better than BOTH
    of its single factors, clears a z-threshold that accounts for the number of combinations screened, and has enough rows."""
    from . import config
    from itertools import combinations
    max_items = max_items or config.INTERACTION_MAX_ITEMS
    min_support = min_support or config.INTERACTION_MIN_SUPPORT
    y = m["y"]
    ok = y.notna()
    n_all = int(ok.sum())
    seg = _segment_frame(df, cols[:max_cols])
    overall = float(y[ok].mean())
    singles = {}
    for c, s in seg.items():
        for lv in s[ok].dropna().unique():
            mask = (s == lv) & ok
            if mask.sum() >= min_support:
                singles[(c, str(lv))] = float(y[mask].mean())
    screened, rows = 0, []
    for c1, c2 in combinations(seg, 2):
        s1, s2 = seg[c1], seg[c2]
        for l1 in s1[ok].dropna().unique():
            for l2 in s2[ok].dropna().unique():
                mask = (s1 == l1) & (s2 == l2) & ok
                n = int(mask.sum())
                if n < min_support or n > 0.6 * n_all:
                    continue
                screened += 1
                a, b = y[mask], y[ok & ~mask]
                if m.get("kind") == "rate":          # pooled two-proportion z: stays honest when a group has 0 events
                    pp = float(y[ok].mean())
                    se = np.sqrt(pp * (1 - pp) * (1 / len(a) + 1 / len(b)))
                else:
                    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
                if not se or np.isnan(se):
                    continue
                z = float((a.mean() - b.mean()) / se)
                s_a, s_b = singles.get((c1, str(l1))), singles.get((c2, str(l2)))
                if s_a is None or s_b is None:
                    continue
                hi = a.mean() > overall
                best_single = max(s_a, s_b) if hi else min(s_a, s_b)
                margin = (a.mean() - best_single) if hi else (best_single - a.mean())
                rel = margin / abs(best_single) if best_single else 0.0
                lift = a.mean() / overall if overall else np.nan
                if abs(z) >= 3.5 and margin > 0 and rel >= 0.10 and (lift >= config.INTERACTION_MIN_LIFT or lift <= 1 / config.INTERACTION_MIN_LIFT):
                    rows.append({"a": c1, "a_level": str(l1), "b": c2, "b_level": str(l2), "n": n, "share_rows": n / n_all,
                                 "mean": float(a.mean()), "rest": float(b.mean()), "single_a": s_a, "single_b": s_b,
                                 "lift": float(lift), "z": z, "share_of_total": float(a.sum() / y[ok].sum()) if y[ok].sum() else np.nan})
    rows.sort(key=lambda r: -abs(r["z"]))
    out, per_pair = [], {}
    for r in rows:
        k = (r["a"], r["b"])
        if per_pair.get(k, 0) < 1:                  # one best combination per column pair
            out.append(r)
            per_pair[k] = 1
        if len(out) >= max_items:
            break
    return {"items": out, "screened": screened, "overall": overall}


def interaction_evidence(res: Dict[str, Any], m: Dict[str, Any], start: int) -> List[Dict[str, Any]]:
    k, ev = m["fmt"], []
    for i, r in enumerate(res["items"]):
        ev.append({"id": f"D{start + i + 1:02d}", "source": "interaction_analysis",
                   "text": (f"Combination {r['a']} = {r['a_level']} AND {r['b']} = {r['b_level']}: {m['name']} is {_f(r['mean'], k)} vs {_f(r['rest'], k)} for everyone else "
                            f"(n={r['n']:,}, {r['share_rows']:.0%} of rows; each factor alone: {_f(r['single_a'], k)} and {_f(r['single_b'], k)}; "
                            f"{res['screened']:,} combinations screened, z={r['z']:.1f})."), "data": r})
    return ev


def driver_groups(df: pd.DataFrame, drivers: List[str], threshold: float = None) -> List[Dict[str, Any]]:
    """Group top model drivers that move together (|association| >= threshold) so they are not presented as independent causes.
    numeric-numeric: |Spearman|; categorical-categorical: Cramer's V; mixed: correlation ratio (eta)."""
    from . import config, viz
    threshold = threshold or config.DRIVER_GROUP_CORR
    cols = [c for c in dict.fromkeys(drivers) if c in df.columns][:12]
    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    assoc = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            na, nb = pd.api.types.is_numeric_dtype(df[a]), pd.api.types.is_numeric_dtype(df[b])
            try:
                if na and nb:
                    v = abs(float(df[[a, b]].dropna().corr(method="spearman").iloc[0, 1]))
                elif not na and not nb:
                    v = float(viz.cramers_v(df[a], df[b]))
                else:
                    v = float(viz.eta_correlation(df[a] if not na else df[b], df[b] if nb else df[a]))
            except Exception:
                continue
            if v == v and v >= threshold:
                assoc[(a, b)] = round(v, 2)
                parent[find(a)] = find(b)
    groups: Dict[str, List[str]] = {}
    for c in cols:
        groups.setdefault(find(c), []).append(c)
    out = []
    for members in groups.values():
        if len(members) >= 2:
            pairs = {f"{a}~{b}": v for (a, b), v in assoc.items() if a in members and b in members}
            out.append({"members": members, "associations": pairs})
    return out
