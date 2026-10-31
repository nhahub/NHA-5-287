"""Trend, seasonality and anomaly analysis on business measures (pure functions)."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .profiling import to_datetime


def choose_freq(dates: pd.Series) -> str:
    """Pick an aggregation frequency that gives ~20-120 periods and respects native cadence."""
    d = pd.Series(dates).dropna().sort_values().drop_duplicates()
    if len(d) < 3:
        return "D"
    step = d.diff().dt.days.median()
    span = (d.iloc[-1] - d.iloc[0]).days
    if step >= 25:
        return "MS"
    if step >= 6:
        return "W"
    if span <= 120:
        return "D"
    return "W" if span <= 900 else "MS"


def measure_value(sub: pd.DataFrame, spec: Dict[str, Any]) -> float:
    a, col = spec["agg"], spec.get("column")
    if a == "count":
        return float(len(sub))
    if a == "nunique":
        return float(sub[col].nunique())
    if a == "rate":
        return float((sub[col].astype(str) == str(spec["positive"])).mean()) if len(sub) else np.nan
    if a == "ratio":
        den = pd.to_numeric(sub[spec["denominator"]], errors="coerce").sum()
        return float(pd.to_numeric(sub[spec["numerator"]], errors="coerce").sum() / den) if den else np.nan
    x = pd.to_numeric(sub[col], errors="coerce")
    return float(getattr(x, a)()) if x.notna().any() else np.nan


def period_series(df: pd.DataFrame, date_col: str, spec: Dict[str, Any], freq: Optional[str] = None) -> pd.Series:
    d = df.copy()
    d[date_col] = to_datetime(d[date_col])
    d = d.dropna(subset=[date_col])
    freq = freq or choose_freq(d[date_col])
    grp = d.groupby(pd.Grouper(key=date_col, freq=freq))
    s = grp.apply(lambda g: measure_value(g, spec) if len(g) else np.nan)
    days = grp[date_col].apply(lambda g: g.dt.normalize().nunique())
    s = s.dropna()
    # drop incomplete edge periods (a 1-day "week" would look like a crash / fake anomaly)
    if len(s) >= 4 and freq != "D":
        med = float(days.reindex(s.index).median())
        if days.reindex(s.index).iloc[-1] < 0.8 * med:
            s = s.iloc[:-1]
        if len(s) and days.reindex(s.index).iloc[0] < 0.8 * med:
            s = s.iloc[1:]
    return s


def analyze_series(s: pd.Series, name: str) -> Dict[str, Any]:
    n = len(s)
    out: Dict[str, Any] = {"measure": name, "n_periods": n, "freq": str(pd.infer_freq(s.index) or "irregular")}
    if n < 8:
        out["skipped"] = f"Only {n} periods; need >= 8 for trend/anomaly analysis."
        return out
    y = s.to_numpy(float)
    idx = np.arange(n)
    slope, intercept = np.polyfit(idx, y, 1)
    mean = float(np.mean(np.abs(y))) or 1.0
    tau, p = stats.kendalltau(idx, y)
    out.update(
        start=str(s.index[0].date()), end=str(s.index[-1].date()), mean=round(float(y.mean()), 4),
        first=round(float(y[0]), 4), last=round(float(y[-1]), 4),
        slope_per_period=round(float(slope), 4), slope_pct_of_mean=round(float(slope / mean * 100), 3),
        mann_kendall_tau=round(float(tau), 3), mann_kendall_p=float(p),
        trend=("increasing" if tau > 0 else "decreasing") if p < config.ALPHA else "no significant trend",
        volatility_cv=round(float(np.std(y) / mean), 3))
    # seasonality: share of variance explained by calendar position (eta squared)
    fq = out["freq"]
    cal = None
    if fq.startswith("D") or fq.startswith("<Day"):
        cal, label = s.index.dayofweek, "day of week"
    elif n >= 24 and (s.index[-1] - s.index[0]).days >= 700:
        cal, label = s.index.month, "month of year"
    if cal is not None:
        g = pd.Series(y).groupby(np.asarray(cal))
        ssb = sum(len(v) * (v.mean() - y.mean()) ** 2 for _, v in g)
        sst = ((y - y.mean()) ** 2).sum() or 1.0
        eta = float(ssb / sst)
        prof = {int(k): round(float(v.mean()), 3) for k, v in g}
        out["seasonality"] = {"by": label, "strength": round(eta, 3), "profile": prof,
                              "present": eta > 0.15}
    # anomalies: residual vs rolling median, robust z (MAD)
    w = 7 if n >= 28 else 5
    base = pd.Series(y).rolling(w, center=True, min_periods=3).median().to_numpy()
    resid = y - base
    mad = np.median(np.abs(resid - np.median(resid))) or (np.std(resid) or 1.0)
    z = 0.6745 * (resid - np.median(resid)) / mad
    flagged = np.where(np.abs(z) > 3.5)[0]
    top = sorted(flagged, key=lambda i: -abs(z[i]))[:10]
    out["anomalies"] = [{"period": str(s.index[i].date()), "value": round(float(y[i]), 4),
                         "expected": round(float(base[i]), 4), "robust_z": round(float(z[i]), 2),
                         "direction": "spike" if z[i] > 0 else "drop"} for i in top]
    out["n_anomalies"] = int(len(flagged))
    # level shift: regress y on [1, t, step(t>=k)] for every k (min 5 periods each side) and keep the k whose
    # step coefficient is most significant. Trend and jump are fitted TOGETHER, so a trending series
    # does not produce a fake break and a real jump is located correctly.
    best = (0.0, None, 0.0)
    for k in range(5, n - 4):
        X = np.column_stack([np.ones(n), idx, (idx >= k).astype(float)])
        beta, res, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = n - 3
        sigma2 = float(resid @ resid) / dof if dof > 0 else 0.0
        try:
            se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[2, 2])
        except np.linalg.LinAlgError:
            continue
        t = abs(beta[2]) / se if se > 0 else 0.0
        if t > best[0]:
            best = (t, k, float(beta[2]))
    if best[1] is not None and best[0] > 6:
        k = best[1]
        out["change_point"] = {"period": str(s.index[k].date()), "shift_vs_trend": round(best[2], 4),
                               "before_mean": round(float(y[:k].mean()), 4), "after_mean": round(float(y[k:].mean()), 4),
                               "t_stat": round(float(best[0]), 2)}
    return out


def numeric_outliers(df: pd.DataFrame, cols: List[str], top: int = 8) -> Dict[str, Any]:
    """Cross-sectional outliers via IQR fences (robust to skew)."""
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(x) < 20 or x.nunique() < 8:
            continue
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((x < lo) | (x > hi)).sum())
        if n_out:
            rows.append({"column": c, "n_outliers": n_out, "pct": round(n_out / len(x) * 100, 2),
                         "lower_fence": round(float(lo), 4), "upper_fence": round(float(hi), 4),
                         "max": round(float(x.max()), 4), "min": round(float(x.min()), 4)})
    rows.sort(key=lambda r: -r["pct"])
    return {"columns": rows[:top], "n_columns_with_outliers": len(rows)}
