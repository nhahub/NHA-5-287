"""The model advisor: look at the data first, then decide WHICH model families are worth trying and WHY.

v4 evaluated the same five families on every dataset. v5 measures the data and builds a shortlist from the measurements; every inclusion and every exclusion carries a
reason that is shown in the report. Nothing here is a black box and nothing needs a language model (a model may later ask for more families; the audit below
catches wrong exclusions either way).

Insights
  time series  season length (ACF/seasonal-profile search), seasonal strength, trend strength, multiplicative (log) scale, random-walk-ness
  tabular      size, feature mix, missing share, skew, imbalance, and two cheap PROBES on a subsample with cross-validation:
               a linear model vs a small boosted-tree model -> is the signal linear, how much do non-linear effects add, what is the achievable ceiling
Plan           shortlist + excluded (with reasons). Session.audit_exclusions() later fits each excluded family once; if one beats the incumbent it is promoted,
               so an unlucky rule can cost a little time but never the result.
"""
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ------------------------------------------------------------------ time series
def _candidate_periods(step_days: float, n: int) -> List[int]:
    if step_days >= 300:
        return []
    if 80 <= step_days < 300:
        c = [4]
    elif 20 <= step_days < 45:
        c = [12]
    elif 5 <= step_days < 9:
        c = [52]
    elif 0.9 <= step_days < 1.5:
        c = [7] + ([365] if n >= 800 else [])
    elif step_days < 0.9:
        h = max(step_days * 24, 1e-6)
        c = [int(round(24 / h))] + ([int(round(168 / h))] if n >= 24 * 21 / h else [])
    else:
        c = []
    return [p for p in c if 2 <= p <= n // 3]


def _detrend(y: np.ndarray, deg: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(-1, 1, len(y))
    trend = np.polyval(np.polyfit(t, y, min(deg, max(len(y) // 20, 1))), t)
    return trend, y - trend


def _seasonal_strength(r: np.ndarray, p: int) -> float:
    if p < 2 or len(r) < 2 * p:
        return 0.0
    ph = np.arange(len(r)) % p
    prof = np.array([r[ph == k].mean() for k in range(p)])
    v = float(np.var(r))
    return float(np.clip(1.0 - np.var(r - prof[ph]) / v, 0.0, 1.0)) if v > 0 else 0.0


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    d = float((x ** 2).sum())
    return float((x[:-lag] * x[lag:]).sum() / d) if d > 0 and 0 < lag < len(x) else 0.0


def series_insights(y: Any, dates: Optional[Any] = None) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    out: Dict[str, Any] = {"n": int(n), "period": 0, "seasonal_strength": 0.0, "trend_strength": 0.0, "log_scale": False, "random_walk": False, "step_days": None}
    if n < 12 or not np.isfinite(y).all():
        return out
    step = None
    if dates is not None:
        d = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
        if len(d) > 2:
            step = float(np.median(np.diff(d.values).astype("timedelta64[s]").astype(float)) / 86400.0)
    out["step_days"] = step
    cands = _candidate_periods(step, n) if step else []
    if not cands:                                                    # unknown / irregular spacing: search the autocorrelation of the detrended series
        _, r0 = _detrend(y)
        lim = min(n // 3, 60)
        sc = [(_acf(r0, p), p) for p in range(2, lim + 1)]
        if sc:
            best = max(sc)
            cands = [best[1]] if best[0] > 0.3 else []
    trend, r = _detrend(y)
    out["trend_strength"] = round(float(np.clip(1 - np.var(r) / np.var(y), 0, 1)) if np.var(y) > 0 else 0.0, 3)
    best_p, best_s = 0, 0.0
    for p in cands:
        s = _seasonal_strength(r, p)
        if s > best_s:
            best_p, best_s = p, s
    if (y > 0).all():                                               # does the season scale with the level? compare the strength on a log scale
        _, rl = _detrend(np.log(y))
        sl = max((_seasonal_strength(rl, p) for p in cands), default=0.0)
        if sl > best_s + 0.02:
            out["log_scale"], best_s = True, sl
    out["period"], out["seasonal_strength"] = (best_p if best_s >= 0.25 else 0), round(best_s, 3)
    dy = np.diff(y)
    a1, ad = _acf(y, 1), (_acf(dy, 1) if len(dy) > 3 else 0.0)
    out["acf1_level"], out["acf1_diff"] = round(a1, 3), round(ad, 3)
    out["random_walk"] = bool(best_s < 0.15 and a1 > 0.9 and abs(ad) < 0.25 and out["trend_strength"] < 0.9)
    return out


def ts_lags(ins: Dict[str, Any], base_lags: List[int], base_rolls: List[int], n: int) -> Tuple[List[int], List[int]]:
    """Seasonal lags on top of the frequency defaults, never so long that they eat more than ~20% of the rows."""
    lags, rolls = list(base_lags), list(base_rolls)
    p = int(ins.get("period") or 0)
    if p:
        lags += [p] + ([p + 1] if p <= 24 else [])
        if p <= n // 6:
            rolls += [p]
    lim = max(3, n // 5)
    return sorted({l for l in lags if 1 <= l <= lim}), sorted({r for r in rolls if 2 <= r <= lim})


# ------------------------------------------------------------------ tabular
def _probe(task: str, X: pd.DataFrame, y: np.ndarray, k: int, seed: int) -> Dict[str, Any]:
    """Linear vs small boosted trees, 3-fold CV on <= 1500 rows. Cheap (seconds), and it answers the question that decides the shortlist."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from . import modeling
    rs = np.random.RandomState(seed)
    if len(y) > 1500:
        idx = np.sort(rs.choice(len(y), 1500, replace=False))
        X, y = X.iloc[idx], y[idx]
    clf = task == "classification"
    if clf:
        if np.bincount(y).min() < 3:
            return {}
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)
        scoring = "roc_auc" if k == 2 else "f1_macro"
        lin, gb = LogisticRegression(max_iter=1000), HistGradientBoostingClassifier(max_iter=60, random_state=seed)
    else:
        cv, scoring = KFold(3, shuffle=True, random_state=seed), "r2"
        lin, gb = Ridge(alpha=1.0), HistGradientBoostingRegressor(max_iter=60, random_state=seed)
    out: Dict[str, Any] = {}
    for name, m in (("linear", lin), ("boosted", gb)):
        try:
            pipe = Pipeline([("prep", modeling.build_preprocessor(X)), ("model", m)])
            out[name] = float(np.mean(cross_val_score(pipe, X, y, cv=cv, scoring=scoring, error_score="raise")))
        except Exception:
            out[name] = None
    if out.get("linear") is not None and out.get("boosted") is not None:
        out["gain"] = round(out["boosted"] - out["linear"], 4)
        out["ceiling"] = round(max(out["linear"], out["boosted"]), 4)
        out["metric"] = scoring
    return out


def tabular_insights(task: str, X: pd.DataFrame, y: np.ndarray, k: int = 0, seed: int = 42, probe: bool = True) -> Dict[str, Any]:
    n, p = X.shape
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    hi_card = [c for c in cat if X[c].nunique() > 30]
    ins: Dict[str, Any] = {"n": int(n), "p": int(p), "numeric_share": round(len(num) / max(p, 1), 2), "categorical": len(cat), "high_cardinality": len(hi_card),
                           "missing_share": round(float(X.isna().to_numpy().mean()), 3), "p_over_n": round(p / max(n, 1), 3)}
    if task == "classification":
        cnt = np.bincount(y)
        ins["minority_share"] = round(float(cnt[cnt > 0].min() / cnt.sum()), 3)
    else:
        ins["target_skew"] = round(float(pd.Series(y).skew()), 2) if n > 2 else 0.0
    if num and len(num) > 1 and n >= 30:
        try:
            c = X[num].corr().abs().to_numpy()
            np.fill_diagonal(c, 0)
            ins["max_feature_correlation"] = round(float(np.nanmax(c)), 3)
        except Exception:
            pass
    if probe and n >= 60:
        ins["probe"] = _probe(task, X, y, k, seed)
    return ins


# ------------------------------------------------------------------ planning
def _entry(family: str, why: str) -> Dict[str, str]:
    return {"family": family, "why": why}


def plan_tabular(task: str, ins: Dict[str, Any], available: List[str]) -> Dict[str, Any]:
    n, pr = ins["n"], ins.get("probe") or {}
    linear = "LogisticRegression" if task == "classification" else "Ridge"
    keep: Dict[str, str] = {}
    drop: Dict[str, str] = {}
    gain, ceil = pr.get("gain"), pr.get("ceiling")
    if gain is None:
        nonlinear, why_sig = None, "no probe (too little data)"
    else:
        rel = gain / max(abs(ceil or 1e-9), 1e-9)
        nonlinear = gain > 0.02 and rel > 0.03
        why_sig = f"probe: linear {pr['linear']:.3f} vs boosted trees {pr['boosted']:.3f} ({pr['metric']}), non-linear gain {gain:+.3f}"
    keep[linear] = "fast, interpretable reference; " + ("it is competitive here (" + why_sig + ")" if nonlinear is False else "shows how much non-linearity adds")
    small = n < 200
    if small:
        if "RandomForest" in available:
            keep["RandomForest"] = f"only {n} rows: bagged trees are the robust non-linear choice at this size"
        for f in ("HistGradientBoosting", "MLP", "KNN"):
            if f in available:
                drop[f] = f"only {n} rows: too few to fit {'a boosted ensemble' if f.startswith('Hist') else 'a neural net' if f == 'MLP' else 'a stable neighbourhood'} reliably"
    else:
        if nonlinear is False:
            keep["HistGradientBoosting"] = "challenger: confirms there is no non-linear structure to exploit"
            for f in ("RandomForest", "KNN", "DecisionTree", "MLP", "ExtraTrees", "AdaBoost"):
                if f in available:
                    drop[f] = f"the data looks linear ({why_sig}); non-linear models would only add variance"
        else:
            keep["HistGradientBoosting"] = "non-linear structure found (" + why_sig + ")" if nonlinear else "strong general-purpose model for mixed tabular data"
            keep["RandomForest"] = "bagged trees: low-variance non-linear model, a good second opinion to boosting"
            if "MLP" in available and n >= 500 and ins["numeric_share"] >= 0.6 and nonlinear:
                keep["MLP"] = f"{n} rows, {int(ins['numeric_share'] * 100)}% numeric features and smooth non-linear signal: a small neural network can fit interactions trees approximate"
            if "KNN" in available:
                if ins["p"] <= 12 and ins["numeric_share"] >= 0.8 and n <= 20000:
                    keep["KNN"] = f"only {ins['p']} mostly numeric features: distances are meaningful"
                else:
                    drop["KNN"] = f"{ins['p']} features / {int((1 - ins['numeric_share']) * 100)}% non-numeric: distances are unreliable"
            drop["DecisionTree"] = "a single tree is dominated by forests and boosting whenever non-linearity is present"
    if ins.get("categorical", 0) / max(ins["p"], 1) > 0.5 and "KNN" in keep:
        drop["KNN"] = keep.pop("KNN") and "categorical-dominated data: distance-based models are a poor fit"
    if ins.get("missing_share", 0) > 0.05 and "HistGradientBoosting" in keep:
        keep["HistGradientBoosting"] += f"; handles the {ins['missing_share']:.0%} missing values natively"
    if ceil is not None and ceil < 0.15 and task == "regression":
        keep[linear] += f"; the achievable score is low ({ceil:.2f} R2), so strong regularisation matters more than model class"
    for f in available:                                   # families nobody ruled on (extra models / boosters when enabled): keep them out unless audited in
        if f not in keep and f not in drop:
            drop[f] = "optional model: only tried if the audit shows the shortlist is leaving accuracy behind"
    return {"shortlist": [_entry(f, w) for f, w in keep.items() if f in available], "excluded": [_entry(f, w) for f, w in drop.items() if f in available]}


def plan_series(ins: Dict[str, Any], available: List[str]) -> Dict[str, Any]:
    keep: Dict[str, str] = {}
    drop: Dict[str, str] = {}
    p, ss, ts = int(ins.get("period") or 0), ins.get("seasonal_strength", 0.0), ins.get("trend_strength", 0.0)
    keep["Naive (last value)"] = "baseline: every forecaster has to beat 'tomorrow = today'"
    if p:
        keep["Seasonal naive"] = f"baseline: 'same period last season' (season length {p}, seasonal strength {ss:.2f})"
        keep["TrendSeasonal"] = f"clear season ({p}) and trend ({ts:.2f}): a trend-plus-season regression extrapolates both, which trees cannot"
        keep["Ridge · seasonal Δ"] = "linear model of the change versus one season ago"
        keep["HistGradientBoosting · seasonal Δ"] = "boosted trees on the seasonal change: non-linear effects without having to extrapolate a level"
    elif ts >= 0.3:
        keep["TrendSeasonal"] = f"strong trend ({ts:.2f}), no reliable season: trend regression"
        keep["Ridge · seasonal Δ"] = "linear model of the step-to-step change (no season found)"
        keep["HistGradientBoosting · seasonal Δ"] = "boosted trees on the step-to-step change"
    elif ins.get("random_walk"):
        keep["Ridge · seasonal Δ"] = "random-walk-like series: only a heavily simple model of the change can add anything"
        drop["TrendSeasonal"] = "no trend or season to model"
        drop["HistGradientBoosting · seasonal Δ"] = "random-walk-like series: flexible models only fit noise"
    else:
        keep["Ridge"] = "weak trend / season: a plain regularised model on lags and date features"
        keep["HistGradientBoosting"] = "weak structure: boosted trees on lag and date features"
        drop["TrendSeasonal"] = "no clear trend or season"
    if ts >= 0.3 or p:
        if "Ridge" in available:                      # found by the exclusion audit: a linear autoregression on (seasonal) lags is a strong forecaster, never rule it out
            keep.setdefault("Ridge", "autoregressive linear model on the lag features: the classic strong forecaster, cheap and hard to beat")
        for f in ("HistGradientBoosting", "RandomForest", "KNN", "DecisionTree", "MLP"):
            if f in available:
                drop[f] = ("trees and neighbours cannot extrapolate a trending level; the seasonal-change variants cover this family" if f != "MLP"
                           else "too little data per parameter for a neural net on a single short series")
    for f in available:
        if f not in keep and f not in drop:
            drop[f] = "not suited to a single time series with this structure"
    return {"shortlist": [_entry(f, w) for f, w in keep.items() if f in available], "excluded": [_entry(f, w) for f, w in drop.items() if f in available]}


def build_plan(task: str, splits: Dict[str, Any], available: List[str], seed: int = 42, probe: bool = True) -> Dict[str, Any]:
    """-> {'task', 'insights', 'shortlist', 'excluded', 'summary'}. Always keeps at least two families."""
    X, y = splits["X_train"], np.asarray(splits["y_train"])
    k = len(splits.get("classes") or [])
    if task == "time_series":
        ins = dict((splits.get("ts_info") or {}).get("insights") or {})
        if not ins:
            ins = series_insights(y, None)
        plan = plan_series(ins, available)
    else:
        ins = tabular_insights(task, X, y, k, seed, probe)
        plan = plan_tabular(task, ins, available)
    if len(plan["shortlist"]) < 2:                                       # never run a one-model "search"
        for e in list(plan["excluded"]):
            plan["shortlist"].append(e)
            plan["excluded"].remove(e)
            if len(plan["shortlist"]) >= 2:
                break
    names = [e["family"] for e in plan["shortlist"]]
    plan.update(task=task, insights=ins, summary=f"{len(names)} of {len(available)} model families shortlisted from the data: " + ", ".join(names))
    return plan
