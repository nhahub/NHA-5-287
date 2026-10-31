"""Statistical tests: a CATALOGUE of tests that are valid for this data, executed with
assumption checks. Nothing here runs "everything".

Principles
  * Candidates are enumerated only where the test is applicable (data types, sample sizes,
    available reference data). The LLM (or a heuristic) then SELECTS from that menu by
    relevance to the business question; it cannot invent or force a test.
  * Shapiro-Wilk is never run standalone. It is used inside group comparisons only, for small
    groups (n < CLT_N), to choose between a parametric and a non-parametric test.
  * K-S drift runs only when there is a legitimate reference: a train/test split or a user
    supplied reference dataset. It is skipped (with a reason) for purely descriptive runs.
  * Correlation is association, never causation. Every result carries that caveat.
  * p-values are Benjamini-Hochberg adjusted across the tests actually run.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import config

CAUSATION_NOTE = "Association only; does not establish causation."


# helpers
def _label(x: float, cuts: Tuple[float, float, float], names=("negligible", "small", "medium", "large")) -> str:
    a = abs(x)
    return names[0] if a < cuts[0] else names[1] if a < cuts[1] else names[2] if a < cuts[2] else names[3]


def benjamini_hochberg(p: List[float]) -> List[float]:
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(n)
    out[order] = adj
    return out.tolist()


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _groups(df: pd.DataFrame, measure: str, group: str, max_levels: int = 10):
    sub = df[[measure, group]].dropna()
    sub[group] = sub[group].astype(str)
    counts = sub[group].value_counts()
    keep = [g for g in counts.index[:max_levels] if counts[g] >= config.MIN_GROUP_N]
    return {g: sub.loc[sub[group] == g, measure].astype(float).to_numpy() for g in keep}


# candidates
def enumerate_candidates(df: pd.DataFrame, profile: Dict[str, Any], measure: Optional[str], problem_type: str,
                         features: List[str], segments: List[str], have_holdout: bool, have_reference: bool,
                         max_candidates: int = 60) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Return (applicable candidate tests, tests deliberately not offered + reason)."""
    cols = profile["columns"]
    num = [c for c in features if c in cols and cols[c]["kind"] == "numeric" and cols[c]["unique_count"] > 2]
    cat = [c for c in features + segments if c in cols and cols[c]["kind"] in ("categorical", "boolean")
           and 2 <= cols[c]["unique_count"] <= 20]
    cat = list(dict.fromkeys(cat))
    skipped: List[Dict[str, str]] = [
        {"test": "Shapiro-Wilk (standalone)",
         "reason": "Not run on every feature. It is only used inside group comparisons with small groups "
                   f"(n < {config.CLT_N}) to choose a parametric vs non-parametric test."}]
    cands: List[Dict[str, Any]] = []

    def add(test, params, why, prior):
        cands.append({"id": f"T{len(cands) + 1:03d}", "test": test, "params": params, "why": why, "prior": round(prior, 3)})

    target_is_num = measure is not None and measure in cols and cols[measure]["kind"] == "numeric"
    target_is_cat = measure is not None and measure in cols and cols[measure]["kind"] in ("categorical", "boolean")

    if measure is not None and measure in cols:
        for f in num:
            if f == measure:
                continue
            if target_is_cat and cols[measure]["unique_count"] <= 10:
                x = pd.to_numeric(df[f], errors="coerce")
                code = df[measure].astype("category").cat.codes.where(df[measure].notna())
                prior = abs(x.corr(code, method="spearman")) if code.nunique() == 2 else 0.3
                add("group_comparison", {"measure": f, "group": measure},
                    f"Does '{f}' differ across '{measure}' groups?", float(np.nan_to_num(prior)))
            elif target_is_num:
                x = pd.to_numeric(df[f], errors="coerce")
                prior = abs(x.corr(pd.to_numeric(df[measure], errors="coerce"), method="spearman"))
                add("correlation", {"a": f, "b": measure}, f"How strongly is '{f}' associated with '{measure}'?",
                    float(np.nan_to_num(prior)))
        for c in cat:
            if c == measure:
                continue
            if target_is_cat and cols[measure]["unique_count"] <= 10:
                add("chi_square", {"a": c, "b": measure}, f"Is '{c}' associated with '{measure}'?", 0.35)
            elif target_is_num:
                add("group_comparison", {"measure": measure, "group": c},
                    f"Does '{measure}' differ across '{c}' groups?", 0.4)
    else:  # descriptive / clustering: compare numeric measures across business segments
        for seg in [s for s in segments if s in cols and 2 <= cols[s]["unique_count"] <= 20][:3]:
            for m in num[:4]:
                add("group_comparison", {"measure": m, "group": seg}, f"Does '{m}' differ across '{seg}'?", 0.3)

    # feature-feature relationships (top by |spearman|)
    if len(num) >= 2:
        sub = df[num[:25]].apply(pd.to_numeric, errors="coerce")
        cm = sub.corr(method="spearman").abs()
        pairs = [(cm.iloc[i, j], num[i], num[j]) for i in range(len(num[:25])) for j in range(i)
                 if measure not in (num[i], num[j])]
        for v, a, b in sorted(pairs, reverse=True)[:5]:
            if pd.notna(v) and v >= 0.3:
                add("correlation", {"a": a, "b": b}, f"Strong feature pair '{a}'/'{b}': redundancy / multicollinearity.", float(v) * 0.8)

    if problem_type in ("classification", "regression", "time_series") and len(num) >= 3:
        add("vif", {"columns": num[:25]}, "Multicollinearity among numeric predictors (matters for linear models).", 0.5)
    else:
        skipped.append({"test": "VIF", "reason": "Needs a supervised problem with >=3 numeric predictors."})

    if have_holdout or have_reference:
        add("ks_drift", {"scope": "reference" if have_reference else "train_vs_test", "columns": num[:30]},
            "Is the hold-out/reference data distributed like the training data?", 0.9)
    else:
        skipped.append({"test": "Kolmogorov-Smirnov drift",
                        "reason": "No train/test split and no reference dataset: drift is undefined here."})

    cands.sort(key=lambda c: -c["prior"])
    for i, c in enumerate(cands):
        c["id"] = f"T{i + 1:03d}"
    return cands[:max_candidates], skipped


# execution
def run_group_comparison(df, measure: str, group: str) -> Dict[str, Any]:
    g = _groups(df, measure, group)
    if len(g) < 2:
        return {"skipped": "Fewer than 2 groups with at least %d observations." % config.MIN_GROUP_N}
    arrs = list(g.values())
    assumptions, normal_ok = [], True
    for name, a in g.items():
        if len(a) >= config.CLT_N:
            sk = float(stats.skew(a))
            ok = abs(sk) < 3
            assumptions.append(f"{name}: n={len(a)} >= {config.CLT_N}, CLT applies (|skew|={abs(sk):.1f})")
        elif len(a) >= 8 and np.ptp(a) > 0:
            p = float(stats.shapiro(a[: config.SHAPIRO_MAX_N]).pvalue)
            ok = p > config.ALPHA
            assumptions.append(f"{name}: n={len(a)}, Shapiro-Wilk p={p:.3f} ({'normal' if ok else 'non-normal'})")
        else:
            ok = False
            assumptions.append(f"{name}: n={len(a)} too small / constant, assuming non-normal")
        normal_ok &= ok
    k, n_total = len(arrs), sum(len(a) for a in arrs)
    out: Dict[str, Any] = {"group_means": {n: round(float(np.mean(a)), 4) for n, a in g.items()},
                           "group_n": {n: int(len(a)) for n, a in g.items()}, "assumptions": assumptions}
    if k == 2:
        a, b = arrs
        if normal_ok:
            r = stats.ttest_ind(a, b, equal_var=False)
            sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)) or 1
            eff = float((a.mean() - b.mean()) / sp)
            out.update(method="Welch t-test", statistic=float(r.statistic), p_value=float(r.pvalue),
                       effect_size=eff, effect_name="Cohen's d", effect_label=_label(eff, (0.2, 0.5, 0.8)))
        else:
            r = stats.mannwhitneyu(a, b, alternative="two-sided")
            eff = float(1 - 2 * r.statistic / (len(a) * len(b)))
            out.update(method="Mann-Whitney U", statistic=float(r.statistic), p_value=float(r.pvalue),
                       effect_size=eff, effect_name="rank-biserial r", effect_label=_label(eff, (0.1, 0.3, 0.5)))
    else:
        lev = float(stats.levene(*arrs).pvalue)
        if normal_ok and lev > config.ALPHA:
            r = stats.f_oneway(*arrs)
            grand = np.concatenate(arrs).mean()
            ssb = sum(len(a) * (a.mean() - grand) ** 2 for a in arrs)
            sst = sum(((a - grand) ** 2).sum() for a in arrs) or 1
            eff = float(ssb / sst)
            out.update(method="One-way ANOVA", statistic=float(r.statistic), p_value=float(r.pvalue),
                       effect_size=eff, effect_name="eta squared", effect_label=_label(eff, (0.01, 0.06, 0.14)))
            assumptions.append(f"Levene p={lev:.3f} (equal variances)")
        else:
            r = stats.kruskal(*arrs)
            eff = float(max(0.0, (r.statistic - k + 1) / max(1, n_total - k)))
            out.update(method="Kruskal-Wallis", statistic=float(r.statistic), p_value=float(r.pvalue),
                       effect_size=eff, effect_name="epsilon squared", effect_label=_label(eff, (0.01, 0.08, 0.26)))
            assumptions.append("Non-parametric chosen (non-normal groups or unequal variances)")
    big = max(out["group_means"], key=out["group_means"].get)
    small = min(out["group_means"], key=out["group_means"].get)
    out["interpretation"] = (f"'{measure}' differs across '{group}' ({out['method']}, p={out['p_value']:.3g}, "
                             f"{out['effect_label']} effect {out['effect_name']}={out['effect_size']:.2f}); "
                             f"highest mean in '{big}' ({out['group_means'][big]}), lowest in '{small}' ({out['group_means'][small]}).")
    return out


def run_chi_square(df, a: str, b: str) -> Dict[str, Any]:
    sub = df[[a, b]].dropna().astype(str)
    for c in (a, b):
        top = sub[c].value_counts().index[:12]
        sub = sub[sub[c].isin(top)]
    ct = pd.crosstab(sub[a], sub[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"skipped": "Fewer than 2 levels after filtering."}
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    low = float((exp < 5).mean())
    assumptions = [f"{low:.0%} of cells have expected count < 5"]
    method = "Chi-square test of independence"
    if low > 0.2:
        if ct.shape == (2, 2):
            p = float(stats.fisher_exact(ct.to_numpy())[1])
            method = "Fisher exact test"
            assumptions.append("Sparse 2x2 table: Fisher exact used instead of chi-square")
        else:
            return {"skipped": "Sparse contingency table (>20% cells expected <5); chi-square invalid."}
    v = float(np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1))))
    return {"method": method, "statistic": float(chi2), "p_value": float(p), "effect_size": v,
            "effect_name": "Cramer's V", "effect_label": _label(v, (0.1, 0.3, 0.5)), "assumptions": assumptions,
            "interpretation": f"'{a}' and '{b}' are associated ({method}, p={p:.3g}, Cramer's V={v:.2f}, "
                              f"{_label(v, (0.1, 0.3, 0.5))}). {CAUSATION_NOTE}"}


def run_correlation(df, a: str, b: str) -> Dict[str, Any]:
    sub = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 10 or sub[a].nunique() < 3 or sub[b].nunique() < 3:
        return {"skipped": "Fewer than 10 complete pairs or (near-)constant column."}
    x, y = sub[a].to_numpy(float), sub[b].to_numpy(float)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    skew = max(abs(stats.skew(x)), abs(stats.skew(y)))
    use_spearman = skew > 1.0
    r, p = (sr, sp) if use_spearman else (pr, pp)
    notes = [f"Pearson r={pr:.3f}, Spearman rho={sr:.3f}",
             "Spearman used (skewed variable(s), |skew|=%.1f)" % skew if use_spearman else "Pearson used (no strong skew)"]
    if abs(pr - sr) > 0.15:
        notes.append("Pearson and Spearman disagree: relationship is non-linear or driven by outliers.")
    return {"method": "Spearman rank correlation" if use_spearman else "Pearson correlation",
            "statistic": float(r), "p_value": float(p), "effect_size": float(r), "effect_name": "correlation",
            "effect_label": _label(r, (0.1, 0.3, 0.5)), "assumptions": notes,
            "interpretation": f"'{a}' vs '{b}': {('positive' if r > 0 else 'negative')} {_label(r, (0.1, 0.3, 0.5))} "
                              f"association (r={r:.2f}, p={p:.3g}, n={len(sub)}). {CAUSATION_NOTE}"}


def run_vif(df, columns: List[str]) -> Dict[str, Any]:
    X = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    X = X.loc[:, X.std() > 0]
    if X.shape[1] < 3 or len(X) < 30:
        return {"skipped": "Too few complete rows / non-constant numeric columns."}
    Z = ((X - X.mean()) / X.std()).to_numpy()
    rows = []
    for j, c in enumerate(X.columns):
        others = np.delete(Z, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        resid = Z[:, j] - others @ beta
        r2 = 1 - resid.var() / Z[:, j].var()
        rows.append({"feature": c, "vif": round(float(1 / max(1e-9, 1 - r2)), 2)})
    rows.sort(key=lambda r: -r["vif"])
    bad = [r for r in rows if r["vif"] > 5]
    return {"method": "Variance inflation factor", "table": rows[:10], "p_value": None,
            "effect_size": rows[0]["vif"], "effect_name": "max VIF",
            "effect_label": "severe" if rows[0]["vif"] > 10 else "moderate" if rows[0]["vif"] > 5 else "low",
            "assumptions": ["VIF > 5 moderate, > 10 severe collinearity"],
            "significant": bool(bad),
            "interpretation": (f"{len(bad)} predictor(s) with VIF > 5 (highest: {rows[0]['feature']} = {rows[0]['vif']}). "
                               "Linear-model coefficients are unstable for these; tree models are unaffected.") if bad
            else "No problematic multicollinearity (all VIF <= 5)."}


def run_ks_drift(ref: pd.DataFrame, cur: pd.DataFrame, columns: List[str], scope: str) -> Dict[str, Any]:
    rows = []
    for c in columns:
        if c not in ref.columns or c not in cur.columns:
            continue
        a = pd.to_numeric(ref[c], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(cur[c], errors="coerce").dropna().to_numpy()
        if len(a) < 30 or len(b) < 30:
            continue
        r = stats.ks_2samp(a, b)
        rows.append({"feature": c, "ks_statistic": round(float(r.statistic), 4), "p_value": float(r.pvalue)})
    if not rows:
        return {"skipped": "No numeric column with >= 30 observations on both sides."}
    adj = benjamini_hochberg([r["p_value"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_adjusted"] = q
        r["drifted"] = bool(q < config.ALPHA and r["ks_statistic"] >= 0.1)
    drifted = [r for r in rows if r["drifted"]]
    rows.sort(key=lambda r: -r["ks_statistic"])
    where = "training vs hold-out" if scope == "train_vs_test" else "dataset vs supplied reference"
    return {"method": "Two-sample Kolmogorov-Smirnov (BH-adjusted)", "table": rows[:10],
            "p_value": min(r["p_adjusted"] for r in rows), "effect_size": rows[0]["ks_statistic"],
            "effect_name": "max KS distance", "effect_label": "drift" if drifted else "none",
            "significant": bool(drifted), "assumptions": [f"Comparison: {where}; continuous features only"],
            "interpretation": (f"Distribution shift detected ({where}) in: {', '.join(r['feature'] for r in drifted[:5])}. "
                               "Model performance on new data may degrade.") if drifted
            else f"No material distribution shift ({where})."}


def execute(cand: Dict[str, Any], df: pd.DataFrame, ref: Optional[pd.DataFrame] = None,
            cur: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    t, p = cand["test"], cand["params"]
    try:
        if t == "group_comparison":
            res = run_group_comparison(df, p["measure"], p["group"])
        elif t == "chi_square":
            res = run_chi_square(df, p["a"], p["b"])
        elif t == "correlation":
            res = run_correlation(df, p["a"], p["b"])
        elif t == "vif":
            res = run_vif(df, p["columns"])
        elif t == "ks_drift":
            res = run_ks_drift(ref, cur, p["columns"], p["scope"]) if ref is not None and cur is not None \
                else {"skipped": "No reference/hold-out data."}
        else:
            res = {"skipped": f"Unknown test {t}"}
    except Exception as e:  # a failing test must never crash the run
        res = {"skipped": f"Test failed: {type(e).__name__}: {e}"}
    return {"id": cand["id"], "test": t, "params": p, "why": cand.get("why", ""), **res}


def narrate(r: Dict[str, Any]) -> str:
    """Sentence generated FROM the significance flag, so text can never contradict the table.
    Uses the BH-adjusted p-value (the one the table shows)."""
    t, p = r.get("test"), r.get("params", {})
    padj, praw, eff, lab = r.get("p_adjusted"), r.get("p_value"), r.get("effect_size"), r.get("effect_label")
    if padj is None or praw is None:
        return r.get("interpretation", "")
    stat = f"{r.get('method')}, adjusted p={padj:.3g}" + (f", {lab} effect {r.get('effect_name')}={eff:.2f}" if eff is not None else "")
    subj = {"group_comparison": f"'{p.get('measure')}' across '{p.get('group')}'",
            "chi_square": f"'{p.get('a')}' and '{p.get('b')}'",
            "correlation": f"'{p.get('a')}' vs '{p.get('b')}'"}.get(t)
    if subj is None:
        return r.get("interpretation", "")
    if r.get("significant"):
        verb = {"group_comparison": "differs", "chi_square": "are associated", "correlation": "are associated"}[t]
        sent = f"{subj}: statistically significant ({stat}). {subj.split(':')[0]} {verb}."
        sent = f"{subj} {verb} ({stat})."
        if t == "group_comparison" and r.get("group_means"):
            gm = r["group_means"]
            big, small = max(gm, key=gm.get), min(gm, key=gm.get)
            sent += f" Highest mean in '{big}' ({gm[big]}), lowest in '{small}' ({gm[small]})."
        return sent + " " + CAUSATION_NOTE
    if lab == "negligible" and padj < config.ALPHA:
        return f"{subj}: detectable but negligible effect ({stat}); not treated as meaningful."
    if praw < config.ALPHA:
        return f"{subj}: nominal signal (raw p={praw:.3g}) did not survive multiple-comparison correction ({stat}); not significant."
    return f"{subj}: no significant relationship found ({stat})."


def finalize(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply multiple-comparison correction across executed tests that have a p-value."""
    idx = [i for i, r in enumerate(results) if r.get("p_value") is not None and "skipped" not in r
           and r["test"] != "ks_drift"]
    adj = benjamini_hochberg([results[i]["p_value"] for i in idx])
    for i, q in zip(idx, adj):
        results[i]["p_adjusted"] = q
        results[i]["significant"] = bool(q < config.ALPHA and results[i].get("effect_label") != "negligible")
        results[i]["interpretation_raw"] = results[i].get("interpretation", "")
        results[i]["interpretation"] = narrate(results[i])
    return results
