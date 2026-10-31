"""Builds the interactive (Plotly) twins of the static charts and attaches them to the Gallery."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config, plotly_viz as pv
from .viz import cramers_v, eta_correlation


def eda(g, df: pd.DataFrame, prof: Dict[str, Any], features: List[str], target: Optional[str], positive: Optional[str]) -> None:
    cols = prof["columns"]
    num = [c for c in features if cols[c]["kind"] == "numeric" and cols[c]["unique_count"] > 2][:25]
    cat = [c for c in features if cols[c]["kind"] in ("categorical", "boolean") and 2 <= cols[c]["unique_count"] <= 15]
    tgt_cat = target is not None and cols[target]["kind"] in ("categorical", "boolean") and cols[target]["unique_count"] <= 10
    tgt_num = target is not None and cols[target]["kind"] == "numeric"

    miss = pd.Series({c: i["missing_pct"] for c, i in cols.items() if i["missing_pct"] > 0}).sort_values()
    if len(miss):
        g.attach("missing", pv.hbar(miss.index.astype(str).tolist(), miss.tolist(), "Missing values by column", xtitle="% missing"),
                 "Missing values", "Hover for exact percentages.", "data_quality")
    if target and (tgt_cat or tgt_num):
        if tgt_cat:
            vc = df[target].astype(str).value_counts()
            g.attach("target", pv.bars(vc.index.tolist(), vc.tolist(), f"Target distribution: {target}", "rows"), "Target distribution", "Class balance of the outcome.", "distributions")
        else:
            g.attach("target", pv.hist(df, [target]), "Target distribution", "Shape of the outcome.", "distributions")
    split = target if tgt_cat else None
    if num:
        g.attach("hist", pv.hist(df, num, split), "Numeric distributions", "Use the dropdown to switch variable; click legend items to hide groups; drag to zoom.", "distributions")
        by = target if tgt_cat else (cat[0] if cat else None)
        g.attach("box", pv.box(df, num[:20], by), "Box plots", "Dropdown selects the variable; points beyond 1.5xIQR are outliers (hover for values).", "outliers")

    sub = df[[c for c in features if cols[c]["kind"] == "numeric" and cols[c]["unique_count"] > 2]].apply(pd.to_numeric, errors="coerce")
    tname = None
    if target and tgt_num:
        sub[target], tname = pd.to_numeric(df[target], errors="coerce"), target
    elif target and tgt_cat and cols[target]["unique_count"] == 2:
        pos = positive or df[target].astype(str).value_counts().index[-1]
        tname = f"{target}={pos}"
        sub[tname] = (df[target].astype(str) == str(pos)).astype(float).where(df[target].notna())
    if sub.shape[1] >= 3:
        if tname and sub.shape[1] > config.MAX_CORR_COLS:
            sub = sub[sub.corrwith(sub[tname], method="spearman").abs().sort_values(ascending=False).index[: config.MAX_CORR_COLS]]
        elif sub.shape[1] > config.MAX_CORR_COLS:
            sub = sub[sub.var().sort_values(ascending=False).index[: config.MAX_CORR_COLS]]
        g.attach("corr", pv.heat({"Pearson (linear)": sub.corr(method="pearson"), "Spearman (rank)": sub.corr(method="spearman")}, "Correlation"),
                 "Correlation maps", "Dropdown switches Pearson / Spearman. Correlation is not causation.", "correlations")
        if tname:
            tc = sub.drop(columns=[tname]).corrwith(sub[tname], method="spearman").dropna().sort_values()
            g.attach("corr_target", pv.hbar(tc.index.astype(str).tolist(), tc.tolist(), f"Spearman correlation with {tname}",
                                           [pv.PAL[0] if v >= 0 else pv.PAL[3] for v in tc]), "Correlation with the target", "Blue moves with the target, red against it.", "correlations")
    cc = cat[:12] + ([target] if tgt_cat and target not in cat else [])
    if len(cc) >= 3:
        mat = pd.DataFrame(index=cc, columns=cc, dtype=float)
        for i, a in enumerate(cc):
            for j, b in enumerate(cc):
                mat.loc[a, b] = 1.0 if a == b else (cramers_v(df[a], df[b]) if j < i else np.nan)
        mat = mat.where(~mat.isna(), mat.T).astype(float)
        g.attach("cramers", pv.heat({"Cramer's V": mat}, "Categorical association", 0, 1, "Blues"), "Categorical association map", "0 = independent, 1 = perfectly associated.", "correlations")
    if target and (tgt_cat or tgt_num):
        assoc = {}
        for c in features:
            if cols[c]["kind"] == "numeric" and cols[c]["unique_count"] > 2 and tname:
                assoc[c] = (abs(pd.to_numeric(df[c], errors="coerce").corr(sub[tname], method="spearman")), "numeric")
            elif c in cat:
                assoc[c] = ((cramers_v(df[c], df[target]) if tgt_cat else eta_correlation(df[c], df[target])), "categorical")
        a = pd.DataFrame(assoc, index=["v", "t"]).T.dropna()
        if len(a) >= 2:
            a["v"] = a["v"].astype(float)
            a = a.sort_values("v").tail(15)
            g.attach("assoc", pv.hbar(a.index.astype(str).tolist(), a["v"].tolist(), f"Association strength with {target}",
                                      [pv.PAL[0] if t == "numeric" else pv.PAL[1] for t in a["t"]], "|Spearman| (blue) / Cramer's V or eta (orange)"),
                     "Which features relate to the target?", "Univariate and descriptive only: not causal.", "correlations")
        if cat:
            if tgt_cat:
                pos = positive or df[target].astype(str).value_counts().index[-1]
                y, ylab = (df[target].astype(str) == str(pos)).astype(float).where(df[target].notna()), f"P({target}={pos})"
            else:
                y, ylab = pd.to_numeric(df[target], errors="coerce"), f"mean {target}"
            g.attach("by_category", pv.by_category(df, y, cat[:10], ylab), "Outcome by segment", "Dropdown picks the segment column; hover shows group size.", "segments")
    if sub.shape[1] >= 2:
        if tname:
            top = sub.drop(columns=[tname]).corrwith(sub[tname], method="spearman").abs().sort_values(ascending=False).index[:8]
            pairs, color = [(c, tname) for c in top], None
        else:
            cm = sub.corr(method="spearman").abs()
            cn = list(cm.columns)
            ranked = sorted(((cm.iloc[i, j], cn[i], cn[j]) for i in range(len(cn)) for j in range(i) if pd.notna(cm.iloc[i, j])), reverse=True)
            pairs, color = [(a, b) for _, a, b in ranked[:8]], None
        if pairs:
            samp = sub.sample(min(len(sub), 3000), random_state=0)
            col = df.loc[samp.index, target].astype(str) if (tgt_cat and target) else None
            if tname and tgt_cat:                                            # binary target is plotted as colour, not as a y axis
                num_top = [c for c, _ in pairs]
                pairs = [(a, b) for a, b in zip(num_top, num_top[1:])][:6] or [(num_top[0], num_top[0])]
                if pairs[0][0] == pairs[0][1]:
                    pairs = []
            if pairs:
                g.attach("scatter", pv.scatter(samp, pairs, col), "Strongest relationships", "Dropdown selects the pair; colour = outcome; zoom and hover for values.", "correlations")


def kpi(g, kpi_results: Dict[str, Any]) -> None:
    a = pv.kpi_segments(kpi_results)
    if a:
        g.attach("kpi_seg", a, "KPIs by segment", "Dropdown picks KPI and segment; green = better than overall.", "kpis")
        for it in g.items:
            if it["id"].startswith("eda_kpi_") and "seg" not in it["id"] and "time" not in it["id"]:
                it["dup"] = True
    b = pv.kpi_time(kpi_results)
    if b:
        g.attach("kpi_time", b, "KPIs over time", "Use the range slider below the chart to zoom.", "kpis")
        for it in g.items:
            if it["id"].startswith("eda_kpi_time_"):
                it["dup"] = True


def trend(g, series: pd.Series, res: Dict[str, Any], measure: str) -> None:
    if "skipped" in res or len(series) < 8:
        return
    y = series.to_numpy(float)
    idx = np.arange(len(y))
    m, k = np.polyfit(idx, y, 1)
    roll = pd.Series(y).rolling(max(3, len(y) // 12), center=True, min_periods=2).mean().tolist()
    g.attach("trend", pv.timeseries([str(i.date()) for i in series.index], y.tolist(), roll, (m * idx + k).tolist(), res.get("anomalies", []),
                                    res.get("change_point"), measure, res["trend"]), f"Trend of {measure}",
             "Zoom with the range slider / buttons; hover an anomaly for its expected value.", "trends")
    s = res.get("seasonality")
    if s:
        names = {"day of week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                 "month of year": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}[s["by"]]
        off = 1 if s["by"] == "month of year" else 0
        prof = sorted(s["profile"].items(), key=lambda kv: int(kv[0]))
        g.attach("season", pv.bars([names[int(k) - off] for k, _ in prof], [v for _, v in prof], f"Seasonality by {s['by']} (strength {s['strength']:.2f})"),
                 "Seasonality", "Average level by calendar position.", "trends")


def model(g, state: Dict[str, Any], diag: Dict[str, Any], importances: List[Dict[str, Any]]) -> None:
    pm = state["primary_metric"]
    g.attach("models", pv.model_compare(state["candidate_metrics"], state["baseline_metrics"], pm), "Model comparison",
             "Selection uses the green bars (CV). Hover for the score source.", "modeling")
    tm = (state.get("tuning_report") or {}).get("models") or {}
    a = pv.trials_scatter(tm, pm)
    if a:
        g.attach("tuning", a, "Hyper-parameter search", "Each dot is a sampled configuration; hover for its parameters.", "tuning")
    b = pv.parallel(tm)
    if b:
        g.attach("parallel", b, "Parallel coordinates (MLflow style)", "Every line is one trial. Drag along an axis to filter and see which parameter ranges give top scores.", "tuning")
    imp = list(reversed(importances[:15]))
    if imp:
        g.attach("importance", pv.hbar([i["feature"] for i in imp], [i["importance"] for i in imp], f"Permutation importance ({pm} drop)",
                                       err=[i["std"] for i in imp]), "Feature importance", "Predictive importance on held-out data, not causal effect.", "explain")
    if "confusion_matrix" in diag:
        g.attach("confusion", pv.confusion(diag["confusion_matrix"], diag["classes"]), "Confusion matrix", "Counts on the held-out set.", "explain")
    if "roc" in diag:
        g.attach("roc", pv.roc(diag["roc"]), "ROC curve", "Ranking quality.", "explain")
        g.attach("pr", pv.pr(diag["pr"]), "Precision-recall curve", "More informative than ROC for rare events.", "explain")
    if "lift" in diag:
        g.attach("lift", pv.lift(diag["lift"], diag["base_rate"]), "Lift and gains", "How well the model concentrates events at the top of its ranking.", "explain")
    if "actual" in diag:
        g.attach("residuals", pv.residuals(diag["actual"], diag["predicted"]), "Actual vs predicted", "Points should hug the diagonal.", "explain")
        g.attach("residual_hist", pv.residual_hist(diag["actual"], diag["predicted"]), "Residual distribution", "Should be centred on zero.", "explain")
    if diag.get("calibration"):
        g.attach("calibration", pv.calibration(diag["calibration"]), "Calibration", "Closer to the diagonal = probabilities can be trusted as risks.", "explain")
    if diag.get("threshold_table"):
        g.attach("threshold", pv.threshold(diag["threshold_table"]), "Decision threshold", "Trade-off between precision and recall.", "explain")
    if diag.get("learning"):
        g.attach("learning", pv.learning(diag["learning"], pm), "Learning curve", "A widening gap or a still-rising validation line means more data should help.", "explain")
    c = pv.pdp(diag.get("pdp") or {})
    if c:
        g.attach("pdp", c, "Partial dependence", "Average predicted outcome as one feature varies (others held at observed values).", "explain")


def cluster(g, cr: Dict[str, Any]) -> None:
    g.attach("k_search", pv.cluster_k(cr), "Segment count search", "Red line = chosen k.", "tuning")
    g.attach("pca", pv.cluster_scatter(cr), "Segments in 2-D", "Click legend entries to isolate a segment.", "segments")
    p = pv.cluster_profile(cr)
    if p:
        g.attach("profile", p, "Segment profiles", "Standardised averages for each segment's most distinctive features.", "segments")
