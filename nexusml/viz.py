"""Charts (matplotlib, headless). Every chart is saved as PNG through the artifact registry and
described by a dict {id, title, caption, key, section} that lives in graph state. The report and
the Streamlit app render from these keys; nothing here holds figures in memory."""
import io
from typing import Any, Dict, List, Optional

import warnings

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from . import config  # noqa: E402

C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*vert: bool.*")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "figure.autolayout": False})


import threading
PLOT_LOCK = threading.RLock()      # matplotlib's pyplot is not thread-safe: stages that draw while other branches run hold this


class Gallery:
    """Collects charts for one phase; `items` is what the node returns into state."""

    def __init__(self, reg, run_id: str, prefix: str):
        self.reg, self.run_id, self.prefix, self.items = reg, run_id, prefix, []

    def add(self, fig, name: str, title: str, caption: str, section: str) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=config.CHART_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        key = self.reg.save_bytes(buf.getvalue(), f"charts/{self.run_id}/{self.prefix}_{name}.png")
        self.items.append({"id": f"{self.prefix}_{name}", "title": title, "caption": caption, "key": key,
                           "spec_key": None, "section": section})

    def attach(self, name: str, spec: Dict[str, Any], title: str, caption: str, section: str) -> None:
        """Store an interactive Plotly spec; attaches to the static chart with the same name, else creates an interactive-only chart."""
        import json
        cid = f"{self.prefix}_{name}"
        skey = self.reg.save_bytes(json.dumps(spec).encode("utf-8"), f"charts/{self.run_id}/{cid}.plotly.json")
        for it in self.items:
            if it["id"] == cid:
                it["spec_key"] = skey
                return
        self.items.append({"id": cid, "title": title, "caption": caption, "key": None, "spec_key": skey, "section": section})


def _grid(n: int, ncols: int = 4, cell=(3.2, 2.6)):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes[:n]


def _heat(ax, mat: pd.DataFrame, title: str, vmin=-1, vmax=1, fmt="{:.2f}"):
    im = ax.imshow(mat.values, cmap="RdBu_r" if vmin < 0 else "Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([str(c)[:18] for c in mat.columns], rotation=60, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([str(c)[:18] for c in mat.index])
    ax.grid(False)
    if len(mat) <= 14:
        for i in range(len(mat)):
            for j in range(len(mat.columns)):
                v = mat.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                            color="white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.32 else "black")
    ax.set_title(title)
    return im


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    sub = pd.concat([a, b], axis=1).dropna().astype(str)
    if len(sub) < 20:
        return np.nan
    ct = pd.crosstab(sub.iloc[:, 0], sub.iloc[:, 1])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.to_numpy().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / d)) if d > 0 else np.nan


def eta_correlation(cat: pd.Series, num: pd.Series) -> float:
    sub = pd.concat([cat.astype(str), pd.to_numeric(num, errors="coerce")], axis=1).dropna()
    if len(sub) < 20 or sub.iloc[:, 0].nunique() < 2:
        return np.nan
    y = sub.iloc[:, 1]
    grand = y.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for _, g in y.groupby(sub.iloc[:, 0]))
    sst = ((y - grand) ** 2).sum()
    return float(np.sqrt(ssb / sst)) if sst else np.nan


# EDA
def eda_charts(g: Gallery, df: pd.DataFrame, profile: Dict[str, Any], features: List[str], target: Optional[str],
               problem_type: str, positive: Optional[str]) -> None:
    cols = profile["columns"]
    num = [c for c in features if cols[c]["kind"] == "numeric" and cols[c]["unique_count"] > 2][: config.MAX_HIST_COLS]
    cat = [c for c in features if cols[c]["kind"] in ("categorical", "boolean") and 2 <= cols[c]["unique_count"] <= 15]
    tgt_cat = target is not None and cols[target]["kind"] in ("categorical", "boolean") and cols[target]["unique_count"] <= 10
    tgt_num = target is not None and cols[target]["kind"] == "numeric"

    # 1 missingness
    miss = pd.Series({c: i["missing_pct"] for c, i in cols.items() if i["missing_pct"] > 0}).sort_values()
    if len(miss):
        fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(miss) + 1.4))
        ax.barh(miss.index.astype(str), miss.values, color=C[3])
        ax.set_xlabel("% missing")
        ax.set_title("Missing values by column")
        g.add(fig, "missing", "Missing values", "Share of missing cells per column. Columns above 60% are proposed for removal.", "data_quality")

    # 2 target distribution
    if target:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        if tgt_cat:
            vc = df[target].astype(str).value_counts()
            ax.bar(vc.index, vc.values, color=C[: len(vc)])
            for i, v in enumerate(vc.values):
                ax.text(i, v, f"{v / vc.sum():.0%}", ha="center", va="bottom")
            ax.set_ylabel("rows")
        elif tgt_num:
            x = df[target].dropna()
            ax.hist(x, bins=30, color=C[0], alpha=0.85)
            ax.axvline(x.mean(), color=C[3], ls="--", label=f"mean {x.mean():,.2f}")
            ax.axvline(x.median(), color=C[1], ls=":", label=f"median {x.median():,.2f}")
            ax.legend()
        ax.set_title(f"Target distribution: {target}")
        g.add(fig, "target", "Target distribution", "Class balance or distribution shape of the outcome being analysed.", "distributions")

    # 3 histograms + 4 box plots
    if num:
        fig, axes = _grid(len(num))
        for ax, c in zip(axes, num):
            x = pd.to_numeric(df[c], errors="coerce").dropna()
            ax.hist(x, bins=25, color=C[0], alpha=0.85)
            ax.axvline(x.mean(), color=C[3], lw=1)
            ax.set_title(f"{c[:22]}  (skew {x.skew():.1f})")
        fig.tight_layout()
        g.add(fig, "hist", "Numeric distributions", "Histograms with the mean (red). Skew above |1| suggests a log transform or rank-based statistics.", "distributions")

        fig, axes = _grid(len(num))
        for ax, c in zip(axes, num):
            x = pd.to_numeric(df[c], errors="coerce").dropna()
            ax.boxplot(x, orientation="horizontal", widths=0.6, patch_artist=True, boxprops=dict(facecolor="#BFD3E6"),
                       flierprops=dict(marker="o", markersize=3, markerfacecolor=C[3], alpha=0.5))
            ax.set_yticks([])
            ax.set_title(c[:26])
        fig.tight_layout()
        g.add(fig, "box", "Box plots (outliers)", "Boxes show the IQR; red dots are points beyond 1.5xIQR whiskers.", "outliers")

        # box plots of the numeric features split by outcome / by a business segment
        split_by, order = None, None
        if tgt_cat:
            split_by = target
        elif cat:
            split_by = cat[0]
        if split_by is not None:
            rank = []
            for c in num:
                rank.append((abs(pd.to_numeric(df[c], errors="coerce").corr(
                    df[split_by].astype("category").cat.codes.where(df[split_by].notna()), method="spearman")), c))
            top = [c for _, c in sorted([(np.nan_to_num(r), c) for r, c in rank], reverse=True)[:6]]
            levels = df[split_by].astype(str).value_counts().index[:6]
            fig, axes = _grid(len(top), ncols=3, cell=(3.6, 3.0))
            for ax, c in zip(axes, top):
                data = [pd.to_numeric(df.loc[df[split_by].astype(str) == lv, c], errors="coerce").dropna() for lv in levels]
                bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                                flierprops=dict(marker="o", markersize=2, alpha=0.4))
                for patch, col in zip(bp["boxes"], C):
                    patch.set_facecolor(col)
                    patch.set_alpha(0.6)
                ax.set_xticklabels([str(l)[:10] for l in levels])
                ax.set_title(f"{c[:24]} by {split_by[:14]}")
            fig.tight_layout()
            g.add(fig, "box_by_group", f"Box plots by {split_by}",
                  f"Numeric features most related to '{split_by}', compared across its groups.", "outliers")

    # 5 correlation heat maps (Pearson + Spearman)
    nm = [c for c in features if cols[c]["kind"] == "numeric" and cols[c]["unique_count"] > 2]
    sub = df[nm].apply(pd.to_numeric, errors="coerce")
    tname = None
    if target and tgt_num:
        sub[target], tname = pd.to_numeric(df[target], errors="coerce"), target
    elif target and tgt_cat and cols[target]["unique_count"] == 2:
        pos = positive or df[target].astype(str).value_counts().index[-1]
        tname = f"{target}={pos}"
        sub[tname] = (df[target].astype(str) == str(pos)).astype(float).where(df[target].notna())
    if sub.shape[1] >= 3:
        if tname and sub.shape[1] > config.MAX_CORR_COLS:
            keep = sub.corrwith(sub[tname], method="spearman").abs().sort_values(ascending=False).index[: config.MAX_CORR_COLS]
            sub = sub[keep]
        elif sub.shape[1] > config.MAX_CORR_COLS:
            sub = sub[sub.var().sort_values(ascending=False).index[: config.MAX_CORR_COLS]]
        pear, spear = sub.corr(method="pearson"), sub.corr(method="spearman")
        w = max(5, 0.5 * len(pear))
        fig, axes = plt.subplots(1, 2, figsize=(2 * w + 3.5, w + 1), gridspec_kw={"wspace": 0.85})
        _heat(axes[0], pear, "Pearson (linear)")
        im = _heat(axes[1], spear, "Spearman (monotonic, rank-based)")
        fig.colorbar(im, ax=axes, shrink=0.7)
        g.add(fig, "corr", "Correlation maps",
              "Pearson measures linear association; Spearman is rank-based and robust to outliers and skew. "
              "Where they disagree the relationship is non-linear or outlier-driven. Correlation is not causation.", "correlations")
        if tname:
            tc = sub.drop(columns=[tname]).corrwith(sub[tname], method="spearman").dropna().sort_values()
            if len(tc):
                fig, ax = plt.subplots(figsize=(6, 0.32 * len(tc) + 1.3))
                ax.barh(tc.index.astype(str), tc.values, color=[C[0] if v >= 0 else C[3] for v in tc.values])
                ax.axvline(0, color="black", lw=0.8)
                ax.set_title(f"Spearman correlation with {tname}")
                g.add(fig, "corr_target", "Correlation with the target", "Blue = moves with the target, red = moves against it.", "correlations")

    # 6 categorical association (Cramer's V)
    cc = cat[:12] + ([target] if tgt_cat and target not in cat else [])
    if len(cc) >= 3:
        mat = pd.DataFrame(index=cc, columns=cc, dtype=float)
        for i, a in enumerate(cc):
            for j, b in enumerate(cc):
                mat.loc[a, b] = 1.0 if a == b else (cramers_v(df[a], df[b]) if j < i else np.nan)
        mat = mat.where(~mat.isna(), mat.T)
        fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(cc) + 2), max(4.5, 0.5 * len(cc) + 1.5)))
        im = _heat(ax, mat.astype(float), "Categorical association (Cramer's V)", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, shrink=0.7)
        g.add(fig, "cramers", "Categorical association map", "Cramer's V (0 = independent, 1 = perfectly associated) between categorical columns.", "correlations")

    # 7 mixed association with target
    if target and (tgt_cat or tgt_num):
        assoc = {}
        for c in features:
            if cols[c]["kind"] == "numeric" and cols[c]["unique_count"] > 2:
                if tname:
                    assoc[c] = (abs(pd.to_numeric(df[c], errors="coerce").corr(sub[tname], method="spearman")), "numeric")
            elif c in cat:
                v = cramers_v(df[c], df[target]) if tgt_cat else eta_correlation(df[c], df[target])
                assoc[c] = (v, "categorical")
        a = pd.DataFrame(assoc, index=["v", "t"]).T.dropna()
        if len(a) >= 2:
            a["v"] = a["v"].astype(float)
            a = a.sort_values("v").tail(15)
            fig, ax = plt.subplots(figsize=(6.2, 0.33 * len(a) + 1.4))
            ax.barh(a.index.astype(str), a["v"], color=[C[0] if t == "numeric" else C[1] for t in a["t"]])
            ax.set_title(f"Association strength with {target}")
            ax.set_xlabel("|Spearman| (numeric, blue)   /   Cramer's V or correlation ratio (categorical, orange)")
            g.add(fig, "assoc", "Which features relate to the target?", "Mixed-type association ranking. Univariate and descriptive only: not causal, and not adjusted for other features.", "correlations")

    # 8 outcome by category
    if target and cat and (tgt_cat or tgt_num):
        use = cat[:6]
        fig, axes = _grid(len(use), ncols=3, cell=(3.8, 3.0))
        if tgt_cat:
            pos = positive or df[target].astype(str).value_counts().index[-1]
            y = (df[target].astype(str) == str(pos)).astype(float).where(df[target].notna())
            ylab = f"P({target}={pos})"
        else:
            y, ylab = pd.to_numeric(df[target], errors="coerce"), f"mean {target}"
        for ax, c in zip(axes, use):
            r = y.groupby(df[c].astype(str)).agg(["mean", "count"])
            r = r[r["count"] >= 5].sort_values("mean", ascending=False).head(8)
            ax.bar([str(i)[:10] for i in r.index], r["mean"], color=C[0])
            ax.axhline(y.mean(), color=C[3], ls="--", lw=1)
            ax.set_title(c[:24])
            ax.set_ylabel(ylab[:22])
            ax.tick_params(axis="x", rotation=40)
        fig.tight_layout()
        g.add(fig, "by_category", "Outcome by segment", "Outcome level in each category (dashed red = overall average).", "segments")

    # 9 scatter of the strongest numeric relationships
    if sub.shape[1] >= 2:
        if tname:
            top = sub.drop(columns=[tname]).corrwith(sub[tname], method="spearman").abs().sort_values(ascending=False).index[:6]
            pairs = [(c, tname) for c in top]
        else:
            cm = sub.corr(method="spearman").abs()
            cn = list(cm.columns)
            ranked = sorted(((cm.iloc[i, j], cn[i], cn[j]) for i in range(len(cn)) for j in range(i) if pd.notna(cm.iloc[i, j])),
                            reverse=True)
            pairs = [(a, b) for _, a, b in ranked[:6]]
        if pairs:
            fig, axes = _grid(len(pairs), ncols=3, cell=(3.8, 3.2))
            samp = sub.sample(min(len(sub), 2500), random_state=0)
            for ax, (a, b) in zip(axes, pairs):
                d = samp[[a, b]].dropna()
                bin_t = tname and sub[tname].nunique() == 2 and b == tname
                if bin_t:
                    for v, col in ((0.0, C[0]), (1.0, C[3])):
                        ax.scatter(d.loc[d[b] == v, a], d.loc[d[b] == v, b] + np.random.RandomState(0).uniform(-0.06, 0.06, (d[b] == v).sum()),
                                   s=5, alpha=0.3, color=col)
                else:
                    ax.scatter(d[a], d[b], s=6, alpha=0.35, color=C[0])
                    if len(d) > 3:
                        m, k = np.polyfit(d[a], d[b], 1)
                        xs = np.linspace(d[a].min(), d[a].max(), 20)
                        ax.plot(xs, m * xs + k, color=C[3], lw=1.5)
                ax.set_xlabel(a[:24])
                ax.set_ylabel(b[:24])
            fig.tight_layout()
            g.add(fig, "scatter", "Strongest relationships", "Scatter plots of the most related numeric pairs (red line = linear fit).", "correlations")

    # 12 outlier share
    out = {c: i for c, i in profile.get("quality", {}).get("outlier_columns_pct", {}).items()}
    if out:
        s = pd.Series(out).sort_values()
        fig, ax = plt.subplots(figsize=(6, 0.33 * len(s) + 1.3))
        ax.barh(s.index.astype(str), s.values, color=C[1])
        ax.set_xlabel("% of values beyond 1.5xIQR")
        ax.set_title("Outlier share by column")
        g.add(fig, "outlier_share", "Outlier share", "Columns with more than 1% IQR outliers.", "outliers")


# business: KPI, trend
def kpi_charts(g: Gallery, kpi_results: Dict[str, Any]) -> None:
    n_seg = n_time = 0
    for name, r in kpi_results.items():
        spec = r["spec"]
        for col, rows in r.get("segments", {}).items():
            if n_seg >= 4:
                break
            v = [x for x in rows if not (isinstance(x["value"], float) and np.isnan(x["value"]))]
            if len(v) < 2:
                continue
            fig, ax = plt.subplots(figsize=(6.3, 0.38 * len(v) + 1.5))
            vals = [x["value"] * (100 if spec["format"] == "percent" else 1) for x in v][::-1]
            labels = [f"{str(x['segment'])[:18]} (n={x['n']:,})" for x in v][::-1]
            ov = r["overall"] * (100 if spec["format"] == "percent" else 1)
            good = spec["higher_is_better"]
            colors = [(C[2] if (val >= ov) == good else C[3]) for val in vals]
            ax.barh(labels, vals, color=colors)
            ax.axvline(ov, color="black", ls="--", lw=1, label=f"overall {r['overall_fmt']}")
            ax.legend(loc="lower right")
            ax.set_title(f"{name} by {col}")
            g.add(fig, f"kpi_{n_seg}", f"{name} by {col}",
                  "Green = better than overall, red = worse (using the KPI's direction of goodness).", "kpis")
            n_seg += 1
        t = r.get("time")
        if t and n_time < 2 and len(t["values"]) >= 6:
            fig, ax = plt.subplots(figsize=(7.5, 3.2))
            x = pd.to_datetime(t["periods"])
            ax.plot(x, t["values"], color=C[0], lw=1.6)
            ax.set_title(f"{name} over time ({t['freq']})")
            g.add(fig, f"kpi_time_{n_time}", f"{name} over time", "Complete periods only (partial first/last periods are excluded).", "kpis")
            n_time += 1


def trend_charts(g: Gallery, series: pd.Series, res: Dict[str, Any], measure: str) -> None:
    if "skipped" in res or len(series) < 8:
        return
    y = series.to_numpy(float)
    idx = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(series.index, y, color=C[0], lw=1.3, label=measure)
    ax.plot(series.index, pd.Series(y).rolling(max(3, len(y) // 12), center=True, min_periods=2).mean(), color=C[1], lw=2, label="rolling mean")
    m, k = np.polyfit(idx, y, 1)
    ax.plot(series.index, m * idx + k, color="black", ls="--", lw=1, label=f"trend ({res['trend']})")
    an = res.get("anomalies", [])
    if an:
        ax.scatter(pd.to_datetime([a["period"] for a in an]), [a["value"] for a in an], s=60, color=C[3], zorder=5,
                   label=f"anomalies ({len(an)})", edgecolor="white")
    cp = res.get("change_point")
    if cp:
        ax.axvline(pd.Timestamp(cp["period"]), color=C[3], ls=":", lw=1.4, label=f"level shift {cp['period']}")
    ax.legend(ncol=3, fontsize=8)
    ax.set_title(f"{measure}: trend, level shift and anomalies")
    g.add(fig, "trend", f"Trend of {measure}", "Robust anomalies are flagged against a rolling median (|z| > 3.5 on MAD). A level shift is only drawn if the series jumps relative to its trend line.", "trends")
    s = res.get("seasonality")
    if s:
        prof = pd.Series(s["profile"])
        names = {"day of week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                 "month of year": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}[s["by"]]
        fig, ax = plt.subplots(figsize=(6.3, 3))
        ax.bar([names[int(i) - (1 if s["by"] == "month of year" else 0)] for i in prof.index], prof.values, color=C[2])
        ax.set_title(f"Seasonality by {s['by']} (strength {s['strength']:.2f})")
        g.add(fig, "season", "Seasonality", "Average level by calendar position. Strength > 0.15 is treated as real seasonality.", "trends")


# modeling
def model_charts(g: Gallery, state: Dict[str, Any], diag: Dict[str, Any], importances: List[Dict[str, Any]]) -> None:
    cands, base, pm = state["candidate_metrics"], state["baseline_metrics"], state["primary_metric"]
    names = [c["name"] for c in cands]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    w = 0.26
    ax.bar(x - w, [c["default_cv"] for c in cands], w, label="Default CV (train)", color=C[0])
    ax.bar(x, [c["select_score"] for c in cands], w, label="Selection score (CV)", color=C[2])
    ax.bar(x + w, [c.get(pm, np.nan) for c in cands], w, label="Held-out test", color=C[1])
    ax.axhline(base["cv_score"], color=C[3], ls="--", label=f"baseline CV ({base['cv_score']})")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(pm)
    ax.legend(fontsize=8)
    ax.set_title("Model comparison")
    g.add(fig, "models", "Model comparison", "Selection uses the green bar: default CV, or nested CV when tuning helped. The test set is shown for reporting only.", "modeling")

    tr = state.get("tuning_report") or {}
    tm = {k: v for k, v in (tr.get("models") or {}).items() if v.get("trials")}
    if tm:
        fig, axes = plt.subplots(1, len(tm), figsize=(5.2 * len(tm), 3.6), squeeze=False)
        for ax, (name, e) in zip(axes[0], tm.items()):
            sc = [t["mean_score"] for t in sorted(e["trials"], key=lambda t: t["rank"])]
            ax.scatter(range(1, len(sc) + 1), sc, color=C[0], s=28, label="search trials (ranked)")
            ax.axhline(e["default_cv"], color=C[3], ls="--", label=f"default CV {e['default_cv']}")
            if e["nested_mean"] is not None:
                ax.axhline(e["nested_mean"], color=C[2], ls="-", label=f"nested CV {e['nested_mean']}")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.set_xlabel("trial rank")
            ax.set_ylabel(f"inner CV {pm}")
            ax.set_title(f"{name} {'(tuned used)' if e['used_tuned'] else '(default kept)'}")
            ax.legend(fontsize=7)
        fig.tight_layout()
        g.add(fig, "tuning", "Hyper-parameter search results",
              "Each dot is one sampled configuration. The green line is the unbiased nested-CV estimate of the tuned model; "
              "tuning is only adopted if it beats the red default line out-of-fold.", "tuning")
        # parameter sensitivity for the best tuned model
        best = max(tm.items(), key=lambda kv: kv[1]["inner_best"])
        name, e = best
        params = [p for p in e["best_params"] if all(isinstance(t["params"].get(p), (int, float)) and t["params"].get(p) is not None for t in e["trials"])][:4]
        if params and len(e["trials"]) >= 5:
            fig, axes = _grid(len(params), ncols=len(params), cell=(3.4, 3))
            for ax, p in zip(axes, params):
                xs = [t["params"][p] for t in e["trials"]]
                ax.scatter(xs, [t["mean_score"] for t in e["trials"]], color=C[0], s=26, alpha=0.8)
                bp = e["best_params"][p]
                ax.scatter([bp], [e["inner_best"]], color=C[3], s=70, marker="*", zorder=5)
                if min(xs) > 0 and max(xs) / min(xs) > 50:
                    ax.set_xscale("log")
                ax.set_xlabel(p)
            axes[0].set_ylabel(f"CV {pm}")
            fig.suptitle(f"{name}: score vs hyper-parameter (star = best)", fontsize=10, fontweight="bold")
            fig.tight_layout()
            g.add(fig, "tuning_params", f"Hyper-parameter sensitivity ({name})", "Shows which parameters the score is sensitive to and where the optimum sits within the searched range.", "tuning")

    imp = list(reversed(importances[:12]))
    if imp:
        fig, ax = plt.subplots(figsize=(6.5, 0.36 * len(imp) + 1.4))
        ax.barh([i["feature"][:28] for i in imp], [i["importance"] for i in imp], xerr=[i["std"] for i in imp], color=C[0])
        ax.set_title(f"Permutation importance ({pm} drop, held-out, raw features)")
        g.add(fig, "importance", "Feature importance", "How much the score falls when a feature is shuffled. Predictive importance, not causal effect.", "explain")

    if "confusion_matrix" in diag:
        cm = np.array(diag["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        ax.set_xticks(range(len(diag["classes"])))
        ax.set_yticks(range(len(diag["classes"])))
        ax.set_xticklabels(diag["classes"])
        ax.set_yticklabels(diag["classes"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix (held-out)")
        g.add(fig, "confusion", "Confusion matrix", "Counts of correct and incorrect predictions on the held-out set.", "explain")
    if "roc" in diag:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        axes[0].plot(diag["roc"]["fpr"], diag["roc"]["tpr"], color=C[0], lw=2)
        axes[0].plot([0, 1], [0, 1], color="grey", ls="--")
        axes[0].set_title(f"ROC curve (AUC {diag['roc']['auc']:.3f})")
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate")
        axes[1].plot(diag["pr"]["recall"], diag["pr"]["precision"], color=C[1], lw=2)
        axes[1].axhline(diag["pr"]["base_rate"], color="grey", ls="--")
        axes[1].set_title(f"Precision-recall (AP {diag['pr']['ap']:.3f})")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        g.add(fig, "roc", "ROC and precision-recall", "Ranking quality. The dashed line is a random model; PR is more informative for rare events.", "explain")
    if "lift" in diag:
        lt = diag["lift"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        axes[0].bar([r["decile"] for r in lt], [r["event_rate"] * 100 for r in lt], color=C[3])
        axes[0].axhline(diag["base_rate"] * 100, color="black", ls="--", lw=1)
        axes[0].set_xlabel("risk decile (1 = highest predicted risk)")
        axes[0].set_ylabel("actual event rate %")
        axes[0].set_title("Event rate by predicted-risk decile")
        axes[1].plot([0] + [r["decile"] * 10 for r in lt], [0] + [r["cum_capture"] * 100 for r in lt], marker="o", color=C[0])
        axes[1].plot([0, 100], [0, 100], color="grey", ls="--")
        axes[1].set_xlabel("% of population contacted (highest risk first)")
        axes[1].set_ylabel("% of all events captured")
        axes[1].set_title("Cumulative gains")
        g.add(fig, "lift", "Lift and gains", "How well the model concentrates events at the top of its ranking: the basis for targeting decisions.", "explain")
    if "actual" in diag:
        a, p = np.array(diag["actual"]), np.array(diag["predicted"])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        axes[0].scatter(a, p, s=8, alpha=0.5, color=C[0])
        lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
        axes[0].plot([lo, hi], [lo, hi], color=C[3], ls="--")
        axes[0].set_xlabel("actual")
        axes[0].set_ylabel("predicted")
        axes[0].set_title("Actual vs predicted (held-out sample)")
        axes[1].hist(a - p, bins=30, color=C[1], alpha=0.85)
        axes[1].axvline(0, color="black")
        axes[1].set_title("Residuals")
        g.add(fig, "residuals", "Regression diagnostics", "Points should hug the diagonal; residuals should be centred on zero without a pattern.", "explain")


def cluster_charts(g: Gallery, cr: Dict[str, Any]) -> None:
    ks = pd.DataFrame(cr["k_scores"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(ks["k"], ks["silhouette"], marker="o", color=C[0])
    axes[0].axvline(cr["k"], color=C[3], ls="--")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("silhouette (higher = better separated)")
    axes[0].set_title("Choosing the number of segments (tuning)")
    axes[1].plot(ks["k"], ks["inertia"], marker="o", color=C[1])
    axes[1].set_xlabel("k")
    axes[1].set_title("Inertia (elbow)")
    g.add(fig, "k_search", "Segment count search", "The number of segments is tuned by silhouette score (red line = chosen k).", "tuning")
    pts = np.array(cr["pca_points"])
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for k in range(cr["k"]):
        m = np.array(cr["pca_labels"]) == k
        ax.scatter(pts[m, 0], pts[m, 1], s=8, alpha=0.5, color=C[k % len(C)], label=f"segment {k}")
    ax.legend(markerscale=2)
    ax.set_xlabel(f"PC1 ({cr['explained_variance'][0]:.0%})")
    ax.set_ylabel(f"PC2 ({cr['explained_variance'][1]:.0%})")
    ax.set_title("Segments in 2-D (PCA projection)")
    g.add(fig, "pca", "Customer / record segments", "2-D projection. Overlap in the plot does not necessarily mean overlap in the full feature space.", "segments")
    feats = list(dict.fromkeys(f["feature"] for p in cr["profiles"] for f in p["defining_features"]))[:10]
    if feats:
        rows = []
        for p in cr["profiles"]:
            d = {f["feature"]: f["z_vs_overall"] for f in p["defining_features"]}
            rows.append([d.get(f, 0.0) for f in feats])
        mat = pd.DataFrame(rows, index=[f"segment {p['cluster']} ({p['share']:.0%})" for p in cr["profiles"]], columns=feats)
        fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(feats) + 2), 0.6 * len(mat) + 2))
        im = _heat(ax, mat, "What defines each segment (z-score vs overall mean)", vmin=-2, vmax=2)
        fig.colorbar(im, ax=ax, shrink=0.7)
        g.add(fig, "profile", "Segment profiles", "Standardised averages: red = above the overall mean, blue = below (shown for each segment's most distinctive features).", "segments")


# thread safety (parallel graph branches)
import functools as _ft
import threading as _th

MPL_LOCK = _th.RLock()


def _locked(fn):
    @_ft.wraps(fn)
    def wrapper(*a, **k):
        with MPL_LOCK:
            return fn(*a, **k)
    return wrapper


for _n in ("eda_charts", "kpi_charts", "trend_charts", "model_charts", "cluster_charts"):
    globals()[_n] = _locked(globals()[_n])
