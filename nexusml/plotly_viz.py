"""Interactive charts as plain Plotly JSON specs ({"data": [...], "layout": {...}}).

Why dicts: they need no plotly import to build or test, serialise to JSON for storage, render in
Streamlit with st.plotly_chart(spec) and in the HTML report with plotly.js. Hover, zoom, pan,
dropdown selectors, range sliders and legend toggling come from Plotly itself (Tableau / Power BI
style exploration). Each builder returns one spec; `Gallery.attach` stores it next to the PNG."""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import palette
from .util import jsonable

PAL = palette.chart_colors()


def _layout(title: str, **kw) -> Dict[str, Any]:
    t = palette.tokens()
    lay = {"title": {"text": title, "x": 0.01, "font": {"size": 15, "color": t["text"]}}, "margin": {"l": 60, "r": 20, "t": 60, "b": 55},
           "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "hovermode": "closest",
           "legend": {"orientation": "h", "y": -0.22}, "font": {"size": 12, "color": t["muted"], "family": "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
           "colorway": PAL,
           "xaxis": {"gridcolor": t["grid"], "zeroline": False, "linecolor": t["border"]},
           "yaxis": {"gridcolor": t["grid"], "zeroline": False, "linecolor": t["border"]}}
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(lay.get(k), dict):
            lay[k] = {**lay[k], **v}
        else:
            lay[k] = v
    return lay


def fig(data: List[Dict[str, Any]], layout: Dict[str, Any]) -> Dict[str, Any]:
    return jsonable({"data": data, "layout": layout})


def _dropdown(labels: Sequence[str], sizes: Sequence[int], titles: Optional[Sequence[str]] = None, y: float = 1.14) -> Dict[str, Any]:
    """One dropdown that shows the traces of ONE option at a time (traces are grouped by option, `sizes` each)."""
    total = sum(sizes)
    buttons, start = [], 0
    for i, (lab, n) in enumerate(zip(labels, sizes)):
        vis = [False] * total
        for j in range(start, start + n):
            vis[j] = True
        args = [{"visible": vis}] + ([{"title.text": titles[i]}] if titles else [])
        buttons.append({"label": str(lab)[:30], "method": "update", "args": args})
        start += n
    return {"updatemenus": [{"buttons": buttons, "direction": "down", "x": 0.0, "y": y, "xanchor": "left", "yanchor": "top",
                             "showactive": True}]}


def _vis_first(data: List[Dict[str, Any]], sizes: Sequence[int]) -> None:
    n0 = sizes[0] if sizes else 0
    for i, t in enumerate(data):
        t["visible"] = i < n0


# distributions
def hist(df: pd.DataFrame, cols: List[str], by: Optional[str] = None) -> Dict[str, Any]:
    data, sizes = [], []
    levels = list(df[by].astype(str).value_counts().index[:6]) if by else []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        if by:
            for k, lv in enumerate(levels):
                data.append({"type": "histogram", "x": x[df[by].astype(str) == lv].dropna().tolist(), "name": f"{by}={lv}", "opacity": 0.65,
                             "marker": {"color": PAL[k % 8]}, "nbinsx": 30, "hovertemplate": f"{c}: %{{x}}<br>count: %{{y}}<extra>{by}={lv}</extra>"})
            sizes.append(len(levels))
        else:
            data.append({"type": "histogram", "x": x.dropna().tolist(), "name": c, "marker": {"color": PAL[0]}, "nbinsx": 30,
                         "hovertemplate": f"{c}: %{{x}}<br>count: %{{y}}<extra></extra>"})
            sizes.append(1)
    _vis_first(data, sizes)
    return fig(data, _layout(f"Distribution of {cols[0]}", barmode="overlay", yaxis={"title": "count"}, xaxis={"title": cols[0]},
                             **(_dropdown(cols, sizes, [f"Distribution of {c}" for c in cols]) if len(cols) > 1 else {})))


def box(df: pd.DataFrame, cols: List[str], by: Optional[str] = None) -> Dict[str, Any]:
    data, sizes = [], []
    levels = list(df[by].astype(str).value_counts().index[:8]) if by else []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        if by:
            for k, lv in enumerate(levels):
                data.append({"type": "box", "y": x[df[by].astype(str) == lv].dropna().tolist(), "name": str(lv), "boxpoints": "outliers",
                             "marker": {"color": PAL[k % 8], "size": 3}})
            sizes.append(len(levels))
        else:
            data.append({"type": "box", "y": x.dropna().tolist(), "name": c, "boxpoints": "outliers", "marker": {"color": PAL[3], "size": 3},
                         "line": {"color": PAL[0]}})
            sizes.append(1)
    _vis_first(data, sizes)
    ttl = lambda c: f"{c}" + (f" by {by}" if by else "") + " (box plot)"
    return fig(data, _layout(ttl(cols[0]), showlegend=False, yaxis={"title": cols[0]},
                             **(_dropdown(cols, sizes, [ttl(c) for c in cols]) if len(cols) > 1 else {})))


def heat(mats: Dict[str, pd.DataFrame], title: str, zmin: float = -1, zmax: float = 1, scale: str = "RdBu") -> Dict[str, Any]:
    names = list(mats)
    data = []
    for i, n in enumerate(names):
        m = mats[n]
        data.append({"type": "heatmap", "z": m.values.tolist(), "x": [str(c) for c in m.columns], "y": [str(c) for c in m.index],
                     "zmin": zmin, "zmax": zmax, "colorscale": scale, "reversescale": scale == "RdBu", "visible": i == 0,
                     "texttemplate": "%{z:.2f}" if len(m) <= 14 else None, "hovertemplate": "%{y} vs %{x}: %{z:.3f}<extra></extra>",
                     "colorbar": {"thickness": 12}})
    n_cols = max(len(m) for m in mats.values())
    return fig(data, _layout(f"{title}: {names[0]}", height=max(380, 28 * n_cols + 160), margin={"l": 130, "b": 130},
                             xaxis={"tickangle": -45, "showgrid": False}, yaxis={"autorange": "reversed", "showgrid": False},
                             **(_dropdown(names, [1] * len(names), [f"{title}: {n}" for n in names]) if len(names) > 1 else {})))


def hbar(labels: List[str], values: List[float], title: str, colors: Optional[List[str]] = None, xtitle: str = "", err: Optional[List[float]] = None) -> Dict[str, Any]:
    t = {"type": "bar", "orientation": "h", "y": labels, "x": values, "marker": {"color": colors or PAL[0]},
         "hovertemplate": "%{y}: %{x:.4g}<extra></extra>"}
    if err:
        t["error_x"] = {"type": "data", "array": err, "visible": True}
    return fig([t], _layout(title, height=max(300, 26 * len(labels) + 120), margin={"l": 170}, xaxis={"title": xtitle}, yaxis={"autorange": "reversed"}))


def by_category(df: pd.DataFrame, y: pd.Series, cats: List[str], ylab: str) -> Dict[str, Any]:
    data, sizes = [], []
    overall = float(y.mean())
    for c in cats:
        r = y.groupby(df[c].astype(str)).agg(["mean", "count"])
        r = r[r["count"] >= 5].sort_values("mean", ascending=False).head(15)
        data.append({"type": "bar", "x": r.index.tolist(), "y": r["mean"].tolist(), "customdata": r["count"].tolist(), "marker": {"color": PAL[0]},
                     "hovertemplate": f"%{{x}}<br>{ylab}: %{{y:.3f}}<br>n=%{{customdata:,}}<extra></extra>"})
        sizes.append(1)
    _vis_first(data, sizes)
    return fig(data, _layout(f"{ylab} by {cats[0]}", yaxis={"title": ylab},
                             shapes=[{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": overall, "y1": overall, "line": {"dash": "dash", "color": PAL[3]}}],
                             annotations=[{"xref": "paper", "x": 1, "y": overall, "text": f"overall {overall:.3g}", "showarrow": False, "yanchor": "bottom", "xanchor": "right"}],
                             **(_dropdown(cats, sizes, [f"{ylab} by {c}" for c in cats]) if len(cats) > 1 else {})))


def scatter(df: pd.DataFrame, pairs: List[tuple], color: Optional[pd.Series] = None, binary: bool = False) -> Dict[str, Any]:
    data, sizes = [], []
    for a, b in pairs:
        d = df[[a, b]].apply(pd.to_numeric, errors="coerce")
        keep = d.dropna().index
        if color is not None and color.nunique() <= 6:
            for k, lv in enumerate(color.dropna().unique()[:6]):
                m = keep[(color.loc[keep] == lv).to_numpy()]
                data.append({"type": "scattergl", "mode": "markers", "x": d.loc[m, a].tolist(), "y": d.loc[m, b].tolist(), "name": str(lv),
                             "marker": {"size": 5, "opacity": 0.45, "color": PAL[k % 8]}})
            sizes.append(min(6, color.nunique()))
        else:
            data.append({"type": "scattergl", "mode": "markers", "x": d.loc[keep, a].tolist(), "y": d.loc[keep, b].tolist(), "name": "records",
                         "marker": {"size": 5, "opacity": 0.45, "color": PAL[0]}})
            n = 1
            if len(keep) > 3:
                m_, k_ = np.polyfit(d.loc[keep, a], d.loc[keep, b], 1)
                xs = [float(d.loc[keep, a].min()), float(d.loc[keep, a].max())]
                data.append({"type": "scatter", "mode": "lines", "x": xs, "y": [m_ * xs[0] + k_, m_ * xs[1] + k_], "name": "linear fit", "line": {"color": PAL[3]}})
                n = 2
            sizes.append(n)
    _vis_first(data, sizes)
    labels = [f"{a} vs {b}" for a, b in pairs]
    return fig(data, _layout(labels[0], xaxis={"title": pairs[0][0]}, yaxis={"title": pairs[0][1]},
                             **(_dropdown(labels, sizes, labels) if len(pairs) > 1 else {})))


def bars(x: List[str], y: List[float], title: str, ytitle: str = "", color: Any = None, ref: Optional[float] = None, ref_label: str = "") -> Dict[str, Any]:
    lay = _layout(title, yaxis={"title": ytitle})
    if ref is not None:
        lay["shapes"] = [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": ref, "y1": ref, "line": {"dash": "dash", "color": "black"}}]
        lay["annotations"] = [{"xref": "paper", "x": 1, "y": ref, "text": ref_label, "showarrow": False, "yanchor": "bottom", "xanchor": "right"}]
    return fig([{"type": "bar", "x": x, "y": y, "marker": {"color": color or PAL[0]}, "hovertemplate": "%{x}: %{y:.4g}<extra></extra>"}], lay)


# KPI / trend
def kpi_segments(kpis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data, labels, sizes = [], [], []
    for name, r in kpis.items():
        sp = r["spec"]
        for col, rows in r.get("segments", {}).items():
            rows = [x for x in rows if x["value"] is not None]
            if len(rows) < 2:
                continue
            k = 100 if sp["format"] == "percent" else 1
            ov = r["overall"] * k
            good = sp["higher_is_better"]
            data.append({"type": "bar", "orientation": "h", "y": [str(x["segment"]) for x in rows], "x": [x["value"] * k for x in rows],
                         "customdata": [[x["n"], x["vs_overall_pct"]] for x in rows],
                         "marker": {"color": [(PAL[2] if ((x["value"] * k) >= ov) == good else PAL[3]) for x in rows]},
                         "hovertemplate": "%{y}<br>value: %{x:.4g}<br>n=%{customdata[0]:,}<br>vs overall: %{customdata[1]}%<extra></extra>"})
            labels.append(f"{name} by {col}")
            sizes.append(1)
    if not data:
        return None
    _vis_first(data, sizes)
    return fig(data, _layout(labels[0], height=420, margin={"l": 150}, yaxis={"autorange": "reversed"}, **(_dropdown(labels, sizes, labels) if len(labels) > 1 else {})))


def kpi_time(kpis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = []
    for i, (name, r) in enumerate(kpis.items()):
        t = r.get("time")
        if t and len(t["values"]) >= 4:
            data.append({"type": "scatter", "mode": "lines+markers", "x": t["periods"], "y": t["values"], "name": name, "line": {"color": PAL[i % 8]},
                         "marker": {"size": 4}})
    if not data:
        return None
    return fig(data, _layout("KPIs over time (complete periods)", xaxis={"rangeslider": {"visible": True}}))


def timeseries(index: List[str], y: List[float], roll: List[float], trend: List[float], anomalies: List[Dict[str, Any]],
               change_point: Optional[Dict[str, Any]], measure: str, trend_label: str) -> Dict[str, Any]:
    data = [{"type": "scatter", "mode": "lines", "x": index, "y": y, "name": measure, "line": {"color": PAL[0], "width": 1.6}},
            {"type": "scatter", "mode": "lines", "x": index, "y": roll, "name": "rolling mean", "line": {"color": PAL[1], "width": 2.5}},
            {"type": "scatter", "mode": "lines", "x": index, "y": trend, "name": f"trend ({trend_label})", "line": {"color": "black", "dash": "dash", "width": 1}}]
    if anomalies:
        data.append({"type": "scatter", "mode": "markers", "x": [a["period"] for a in anomalies], "y": [a["value"] for a in anomalies], "name": "anomalies",
                     "marker": {"color": PAL[3], "size": 11, "line": {"color": "white", "width": 1.5}},
                     "customdata": [[a["expected"], a["robust_z"]] for a in anomalies],
                     "hovertemplate": "%{x}<br>value %{y:,.2f}<br>expected %{customdata[0]:,.2f}<br>robust z %{customdata[1]}<extra>anomaly</extra>"})
    lay = _layout(f"{measure}: trend, level shift and anomalies", yaxis={"title": measure},
                  xaxis={"rangeslider": {"visible": True}, "rangeselector": {"buttons": [
                      {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"}, {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                      {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"}, {"step": "all"}]}})
    if change_point:
        lay["shapes"] = [{"type": "line", "x0": change_point["period"], "x1": change_point["period"], "yref": "paper", "y0": 0, "y1": 1,
                          "line": {"color": PAL[3], "dash": "dot"}}]
        lay["annotations"] = [{"x": change_point["period"], "yref": "paper", "y": 1, "text": "level shift", "showarrow": False, "font": {"color": PAL[3]}}]
    return fig(data, lay)


# modeling
def model_compare(cands: List[Dict[str, Any]], base: Dict[str, Any], pm: str) -> Dict[str, Any]:
    n = [c["name"] for c in cands]
    data = [{"type": "bar", "name": "Default CV (train)", "x": n, "y": [c["default_cv"] for c in cands], "marker": {"color": PAL[0]}},
            {"type": "bar", "name": "Selection score (CV)", "x": n, "y": [c["select_score"] for c in cands], "marker": {"color": PAL[2]},
             "customdata": [c["select_source"] for c in cands], "hovertemplate": "%{x}<br>%{y:.4f}<br>%{customdata}<extra>selection</extra>"},
            {"type": "bar", "name": "Held-out test", "x": n, "y": [c.get(pm) for c in cands], "marker": {"color": PAL[1]}}]
    return fig(data, _layout("Model comparison", barmode="group", yaxis={"title": pm},
                             shapes=[{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": base["cv_score"], "y1": base["cv_score"], "line": {"dash": "dash", "color": PAL[3]}}],
                             annotations=[{"xref": "paper", "x": 1, "y": base["cv_score"], "text": f"baseline CV {base['cv_score']}", "showarrow": False, "yanchor": "bottom", "xanchor": "right"}]))


def trials_scatter(models: Dict[str, Any], pm: str) -> Optional[Dict[str, Any]]:
    data, sizes, names = [], [], []
    for m, e in models.items():
        if not e.get("trials"):
            continue
        ts = sorted(e["trials"], key=lambda t: t["rank"])
        data.append({"type": "scatter", "mode": "markers", "x": [t["rank"] for t in ts], "y": [t["mean_score"] for t in ts],
                     "error_y": {"type": "data", "array": [t["std_score"] for t in ts], "visible": True},
                     "customdata": [str(t["params"]) for t in ts], "marker": {"size": 10, "color": PAL[0]},
                     "hovertemplate": "rank %{x}<br>score %{y:.4f}<br>%{customdata}<extra></extra>", "name": "trials"})
        data.append({"type": "scatter", "mode": "lines", "x": [1, len(ts)], "y": [e["default_cv"]] * 2, "name": "default CV", "line": {"color": PAL[3], "dash": "dash"}})
        if e.get("nested_mean") is not None:
            data.append({"type": "scatter", "mode": "lines", "x": [1, len(ts)], "y": [e["nested_mean"]] * 2, "name": "nested CV (honest)", "line": {"color": PAL[2]}})
        sizes.append(len(data) - sum(sizes))
        names.append(m)
    if not data:
        return None
    _vis_first(data, sizes)
    return fig(data, _layout(f"Hyper-parameter search: {names[0]}", xaxis={"title": "trial rank"}, yaxis={"title": f"inner CV {pm}"},
                             **(_dropdown(names, sizes, [f"Hyper-parameter search: {n}" for n in names]) if len(names) > 1 else {})))


def parallel(models: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """MLflow-style parallel-coordinates plot: every trial is a line through its parameter values to its score."""
    data, sizes, names = [], [], []
    for m, e in models.items():
        trials = e.get("trials") or []
        if len(trials) < 3:
            continue
        dims = []
        for p in trials[0]["params"]:
            vals = [t["params"].get(p) for t in trials]
            num = [isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals]
            if all(num):
                arr = np.array(vals, dtype=float)
                if arr.min() > 0 and arr.max() / arr.min() > 50:
                    dims.append({"label": f"log10 {p}", "values": np.log10(arr).tolist()})
                else:
                    dims.append({"label": p, "values": arr.tolist()})
            else:
                cats = sorted({str(v) for v in vals})
                dims.append({"label": p, "values": [cats.index(str(v)) for v in vals], "tickvals": list(range(len(cats))), "ticktext": cats})
        scores = [t["mean_score"] for t in trials]
        dims.append({"label": "CV score", "values": scores})
        data.append({"type": "parcoords", "line": {"color": scores, "colorscale": "Viridis", "showscale": True, "colorbar": {"thickness": 12}},
                     "dimensions": dims})
        sizes.append(1)
        names.append(m)
    if not data:
        return None
    _vis_first(data, sizes)
    return fig(data, _layout(f"Parallel coordinates: {names[0]}", margin={"t": 90}, **(_dropdown(names, sizes, [f"Parallel coordinates: {n}" for n in names], y=1.2) if len(names) > 1 else {})))


def roc(d: Dict[str, Any]) -> Dict[str, Any]:
    return fig([{"type": "scatter", "mode": "lines", "x": d["fpr"], "y": d["tpr"], "name": f"AUC {d['auc']:.3f}", "line": {"color": PAL[0], "width": 3}},
                {"type": "scatter", "mode": "lines", "x": [0, 1], "y": [0, 1], "name": "random", "line": {"dash": "dash", "color": "grey"}}],
               _layout("ROC curve", xaxis={"title": "False positive rate"}, yaxis={"title": "True positive rate"}))


def pr(d: Dict[str, Any]) -> Dict[str, Any]:
    return fig([{"type": "scatter", "mode": "lines", "x": d["recall"], "y": d["precision"], "name": f"AP {d['ap']:.3f}", "line": {"color": PAL[1], "width": 3}},
                {"type": "scatter", "mode": "lines", "x": [0, 1], "y": [d["base_rate"]] * 2, "name": "base rate", "line": {"dash": "dash", "color": "grey"}}],
               _layout("Precision-recall curve", xaxis={"title": "Recall"}, yaxis={"title": "Precision"}))


def lift(lt: List[Dict[str, Any]], base_rate: float) -> Dict[str, Any]:
    return fig([{"type": "bar", "x": [r["decile"] for r in lt], "y": [r["event_rate"] for r in lt], "name": "event rate", "marker": {"color": PAL[3]},
                 "hovertemplate": "decile %{x}<br>event rate %{y:.1%}<extra></extra>"},
                {"type": "scatter", "mode": "lines+markers", "x": [r["decile"] for r in lt], "y": [r["cum_capture"] for r in lt], "name": "cumulative capture",
                 "yaxis": "y2", "line": {"color": PAL[0]}, "hovertemplate": "top %{x}0%: %{y:.1%} of all events<extra></extra>"}],
               _layout("Lift and cumulative gains by predicted-risk decile", xaxis={"title": "risk decile (1 = highest predicted risk)"},
                       yaxis={"title": "event rate", "tickformat": ".0%"}, yaxis2={"overlaying": "y", "side": "right", "title": "cumulative capture", "tickformat": ".0%", "range": [0, 1]},
                       shapes=[{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": base_rate, "y1": base_rate, "line": {"dash": "dash", "color": "black"}}]))


def confusion(cm: List[List[int]], classes: List[str]) -> Dict[str, Any]:
    return fig([{"type": "heatmap", "z": cm, "x": classes, "y": classes, "colorscale": "Blues", "texttemplate": "%{z}", "showscale": False,
                 "hovertemplate": "actual %{y}<br>predicted %{x}<br>count %{z}<extra></extra>"}],
               _layout("Confusion matrix (held-out)", xaxis={"title": "Predicted", "showgrid": False}, yaxis={"title": "Actual", "autorange": "reversed", "showgrid": False}))


def residuals(actual: List[float], pred: List[float]) -> Dict[str, Any]:
    a, p = np.array(actual), np.array(pred)
    lo, hi = float(min(a.min(), p.min())), float(max(a.max(), p.max()))
    return fig([{"type": "scattergl", "mode": "markers", "x": a.tolist(), "y": p.tolist(), "name": "held-out", "marker": {"size": 5, "opacity": 0.5, "color": PAL[0]}},
                {"type": "scatter", "mode": "lines", "x": [lo, hi], "y": [lo, hi], "name": "perfect", "line": {"dash": "dash", "color": PAL[3]}}],
               _layout("Actual vs predicted", xaxis={"title": "actual"}, yaxis={"title": "predicted"}))


def residual_hist(actual: List[float], pred: List[float]) -> Dict[str, Any]:
    r = (np.array(actual) - np.array(pred)).tolist()
    return fig([{"type": "histogram", "x": r, "nbinsx": 30, "marker": {"color": PAL[1]}, "name": "residual"}],
               _layout("Residual distribution", xaxis={"title": "actual - predicted"}, yaxis={"title": "count"}, showlegend=False))


def calibration(d: Dict[str, Any]) -> Dict[str, Any]:
    return fig([{"type": "scatter", "mode": "lines+markers", "x": d["mean_predicted"], "y": d["fraction_positive"], "name": f"model (Brier {d['brier']:.3f})",
                 "line": {"color": PAL[0]}}, {"type": "scatter", "mode": "lines", "x": [0, 1], "y": [0, 1], "name": "perfectly calibrated", "line": {"dash": "dash", "color": "grey"}}],
               _layout("Calibration: do predicted probabilities match reality?", xaxis={"title": "mean predicted probability"}, yaxis={"title": "observed event rate"}))


def learning(d: Dict[str, Any], pm: str) -> Dict[str, Any]:
    x = d["train_sizes"]

    def band(m, s, rgba):
        lo, hi = [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)]
        return {"type": "scatter", "x": x + x[::-1], "y": hi + lo[::-1], "fill": "toself", "fillcolor": rgba, "line": {"width": 0},
                "showlegend": False, "hoverinfo": "skip"}
    data = [band(d["train_mean"], d["train_std"], "rgba(245,133,24,0.15)"), band(d["val_mean"], d["val_std"], "rgba(76,120,168,0.15)"),
            {"type": "scatter", "mode": "lines+markers", "x": x, "y": d["train_mean"], "name": "training score", "line": {"color": PAL[1]}},
            {"type": "scatter", "mode": "lines+markers", "x": x, "y": d["val_mean"], "name": "cross-validated score", "line": {"color": PAL[0]}}]
    return fig(data, _layout("Learning curve: would more data help?", xaxis={"title": "training rows"}, yaxis={"title": pm}))


def pdp(curves: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not curves:
        return None
    names = list(curves)
    data = [{"type": "scatter", "mode": "lines+markers", "x": c["x"], "y": c["y"], "name": n, "line": {"color": PAL[i % 8]}, "visible": i == 0}
            for i, (n, c) in enumerate(curves.items())]
    return fig(data, _layout(f"Partial dependence: {names[0]}", yaxis={"title": "average prediction"}, showlegend=False,
                             **(_dropdown(names, [1] * len(names), [f"Partial dependence: {n}" for n in names]) if len(names) > 1 else {})))


def threshold(table: List[Dict[str, Any]]) -> Dict[str, Any]:
    t = [r["threshold"] for r in table]
    return fig([{"type": "scatter", "mode": "lines", "x": t, "y": [r["precision"] for r in table], "name": "precision", "line": {"color": PAL[0]}},
                {"type": "scatter", "mode": "lines", "x": t, "y": [r["recall"] for r in table], "name": "recall", "line": {"color": PAL[1]}},
                {"type": "scatter", "mode": "lines", "x": t, "y": [r["f1"] for r in table], "name": "F1", "line": {"color": PAL[2], "width": 3}}],
               _layout("Decision threshold: precision, recall and F1", xaxis={"title": "probability threshold"}, yaxis={"title": "score", "range": [0, 1]}))


def cost_curve(table: List[Dict[str, Any]], c_fp: float, c_fn: float) -> Dict[str, Any]:
    t = [r["threshold"] for r in table]
    cost = [r["fp"] * c_fp + r["fn"] * c_fn for r in table]
    i = int(np.argmin(cost))
    return fig([{"type": "scatter", "mode": "lines", "x": t, "y": cost, "name": "total cost", "line": {"color": PAL[3], "width": 3}},
                {"type": "scatter", "mode": "markers", "x": [t[i]], "y": [cost[i]], "name": f"best threshold {t[i]:.2f}", "marker": {"size": 14, "color": PAL[2]}}],
               _layout(f"Business cost by threshold (FP={c_fp:g}, FN={c_fn:g}): best at {t[i]:.2f}", xaxis={"title": "probability threshold"}, yaxis={"title": "total cost on held-out set"}))


# clustering
def cluster_scatter(cr: Dict[str, Any]) -> Dict[str, Any]:
    pts, lab = np.array(cr["pca_points"]), np.array(cr["pca_labels"])
    names = {p["cluster"]: p.get("name", f"segment {p['cluster']}") for p in cr["profiles"]}
    data = [{"type": "scattergl", "mode": "markers", "x": pts[lab == k, 0].tolist(), "y": pts[lab == k, 1].tolist(), "name": names.get(k, str(k)),
             "marker": {"size": 5, "opacity": 0.55, "color": PAL[k % 8]}} for k in range(cr["k"])]
    return fig(data, _layout("Segments in 2-D (PCA)", xaxis={"title": f"PC1 ({cr['explained_variance'][0]:.0%})"}, yaxis={"title": f"PC2 ({cr['explained_variance'][1]:.0%})"}))


def cluster_k(cr: Dict[str, Any]) -> Dict[str, Any]:
    ks = cr["k_scores"]
    return fig([{"type": "scatter", "mode": "lines+markers", "x": [r["k"] for r in ks], "y": [r["silhouette"] for r in ks], "name": "silhouette", "line": {"color": PAL[0]}},
                {"type": "scatter", "mode": "lines+markers", "x": [r["k"] for r in ks], "y": [r["davies_bouldin"] for r in ks], "name": "Davies-Bouldin (lower is better)", "yaxis": "y2", "line": {"color": PAL[1]}}],
               _layout("Choosing the number of segments", xaxis={"title": "k", "dtick": 1}, yaxis={"title": "silhouette"}, yaxis2={"overlaying": "y", "side": "right", "title": "Davies-Bouldin"},
                       shapes=[{"type": "line", "x0": cr["k"], "x1": cr["k"], "yref": "paper", "y0": 0, "y1": 1, "line": {"color": PAL[3], "dash": "dash"}}]))


def cluster_profile(cr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    feats = list(dict.fromkeys(f["feature"] for p in cr["profiles"] for f in p["defining_features"]))[:12]
    if not feats:
        return None
    z = [[next((f["z_vs_overall"] for f in p["defining_features"] if f["feature"] == ft), 0.0) for ft in feats] for p in cr["profiles"]]
    ylab = [f"{p.get('name', 'segment ' + str(p['cluster']))} ({p['share']:.0%})" for p in cr["profiles"]]
    return fig([{"type": "heatmap", "z": z, "x": feats, "y": ylab, "zmin": -2, "zmax": 2, "colorscale": "RdBu", "reversescale": True, "texttemplate": "%{z:.1f}",
                 "hovertemplate": "%{y}<br>%{x}: %{z:.2f} sd vs overall<extra></extra>"}],
               _layout("What defines each segment (z-score vs overall)", margin={"l": 190, "b": 120}, xaxis={"tickangle": -40, "showgrid": False}, yaxis={"autorange": "reversed", "showgrid": False}))


# Tableau-style explorer
AGGS = ["count", "sum", "mean", "median", "min", "max", "nunique"]
EXPLORE_TYPES = ["Bar", "Line", "Scatter", "Histogram", "Box", "Heatmap"]


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """filters: {col: ("range", lo, hi) | ("in", [values])}  (slicers)."""
    m = pd.Series(True, index=df.index)
    for c, f in (filters or {}).items():
        if c not in df.columns:
            continue
        if f[0] == "range":
            x = pd.to_numeric(df[c], errors="coerce")
            m &= x.between(f[1], f[2])
        elif f[0] == "in" and f[1]:
            m &= df[c].astype(str).isin([str(v) for v in f[1]])
    return df[m]


def explore(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """spec: {type, x, y, color, agg, top_n, bins, filters}. Returns a Plotly spec (drag-and-drop BI style)."""
    d = apply_filters(df, spec.get("filters"))
    t, x, y, c, agg = spec["type"], spec.get("x"), spec.get("y"), spec.get("color"), spec.get("agg", "count")
    n = len(d)
    if n == 0:
        return fig([], _layout("No rows match the filters"))
    top = int(spec.get("top_n", 20))

    def grouped(keys: List[str], val: Optional[str]) -> pd.DataFrame:
        g = d.groupby(keys, dropna=True)
        if agg == "count" or not val:
            r = g.size().rename("value")
        else:
            col = pd.to_numeric(d[val], errors="coerce") if agg != "nunique" else d[val]
            r = col.groupby([d[k] for k in keys]).agg(agg).rename("value")
        return r.reset_index()

    if t == "Bar":
        if not x:
            return fig([], _layout("Choose an X field"))
        r = grouped([x] + ([c] if c and c != x else []), y)
        if c and c != x:
            keep = r.groupby(x)["value"].sum().nlargest(top).index
            r = r[r[x].isin(keep)]
            data = [{"type": "bar", "x": s[x].astype(str).tolist(), "y": s["value"].tolist(), "name": str(k), "marker": {"color": PAL[i % 8]}}
                    for i, (k, s) in enumerate(r.groupby(c))]
            lay = _layout(f"{agg}({y or 'rows'}) by {x} and {c}", barmode="stack", xaxis={"title": x}, yaxis={"title": f"{agg} {y or ''}"})
        else:
            r = r.sort_values("value", ascending=False).head(top)
            data = [{"type": "bar", "x": r[x].astype(str).tolist(), "y": r["value"].tolist(), "marker": {"color": PAL[0]}}]
            lay = _layout(f"{agg}({y or 'rows'}) by {x}", xaxis={"title": x}, yaxis={"title": f"{agg} {y or ''}"}, showlegend=False)
        return fig(data, lay)
    if t == "Line":
        if not x:
            return fig([], _layout("Choose an X field"))
        xs = pd.to_datetime(d[x], errors="coerce") if not pd.api.types.is_numeric_dtype(d[x]) else d[x]
        dd = d.assign(**{x: xs}).dropna(subset=[x])
        if pd.api.types.is_datetime64_any_dtype(dd[x]):
            span = (dd[x].max() - dd[x].min()).days
            dd[x] = dd[x].dt.to_period("M" if span > 400 else "W" if span > 90 else "D").dt.start_time
        d2 = d
        d = dd
        r = grouped([x] + ([c] if c else []), y).sort_values(x)
        d = d2
        data = ([{"type": "scatter", "mode": "lines+markers", "x": s[x].astype(str).tolist(), "y": s["value"].tolist(), "name": str(k), "line": {"color": PAL[i % 8]}}
                 for i, (k, s) in enumerate(r.groupby(c))] if c else
                [{"type": "scatter", "mode": "lines+markers", "x": r[x].astype(str).tolist(), "y": r["value"].tolist(), "name": f"{agg} {y or 'rows'}", "line": {"color": PAL[0]}}])
        return fig(data, _layout(f"{agg}({y or 'rows'}) over {x}", xaxis={"title": x, "rangeslider": {"visible": True}}, yaxis={"title": f"{agg} {y or ''}"}))
    if t == "Scatter":
        if not (x and y):
            return fig([], _layout("Choose X and Y fields"))
        s = d[[x, y] + ([c] if c else [])].dropna().head(8000)
        if c and s[c].nunique() <= 10:
            data = [{"type": "scattergl", "mode": "markers", "x": g[x].tolist(), "y": g[y].tolist(), "name": str(k), "marker": {"size": 6, "opacity": 0.55, "color": PAL[i % 8]}}
                    for i, (k, g) in enumerate(s.groupby(c))]
        elif c and pd.api.types.is_numeric_dtype(s[c]):
            data = [{"type": "scattergl", "mode": "markers", "x": s[x].tolist(), "y": s[y].tolist(), "marker": {"size": 6, "opacity": 0.6, "color": s[c].tolist(), "colorscale": "Viridis", "showscale": True,
                                                                                                             "colorbar": {"title": c, "thickness": 12}}, "name": "records"}]
        else:
            data = [{"type": "scattergl", "mode": "markers", "x": s[x].tolist(), "y": s[y].tolist(), "marker": {"size": 6, "opacity": 0.5, "color": PAL[0]}, "name": "records"}]
        return fig(data, _layout(f"{y} vs {x}", xaxis={"title": x}, yaxis={"title": y}))
    if t == "Histogram":
        if not x:
            return fig([], _layout("Choose an X field"))
        if pd.api.types.is_numeric_dtype(d[x]):
            data = ([{"type": "histogram", "x": g[x].dropna().tolist(), "name": str(k), "opacity": 0.65, "nbinsx": int(spec.get("bins", 30)), "marker": {"color": PAL[i % 8]}}
                     for i, (k, g) in enumerate(d.groupby(c))] if c and d[c].nunique() <= 8 else
                    [{"type": "histogram", "x": d[x].dropna().tolist(), "nbinsx": int(spec.get("bins", 30)), "marker": {"color": PAL[0]}}])
        else:
            vc = d[x].astype(str).value_counts().head(top)
            data = [{"type": "bar", "x": vc.index.tolist(), "y": vc.tolist(), "marker": {"color": PAL[0]}}]
        return fig(data, _layout(f"Distribution of {x}", barmode="overlay", xaxis={"title": x}, yaxis={"title": "count"}))
    if t == "Box":
        if not y and not x:
            return fig([], _layout("Choose a Y field"))
        val = y or x
        grp = x if y else None
        if grp and d[grp].nunique() <= 25:
            top_levels = d[grp].astype(str).value_counts().index[:top]
            data = [{"type": "box", "y": pd.to_numeric(d.loc[d[grp].astype(str) == lv, val], errors="coerce").dropna().tolist(), "name": str(lv), "boxpoints": "outliers",
                     "marker": {"size": 3, "color": PAL[i % 8]}} for i, lv in enumerate(top_levels)]
        else:
            data = [{"type": "box", "y": pd.to_numeric(d[val], errors="coerce").dropna().tolist(), "name": val, "boxpoints": "outliers", "marker": {"color": PAL[0], "size": 3}}]
        return fig(data, _layout(f"{val}" + (f" by {grp}" if grp else ""), showlegend=False, yaxis={"title": val}))
    if t == "Heatmap":
        if not (x and y):
            return fig([], _layout("Choose X and Y fields"))
        xs = d[x].astype(str).value_counts().index[:top]
        ys = d[y].astype(str).value_counts().index[:top]
        s = d[d[x].astype(str).isin(xs) & d[y].astype(str).isin(ys)]
        if agg == "count" or not c:
            piv = pd.crosstab(s[y].astype(str), s[x].astype(str))
            ttl = f"count of rows: {y} x {x}"
        else:
            piv = pd.pivot_table(s.assign(**{c: pd.to_numeric(s[c], errors="coerce")}), index=s[y].astype(str), columns=s[x].astype(str), values=c, aggfunc=agg)
            ttl = f"{agg}({c}): {y} x {x}"
        return fig([{"type": "heatmap", "z": piv.values.tolist(), "x": piv.columns.tolist(), "y": piv.index.tolist(), "colorscale": "Blues", "texttemplate": "%{z:.3g}" if piv.size <= 120 else None,
                     "hovertemplate": "%{y} / %{x}: %{z:.4g}<extra></extra>"}], _layout(ttl, xaxis={"showgrid": False}, yaxis={"showgrid": False, "autorange": "reversed"}, margin={"l": 120, "b": 110}))
    return fig([], _layout("Unsupported chart type"))


# v5: leaderboard comparison
def leaderboard_scores(rows: List[Dict[str, Any]], pm: str, baseline: Optional[float] = None) -> Dict[str, Any]:
    """CV score +- one SD for EVERY candidate, with the held-out score as a marker (a gap between them = optimism or luck). Winner highlighted."""
    rows = sorted(rows, key=lambda r: -(r.get("select_score") or -1e9))
    names = [r["name"] + (" ★" if r.get("selected") else "") for r in rows]
    cols = [PAL[0] if r.get("selected") else "#9bb0d3" for r in rows]
    data = [{"type": "bar", "orientation": "h", "y": names, "x": [r.get("select_score") for r in rows], "name": f"CV {pm}", "marker": {"color": cols},
             "error_x": {"type": "data", "array": [r.get("cv_std") or 0 for r in rows], "visible": True},
             "hovertext": [r.get("verdict", "") for r in rows], "hovertemplate": "%{y}: %{x:.4g}<br>%{hovertext}<extra></extra>"},
            {"type": "scatter", "mode": "markers", "y": names, "x": [(r.get("heldout") or {}).get(pm) for r in rows], "name": f"held-out {pm}",
             "marker": {"symbol": "diamond", "size": 11, "color": PAL[3], "line": {"width": 1, "color": "white"}}}]
    lay = _layout(f"All candidates: CV {pm} ± SD vs held-out", height=max(320, 34 * len(rows) + 130), margin={"l": 230}, xaxis={"title": pm}, yaxis={"autorange": "reversed"})
    if baseline is not None:
        lay["shapes"] = [{"type": "line", "yref": "paper", "x0": baseline, "x1": baseline, "y0": 0, "y1": 1, "line": {"dash": "dash", "color": "black"}}]
        lay["annotations"] = [{"yref": "paper", "x": baseline, "y": 1.0, "text": "baseline", "showarrow": False, "yanchor": "bottom"}]
    return fig(data, lay)


def leaderboard_folds(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = [r for r in rows if r.get("fold_scores_used") or r.get("cv_scores")]
    if not rows:
        return None
    return fig([{"type": "box", "y": r.get("fold_scores_used") or r["cv_scores"], "name": r["name"][:28], "boxpoints": "all", "jitter": 0.4,
                 "marker": {"color": PAL[0] if r.get("selected") else "#9bb0d3"}} for r in rows],
               _layout("Per-fold scores: overlapping boxes cannot be told apart", showlegend=False, yaxis={"title": "fold score"}))


def roc_overlay(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rs = [r for r in rows if (r.get("diagnostics") or {}).get("roc")]
    if not rs:
        return None
    data = [{"type": "scatter", "mode": "lines", "x": r["diagnostics"]["roc"]["fpr"], "y": r["diagnostics"]["roc"]["tpr"],
             "name": f"{r['name'][:26]} (AUC {r['diagnostics']['roc']['auc']})", "line": {"width": 3 if r.get("selected") else 1.5}} for r in rs]
    data.append({"type": "scatter", "mode": "lines", "x": [0, 1], "y": [0, 1], "name": "random", "line": {"dash": "dash", "color": "grey"}})
    return fig(data, _layout("ROC curves of all candidates (held-out)", xaxis={"title": "False positive rate"}, yaxis={"title": "True positive rate"}))


def stress_bars(st: Dict[str, Any], what: str = "degradation") -> Optional[Dict[str, Any]]:
    rows = [r for r in st.get("results", []) if "score" in r]
    if not rows:
        return None
    return bars([r["perturbation"] for r in rows], [r["score"] for r in rows], f"Held-out {st['metric']} under {what} (clean = {st['clean']})", st["metric"], ref=st["clean"], ref_label="clean")


# v6: AutoML, goal, crew timeline
def automl_trials(trials: List[Dict[str, Any]], pm: str = "") -> Optional[Dict[str, Any]]:
    """Every full-fidelity trial in order (colour = family) with the best-so-far line: shows whether the search is still paying."""
    ts = [t for t in trials if t.get("mean") is not None and t.get("fidelity", 1) >= 0.999]
    if not ts:
        return None
    fams = sorted({t["family"] for t in ts})
    data = [{"type": "scatter", "mode": "markers", "name": f, "x": [t["id"] for t in ts if t["family"] == f], "y": [t["mean"] for t in ts if t["family"] == f],
             "text": [t["label"] for t in ts if t["family"] == f], "marker": {"size": 9, "color": PAL[i % len(PAL)]}, "hovertemplate": "%{text}<br>trial %{x}: %{y:.4f}<extra></extra>"} for i, f in enumerate(fams)]
    best = np.maximum.accumulate([t["mean"] for t in ts])
    data.append({"type": "scatter", "mode": "lines", "name": "best so far", "x": [t["id"] for t in ts], "y": [float(b) for b in best], "line": {"color": "black", "width": 2, "shape": "hv"}})
    return fig(data, _layout(f"AutoML search: every trial's CV {pm}", xaxis={"title": "trial #"}, yaxis={"title": f"CV {pm}"}, height=380))


def goal_ladder(gr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The three looks at the goal: optimistic search estimate -> fresh folds -> held-out (scored once), against the target line."""
    rows = [("search estimate (optimistic)", (gr.get("search_estimate") or {}).get("value")), ("fresh folds (after search)", (gr.get("confirmed_on_fresh_folds") or {}).get("value")),
            ("held-out (scored once)", (gr.get("heldout") or {}).get("value"))]
    rows = [(a, b) for a, b in rows if b is not None]
    if not rows or gr.get("target") is None:
        return None
    ci = (gr.get("heldout") or {}).get("ci")
    d = [{"type": "bar", "orientation": "h", "y": [a for a, _ in rows], "x": [b for _, b in rows], "marker": {"color": ["#9bb0d3", "#5b84c4", PAL[0]][:len(rows)]}}]
    lay = _layout(f"Goal {gr.get('text')}: {str(gr.get('status')).replace('_', ' ')}", height=260, margin={"l": 210}, yaxis={"autorange": "reversed"}, showlegend=False,
                  shapes=[{"type": "line", "yref": "paper", "x0": gr["target"], "x1": gr["target"], "y0": 0, "y1": 1, "line": {"dash": "dash", "color": "#d73a49", "width": 2}}],
                  annotations=[{"yref": "paper", "x": gr["target"], "y": 1.08, "text": "target", "showarrow": False, "font": {"color": "#d73a49"}}])
    if ci:
        d[0]["error_x"] = {"type": "data", "symmetric": False, "array": [0, 0, ci[1] - rows[-1][1]], "arrayminus": [0, 0, rows[-1][1] - ci[0]][:len(rows)]}
    return fig(d, lay)


def crew_gantt(timings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """When each stage actually ran. Overlapping bars = real parallelism."""
    rows = [(k, v["first_start"], v["last_end"]) for k, v in timings.items() if isinstance(v, dict) and "." not in k and v.get("first_start")]
    if not rows:
        return None
    t0 = min(r[1] for r in rows)
    rows.sort(key=lambda r: r[1])
    return fig([{"type": "bar", "orientation": "h", "y": [r[0] for r in rows], "x": [round(r[2] - r[1], 2) for r in rows], "base": [round(r[1] - t0, 2) for r in rows],
                 "marker": {"color": [PAL[0] if r[0] in ("sentinel", "explore", "scout", "automl") else "#9bb0d3" for r in rows]}, "hovertemplate": "%{y}: %{base:.1f}s → %{x:.1f}s longer<extra></extra>"}],
               _layout("When each stage ran (blue = the four parallel crew branches)", xaxis={"title": "seconds from start"}, yaxis={"autorange": "reversed"}, showlegend=False, height=max(300, 26 * len(rows) + 100)))
