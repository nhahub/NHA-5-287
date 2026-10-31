"""Plain Plotly specs (dicts) for the assistant's answers, styled by the shared palette. No plotly import is needed to build them."""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import palette
from ..plotly_viz import PAL, _layout, fig

MAX_POINTS = 6000


def _thin(a, k: int = MAX_POINTS):
    a = np.asarray(a)
    if len(a) <= k:
        return a
    idx = np.linspace(0, len(a) - 1, k).astype(int)
    return a[idx]


def histogram(values, name: str, bins: int = 30) -> Dict[str, Any]:
    return fig([{"type": "histogram", "x": [float(v) for v in _thin(values, 20000)], "nbinsx": bins, "marker": {"color": PAL[0]}, "opacity": 0.9}],
               _layout(f"Distribution of {name}", xaxis={"title": name}, yaxis={"title": "rows"}, showlegend=False, bargap=0.04, height=340))


def box(groups: Dict[str, Sequence[float]], title: str, ytitle: str = "") -> Dict[str, Any]:
    return fig([{"type": "box", "y": [float(v) for v in _thin(v_, 2000)], "name": str(k)[:24], "marker": {"color": PAL[i % len(PAL)]}, "boxpoints": "outliers"} for i, (k, v_) in enumerate(groups.items())],
               _layout(title, yaxis={"title": ytitle}, showlegend=False, height=360))


def scatter(x, y, xname: str, yname: str, flagged: Optional[Sequence[bool]] = None, title: str = "", trend: bool = False, labels: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    fl = np.asarray(flagged, bool)[ok] if flagged is not None else np.zeros(len(x), bool)
    if len(x) > MAX_POINTS:                                           # keep every flagged point, thin the rest
        keep = np.where(fl)[0]
        rest = np.where(~fl)[0]
        rest = rest[np.linspace(0, len(rest) - 1, MAX_POINTS - min(len(keep), MAX_POINTS // 2)).astype(int)]
        sel = np.sort(np.r_[keep[:MAX_POINTS // 2], rest])
        x, y, fl = x[sel], y[sel], fl[sel]
    data = [{"type": "scattergl", "mode": "markers", "x": x[~fl].tolist(), "y": y[~fl].tolist(), "name": "other rows", "marker": {"size": 5, "color": PAL[2], "opacity": 0.45}}]
    if fl.any():
        data.append({"type": "scattergl", "mode": "markers", "x": x[fl].tolist(), "y": y[fl].tolist(), "name": "flagged", "marker": {"size": 9, "color": PAL[1], "line": {"width": 1, "color": palette.tokens()["surface"]}}})
    if trend and len(x) > 2 and np.ptp(x) > 0:
        b, a = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        data.append({"type": "scatter", "mode": "lines", "x": xs.tolist(), "y": (a + b * xs).tolist(), "name": "linear fit", "line": {"color": PAL[0], "width": 2}})
    return fig(data, _layout(title or f"{yname} against {xname}", xaxis={"title": xname}, yaxis={"title": yname}, height=380))


def bar(labels: List[str], values: List[float], title: str, xtitle: str = "", horizontal: bool = True) -> Dict[str, Any]:
    labels = [str(l)[:32] for l in labels]
    if horizontal:
        return fig([{"type": "bar", "orientation": "h", "y": labels, "x": [float(v) for v in values], "marker": {"color": PAL[0]}}],
                   _layout(title, xaxis={"title": xtitle}, yaxis={"autorange": "reversed"}, showlegend=False, margin={"l": 170}, height=max(280, 28 * len(labels) + 120)))
    return fig([{"type": "bar", "x": labels, "y": [float(v) for v in values], "marker": {"color": PAL[0]}}], _layout(title, yaxis={"title": xtitle}, showlegend=False, height=340))


def line(x: List[Any], series: Dict[str, Sequence[float]], title: str, ytitle: str = "") -> Dict[str, Any]:
    data = [{"type": "scatter", "mode": "lines", "x": [str(v) for v in x], "y": [None if not np.isfinite(v) else float(v) for v in vals], "name": k,
             "line": {"color": PAL[i % len(PAL)], "width": 2 if i == 0 else 2.5, "dash": "solid" if i == 0 else "dot"}} for i, (k, vals) in enumerate(series.items())]
    return fig(data, _layout(title, yaxis={"title": ytitle}, height=360, showlegend=len(series) > 1))


def heatmap(matrix, labels: List[str], title: str, zmin: float = -1, zmax: float = 1) -> Dict[str, Any]:
    t = palette.tokens()
    return fig([{"type": "heatmap", "z": np.round(np.asarray(matrix, float), 3).tolist(), "x": [str(l)[:18] for l in labels], "y": [str(l)[:18] for l in labels], "zmin": zmin, "zmax": zmax,
                 "colorscale": [[0, PAL[1]], [0.5, t["surface"]], [1, PAL[0]]], "hovertemplate": "%{y} / %{x}: %{z}<extra></extra>"}],
               _layout(title, height=max(340, 26 * len(labels) + 140), yaxis={"autorange": "reversed"}, showlegend=False, margin={"l": 120, "b": 110}))
