"""Small, consistent building blocks: page headers, cards, badges, empty states, tables and chart/artifact rendering."""
import html
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from ._st import st

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
TONES = {"ok": "nx-ok", "warn": "nx-warn", "bad": "nx-bad", "info": "nx-info", "neutral": "nx-neutral"}


def greeting(hour: Optional[int] = None) -> str:
    h = time.localtime().tm_hour if hour is None else hour
    return "Good morning" if h < 12 else "Good afternoon" if h < 18 else "Good evening"


def time_ago(ts: float, now: Optional[float] = None) -> str:
    d = (now or time.time()) - float(ts)
    if d < 60:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)} min ago"
    if d < 86400:
        return f"{int(d // 3600)} h ago"
    if d < 86400 * 14:
        return f"{int(d // 86400)} days ago"
    return time.strftime("%d %b %Y", time.localtime(ts))


def fmt_bytes(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} GB"


def badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="nx-badge {TONES.get(tone, "nx-neutral")}">{html.escape(str(text))}</span>'


def goal_badge(status: Optional[str]) -> str:
    return {"met": badge("Goal met", "ok"), "met_on_test_only": badge("Borderline", "warn"), "not_met": badge("Goal not met", "bad"), "no_goal": badge("No target set", "neutral")}.get(status or "", badge("", "neutral"))


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f'<p class="nx-sub">{html.escape(subtitle)}</p>', unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="nx-eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str, button: Optional[str] = None, key: Optional[str] = None) -> bool:
    with st.container(border=True):
        st.markdown(f'<div class="nx-empty"><h3>{html.escape(title)}</h3><div>{html.escape(body)}</div></div>', unsafe_allow_html=True)
        if button:
            _, mid, _ = st.columns([2, 1, 2])
            with mid:
                return bool(st.button(button, key=key, type="primary", use_container_width=True))
    return False


def stat_cards(items: Sequence[Tuple[str, Any, str]]) -> None:
    cols = st.columns(len(items))
    for c, (label, value, hint) in zip(cols, items):
        with c:
            st.metric(label, value, help=hint or None)


def plot(spec: Optional[Dict[str, Any]], key: str) -> None:
    if spec:
        st.plotly_chart(spec, use_container_width=True, key=key, config=PLOT_CONFIG, theme=None)


def frame(table: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """A stored table payload ({columns, rows}) back to a DataFrame."""
    if not table:
        return None
    return pd.DataFrame(table["rows"], columns=table["columns"])


def show_table(df: pd.DataFrame, key: Optional[str] = None, height: Optional[int] = None) -> None:
    st.dataframe(df, use_container_width=True, hide_index=True, height=height, key=key)


def render_artifact(a: Dict[str, Any], key: str) -> None:
    """One result of an analysis step: title, chart, table, notes. Used by the assistant and by the Data page, so both look the same."""
    with st.container(border=True):
        st.markdown(f"**{a.get('title', 'Result')}**")
        if a.get("figure"):
            plot(a["figure"], f"{key}-fig")
        df = frame(a.get("table"))
        if df is not None and len(df):
            show_table(df, key=f"{key}-tbl", height=min(420, 38 + 35 * len(df)))
            total = (a.get("table") or {}).get("total_rows", len(df))
            if total > len(df):
                st.caption(f"Showing {len(df)} of {total} rows.")
        for n in a.get("notes", []) or []:
            st.caption(n)


def status_tone(status: str) -> str:
    return {"done": "ok", "running": "info", "queued": "info", "failed": "bad", "cancelled": "neutral", "needs_input": "warn"}.get(status, "neutral")
