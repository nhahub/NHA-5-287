"""The conversation with the assistant, shared by the Assistant page and the assistant dock that is available on every page."""
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .. import workspace
from ..assistant import agent, attachments, tools, workspace_ctx
from . import components as c
from . import context, nav, state
from ._st import ss, st

TOOL_LABEL = {"overview": "Profiling the dataset", "distribution": "Describing a column", "value_counts": "Counting values", "outliers": "Checking for outliers", "correlations": "Computing correlations",
              "missing": "Checking missing values", "group_summary": "Summarising by group", "relationship": "Testing a relationship", "plot": "Drawing a chart", "filter_rows": "Filtering rows",
              "time_trend": "Analysing the trend", "duplicates": "Looking for duplicates", "drivers": "Fitting a quick model", "propose_analysis": "Preparing an analysis", "ask_run": "Reading the open analysis",
              "check": "Checking the answer", "analyses": "Looking through your analyses", "analysis": "Reading an analysis", "analysis_chart": "Finding a chart", "jobs_status": "Checking what is running",
              "workspace_info": "Looking at the workspace", "packages": "Checking installed packages", "note_concern": "Saving your concern", "propose_extension": "Preparing a follow-up"}


def args_text(args: Dict[str, Any]) -> str:
    parts = []
    for k, v in (args or {}).items():
        if v in (None, "", [], {}):
            continue
        parts.append(f"{k.replace('_', ' ')}: {', '.join(map(str, v)) if isinstance(v, list) and v and not isinstance(v[0], dict) else v if not isinstance(v, (list, dict)) else '…'}")
    return "; ".join(parts)


def proposal_card(p: Dict[str, Any], i: int, kp: str = "") -> None:
    extend = p.get("kind") == "extend"
    with st.container(border=True):
        c.eyebrow("Proposed follow-up" if extend else "Proposed analysis")
        st.markdown(f"**{p.get('objective', '')}**")
        bits = [f"Predict `{p['target']}`" if p.get("target") else None, f"Aim for {p['goal_metric']} ≥ {p['goal_value']}" if p.get("goal_metric") and p.get("goal_value") else None,
                ("Reuses: " + ", ".join(p["reuse"])) if extend and p.get("reuse") else None]
        if any(bits):
            st.caption(" · ".join(b for b in bits if b))
        st.caption("This builds on the earlier analysis and does only the new work. Nothing has started yet." if extend else
                   "This runs a complete, validated analysis in the background and can take several minutes. Nothing has started yet.")
        if st.button("Set up this follow-up" if extend else "Set up this analysis", key=f"prop-{kp}{ss.uid}-{i}", type="primary"):
            ss.prefill = {**p, "dataset_id": ss.get("ds_id")}
            if extend:
                ss.extend_from = p["parent"]
            nav.go("analyze")


def render_message(m: Dict[str, Any], i: int, kp: str = "") -> None:
    role = m.get("role")
    if role == "note":
        st.caption(m.get("text", ""))
        return
    with st.chat_message(role if role in ("user", "assistant") else "assistant"):
        st.markdown(m.get("text", ""))
        if role != "assistant":
            return
        steps = [s for s in m.get("steps", []) if s.get("tool") != "check"]
        if steps:
            with st.expander(f"What I did ({len(steps)} step{'s' if len(steps) != 1 else ''})"):
                for s in steps:
                    line = f"**{TOOL_LABEL.get(s['tool'], s['tool'])}**" + (f" · {args_text(s.get('args'))}" if s.get("args") else "")
                    st.markdown(line + ("" if s.get("ok") else f"  \n:red[{s.get('note', 'failed')}]"))
                    if s.get("ok") and s.get("note"):
                        st.caption(s["note"])
        for j, a in enumerate(m.get("artifacts", [])):
            c.render_artifact(a, f"{kp}a{ss.uid}-{i}-{j}")
        if m.get("proposal"):
            proposal_card(m["proposal"], i, kp)
        for w in m.get("warnings", []):
            st.caption(w)
        if m.get("followups") and i == len(ss.messages) - 1:
            cols = st.columns(len(m["followups"]))
            for k, (col, q) in enumerate(zip(cols, m["followups"])):
                with col:
                    if st.button(q, key=f"{kp}fu-{ss.uid}-{i}-{k}", use_container_width=True):
                        ss.pending_prompt = q
                        st.rerun(scope="fragment") if kp else st.rerun()


def run_context() -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
    exp = ss.get("open_run")
    values = state.load_run(exp) if exp else None
    if not values:
        return None, None

    def answer(q: str) -> str:
        from .. import qa
        return qa.answer(values, q, [m for m in ss.messages if m.get("role") in ("user", "assistant")][-6:]).get("answer", "")
    return values, answer


def _attach(f: Any) -> None:
    """A picture is described, a document is summarised, right in the conversation. Nothing is added to the workspace's datasets (a PDF that holds tables still is)."""
    name = f.name
    with st.spinner("Looking at the picture…" if attachments.kind_of(name) == "image" else "Reading the document…"):
        att = attachments.read(name, f.getvalue())
    ss.messages.append({"role": "user", "text": f"Attached {name}"})
    ss.messages.append({"role": "assistant", "text": attachments.message_text(att), "steps": [], "artifacts": [], "proposal": None, "followups": attachments.followups(att), "warnings": []})
    if att["ok"]:
        ss.attachments = (list(ss.get("attachments") or []) + [attachments.context(att)])[-6:]
    if name.lower().endswith(".pdf"):
        try:
            meta = workspace._add_pdf_tables(name, f.getvalue())
        except Exception:
            meta = None
        if meta:
            ss.ds_id = meta["id"]
            ss.messages.append({"role": "note", "text": f"The PDF also contains a table, so I added it as the dataset {meta['name']} ({meta['rows']:,} rows, {meta['cols']} columns)."})


def ingest(files: List[Any]) -> None:
    for f in files:
        if attachments.kind_of(f.name):
            _attach(f)
            continue
        try:
            meta = workspace.add_upload(f.name, f.getvalue())
            ss.ds_id = meta["id"]
            ss.messages.append({"role": "note", "text": f"Added {meta['name']} ({meta['rows']:,} rows, {meta['cols']} columns). I'll use it for your questions."})
        except ValueError as e:
            ss.messages.append({"role": "note", "text": f"Could not add {f.name}: {e}"})


def answer(text: str, meta: Optional[Dict[str, Any]], df: Any, kp: str = "", show_user: bool = True) -> None:
    """Ask the assistant (it knows the page, the open analysis, the dataset and the whole workspace), record the exchange and redraw."""
    ss.messages.append({"role": "user", "text": text})
    if show_user:
        with st.chat_message("user"):
            st.markdown(text)
    values, answer_run = run_context()
    view = context.get_view()
    ctx = tools.ToolContext(df if df is not None else pd.DataFrame(), meta["name"] if meta else "dataset", run_state=values, answer_run=answer_run, view=view)
    try:
        ws = workspace_ctx.digest(view, ss.get("open_run"))
    except Exception:
        ws = None
    with st.chat_message("assistant"):
        status = st.status("Working on it…", expanded=False)

        def on_event(e: Dict[str, Any]) -> None:
            if e["type"] == "thinking" and e["step"] > 1:
                status.update(label="Reading the results…")
            elif e["type"] == "tool_start":
                status.update(label=TOOL_LABEL.get(e["tool"], "Working") + "…")
        reply = agent.respond(text, [m for m in ss.messages[:-1] if m.get("role") in ("user", "assistant")], ctx, share_labels=bool(ss.get("share_labels", True)), on_event=on_event, workspace=ws, attachments=ss.get("attachments"))
        status.update(label=f"Done in {reply['seconds']:.1f} s", state="complete")
    ss.messages.append({"role": "assistant", "text": reply["text"], "steps": reply["steps"], "artifacts": reply["artifacts"], "proposal": reply["proposal"],
                        "followups": reply["followups"], "warnings": reply["warnings"]})
    state.save_chat()
    st.rerun(scope="fragment") if kp else st.rerun()
