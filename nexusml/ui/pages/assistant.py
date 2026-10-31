"""Assistant: ask anything about the data in plain language. Quick questions are answered directly; a full analysis is only ever started from a proposal you confirm."""
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ... import llm, workspace
from ...assistant import agent, tools
from .. import components as c
from .. import nav, state
from .._st import ss, st

TOOL_LABEL = {"overview": "Profiling the dataset", "distribution": "Describing a column", "value_counts": "Counting values", "outliers": "Checking for outliers", "correlations": "Computing correlations",
              "missing": "Checking missing values", "group_summary": "Summarising by group", "relationship": "Testing a relationship", "plot": "Drawing a chart", "filter_rows": "Filtering rows",
              "time_trend": "Analysing the trend", "duplicates": "Looking for duplicates", "drivers": "Fitting a quick model", "propose_analysis": "Preparing an analysis", "ask_run": "Reading the open analysis",
              "check": "Checking the answer"}
UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xls", "json", "pdf", "docx", "odt", "html", "md", "png", "jpg", "jpeg", "zip"]


def _unpack(v: Any) -> Tuple[str, List[Any]]:
    if v is None:
        return "", []
    if isinstance(v, str):
        return v.strip(), []
    text = (v.get("text") if hasattr(v, "get") else getattr(v, "text", "")) or ""
    files = (v.get("files") if hasattr(v, "get") else getattr(v, "files", [])) or []
    return text.strip(), list(files)


def _args_text(args: Dict[str, Any]) -> str:
    parts = []
    for k, v in (args or {}).items():
        if v in (None, "", [], {}):
            continue
        parts.append(f"{k.replace('_', ' ')}: {', '.join(map(str, v)) if isinstance(v, list) and v and not isinstance(v[0], dict) else v if not isinstance(v, (list, dict)) else '…'}")
    return "; ".join(parts)


def _render_message(m: Dict[str, Any], i: int) -> None:
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
                    line = f"**{TOOL_LABEL.get(s['tool'], s['tool'])}**" + (f" · {_args_text(s.get('args'))}" if s.get("args") else "")
                    st.markdown(line + ("" if s.get("ok") else f"  \n:red[{s.get('note', 'failed')}]"))
                    if s.get("ok") and s.get("note"):
                        st.caption(s["note"])
        for j, a in enumerate(m.get("artifacts", [])):
            c.render_artifact(a, f"a{ss.uid}-{i}-{j}")
        if m.get("proposal"):
            _proposal_card(m["proposal"], i)
        for w in m.get("warnings", []):
            st.caption(w)
        if m.get("followups") and i == len(ss.messages) - 1:
            cols = st.columns(len(m["followups"]))
            for k, (col, q) in enumerate(zip(cols, m["followups"])):
                with col:
                    if st.button(q, key=f"fu-{ss.uid}-{i}-{k}", use_container_width=True):
                        ss.pending_prompt = q
                        st.rerun()


def _proposal_card(p: Dict[str, Any], i: int) -> None:
    with st.container(border=True):
        c.eyebrow("Proposed analysis")
        st.markdown(f"**{p.get('objective', '')}**")
        bits = [f"Predict `{p['target']}`" if p.get("target") else None, f"Aim for {p['goal_metric']} ≥ {p['goal_value']}" if p.get("goal_metric") and p.get("goal_value") else None]
        if any(bits):
            st.caption(" · ".join(b for b in bits if b))
        st.caption("This runs a complete, validated analysis in the background and can take several minutes. Nothing has started yet.")
        if st.button("Set up this analysis", key=f"prop-{ss.uid}-{i}", type="primary"):
            ss.prefill = {**p, "dataset_id": ss.get("ds_id")}
            nav.go("analyze")


def _history_menu() -> None:
    items = workspace.list_conversations(limit=30)
    with st.popover("History", use_container_width=True):
        if not items:
            st.caption("Conversations you have will appear here.")
        for it in items:
            lab = ("● " if it["pinned"] else "") + it["title"]
            if st.button(lab, key=f"conv-{it['id']}", use_container_width=True):
                if state.open_chat(it["id"]):
                    st.rerun()


def _run_context() -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
    exp = ss.get("open_run")
    values = state.load_run(exp) if exp else None
    if not values:
        return None, None

    def answer(q: str) -> str:
        from ... import qa
        return qa.answer(values, q, [m for m in ss.messages if m.get("role") in ("user", "assistant")][-6:]).get("answer", "")
    return values, answer


def _ingest(files: List[Any]) -> None:
    for f in files:
        try:
            meta = workspace.add_upload(f.name, f.getvalue())
            ss.ds_id = meta["id"]
            ss.messages.append({"role": "note", "text": f"Added {meta['name']} ({meta['rows']:,} rows, {meta['cols']} columns). I'll use it for your questions."})
        except ValueError as e:
            ss.messages.append({"role": "note", "text": f"Could not add {f.name}: {e}"})


def _answer(text: str, meta: Optional[Dict[str, Any]], df: Any) -> None:
    ss.messages.append({"role": "user", "text": text})
    with st.chat_message("user"):
        st.markdown(text)
    values, answer_run = _run_context()
    ctx = tools.ToolContext(df if df is not None else pd.DataFrame(), meta["name"] if meta else "dataset", run_state=values, answer_run=answer_run)
    with st.chat_message("assistant"):
        status = st.status("Working on it…", expanded=False)

        def on_event(e: Dict[str, Any]) -> None:
            if e["type"] == "thinking" and e["step"] > 1:
                status.update(label="Reading the results…")
            elif e["type"] == "tool_start":
                status.update(label=TOOL_LABEL.get(e["tool"], "Working") + "…")
        reply = agent.respond(text, [m for m in ss.messages[:-1] if m.get("role") in ("user", "assistant")], ctx, share_labels=bool(ss.get("share_labels", True)), on_event=on_event)
        status.update(label=f"Done in {reply['seconds']:.1f} s", state="complete")
    ss.messages.append({"role": "assistant", "text": reply["text"], "steps": reply["steps"], "artifacts": reply["artifacts"], "proposal": reply["proposal"],
                        "followups": reply["followups"], "warnings": reply["warnings"]})
    state.save_chat()
    st.rerun()


def render() -> None:
    state.init()
    meta, df = state.active_dataset()
    datasets = workspace.list_datasets()
    c.page_header("Assistant", "Ask about your data in plain language. Quick questions are answered right away; a full analysis only starts when you confirm it.")
    top = st.columns([3, 1, 1])
    with top[0]:
        names = {d["id"]: f"{d['name']}  ·  {d['rows']:,} rows" for d in datasets}
        opts = ["No dataset"] + list(names)
        cur = ss.get("ds_id") if ss.get("ds_id") in names else None
        pick = st.selectbox("Dataset", opts, index=(opts.index(cur) if cur else 0), format_func=lambda k: names.get(k, "No dataset"), label_visibility="collapsed", key="asst-ds")
        new_id = None if pick == "No dataset" else pick
        if new_id != ss.get("ds_id") and (new_id or datasets):
            ss.ds_id = new_id
            st.rerun()
    with top[1]:
        _history_menu()
    with top[2]:
        if st.button("New chat", use_container_width=True, key="asst-new"):
            state.new_chat()
            st.rerun()

    ready = llm.available()
    if not ready:
        st.warning("The language model is not connected, so the assistant can't answer yet. Add your API key in Settings → Connections.")

    for i, m in enumerate(ss.messages):
        _render_message(m, i)

    if not ss.messages:
        with st.container(border=True):
            c.eyebrow("Try asking")
            sugg = agent.suggestions(df)
            cols = st.columns(2)
            for k, q in enumerate(sugg):
                with cols[k % 2]:
                    if st.button(q, key=f"sg-{k}", use_container_width=True, disabled=not ready):
                        ss.pending_prompt = q
                        st.rerun()
            if df is None:
                st.caption("No dataset yet: drop a file into the box below, or add one on the Data page.")

    raw = st.chat_input("Ask about your data, or drop a file here…", accept_file=True, file_type=UPLOAD_TYPES, disabled=not ready, key="asst-input")
    text, files = _unpack(raw)
    if ss.get("pending_prompt"):
        text, ss.pending_prompt = ss.pending_prompt, None
    if files:
        _ingest(files)
        meta, df = state.active_dataset()
        if not text:
            state.save_chat()
            st.rerun()
    if text:
        _answer(text, meta, df)
    st.caption("The assistant sees column names, summary statistics and category counts, never individual rows. Tables and charts are shown to you only. You can change this in Settings.")
