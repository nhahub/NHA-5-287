"""Assistant: ask anything about the data in plain language. Quick questions are answered directly; a full analysis is only ever started from a proposal you confirm."""
from typing import Any, Dict, List, Optional, Tuple

from ... import llm, workspace
from ...assistant import agent
from .. import components as c
from .. import chat, context, state
from .._st import ss, st

UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xls", "json", "pdf", "docx", "odt", "html", "md", "png", "jpg", "jpeg", "webp", "gif", "zip"]


def _unpack(v: Any) -> Tuple[str, List[Any]]:
    if v is None:
        return "", []
    if isinstance(v, str):
        return v.strip(), []
    text = (v.get("text") if hasattr(v, "get") else getattr(v, "text", "")) or ""
    files = (v.get("files") if hasattr(v, "get") else getattr(v, "files", [])) or []
    return text.strip(), list(files)


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


def render() -> None:
    state.init()
    context.set_view("assistant", dataset=(state.active_dataset()[0] or {}).get("name"), run=ss.get("open_run"))
    meta, df = state.active_dataset()
    datasets = workspace.list_datasets()
    c.page_header("Assistant", "Ask anything. It can also look at your data and analyses, describe a picture, or summarise a document. A full analysis only starts when you confirm it.")
    top = st.columns([3, 1, 1])
    with top[0]:
        names = {d["id"]: f"{d['name']}  ·  {d['rows']:,} rows" for d in datasets}
        opts = ["No dataset"] + list(names)
        state.sync_pick("asst-ds", names, "No dataset")
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

    if ss.get("open_run"):
        st.caption(f"Open analysis: {ss.open_run}. Ask about its findings, metrics, charts or report, or say what worries you.")
    for i, m in enumerate(ss.messages):
        chat.render_message(m, i)

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
                st.caption("No dataset needed for general questions. Drop a picture or a PDF below and I will describe or summarise it, or add a dataset to ask about your data.")

    raw = st.chat_input("Ask anything, or drop a picture, document or dataset here…", accept_file=True, file_type=UPLOAD_TYPES, disabled=not ready, key="asst-input")
    text, files = _unpack(raw)
    if ss.get("pending_prompt"):
        text, ss.pending_prompt = ss.pending_prompt, None
    if files:
        chat.ingest(files)
        meta, df = state.active_dataset()
        if not text:
            state.save_chat()
            st.rerun()
    if text:
        from ... import linkfetch
        for u in linkfetch.find_urls(text)[:2]:                       # a pasted link is fetched and added, then the question is answered about it
            try:
                with st.spinner("Downloading the link…"):
                    r = linkfetch.add_from_link(u)
                ss.ds_id = r["primary"]["id"]
                ss.messages.append({"role": "note", "text": f"Added {', '.join(d['name'] for d in r['datasets'])} from the link ({r['primary']['rows']:,} rows). " + " ".join(r["notes"])})
                meta, df = state.active_dataset()
            except linkfetch.LinkError as e:
                ss.messages.append({"role": "note", "text": f"Could not use the link: {e}"})
        chat.answer(text, meta, df)
    st.caption("The assistant sees column names, summary statistics and category counts, never individual rows. Tables and charts are shown to you only. You can change this in Settings.")
