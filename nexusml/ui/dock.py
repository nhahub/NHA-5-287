"""The assistant, reachable from every page.

A button at the foot of the sidebar opens the same conversation as the Assistant page in a dialog. The assistant is told which page the person is on and which analysis
or dataset is open, and can read every analysis, chart, report, running job, model and installed package (see assistant/tools.py), so a question asked anywhere can be
answered with the whole workspace in mind. It is drawn AFTER the page has run, so it always sees the page's own view.
"""
from typing import Any, Dict

from .. import llm
from . import chat, context, state
from ._st import ss, st

SHOWN = 8                                                          # messages drawn in the dialog; the Assistant page keeps the whole conversation


def _body() -> None:
    meta, df = state.active_dataset()
    view = context.get_view()
    st.caption(context.describe(view) + ". The assistant sees summaries, never individual rows.")
    ready = llm.available()
    if not ready:
        st.warning("The language model is not connected, so the assistant can't answer yet. Add your API key in Settings → Connections.")
    msgs = ss.messages
    if len(msgs) > SHOWN:
        st.caption(f"Showing the last {SHOWN} messages. The full conversation is on the Assistant page.")
    for i in range(max(0, len(msgs) - SHOWN), len(msgs)):
        chat.render_message(msgs[i], i, "dock-")
    if not msgs:
        for k, q in enumerate(context.suggestions(view)):
            if st.button(q, key=f"dock-sg-{k}", use_container_width=True, disabled=not ready):
                ss.pending_prompt = q
                st.rerun(scope="fragment")
    with st.form("dock-form", clear_on_submit=True):
        text = st.text_area("Ask the assistant", height=80, key="dock-text", label_visibility="collapsed", placeholder="Ask about this page, an analysis, a result, or say what worries you…")
        sent = st.form_submit_button("Send", type="primary", disabled=not ready)
    q = (text or "").strip() if sent else ""
    if ss.get("pending_prompt"):
        q, ss.pending_prompt = ss.pending_prompt, None
    if q:
        chat.answer(q, meta, df, kp="dock-")


def render() -> None:
    """Draw the dock in the sidebar. Not shown on the Assistant page, which is the same conversation at full size."""
    state.init()                                                      # the dock must work even if the page before it never initialised the session
    view = context.get_view()
    if view.get("page") == "assistant":
        return
    with st.sidebar:
        st.divider()
        st.caption(context.describe(view))
        if st.button("Ask the assistant", key="dock-open", icon=":material/forum:", use_container_width=True):
            _open()


def _open() -> None:
    @st.dialog("Assistant", width="large")
    def _dialog() -> None:
        _body()
    _dialog()
