"""Settings: connections, privacy, defaults, storage and diagnostics."""
import platform
import sys
from pathlib import Path

from ... import __version__, config, llm, media, tracking, workspace
from .. import components as c
from .. import state
from .._st import ss, st
from ..theme import APP_NAME


def _confirm_clear() -> None:
    @st.dialog("Clear conversations")
    def _dialog() -> None:
        st.write("Delete every saved conversation? Your datasets and analyses are not affected.")
        a, b = st.columns(2)
        if a.button("Delete", type="primary", key="clr-yes"):
            n = workspace.clear_conversations()
            ss.messages = []
            st.toast(f"Deleted {n} conversations")
            st.rerun()
        if b.button("Cancel", key="clr-no"):
            st.rerun()
    _dialog()


def render() -> None:
    state.init()
    c.page_header("Settings", "Connections, privacy and defaults.")
    t_conn, t_priv, t_data, t_about = st.tabs(["Connections", "Privacy and defaults", "Storage", "About"])
    with t_conn:
        with st.container(border=True):
            c.eyebrow("Language model")
            ok = llm.available()
            st.markdown(c.badge("Connected" if ok else "Not configured", "ok" if ok else "bad"), unsafe_allow_html=True)
            st.caption(f"{config.LLM_PROVIDER_NAME}, model: {config.LLM_MODEL}")
            if not ok:
                st.markdown(f"Set the `{config.LLM_KEY_ENV}` environment variable (or put it in a `.env` file next to the app) and restart. Use `GEMINI_API_KEY` for Google Gemini or `GROQ_API_KEY` for Groq.")
            elif st.button("Test the connection", key="set-test"):
                try:
                    llm.preflight()
                    st.success("The language model answered.")
                except llm.LLMRequired as e:
                    st.error(str(e))
        with st.container(border=True):
            c.eyebrow("Experiment tracking")
            ml = tracking.mlflow_status()
            st.markdown(c.badge("MLflow active" if ml.get("active") else "Built-in store only", "ok" if ml.get("active") else "neutral"), unsafe_allow_html=True)
            st.caption(ml.get("uri") or ml.get("reason", ""))
        with st.container(border=True):
            c.eyebrow("File support")
            dep = media.dependency_status()
            st.dataframe([{"Format": k, "Status": "Ready" if v.get("ok") else "Unavailable", "How": v.get("how", ""), "To enable": v.get("fix", "")} for k, v in dep.items()], use_container_width=True, hide_index=True)
    with t_priv:
        with st.container(border=True):
            c.eyebrow("What the assistant can see")
            st.markdown("The assistant sees **column names, summary statistics and category counts**. It never sees individual rows: tables and charts are rendered for you only.")
            share = st.toggle("Include category labels (for example region names) with their counts", value=bool(ss.get("share_labels", True)), key="set-labels")
            if share != ss.get("share_labels"):
                ss.share_labels = share
                workspace.set_pref("share_labels", share)
        with st.container(border=True):
            c.eyebrow("Defaults for new analyses")
            m = st.slider("Time for the model search (minutes)", 1, 60, int(workspace.get_pref("minutes", 5)), key="set-min")
            w = st.slider("Parallel workers", 1, 8, int(workspace.get_pref("workers", 2)), key="set-workers")
            if st.button("Save defaults", key="set-save"):
                workspace.set_pref("minutes", m)
                workspace.set_pref("workers", w)
                st.toast("Saved")
    with t_data:
        ds = workspace.list_datasets()
        c.stat_cards([("Datasets", len(ds), ""), ("Stored", c.fmt_bytes(sum(d["bytes"] for d in ds)), ""), ("Conversations", len(workspace.list_conversations(limit=1000)), "")])
        st.write("")
        if st.button("Delete all conversations", key="set-clear"):
            _confirm_clear()
        log = Path(config.artifacts_dir()) / "app_errors.log"
        if log.exists():
            st.download_button("Download the error log", log.read_bytes(), file_name="app_errors.log", key="set-log")
            st.caption("Include this file if you ask for help.")
    with t_about:
        st.markdown(f"**{APP_NAME}** · version {__version__}")
        st.caption(f"Python {sys.version.split()[0]} · {platform.system()}")
        st.caption(f"Signed in as {ss.user}")
