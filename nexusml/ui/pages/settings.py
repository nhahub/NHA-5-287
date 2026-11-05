"""Settings: connections, privacy, defaults, storage and diagnostics."""
import platform
import sys
from pathlib import Path

from ... import __version__, config, llm, media, tracking, workspace
from .. import components as c
from .. import context, state
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
    context.set_view("settings")
    c.page_header("Settings", "Connections, privacy and defaults.")
    t_conn, t_priv, t_data, t_about = st.tabs(["Connections", "Privacy and defaults", "Storage", "About"])
    with t_conn:
        with st.container(border=True):
            c.eyebrow("Language model")
            rows = config.provider_status()
            ready = [r for r in rows if r["configured"]]
            ids = [r["id"] for r in rows]
            label = {r["id"]: f"{r['name']}  ·  {'key found' if r['configured'] else 'no key'}" for r in rows}
            pick = st.selectbox("Provider", ids, index=ids.index(config.LLM_PROVIDER), format_func=lambda i: label[i], key="set-provider")
            row = next(r for r in rows if r["id"] == pick)
            model = st.text_input("Model", value=config.LLM_MODEL if pick == config.LLM_PROVIDER else row["default_model"], key=f"set-model-{pick}",
                                  help="Any model id your key can use. Leave the default if unsure.")
            if pick != config.LLM_PROVIDER or model != config.LLM_MODEL:
                if row["configured"]:
                    config.set_provider(pick, model.strip())
                    workspace.set_pref("llm_provider", pick)
                    workspace.set_pref("llm_model", model.strip() if model.strip() != row["default_model"] else "")
                    st.toast(f"Using {row['name']}")
                else:
                    st.warning(f"No key found for {row['name']}. Set the `{row['key_env']}` environment variable (or add it to a git-ignored `.env` file) and restart.")
            ok = llm.available()
            st.markdown(c.badge("Connected" if ok else "Not configured", "ok" if ok else "bad"), unsafe_allow_html=True)
            st.caption(f"Active: {config.LLM_PROVIDER_NAME}, model {config.LLM_MODEL}. Keys are read only from environment variables "
                       f"({', '.join('`' + r['key_env'] + '`' for r in rows[:2])}, ...) and are never stored, shown or logged by this app.")
            if not ready:
                st.markdown("No provider key was found. Set `GEMINI_API_KEY` (Gemini) or `GROQ_API_KEY` (Groq) and restart.")
            elif ok and st.button("Test the connection", key="set-test"):
                try:
                    llm.preflight()
                    st.success("The language model answered.")
                except llm.LLMRequired as e:
                    st.error(str(e))
        with st.container(border=True):
            c.eyebrow("Data from links")
            from ... import linkfetch
            ok_k = linkfetch.kaggle_configured()
            st.markdown(c.badge("Kaggle key found" if ok_k else "No Kaggle key", "ok" if ok_k else "bad"), unsafe_allow_html=True)
            st.caption("Paste a Kaggle, GitHub or direct file link on the Home or Data page and it is downloaded for you (up to "
                       f"{linkfetch.max_bytes() // 1048576} MB). Kaggle needs `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` and `KAGGLE_KEY`) in your environment; it is never stored or shown.")
            if ok_k and st.button("Test the Kaggle key", key="set-test-kaggle"):
                try:
                    st.success(linkfetch.kaggle_check())
                except linkfetch.LinkError as e:
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
