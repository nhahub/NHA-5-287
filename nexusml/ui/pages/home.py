"""Home: where things stand, and the quickest way to the next thing you want to do."""

from ... import jobs, runstore, workspace
from ...assistant import agent
from .. import components as c
from .. import nav, state
from .._st import ss, st

UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xls", "json", "pdf", "docx", "odt", "html", "md", "png", "jpg", "jpeg", "zip"]


def add_files(files, key: str) -> int:
    added = 0
    for f in files or []:
        try:
            meta = workspace.add_upload(f.name, f.getvalue())
            ss.ds_id = meta["id"]
            added += 1
        except ValueError as e:
            st.error(f"{f.name}: {e}")
    return added


def render() -> None:
    state.init()
    meta, df = state.active_dataset()
    datasets, runs, chats = workspace.list_datasets(), runstore.list_runs(60), workspace.list_conversations(limit=5)
    who = "" if ss.user in ("local", "") else f", {ss.user}"
    c.page_header(f"{c.greeting()}{who}", "Here is where things stand.")

    live = [j for j in jobs.active() if j["status"] in ("running", "queued")]
    if live:
        with st.container(border=True):
            cols = st.columns([5, 1])
            cols[0].markdown(f"**{len(live)} analysis{'es' if len(live) > 1 else ''} running** · {live[0]['stage_label'] or 'starting'} · {live[0]['dataset_name']}")
            with cols[1]:
                if st.button("View progress", key="home-live", use_container_width=True):
                    ss.job_id = live[0]["id"]
                    nav.go("analyze")

    if not datasets:
        with st.container(border=True):
            c.eyebrow("Get started")
            st.markdown("### Add your first dataset")
            st.caption("A CSV or Excel table, or a folder of documents or images (as a zip). Everything stays in this workspace.")
            files = st.file_uploader("Choose files", type=UPLOAD_TYPES, accept_multiple_files=True, key="home-upload", label_visibility="collapsed")
            if st.button("Add to workspace", type="primary", key="home-add", disabled=not files):
                if add_files(files, "home"):
                    st.rerun()
        return

    goals = [r for r in runs if r.get("passed") is not None]
    c.stat_cards([("Datasets", len(datasets), ""), ("Analyses", len(runs), "Completed full analyses"), ("Passed review", f"{sum(1 for r in goals if r['passed'])} of {len(goals)}" if goals else "–", "Analyses that passed the quality review"),
                  ("Conversations", len(workspace.list_conversations(limit=500)), "")])
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            c.eyebrow("Ask something")
            st.markdown(f"Working with **{meta['name']}**" if meta else "No dataset selected")
            sugg = agent.suggestions(df)[:4]
            cols = st.columns(2)
            for k, q in enumerate(sugg):
                with cols[k % 2]:
                    if st.button(q, key=f"home-sg-{k}", use_container_width=True):
                        ss.pending_prompt = q
                        nav.go("assistant")
    with right:
        with st.container(border=True):
            c.eyebrow("Jump to")
            if st.button("Explore the data", key="home-data", use_container_width=True):
                nav.go("data")
            if st.button("Start an analysis", key="home-new", use_container_width=True, type="primary"):
                nav.go("analyze")
            if st.button("Browse models", key="home-models", use_container_width=True):
                nav.go("models")

    st.write("")
    a, b = st.columns(2)
    with a:
        c.eyebrow("Recent analyses")
        if not runs:
            st.caption("No analyses yet. Start one from a proposal in the assistant, or from New analysis.")
        for r in runs[:5]:
            with st.container(border=True):
                cols = st.columns([4, 1])
                cols[0].markdown(f"**{(r['request'] or 'Analysis')[:70]}**")
                cols[0].caption(f"{r.get('target') or '–'} · {r.get('best_model') or '–'} · {c.time_ago(r['created'])}")
                with cols[1]:
                    if st.button("Open", key=f"home-run-{r['run_id']}", use_container_width=True):
                        ss.open_run = ss.view_run = r["run_id"]
                        nav.go("history")
    with b:
        c.eyebrow("Recent conversations")
        if not chats:
            st.caption("Your conversations with the assistant appear here.")
        for it in chats:
            with st.container(border=True):
                cols = st.columns([4, 1])
                cols[0].markdown(f"**{it['title']}**")
                cols[0].caption(c.time_ago(it["updated"]))
                with cols[1]:
                    if st.button("Open", key=f"home-chat-{it['id']}", use_container_width=True):
                        state.open_chat(it["id"])
                        nav.go("assistant")
