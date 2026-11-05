"""Home: where things stand, and the quickest way to the next thing you want to do."""

from ... import jobs, runstore, workspace
from ...assistant import agent
from .. import components as c
from .. import context
from .. import nav, state
from .._st import ss, st

UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xls", "json", "pdf", "docx", "odt", "html", "md", "png", "jpg", "jpeg", "zip"]


def uploader_key(key: str) -> str:
    """A file box cannot be emptied from code, so it gets a new key after each successful add (the key is unchanged until then)."""
    n = ss.get(f"_up:{key}", 0)
    return f"{key}-upload" if not n else f"{key}-upload-{n}"


def add_files(files, key: str) -> int:
    """Add the chosen files. What happened to each file is kept in `ss.add_notice`, so it is still on screen after the page reruns."""
    added, notice = 0, []
    for f in files or []:
        try:
            meta = workspace.add_upload(f.name, f.getvalue())
            ss.ds_id = meta["id"]
            added += 1
            notice.append({"ok": True, "name": meta["name"], "rows": meta["rows"], "cols": meta["cols"], "id": meta["id"]})
        except ValueError as e:
            notice.append({"ok": False, "name": f.name, "error": str(e)})
            st.error(f"{f.name}: {e}")
    if notice:
        ss.add_notice = notice
    if added:
        ss[f"_up:{key}"] = ss.get(f"_up:{key}", 0) + 1                # the chosen files are now in the workspace: show an empty file box
    return added


def show_notice() -> None:
    """Say what was just added (once): each dataset with its size, the last one noted as selected, and any file that could not be read."""
    notice = ss.pop("add_notice", None) or []
    last_ok = max((i for i, n in enumerate(notice) if n["ok"]), default=None)
    for i, n in enumerate(notice):
        if not n["ok"]:
            st.error(f"Could not add {n['name']}: {n['error']}")
            continue
        src = f" from {n['source']}" if n.get("source") else ""
        st.success(f"Added **{n['name']}**{src}: {n['rows']:,} rows and {n['cols']} columns." + (" It is now the selected dataset." if i == last_ok else ""), icon=":material/check_circle:")


def add_link(url: str) -> int:
    """Fetch a link (Kaggle dataset / competition, GitHub or direct file) into the workspace. Returns how many datasets were added."""
    from ... import linkfetch
    try:
        with st.spinner("Downloading and reading the link…"):
            r = linkfetch.add_from_link(url.strip())
    except linkfetch.LinkError as e:
        st.error(str(e))
        return 0
    ss.ds_id = r["primary"]["id"]
    names = ", ".join(d["name"] for d in r["datasets"])
    ss.add_notice = [{"ok": True, "name": d["name"], "rows": d["rows"], "cols": d["cols"], "id": d["id"], "source": r["source"]} for d in r["datasets"]]
    if r["notes"]:
        st.caption(" ".join(r["notes"]))
    return len(r["datasets"])


def link_box(key: str) -> None:
    """The 'paste a link' control shared by the Home and Data pages."""
    from ... import linkfetch
    url = st.text_input("Or paste a link", key=f"{key}-link", placeholder="https://www.kaggle.com/datasets/owner/name   (or a GitHub / direct file link)")
    if url and "kaggle.com" in url and not linkfetch.kaggle_configured():
        st.caption("Kaggle needs your API token: set `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` and `KAGGLE_KEY`) as an environment variable and restart.")
    if st.button("Fetch link", key=f"{key}-fetch", disabled=not url):
        if add_link(url):
            st.rerun()


def render() -> None:
    state.init()
    context.set_view("home")
    meta, df = state.active_dataset()
    datasets, runs, chats = workspace.list_datasets(), runstore.list_runs(60), workspace.list_conversations(limit=5)
    who = "" if ss.user in ("local", "") else f", {ss.user}"
    c.page_header(f"{c.greeting()}{who}", "Here is where things stand.")
    show_notice()

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
            files = st.file_uploader("Choose files", type=UPLOAD_TYPES, accept_multiple_files=True, key=uploader_key("home"), label_visibility="collapsed")
            if st.button("Add to workspace", type="primary", key="home-add", disabled=not files):
                if add_files(files, "home"):
                    st.rerun()
            link_box("home")
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
