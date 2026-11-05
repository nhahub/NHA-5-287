"""Data: your datasets, how they look, how healthy they are, and how their columns relate."""

from pathlib import Path

from ... import multitable, workspace
from ...assistant import agent, tools
from .. import components as c
from .. import context
from .. import state
from .._st import ss, st
from .home import UPLOAD_TYPES, add_files, link_box, show_notice, uploader_key


def _tool(df, name, key, **args):
    try:
        res = tools.run_tool(tools.ToolContext(df, "dataset"), name, args)
    except tools.ToolError as e:
        st.info(str(e))
        return None
    c.render_artifact({"title": res.title, "table": agent.table_payload(res.table), "figure": res.figure, "notes": res.notes}, key)
    return res


def _confirm_delete(meta) -> None:
    @st.dialog("Delete dataset")
    def _dialog() -> None:
        st.write(f"Delete **{meta['name']}**? Analyses already run on it are kept.")
        a, b = st.columns(2)
        if a.button("Delete", type="primary", key="del-yes"):
            workspace.delete_dataset(meta["id"])
            ss.ds_id = None
            st.rerun()
        if b.button("Cancel", key="del-no"):
            st.rerun()
    _dialog()


def _frames(ids, names):
    """Workspace datasets -> {table name: DataFrame}; names come from the file names, made unique."""
    out = {}
    for i in ids:
        base = Path(names[i]).stem or "table"
        name, n = base, 2
        while name in out:
            name, n = f"{base}_{n}", n + 1
        df = workspace.load_df(i)
        if df is not None:
            out[name] = df
    return out


def _combine_panel(datasets) -> None:
    names = {d["id"]: d["name"] for d in datasets}
    with st.expander("Combine tables", expanded=False):
        st.caption("Link related tables (orders, items, customers...) into one dataset. Child tables are summarised to one row per record "
                   "of the main table, and nothing is joined until you approve the plan.")
        picked = st.multiselect("Tables to combine", list(names), format_func=lambda k: names[k], key="data-comb-pick")
        frames = _frames(picked, names) if len(picked) >= 2 else {}
        base = st.selectbox("Main table", ["Choose automatically"] + list(frames), key="data-comb-base", disabled=len(frames) < 2)
        if st.button("Propose joins", disabled=len(frames) < 2, key="data-comb-go"):
            try:
                ss.comb_plan = multitable.propose(frames, None if base.startswith("Choose") else base)
            except ValueError as e:
                ss.comb_plan = None
                st.error(str(e))
        plan = ss.get("comb_plan")
        if not plan or len(frames) < 2 or plan["base"] not in frames:
            return
        st.code(multitable.describe(plan), language=None)
        if not plan["joins"]:
            st.warning("No reliable key links these tables, so there is nothing to join. Pick a different main table or different tables.")
            return
        name = st.text_input("Name for the combined dataset", value=f"combined_{plan['base']}", key="data-comb-name")
        if st.button("Approve and create dataset", type="primary", key="data-comb-ok", disabled=not name.strip()):
            try:
                merged = multitable.apply(frames, {**plan, "approved": True})
                meta = workspace.add_dataset(name.strip() + ".csv", merged.to_csv(index=False).encode(),
                                             note="Combined from " + ", ".join(frames))
            except (multitable.JoinGuardError, ValueError) as e:
                st.error(f"Could not combine these tables: {e}")
                return
            ss.comb_plan = None
            ss.ds_id = meta["id"]
            st.rerun()


def render() -> None:
    state.init()
    context.set_view("data", dataset=(workspace.get_dataset(ss.get("ds_id")) or {}).get("name"))
    c.page_header("Data", "Browse, profile and check the datasets in your workspace.")
    show_notice()
    datasets = workspace.list_datasets()
    with st.expander("Add data", expanded=not datasets):
        files = st.file_uploader("Choose files", type=UPLOAD_TYPES, accept_multiple_files=True, key=uploader_key("data"), label_visibility="collapsed")
        if st.button("Add to workspace", type="primary", disabled=not files, key="data-add"):
            if add_files(files, "data"):
                st.rerun()
        link_box("data")
    if not datasets:
        c.empty_state("No datasets yet", "Add a table above and it will appear here, ready to explore.")
        return
    if len(datasets) >= 2:
        _combine_panel(datasets)

    meta, df = state.active_dataset()
    names = {d["id"]: d["name"] for d in datasets}
    state.sync_pick("data-pick", names)                              # after an upload the picker must show the new dataset, not the one chosen before
    head = st.columns([3, 1, 1])
    with head[0]:
        pick = st.selectbox("Dataset", list(names), index=list(names).index(meta["id"]) if meta else 0, format_func=lambda k: names[k], label_visibility="collapsed", key="data-pick")
        if meta is None or pick != meta["id"]:
            ss.ds_id = pick
            st.rerun()
    with head[1]:
        if df is not None:
            st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name=meta["name"], mime="text/csv", use_container_width=True, key="data-dl")
    with head[2]:
        if st.button("Delete", use_container_width=True, key="data-del"):
            _confirm_delete(meta)
    if df is None:
        st.error("This dataset's file is missing from storage.")
        return
    c.stat_cards([("Rows", f"{len(df):,}", ""), ("Columns", df.shape[1], ""), ("Missing cells", f"{100 * df.isna().sum().sum() / max(1, df.size):.1f}%", ""), ("Size", c.fmt_bytes(meta["bytes"]), "")])
    st.write("")
    t_prev, t_cols, t_qual, t_rel = st.tabs(["Preview", "Columns", "Quality", "Relationships"])
    with t_prev:
        q = st.text_input("Search", placeholder="Find rows containing…", key="data-search")
        shown = st.multiselect("Columns", list(df.columns), default=list(df.columns[:12]), key="data-cols")
        view = df[shown or list(df.columns[:12])]
        if q.strip():
            mask = view.astype(str).apply(lambda col: col.str.contains(q.strip(), case=False, regex=False, na=False)).any(axis=1)
            view = view[mask]
            st.caption(f"{len(view):,} matching rows")
        st.dataframe(view.head(1000), use_container_width=True, hide_index=False, height=420, key="data-grid")
        if len(view) > 1000:
            st.caption("Showing the first 1,000 rows. Download the CSV for everything.")
    with t_cols:
        _tool(df, "overview", "dv-ov")
        col = st.selectbox("Look closely at a column", list(df.columns), key="data-col")
        _tool(df, "distribution", "dv-dist", column=col)
    with t_qual:
        _tool(df, "missing", "dv-miss")
        _tool(df, "duplicates", "dv-dup")
        if st.button("Check for outliers", key="data-out"):
            _tool(df, "outliers", "dv-outl")
    with t_rel:
        _tool(df, "correlations", "dv-corr")
        a, b = st.columns(2)
        ca = a.selectbox("First column", list(df.columns), key="data-rel-a")
        cb = b.selectbox("Second column", list(df.columns), index=min(1, len(df.columns) - 1), key="data-rel-b")
        if ca != cb:
            _tool(df, "relationship", "dv-rel", a=ca, b=cb)
