"""The history of analyses as a compact rail, and the actions on each entry: open, extend, retry (with a changed question) and delete.

Three kinds of entry share one list: finished analyses (saved, with results), analyses running in this session, and analyses that did not finish (failed, cancelled or
waiting for an answer; kept so they can be retried or removed even after a restart).
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import jobs, lineage, runstore, workspace
from . import components as c
from . import context
from ._st import ss, st
from .options import PROBLEMS

UNFINISHED = ("failed", "cancelled", "needs_input")
LABEL = {"running": "Running", "queued": "Queued", "done": "Finished", "failed": "Failed", "cancelled": "Cancelled", "needs_input": "Needs input"}


def entries(query: str = "", limit: int = 80) -> List[Dict[str, Any]]:
    """Every analysis worth showing, newest first: {kind: run|job|attempt, id, request, dataset, when, status, detail, parent}."""
    out: List[Dict[str, Any]] = []
    live = {j["id"]: j for j in jobs.active() if j["status"] in ("running", "queued", "needs_input", "failed", "cancelled")}
    for j in live.values():
        out.append({"kind": "job", "id": j["id"], "request": j["request"], "dataset": j["dataset_name"], "when": j.get("started") or 0, "status": j["status"],
                    "detail": j["stage_label"] or "", "parent": j.get("parent")})
    for a in runstore.list_attempts(60):
        if a["id"] not in live:
            out.append({"kind": "attempt", "id": a["id"], "request": a["request"], "dataset": a["dataset_name"], "when": a["created"], "status": a["status"],
                        "detail": (a["error"] or a["message"] or "")[:90], "parent": a.get("parent")})
    for r in runstore.list_runs(200):
        score = f"{r.get('primary_metric') or 'score'} {r['heldout']:.3f}" if r.get("heldout") is not None else (r.get("best_model") or "No model")
        out.append({"kind": "run", "id": r["run_id"], "request": r["request"], "dataset": Path(r.get("dataset") or "").name, "when": r["created"], "status": "done",
                    "detail": score, "passed": bool(r.get("passed")), "parent": runstore.parent_of(r["run_id"])})
    q = query.strip().lower()
    if q:
        out = [e for e in out if q in f"{e['request']} {e['dataset']} {e['detail']}".lower()]
    return sorted(out, key=lambda e: -(e["when"] or 0))[:limit]


# ---- actions
def open_run(exp: str) -> None:
    ss.open_run = ss.view_run = exp
    st.rerun()


def extend(exp: str) -> None:
    ss.extend_from = exp
    ss.prefill = None
    ss.job_id = None
    ss.open_run = None
    try:
        st.query_params.clear()
    except Exception:
        pass
    st.rerun()


def delete(entry: Dict[str, Any]) -> None:
    if entry["kind"] == "run":
        lineage.delete_run(entry["id"])
    else:
        lineage.delete_attempt(entry["id"])
    if ss.get("open_run") == entry["id"]:
        ss.open_run = None
    if ss.get("job_id") == entry["id"]:
        ss.job_id = None
    if ss.get("extend_from") == entry["id"]:
        ss.extend_from = None
    if st.query_params.get("run") == entry["id"]:
        st.query_params.clear()
    st.toast("Deleted.")
    st.rerun()


def clear_unfinished() -> int:
    n = 0
    for a in runstore.list_attempts(500):
        if a["status"] in UNFINISHED:
            lineage.delete_attempt(a["id"])
            n += 1
    return n


def _dataset_for(path: Optional[str]) -> Dict[str, Any]:
    for d in workspace.list_datasets():
        try:
            if path and Path(workspace.dataset_path(d["id"])).resolve() == Path(path).resolve():
                return d
        except OSError:
            continue
    return {}


def retry_dialog(entry: Dict[str, Any]) -> None:
    """Run an analysis again, with the question (and optionally the target and kind) changed."""
    spec = lineage.retry_spec(exp=entry["id"]) if entry["kind"] == "run" else lineage.retry_spec(attempt_id=entry["id"])

    @st.dialog("Retry this analysis", width="large")
    def _dialog() -> None:
        if not spec:
            st.error("The details of this analysis are no longer available, so it cannot be retried.")
            return
        meta = _dataset_for(spec.get("dataset_path")) or (workspace.get_dataset(spec.get("dataset_id")) or {})
        path = spec.get("dataset_path") or (workspace.dataset_path(meta["id"]) if meta else None)
        if not path or not Path(str(path)).exists():
            st.error("The dataset this analysis used is no longer in the workspace. Add it again on the Data page.")
            return
        st.caption(f"Dataset: {meta.get('name') or Path(str(path)).name}. Change the question below, or run it again as it was.")
        kw = dict(spec["kwargs"])
        request = st.text_area("What do you want to find out?", value=spec["request"], height=110, key=f"rt-req-{entry['id']}")
        df = workspace.load_df(meta["id"]) if meta else None
        cols = list(df.columns) if df is not None else []
        a, b = st.columns(2)
        keep = "Keep as before"
        t_opts = [keep, "Decide for me"] + cols
        target = a.selectbox("Column to predict", t_opts, key=f"rt-target-{entry['id']}")
        kinds = [keep] + list(PROBLEMS)
        kind = b.selectbox("Kind of analysis", kinds, key=f"rt-kind-{entry['id']}")
        reuse = False
        if entry["kind"] == "run":
            reuse = st.checkbox("Keep results from the earlier analysis that are still valid (faster)", value=False, key=f"rt-reuse-{entry['id']}",
                                help="Reuses the data profile, KPIs and trends when the dataset has not changed. The model search and the report are always redone.")
        if target != keep:
            kw.pop("target_column", None)
            if target != "Decide for me":
                kw["target_column"] = target
        if kind != keep:
            kw.pop("problem_type", None)
            if PROBLEMS[kind]:
                kw["problem_type"] = PROBLEMS[kind]
        go = st.button("Run again", type="primary", key=f"rt-go-{entry['id']}", disabled=not request.strip())
        if go:
            if reuse:
                kw["resume_from"] = entry["id"]
            elif spec.get("parent") and entry["kind"] != "run":
                kw["resume_from"] = spec["parent"]
            try:
                ss.job_id = jobs.submit(str(path), request.strip(), meta.get("id"), meta.get("name") or Path(str(path)).name, options=spec.get("options") or {}, **kw)
            except Exception as e:                                   # e.g. a resume that is no longer possible: say why instead of failing silently
                st.error(str(e)[:300])
                return
            if entry["kind"] != "run":
                lineage.delete_attempt(entry["id"])                   # the new run replaces the attempt that did not finish
            ss.open_run = None
            st.query_params.clear()
            st.rerun()
    _dialog()


def delete_menu(e: Dict[str, Any], kp: str = "h") -> None:
    """Delete with an explicit second click (the confirmation lives inside the popover)."""
    with st.popover("Delete", icon=":material/delete:", use_container_width=True):
        kids = runstore.children_of(e["id"]) if e["kind"] == "run" else []
        st.caption("Deleting removes the results, report, charts and model of this analysis. It cannot be undone." if e["kind"] == "run" else "Deleting removes this entry.")
        if kids:
            st.caption(f"{len(kids)} later analysis(es) were built on this one. They keep their own results.")
        if st.button("Delete permanently", key=f"{kp}-del-{e['id']}", type="primary", use_container_width=True):
            delete(e)


def manage_menu(e: Dict[str, Any], kp: str = "h") -> None:
    """Extend / Retry / Delete for one entry, inside a popover so the rail stays compact. Delete needs a second, explicit click."""
    with st.popover("Manage", icon=":material/more_horiz:", use_container_width=True):
        kids = [r for r in runstore.children_of(e["id"])] if e["kind"] == "run" else []
        if e["kind"] == "run" and st.button("Build on this", key=f"{kp}-ext-{e['id']}", use_container_width=True, icon=":material/add_circle:"):
            extend(e["id"])
        if e["kind"] != "job" or e["status"] in UNFINISHED:
            if st.button("Retry with changes", key=f"{kp}-retry-{e['id']}", use_container_width=True, icon=":material/replay:"):
                retry_dialog(e)
        if e["kind"] == "job" and e["status"] in ("running", "queued"):
            if st.button("Cancel", key=f"{kp}-cancel-{e['id']}", use_container_width=True):
                jobs.cancel(e["id"])
                st.rerun()
            return
        st.divider()
        st.caption("Deleting removes the results, report, charts and model of this analysis." if e["kind"] == "run" else "Deleting removes this entry.")
        if kids:
            st.caption(f"{len(kids)} later analysis(es) were built on this one. They keep their own results.")
        if st.button("Delete", key=f"{kp}-del-{e['id']}", type="primary", use_container_width=True, icon=":material/delete:"):
            delete(e)


def _entry(e: Dict[str, Any]) -> None:
    with st.container(border=True):
        title = (e["request"] or "Analysis").strip()
        st.markdown(f"**{title[:58]}{'…' if len(title) > 58 else ''}**")
        tone = c.status_tone(e["status"])
        bits = [LABEL.get(e["status"], e["status"]), e["dataset"] or "", c.time_ago(e["when"]) if e["when"] else ""]
        st.markdown(c.badge(bits[0], tone) + f" <span class='nx-muted'>{' · '.join(b for b in bits[1:] if b)}</span>", unsafe_allow_html=True)
        if e.get("detail"):
            st.caption(e["detail"][:70])
        if e.get("parent"):
            st.caption("Follow-up" + ("" if runstore.get_run(e["parent"]) else " (earlier analysis deleted)"))
        a, b = st.columns(2)
        with a:
            if e["kind"] == "run":
                if st.button("Open", key=f"h-open-{e['id']}", use_container_width=True):
                    open_run(e["id"])
            elif e["kind"] == "job":
                if st.button("View", key=f"h-view-{e['id']}", use_container_width=True):
                    ss.job_id = e["id"]
                    st.rerun()
            else:
                st.button("Open", key=f"h-open-{e['id']}", use_container_width=True, disabled=True)
        with b:
            manage_menu(e)


def render_rail() -> None:
    """The compact history column of the Analyses page."""
    c.eyebrow("History")
    q = st.text_input("Search", placeholder="Search analyses", key="hist-q", label_visibility="collapsed")
    items = entries(q)
    if not items:
        st.caption("Nothing here yet. Your analyses will be listed as you run them." if not q.strip() else "No analysis matches that search.")
        return
    box = st.container(height=640, border=False)
    with box:
        for e in items:
            _entry(e)
    stale = [e for e in items if e["kind"] != "run" and e["status"] in UNFINISHED]
    if stale:
        with st.popover(f"Clear {len(stale)} unfinished", use_container_width=True):
            st.caption("Removes analyses that failed, were cancelled or never got an answer. Finished analyses are not touched.")
            if st.button("Clear them", key="hist-clear", type="primary", use_container_width=True):
                clear_unfinished()
                st.rerun()
