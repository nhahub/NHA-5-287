"""Analyses: the composer and history landing, and the full results of one analysis when it is open."""
from pathlib import Path

from ... import lineage, runstore
from ...storage import get_registry
from .. import components as c
from .. import context, history, legacy, nav, state
from .._st import ss, st

def _metric_text(r) -> str:
    if r.get("heldout") is None:
        return "–"
    ci = f" ({r['ci_lo']:.3f} to {r['ci_hi']:.3f})" if r.get("ci_lo") is not None and r.get("ci_hi") is not None else ""
    return f"{r.get('primary_metric') or 'score'} {r['heldout']:.3f}{ci}"


def _lineage(exp: str) -> None:
    fam = lineage.family(exp)
    bits = []
    if fam["parent"]:
        p = runstore.get_run(fam["parent"])
        bits.append(("Built on", fam["parent"], (p or {}).get("request")))
    elif fam["parent_deleted"]:
        st.caption("Built on an earlier analysis that has since been deleted. This analysis keeps its own copy of everything it reused.")
    for kid in fam["children"][:5]:
        k = runstore.get_run(kid)
        bits.append(("Follow-up", kid, (k or {}).get("request")))
    for label, rid, req in bits:
        a, b = st.columns([6, 1])
        a.caption(f"{label}: {(req or 'Analysis')[:90]}")
        if b.button("Open", key=f"lin-{rid}", use_container_width=True):
            history.open_run(rid)


def _notes(exp: str) -> None:
    notes = runstore.list_notes(exp)
    with st.expander(f"Notes and concerns ({len(notes)})", expanded=bool(notes)):
        for n in notes:
            st.markdown(f"- {n['text']}")
        if not notes:
            st.caption("Write down a doubt or a question to come back to. The assistant can see these notes, and you can also tell it a concern directly.")
        with st.form(f"note-{exp}", clear_on_submit=True):
            text = st.text_input("Add a note", placeholder="For example: the score seems too high for this kind of data", key=f"note-text-{exp}")
            if st.form_submit_button("Save note") and text.strip():
                runstore.add_note(exp, text.strip(), "concern")
                st.rerun()


def _detail(exp: str) -> None:
    values = state.load_run(exp)
    top = st.columns([1, 5])
    with top[0]:
        if st.button("Back", key="run-back"):
            st.query_params.clear()
            ss.open_run = None
            st.rerun()
    if not values:
        st.error("This analysis could not be found. It may have been removed.")
        return
    ss.open_run = exp
    context.set_view("analysis", run=exp, request=(values.get("user_request") or "")[:120], target=(values.get("analysis_plan") or {}).get("target"))
    gr = (values.get("automl_report") or {}).get("goal") or {}
    c.page_header((values.get("user_request") or "Analysis")[:110], f"{Path(values.get('dataset_path') or '').name or 'dataset'} · {values.get('best_model_name') or 'no model'}")
    badges = [c.goal_badge(gr.get("status")) if gr.get("status") not in (None, "no_goal") else "", c.badge("Passed review", "ok") if values.get("quality_passed") else c.badge("Needs attention", "warn")]
    st.markdown(" ".join(b for b in badges if b), unsafe_allow_html=True)
    entry = {"kind": "run", "id": exp, "request": values.get("user_request"), "status": "done"}
    act = st.columns(5)
    with act[0]:
        if st.button("Build on this", key="run-extend", type="primary", use_container_width=True, icon=":material/add_circle:"):
            history.extend(exp)
    with act[1]:
        if st.button("Retry with changes", key="run-retry", use_container_width=True, icon=":material/replay:"):
            history.retry_dialog(entry)
    with act[2]:
        if st.button("Ask the assistant", key="run-ask", use_container_width=True, icon=":material/forum:"):
            nav.go("assistant")
    with act[3]:
        html_key = values.get("report_html_key")
        p = get_registry().path_of(html_key) if html_key else None
        if p and p.exists():
            st.download_button("Download report", p.read_bytes(), file_name=f"report_{exp}.html", mime="text/html", use_container_width=True, key="run-report")
    with act[4]:
        history.delete_menu(entry, kp="run")
    _lineage(exp)
    _notes(exp)
    st.write("")
    with legacy.guarded("Results"):
        legacy.result_view(values, f"run{exp[-8:]}")


def render() -> None:
    """The Analyses page: one analysis in full when a run is open, otherwise the composer with the history rail."""
    state.init()
    pending = ss.pop("view_run", None)
    if pending:
        st.query_params["run"] = pending
    exp = st.query_params.get("run")
    if exp:
        _detail(exp)
    else:
        from . import analyze
        analyze.render()
