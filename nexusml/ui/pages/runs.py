"""Analyses: every analysis that has run (or is running), and the full results of each."""
from pathlib import Path

from ... import jobs, runstore
from ...storage import get_registry
from .. import components as c
from .. import legacy, nav, state
from .._st import ss, st

TERMINAL_SHOWN = ("running", "queued", "needs_input", "failed")


def _metric_text(r) -> str:
    if r.get("heldout") is None:
        return "–"
    ci = f" ({r['ci_lo']:.3f} to {r['ci_hi']:.3f})" if r.get("ci_lo") is not None and r.get("ci_hi") is not None else ""
    return f"{r.get('primary_metric') or 'score'} {r['heldout']:.3f}{ci}"


def _open(exp: str) -> None:
    ss.open_run = ss.view_run = exp
    st.rerun()


def _list() -> None:
    c.page_header("Analyses", "Every analysis you have run, and the ones running now.")
    for j in [j for j in jobs.active() if j["status"] in TERMINAL_SHOWN][:4]:
        with st.container(border=True):
            cols = st.columns([5, 2, 1])
            cols[0].markdown(f"**{j['request'][:80] or 'Analysis'}**")
            cols[0].caption(f"{j['dataset_name']} · {j['stage_label'] or 'starting'}")
            cols[1].markdown(c.badge(j["status"].replace("_", " ").capitalize(), c.status_tone(j["status"])), unsafe_allow_html=True)
            with cols[2]:
                if st.button("View", key=f"job-{j['id']}", use_container_width=True):
                    ss.job_id = j["id"]
                    nav.go("analyze")
    runs = runstore.list_runs(200)
    f1, f2 = st.columns([3, 1])
    q = f1.text_input("Search", placeholder="Search by question, target or model", key="runs-q", label_visibility="collapsed")
    only = f2.toggle("Passed review only", key="runs-pass")
    if q.strip():
        k = q.strip().lower()
        runs = [r for r in runs if k in " ".join(str(r.get(x) or "") for x in ("request", "target", "best_model", "analysis_type")).lower()]
    if only:
        runs = [r for r in runs if r.get("passed")]
    if not runs:
        if c.empty_state("No analyses yet", "Start one from New analysis, or ask the assistant to propose one.", "New analysis", key="runs-empty"):
            nav.go("analyze")
        return
    for r in runs[:60]:
        with st.container(border=True):
            cols = st.columns([5, 3, 2, 1])
            cols[0].markdown(f"**{(r['request'] or 'Analysis')[:90]}**")
            cols[0].caption(f"{Path(r['dataset'] or '').name or '–'} · {c.time_ago(r['created'])}")
            cols[1].caption(f"{r.get('target') or '–'} · {r.get('best_model') or '–'}")
            cols[1].caption(_metric_text(r))
            cols[2].markdown(c.badge("Passed review" if r.get("passed") else "Needs attention", "ok" if r.get("passed") else "warn"), unsafe_allow_html=True)
            with cols[3]:
                if st.button("Open", key=f"run-{r['run_id']}", use_container_width=True):
                    _open(r["run_id"])


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
    gr = (values.get("automl_report") or {}).get("goal") or {}
    c.page_header((values.get("user_request") or "Analysis")[:110], f"{Path(values.get('dataset_path') or '').name or 'dataset'} · {values.get('best_model_name') or 'no model'}")
    badges = [c.goal_badge(gr.get("status")) if gr.get("status") not in (None, "no_goal") else "", c.badge("Passed review", "ok") if values.get("quality_passed") else c.badge("Needs attention", "warn")]
    st.markdown(" ".join(b for b in badges if b), unsafe_allow_html=True)
    act = st.columns([2, 2, 4])
    with act[0]:
        if st.button("Ask the assistant", key="run-ask", use_container_width=True):
            nav.go("assistant")
    with act[1]:
        html_key = values.get("report_html_key")
        p = get_registry().path_of(html_key) if html_key else None
        if p and p.exists():
            st.download_button("Download report", p.read_bytes(), file_name=f"report_{exp}.html", mime="text/html", use_container_width=True, key="run-report")
    st.write("")
    with legacy.guarded("Results"):
        legacy.result_view(values, f"run{exp[-8:]}")


def render() -> None:
    state.init()
    pending = ss.pop("view_run", None)
    if pending:
        st.query_params["run"] = pending
    exp = st.query_params.get("run")
    _detail(exp) if exp else _list()
