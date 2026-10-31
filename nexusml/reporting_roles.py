"""Role-based reports: a one-page executive brief (this module) alongside the full analyst report (reporting.build)."""
from typing import Any, Dict

from .reporting import Doc


def build_executive(state: Dict[str, Any], reg) -> tuple:
    d = Doc()
    ins = state.get("insights") or {}
    plan = state["analysis_plan"]
    d.h(1, "Executive brief")
    d.p(f"Question: {state.get('user_request') or state.get('business_question') or ''}")
    d.p(ins.get("headline", ""))
    d.p(ins.get("executive_summary", ""))
    ci = state.get("metric_ci") or {}
    if ci.get("value") is not None:
        rng = f" (95% CI {ci['lo']}-{ci['hi']})" if ci.get("lo") is not None else ""
        d.p(f"How reliable is the prediction model? Held-out {ci['metric']} = {ci['value']}{rng}"
            + (f"; only {ci['n_test_positives']} positive cases were available to test it, so treat the figure as approximate." if ci.get("n_test_positives") is not None and ci["n_test_positives"] < 60 else "."))
    ct = state.get("cost_threshold") or {}
    if ct:
        t = ct["test"]
        d.p(f"Who to act on: using a cutoff of {ct['threshold']}, the model flags {t['flagged']} of the held-out records, catching {t['recall']:.0%} of true cases with {t['precision']:.0%} precision"
            + ("." if ct.get("costs_from_user") else " (based on DEFAULT cost assumptions; replace them with your real costs)."))
    kp = state.get("kpi_results") or {}
    if kp:
        d.table(["KPI", "Value"], [[n, r["overall_fmt"]] for n, r in list(kp.items())[:6]])
    d.h(2, "Top actions")
    d.table(["Priority", "Action", "Why", "Track"], [[r["priority"], r["action"], r["rationale"], r["kpi_to_track"]] for r in (state.get("recommendations") or [])[:3]])
    cav = list(ins.get("caveats", []))
    if state.get("sensitive_columns") and not state.get("exclude_sensitive"):
        cav.append(f"The model uses sensitive attributes {state['sensitive_columns']}; review fairness before using scores about individuals.")
    if any(r.get("unverified") for r in state.get("recommendations", [])):
        cav.append("Some figures in the recommendations could not be matched to the evidence; see the full report.")
    if cav:
        d.h(2, "Read this before acting")
        d.bullets(cav[:5])
    d.p("Findings are associations in observed data, not proof of cause. Full detail, evidence and methods are in the analyst report.")
    return d.markdown(state["run_id"]), d.html(f"Executive brief {state['experiment_id']}", reg)
