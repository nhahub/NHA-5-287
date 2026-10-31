"""Follow-up questions about a finished analysis ("chat with your analysis").
The LLM answers ONLY from the evidence register, insights, root causes and recommendations of that run."""
import pandas as pd
from typing import Any, Dict, List, Tuple

from . import drilldown
from .agents._util import ANALYST_PERSONA, ask, evidence_brief, valid_ids


def _drill(state: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
    """If the question asks for an analysis the run did not do, let the LLM pick up to 3 whitelisted actions and run them."""
    if not drilldown.wants_drilldown(question) or not state.get("dataset_path"):
        return []
    try:
        cols = list(pd.read_csv(state["dataset_path"], nrows=5).columns)
        payload = {"question": question, "columns": cols, "actions": drilldown.ACTIONS,
                   "rules": ["Pick 0-3 actions that would answer the question with NEW numbers; pick none if existing evidence suffices."],
                   "output_schema": {"actions": [{"action": "segment_profile", "column": "col"}]}}
        plan, _, _ = ask(state, "plan_drilldown", ANALYST_PERSONA, payload, 500,
                         lambda d: {"actions": drilldown.validate_actions(d.get("actions"), cols)})
        return drilldown.run(state, plan["actions"]) if plan["actions"] else []
    except Exception:
        return []                                         # drill-down is optional; the normal answer still works


def answer(state: Dict[str, Any], question: str, history: List[Tuple[str, str]]) -> Dict[str, Any]:
    new = _drill(state, question)
    if new:
        state.setdefault("evidence", []).extend(new)      # later questions can cite these too
    ev = state.get("evidence", [])
    payload = {"question": question, "business_question": state.get("business_question"), "analysis_type": state.get("analysis_type"),
               "headline": (state.get("insights") or {}).get("headline"), "executive_summary": (state.get("insights") or {}).get("executive_summary"),
               "root_causes": [{"finding": r["finding"], "hypothesis": r["hypothesis"], "ids": r["evidence_ids"]} for r in state.get("root_causes", [])],
               "recommendations": [{"action": r["action"], "ids": r["evidence_ids"]} for r in state.get("recommendations", [])],
               "kpis": {n: r["overall_fmt"] for n, r in (state.get("kpi_results") or {}).items()},
               "evidence": evidence_brief(ev, 60), "previous_turns": [{"q": q, "a": a[:400]} for q, a in history[-4:]],
               "rules": ["Answer ONLY from the facts provided. If the analysis does not cover the question, say so and suggest what analysis would.",
                         "Cite evidence ids you used. Do not invent numbers. Correlation is not causation.",
                         "Be concise: 2-6 sentences, plain business language."],
               "output_schema": {"answer": "text", "evidence_ids": ["D01"], "follow_ups": ["up to 2 useful next questions"]}}

    def v(d):
        a = str(d.get("answer", "")).strip()
        if not a:
            raise ValueError("provide a non-empty 'answer'")
        return {"answer": a[:1500], "evidence_ids": valid_ids(d.get("evidence_ids"), ev),
                "follow_ups": [str(x)[:120] for x in (d.get("follow_ups") or [])][:2]}
    out, _, _ = ask(state, "followup_question", ANALYST_PERSONA, payload, 700, v)
    out["new_analysis"] = [{"id": e["id"], "text": e["text"]} for e in new]
    return out
