"""Context Researcher: optional web context (Tavily) for the root-cause step.

Runs only when TAVILY_API_KEY is set. The LLM writes up to 3 generic, public-context search queries
(never dataset values). Results become evidence ids W01.. clearly labelled as EXTERNAL and unverified:
background that may help explain a pattern, never proof."""
from typing import Any, Dict, List

from .. import external
from ._util import ANALYST_PERSONA, ask, jsonable, log, make_evidence

AGENT = "Context Researcher"


def context_research_node(state) -> Dict[str, Any]:
    if not external.available():
        return {"external_context": {"status": "skipped", "reason": "TAVILY_API_KEY is not set (optional web context disabled)."},
                "current_status": "Web context skipped.", "active_agent": AGENT,
                "agent_log": [log(AGENT, "Skipped: no TAVILY_API_KEY (optional).")]}
    tr = state.get("trend_results") or {}
    payload = {"business_question": state.get("business_question"), "analysis_type": state.get("analysis_type"),
               "problem_type": state["analysis_plan"]["problem_type"], "column_names": list(state["dataset_profile"]["columns"])[:40],
               "kpis": list((state.get("kpi_results") or {}))[:6],
               "period": [tr.get("start"), tr.get("end")] if tr and "skipped" not in tr else None,
               "anomaly_dates": [a["period"] for a in (tr.get("anomalies") or [])[:3]],
               "rules": ["Write 1-3 web search queries about PUBLIC context that could explain the business pattern "
                         "(industry trends, seasonality, regulation, market events, benchmarks).",
                         "NEVER include numbers, names, IDs or any values from the dataset; use generic wording.",
                         "If the data suggests no useful external context, return an empty list."],
               "output_schema": {"queries": ["query text"]}}

    def v_q(d):
        qs = [str(q).strip()[:140] for q in (d.get("queries") or []) if isinstance(q, str) and len(str(q).strip()) >= 8][:3]
        return qs
    qs, _, calls = ask(state, "plan_context_queries", ANALYST_PERSONA, payload, 500, v_q)
    results: List[Dict[str, Any]] = []
    warns: List[str] = []
    for q in qs:
        try:
            for r in external.search(q, 4):
                results.append({**r, "query": q})
        except external.ExternalError as e:
            warns.append(f"[Web context] {e}; continuing without it.")
            break
    items = []
    seen = set()
    for r in results:
        if r["url"] in seen or not r["snippet"]:
            continue
        seen.add(r["url"])
        items.append(("external_web", f"EXTERNAL context (unverified, from {r['domain']}): {r['title']}: {r['snippet']}", {"url": r["url"], "query": r["query"]}))
    items = items[:8]
    return {"external_context": jsonable({"status": "ok" if items else "no_results", "queries": qs, "sources": [i[2] for i in items]}),
            "evidence": make_evidence("W", items), "warnings": warns, "llm_calls": calls,
            "current_status": "Web context gathered.", "active_agent": AGENT,
            "agent_log": [log(AGENT, f"{len(qs)} generic quer(ies) sent to Tavily; {len(items)} external source(s) added as evidence (labelled unverified).")]}
