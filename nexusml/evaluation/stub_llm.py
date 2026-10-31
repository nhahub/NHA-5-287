"""A deterministic, well-behaved stand-in for the LLM, built from each prompt's own payload. It lets the whole pipeline (and the
evaluation harness) run offline and reproducibly. `frame` describes how the 'LLM' frames the business problem for one dataset."""
import json
from typing import Any, Callable, Dict, Optional

from ..agents.supervisor import MENU


def _stub_round(pl: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic AutoML research lead: drop quarantined columns, try untried recipes, refine the leader, stop when the goal is met or rounds stop paying."""
    acts = []
    for m in pl.get("messages_from_other_agents", []):
        if m.get("kind") == "quarantine_proposal" and m.get("columns"):
            acts.append({"type": "drop_features", "params": {"columns": m["columns"]}, "why": "sentinel flagged leakage"})
    if any(m.get("kind") == "feature_ideas" for m in pl.get("messages_from_other_agents", [])):
        acts.append({"type": "try_ideas", "params": {}, "why": "scout proposed ideas"})
    for r in (pl.get("recipes_untried_and_applicable") or [])[:2]:
        acts.append({"type": "try_recipe", "params": {"recipe": r}, "why": f"try {r}"})
    lb = pl.get("leaderboard") or []
    if lb:
        acts.append({"type": "search_family", "params": {"family": lb[0]["family"], "n_trials": 4, "focus": "exploit"}, "why": "refine the leader"})
    if pl.get("budget", {}).get("rounds_left", 1) <= 1 or pl.get("plateau_rounds", 0) >= 1:
        acts.append({"type": "ensemble", "params": {"method": "greedy", "size": 15}, "why": "average the strong models"})
    stop = None
    if pl.get("goal", {}).get("estimate_meets_goal"):
        stop = {"reason": "goal_met", "why": "the search estimate meets the goal"}
    elif pl.get("plateau_rounds", 0) >= 2 or (not acts):
        stop = {"reason": "plateau", "why": "no further gain in recent rounds"}
    return {"diagnosis": "Stub research lead: follow the standard improvement ladder.", "actions": acts[:4], "stop": stop}


def _stub_scout(pl: Dict[str, Any]) -> Dict[str, Any]:
    nums = [c["name"] for c in pl.get("columns", []) if c.get("kind") == "numeric" and c.get("n_unique", 0) > 10 and c.get("name") != pl.get("target")]
    ideas = [{"op": "ratio", "cols": nums[:2], "name": f"{nums[0]}_over_{nums[1]}", "why": "stub ratio"}] if len(nums) >= 2 else []
    return {"ideas": ideas, "hypotheses": ["stub hypothesis"], "rationale": "Stub scout: one ratio of the first two numeric columns."}


def _stub_assistant(pl: Dict[str, Any]) -> Dict[str, Any]:
    """A deterministic stand-in for the workspace assistant (test double only): keyword -> one tool, then a reply built from the observation."""
    obs, msg = pl.get("observations") or [], str(pl.get("message", "")).lower()
    cols = (pl.get("dataset") or {}).get("column_list", [])
    num = [c["name"] for c in cols if c["type"] == "numeric" and c["distinct"] > 8]
    dts = [c["name"] for c in cols if c["type"] == "datetime"]
    if pl.get("problem_with_your_last_reply"):
        last = obs[-1]["summary"] if obs else {}
        return {"reply": f"Here is what the data shows: {json.dumps({k: v for k, v in last.items() if isinstance(v, (int, str))})[:200]}.", "followups": []}
    if obs:
        o = obs[-1]
        if "error" in o:
            return {"reply": f"I could not do that: {o['error']}", "followups": []}
        sm = o["summary"]
        facts = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sm.items() if isinstance(v, (int, float)) and not k.startswith("_"))[:240]
        return {"reply": f"**{o['title']}.** {facts}.", "followups": ["Show me the rows", "Plot this"]}
    if not (pl.get("dataset")) and any(w in msg for w in ("outlier", "missing", "correl", "trend", "overview")):
        return {"reply": "I need a dataset first: add one on the Data page and ask again."}
    pick = None
    if "outlier" in msg: pick = ("outliers", {})
    elif "missing" in msg: pick = ("missing", {})
    elif "correl" in msg: pick = ("correlations", {})
    elif "overview" in msg or "describe the data" in msg: pick = ("overview", {})
    elif ("trend" in msg or "over time" in msg or "change" in msg) and dts and num: pick = ("time_trend", {"date": dts[0], "value": num[0], "agg": "sum"})
    elif "drive" in msg and num: pick = ("drivers", {"target": num[-1]})
    elif "differ by" in msg and num: pick = ("relationship", {"a": num[0], "b": next((c["name"] for c in cols if c["type"] == "categorical"), num[0])})
    elif "full analysis" in msg or "build a model" in msg: pick = ("propose_analysis", {"objective": pl.get("message", "")})
    elif "ghost" in msg: pick = ("outliers", {"columns": ["ghost_column"]})
    if pick:
        return {"reasoning": "stub routing", "action": {"tool": pick[0], "args": pick[1]}}
    return {"reply": "Happy to help. Ask me about your data, or about how to approach an analysis.", "followups": []}


def v6_answer(purpose: str, pl: Dict[str, Any]):
    """Deterministic answers for the v6 decision points (Director brief, sentinel review, scout, AutoML round, route verdict). None for any other purpose."""
    if purpose == "brief_crew":
        return {"crew": {"sentinel": True, "scout": True, "explore": True}, "automl": {"max_seconds": None, "max_rounds": 3, "workers": 1, "style": "balanced"}, "goal": None,
                "rationale": "Stub director: deploy the whole crew with a balanced budget."}
    if purpose == "automl_round":
        return _stub_round(pl)
    if purpose == "scout_ideas":
        return _stub_scout(pl)
    if purpose == "sentinel_review":
        return {"decisions": [{"column": c["column"], "action": "quarantine" if c.get("severity") == "high" else "keep", "why": "stub"} for c in pl.get("suspects", [])],
                "rationale": "Stub sentinel: quarantine the high-severity suspects."}
    if purpose == "assistant_step":
        return _stub_assistant(pl)
    if purpose == "route_verdict":
        return {"route": "accept", "directive": "", "rationale": "Stub director accepts the result."}
    return None


def make_stub(frame: Dict[str, Any], overrides: Optional[Dict[str, Callable]] = None) -> Callable:
    overrides = overrides or {}

    def fake(system, prompt, purpose="", max_tokens=0, temperature=0):
        rec = {"purpose": purpose, "model": "stub", "ok": True, "seconds": 0.0, "tokens": 10}
        pl = json.loads(prompt)
        ev = [e["id"] for e in pl.get("evidence", [])]
        if purpose in overrides:
            return overrides[purpose](pl), None, rec
        ov = pl.get("user_choices_you_must_respect", {})
        fr = {**frame, "business_question": "Q?", "hypotheses": ["H1"]}
        if ov.get("target_column"):
            fr["target"] = ov["target_column"]
        if ov.get("problem_type"):
            fr["problem_type"] = ov["problem_type"]
        d = {
            "preflight": {"ok": True},
            "frame_business_problem": fr,
            "select_statistical_tests": {"selected": [{"id": c["id"], "reason": "relevant"} for c in pl.get("candidates", [])[:6]]},
            "review_column_drops": {"reviews": [{"column": a["column"], "verdict": "drop", "why": "not useful"} for a in pl.get("proposed_drops", [])]},
            "advise_hyperparameter_ranges": {"spaces": {}, "rationale": "defaults suit this sample size"},
            "choose_retry_strategy": {"levers": list(pl.get("available_levers", {}))[:1], "rationale": "address the failure"},
            "name_segments": {"segments": [{"cluster": s["cluster"], "name": f"Segment {s['cluster']}", "description": "desc"} for s in pl.get("segments", [])]},
            "root_cause_analysis": {"root_causes": [{"finding": "f", "hypothesis": "h", "evidence_ids": ev[:2], "confidence": "medium", "validation_step": "pilot"}]},
            "business_insights": {"headline": "Headline", "executive_summary": "Summary.", "key_insights": [{"title": "T", "insight": "I", "so_what": "S", "evidence_ids": ev[:2]}], "caveats": []},
            "recommendations": {"recommendations": [{"action": "Act", "rationale": "r", "evidence_ids": ev[:1], "priority": "high", "effort": "low",
                                                     "expected_impact": "i", "kpi_to_track": "k", "risk": "r", "owner": "o"}]},
            "powerbi_report_layout": {"pages": [{"title": "Overview", "visuals": [{"type": "card", "table": "KPI_Summary", "fields": ["kpi", "value"], "title": "KPIs"}]}]},
            "plan_phases": {"run": list(MENU), "rationale": "stub runs everything"},
            "plan_drilldown": {"actions": []},
            "name_doc_topics": {"names": [{"theme": t["theme"], "name": f"Theme {t['theme']}"} for t in pl.get("themes", [])]},
            "document_question": {"answer": "From the passages.", "sources": [p["document"] for p in pl.get("passages", [])][:1]},
            "general_chat": {"answer": "General answer.", "needs_data": False, "next_steps": []},
            "summarize_document": {"summary": "Stub summary.", "key_points": ["p1"], "action_items": [], "topics": ["t"], "questions_to_ask": ["q?"]},
            "summarize_chunk": {"summary": "part", "key_points": ["k"]},
            "followup_question": {"answer": "From the evidence.", "evidence_ids": ev[:1], "follow_ups": []},
        }
        if purpose not in d:
            v6 = v6_answer(purpose, pl)
            if v6 is None:
                raise KeyError(f"stub has no answer for purpose {purpose!r}")
            return v6, None, rec
        return d[purpose], None, rec
    return fake
