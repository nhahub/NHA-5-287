"""Strategist: when the Critic fails a run, choose NEW training levers (never repeat one).
The LLM picks and justifies the levers; code restricts it to levers that are applicable and unused."""
from typing import Any, Dict, List

import numpy as np

from .. import config
from ._util import ANALYST_PERSONA, ask, log

AGENT = "Strategist"
LEVERS = {
    "class_weight": "Use balanced class weights (classification with a rare class)",
    "log_target": "Model log(1+target) (regression with a skewed, non-negative target)",
    "delta_target": "Model the change vs the previous value (time series with a trend; lets trees forecast levels)",
    "extra_models": "Add ExtraTrees to the candidate set",
    "tune": "Turn on hyper-parameter tuning (randomized search + nested CV)",
    "wider_search": "Double the number of hyper-parameter search iterations (tuning already on)",
    "regularize": "Use stronger regularisation (smaller C / larger alpha / shallower, smoother trees) to reduce overfitting",
}
def _applicable(state, applied: List[str]) -> List[str]:
    pt = state["analysis_plan"]["problem_type"]
    s = state.get("strategy") or {}
    ok = []
    for lv in LEVERS:
        if lv in applied:
            continue
        if lv == "class_weight" and pt != "classification":
            continue
        if lv == "log_target" and (pt != "regression" or state.get("data_summary", {}).get("target_min", -1) < 0):
            continue
        if lv == "delta_target" and pt != "time_series":
            continue
        if lv == "tune" and s.get("tune"):
            continue
        if lv == "wider_search" and not s.get("tune"):
            continue
        ok.append(lv)
    return ok


def revise_strategy_node(state) -> Dict[str, Any]:
    applied = list(state.get("strategies_applied", []))
    avail = _applicable(state, applied)
    s = dict(state.get("strategy") or {})
    if not avail:
        return {"strategy_exhausted": True, "active_agent": AGENT, "current_status": "No new strategy available.",
                "agent_log": [log(AGENT, "No untried applicable lever left; proceeding with the best result.")]}
    rng = np.random.RandomState(int(state.get("seed", config.RANDOM_STATE)) * 1000 + int(state.get("retry_count", 0)))
    if state.get("explore"):                                 # seeded exploration: vary which levers the LLM sees first
        avail = [str(x) for x in rng.permutation(avail)]
    payload = {"problem_type": state["analysis_plan"]["problem_type"], "issues": state["audit"]["issues"],
               "already_tried": applied, "available_levers": {k: LEVERS[k] for k in avail},
               "recent_attempts": state.get("attempts", [])[-3:], "max_levers": 2,
               "output_schema": {"levers": ["lever_name"], "rationale": "why these will address the failure"}}

    def v_lev(d):
        pk = [lv for lv in dict.fromkeys(d.get("levers") or []) if lv in avail][:2]
        if not pk:
            raise ValueError(f"choose 1-2 levers from {avail}")
        return {"levers": pk, "rationale": str(d.get("rationale", ""))[:250]}

    ch, warns, calls = ask(state, "choose_retry_strategy", ANALYST_PERSONA + " You are the ML strategist diagnosing a failed model run.", payload, 600, v_lev)
    pick, why, by = ch["levers"], ch["rationale"], "LLM"
    if state.get("explore") and len(pick) < 2 and len(avail) > len(pick):
        extra = [x for x in avail if x not in pick]
        pick = pick + [str(rng.choice(extra))]
        why = (why + " " if why else "") + "(+1 lever added by seeded exploration)"
    for lv in pick:
        if lv == "class_weight":
            s["class_weight"] = "balanced"
        elif lv == "regularize":
            s["regularize"] = True
        elif lv == "wider_search":
            s["tune_n_iter"] = min(48, int(s.get("tune_n_iter") or config.TUNE_N_ITER) * 2)
        else:
            s[lv] = True
    return {"strategy": s, "strategies_applied": pick, "retry_count": state.get("retry_count", 0) + 1, "warnings": warns,
            "llm_calls": calls, "current_status": "Strategy revised.", "active_agent": AGENT,
            "agent_log": [log(AGENT, f"Retry {state.get('retry_count', 0) + 1}: applying {pick} (chosen by {by})" + (f": {why}" if why else "."))]}
