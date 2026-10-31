"""Bounded supervisor: decides WHICH analysis phases a request needs.

It is deliberately limited:
  * it is only consulted when the request looks narrow ("just check outliers", "how many ...", "quick look");
  * the LLM can only choose from a fixed menu; code validates the choice against hard rules;
  * any failure, doubt or invalid answer falls back to the standard full pipeline;
  * at most SELF_CONSISTENCY_N LLM calls per run (default 1); with N > 1 a phase is skipped only if a MAJORITY of samples skip it;
  * the critic gate, retry caps and approval pauses stay deterministic and are never delegated.
"""
import re
from typing import Any, Dict, List

from .. import config, runlog
from ._util import ANALYST_PERSONA, ask

MENU = {
    "media": "analysis of image / document collections (quality, duplicates, themes)",
    "statistics": "hypothesis tests of relationships and group differences",
    "kpis": "business KPI calculation and breakdowns by segment",
    "trends": "trend, seasonality and anomaly analysis over time",
    "charts": "exploratory charts",
    "context": "optional external web context (needs a search key)",
    "modeling": "train/tune predictive models (or clusters) and explain them",
}
NARROW = re.compile(r"\b(just|only|simply|quick(ly)?|briefly|check|list|how many|what (is|was|were)|which|count|average|total|"
                    r"outliers?|distribution|summar(y|ise|ize)|describe|no model|without (a )?model|don'?t (train|model))\b", re.I)
KEEP_MODELING = re.compile(r"\b(predict\w*|forecast\w*|model\w*|classif\w*|cluster\w*|segment\w*|scor(e|ing)|risk|probab\w*)\b", re.I)


NEGATED_MODEL = re.compile(r"\b(no|without( a)?|don'?t (train|use|build|run)( a| any)?|skip( the)?)\s+(predictive\s+)?model\w*", re.I)


def wants_modeling(request: str) -> bool:
    """True if the request asks for prediction/modeling (a negated mention such as 'no model' does not count)."""
    return bool(KEEP_MODELING.search(NEGATED_MODEL.sub("", request or "")))


def looks_narrow(request: str) -> bool:
    return bool(request and NARROW.search(request))


def _default(reason: str) -> Dict[str, Any]:
    return {"skip_phases": [], "source": "default", "rationale": reason, "samples": 0}


def plan_phases(state, problem_type: str) -> Dict[str, Any]:
    req = state.get("user_request", "") or ""
    if not state.get("flexible", True):
        return _default("flexible mode is off: standard pipeline")
    if not looks_narrow(req):
        return _default("request is not narrow: standard pipeline")
    n = max(1, int(config.SELF_CONSISTENCY_N))
    payload = {"request": req, "business_question": state.get("business_question"), "problem_type": problem_type,
               "phases": MENU, "instructions": ["Return the phases that are NEEDED to answer the request. Default to running everything.",
                                                "Skip a phase only if the request clearly does not need it.",
                                                "Never return an empty list."],
               "output_schema": {"run": ["phase_name"], "rationale": "one sentence"}}

    def v(d):
        run = [p for p in dict.fromkeys(d.get("run") or []) if p in MENU]
        if not run:
            raise ValueError(f"'run' must list phases from {list(MENU)}")
        return {"run": run, "rationale": str(d.get("rationale", ""))[:250]}
    votes: List[List[str]] = []
    why = ""
    try:
        for i in range(n):
            pl = {**payload, "sample": i} if n > 1 else payload
            r, _, _ = ask(state, "plan_phases", ANALYST_PERSONA + " You decide which analysis phases a request needs.", pl, 500, v,
                          temperature=0.7 if n > 1 else None)
            votes.append(r["run"])
            why = why or r["rationale"]
    except Exception as e:                                           # LLM failure, stub without this purpose, ... -> standard pipeline
        runlog.decision("supervisor", "default", reason=f"{type(e).__name__}")
        return _default(f"supervisor unavailable ({type(e).__name__}): standard pipeline")
    skip = [p for p in MENU if sum(p not in run for run in votes) > len(votes) / 2]      # strict majority to skip
    # ---- hard rules (code overrides the LLM)
    if wants_modeling(req) and "modeling" in skip:
        skip.remove("modeling")
    if problem_type in ("classification", "regression", "time_series") and "modeling" in skip and not NARROW.search(req):
        skip.remove("modeling")
    analysis = {"statistics", "kpis", "trends", "modeling"}
    if analysis <= set(skip):                               # nothing analytical would remain
        runlog.decision("supervisor", "default", reason="would skip every analysis phase")
        return _default("plan would skip every analysis phase: standard pipeline")
    runlog.decision("supervisor", {"skip": skip}, samples=len(votes))
    return {"skip_phases": skip, "source": "llm", "rationale": why, "samples": len(votes), "votes": votes}
