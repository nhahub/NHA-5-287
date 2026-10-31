"""LLM decision points ("the Director").

A deterministic router answers "is this valid / is there budget left?". The Director answers "what is the SMARTEST thing to do next?", which needs judgement:
which crew to deploy, how to split the budget, whether to refine or accept, which columns to quarantine. Pattern at every junction:

   facts (deterministic)  ->  LLM picks among the options the guards ALLOW  ->  guards re-validate the pick  ->  the choice and its reasoning are logged

If the LLM cannot answer (outage, unusable output twice) the junction falls back to its deterministic default and says so (`source` = "default"), unless
NEXUSML_STRICT_LLM=1, which makes any failed decision stop the run like every other LLM call does.
"""
import os
import time
from typing import Any, Callable, Dict, List, Tuple

from . import llm
from .agents._util import ask


def strict() -> bool:
    return os.environ.get("NEXUSML_STRICT_LLM", "0") == "1"


def decide(state, junction: str, persona: str, payload: Dict[str, Any], validate: Callable[[Any], Any], default: Callable[[], Dict[str, Any]],
           max_tokens: int = 1200) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]], List[str]]:
    """-> (decision, source, llm_call_records, warnings). `source` is 'llm' or 'default'."""
    try:
        data, warns, calls = ask(state, junction, persona, payload, max_tokens, validate)
        return data, "llm", calls, warns
    except llm.LLMRequired as e:
        if strict():
            raise
        return default(), "default", [], [f"Decision '{junction}' used the deterministic default because the LLM could not answer ({str(e)[:140]})."]


def record(junction: str, source: str, decision: Any, rationale: str = "", options: Any = None, guard: str = "") -> Dict[str, Any]:
    return {"junction": junction, "source": source, "decision": decision, "rationale": str(rationale)[:600], "options": options, "guard": guard, "ts": time.strftime("%H:%M:%S")}
