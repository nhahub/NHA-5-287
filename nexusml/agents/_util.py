"""Helpers shared by agents: logging, safe LLM calls, evidence handling, JSON sanitising."""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


from .. import config, llm, runlog
from ..util import jsonable  # noqa: F401  (re-exported for agents)

ANALYST_PERSONA = ("You are a senior BUSINESS analyst (not just a data analyst). You explain what the data means for "
                   "the business, why it is happening, and what to do. Use ONLY the facts provided; never invent numbers, "
                   "columns, or evidence ids. Correlation is not causation: phrase drivers as hypotheses to validate.")


def log(agent: str, message: str) -> Dict[str, str]:
    return {"agent": agent, "message": message, "ts": datetime.now().strftime("%H:%M:%S")}


def ask(state, purpose: str, system: str, payload: Dict[str, Any], max_tokens: int = 1500, validate=None, temperature: Optional[float] = None
        ) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
    """Call the LLM (mandatory: there is no rule-based fallback).

    `validate(data)` must return the cleaned result or raise ValueError("why it is unacceptable").
    On rejection the model gets ONE corrective retry that includes the reason; if it still fails, or
    the call itself fails, LLMRequired is raised and the run stops with a clear message.
    Returns (clean_result, warnings, call_records) so existing callers keep their shape."""
    calls: List[Dict[str, Any]] = []
    feedback = None
    for _ in range(2):
        pl = payload if feedback is None else {**payload, "your_previous_answer_was_rejected_because": feedback,
                                               "instruction": "Fix the problem and answer again."}
        text = json.dumps(jsonable(pl), default=str)
        h = runlog.phash(system, text)
        replayed = runlog.replay_lookup(purpose, h) if runlog.replay_active() else None
        if replayed is not None:
            data, err, rec = replayed, None, {"purpose": purpose, "model": "replay", "ok": True, "seconds": 0.0, "tokens": 0}
        else:
            data, err, rec = (llm.ask_json(system, text, purpose=purpose, max_tokens=max_tokens) if temperature is None
                              else llm.ask_json(system, text, purpose=purpose, max_tokens=max_tokens, temperature=temperature))
        runlog.llm_event(purpose, h, data, err, rec)
        calls.append(rec)
        if err:
            raise llm.LLMRequired(f"LLM call '{purpose}' failed: {err}. Check the API key ({llm.config.LLM_KEY_ENV}), NEXUSML_LLM_MODEL and your {llm.config.LLM_PROVIDER_NAME} rate limits.")
        if validate is None:
            return data, [], calls
        try:
            return validate(data), [], calls
        except ValueError as e:
            feedback = str(e)
    raise llm.LLMRequired(f"LLM call '{purpose}' returned unusable output twice: {feedback}")


def compact_profile(profile: Dict[str, Any], max_cols: int = 60) -> List[Dict[str, Any]]:
    out = []
    for c, i in list(profile["columns"].items())[:max_cols]:
        row = {"col": c, "kind": i["kind"], "missing_pct": i["missing_pct"], "unique": i["unique_count"]}
        if i["kind"] == "numeric" and "mean" in i:
            row.update(min=round(i["min"], 3), max=round(i["max"], 3), mean=round(i["mean"], 3))
        if config.LLM_SEND_SAMPLES:
            row["examples"] = i.get("sample")
        out.append(row)
    return out


PRIORITY = {"D": 0, "R": 1, "T": 1, "M": 2, "B": 2, "L": 2, "X": 2, "V": 3, "S": 3, "G": 3, "K": 4, "C": 5, "Q": 6, "W": 7}


def evidence_brief(evidence: List[Dict[str, Any]], limit: int = 45) -> List[Dict[str, str]]:
    ev = sorted(evidence, key=lambda e: (PRIORITY.get(e["id"][0], 9), e["id"]))[:limit]
    return [{"id": e["id"], "fact": e["text"]} for e in ev]


def valid_ids(ids: Any, evidence: List[Dict[str, Any]]) -> List[str]:
    known = {e["id"] for e in evidence}
    return [i for i in (ids if isinstance(ids, list) else []) if i in known]


def make_evidence(prefix: str, items: List[Tuple[str, str, Any]]) -> List[Dict[str, Any]]:
    """items = [(source, text, data)] -> evidence dicts with ids like K01, T02 ..."""
    return [{"id": f"{prefix}{i + 1:02d}", "source": s, "text": t, "data": jsonable(d)} for i, (s, t, d) in enumerate(items)]
