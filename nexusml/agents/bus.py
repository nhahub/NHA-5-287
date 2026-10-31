"""Agent-to-agent messages. An agent can ask another agent's SERVICE for work; every request/response is logged in
state["messages"], bounded by MAX_AGENT_REQUESTS per run, and executed by deterministic code (no extra LLM calls)."""
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .. import config, stats_tests

SERVICES: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}


def service(topic: str):
    def deco(fn):
        SERVICES[topic] = fn
        return fn
    return deco


def _msg(n: int, frm: str, to: str, kind: str, topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": f"MSG{n:03d}", "from": frm, "to": to, "kind": kind, "topic": topic, "payload": payload, "ts": time.strftime("%H:%M:%S")}


def request(state, frm: str, to: str, topic: str, payload: Dict[str, Any], used: int) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], int]:
    """Returns (messages_to_log, result_or_None, new_used_count)."""
    base = len(state.get("messages") or [])
    req = _msg(base + 1, frm, to, "request", topic, payload)
    if used >= config.MAX_AGENT_REQUESTS or topic not in SERVICES:
        why = "request cap reached" if used >= config.MAX_AGENT_REQUESTS else "no such service"
        return [req, _msg(base + 2, to, frm, "response", topic, {"refused": why})], None, used
    try:
        res = SERVICES[topic](state, payload)
    except Exception as e:
        return [req, _msg(base + 2, to, frm, "response", topic, {"error": f"{type(e).__name__}: {e}"})], None, used + 1
    return [req, _msg(base + 2, to, frm, "response", topic, {"ok": True, "summary": res.get("summary", "")})], res, used + 1


def pick_test(df: pd.DataFrame, a: str, b: str) -> Optional[Dict[str, Any]]:
    """Choose the appropriate test for a pair of columns by their types."""
    na, nb = pd.api.types.is_numeric_dtype(df[a]) and df[a].nunique() > 8, pd.api.types.is_numeric_dtype(df[b]) and df[b].nunique() > 8
    if na and nb:
        return {"test": "correlation", "params": {"a": a, "b": b}}
    if na != nb:
        num, cat = (a, b) if na else (b, a)
        return {"test": "group_comparison", "params": {"measure": num, "group": cat}}
    return {"test": "chi_square", "params": {"a": a, "b": b}}


def run_single_test(df: pd.DataFrame, a: str, b: str, tid: str = "REQ") -> Optional[Dict[str, Any]]:
    cand = pick_test(df, a, b)
    if cand is None:
        return None
    r = stats_tests.execute({"id": tid, **cand}, df)
    if "skipped" in r:
        return r
    r["p_adjusted"] = r["p_value"]                      # a single requested test: no multiplicity to correct
    r["significant"] = bool(r["p_value"] < config.ALPHA and r.get("effect_label") != "negligible")
    r["interpretation"] = stats_tests.narrate(r)
    return r


@service("statistician/test_association")
def _test_association(state, payload):
    df = pd.read_csv(state["dataset_path"])
    a, b = payload["a"], payload["b"]
    if a not in df.columns or b not in df.columns:
        raise ValueError("unknown column")
    r = run_single_test(df, a, b)
    if r is None or "skipped" in r:
        raise ValueError((r or {}).get("skipped", "no applicable test"))
    return {"summary": r["interpretation"], "result": r}
