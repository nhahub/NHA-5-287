"""Convenience driver: runs a graph to completion, answering interrupts via callbacks."""
import uuid
from typing import Any, Callable, Dict, Optional

from .graph import build_graph
from .state import initial_state


def new_thread() -> Dict[str, Any]:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def snapshot(graph, config):
    s = graph.get_state(config)
    return s.values, tuple(s.next)


def run_to_completion(
    dataset_path: str,
    request: str,
    graph=None,
    on_event: Optional[Callable[[str, Dict], None]] = None,
    on_clarify: Optional[Callable[[Dict], Dict]] = None,
    on_approval: Optional[Callable[[Dict], Dict]] = None,
    **state_kwargs,
) -> Dict[str, Any]:
    """on_clarify(values) -> {"target_column": ..., "problem_type": ..., "analysis_type": ...} (all optional)
       on_approval(values) -> {"approved": bool, "drop_columns": [...] (optional)}
    Defaults: approve everything; no clarification (returns state paused on the question)."""
    graph = graph or build_graph()
    config = new_thread()
    state_kwargs.setdefault("run_args", {"dataset_path": dataset_path, "request": request,
                                         **{k: v for k, v in state_kwargs.items() if k != "run_args"}})
    inp: Optional[Dict] = initial_state(dataset_path, request, thread_id=config["configurable"]["thread_id"],
                                        **state_kwargs)
    for _ in range(20):
        for ev in graph.stream(inp, config=config):
            for node, upd in ev.items():
                if on_event and not node.startswith("__"):
                    on_event(node, upd or {})
        inp = None
        values, nxt = snapshot(graph, config)
        if not nxt:
            return values
        if "await_clarification" in nxt:
            if on_clarify is None:
                return values
            upd = dict(on_clarify(values))
            if "task_type" in upd:                         # legacy key
                upd["problem_type"] = upd.pop("task_type")
            graph.update_state(config, upd)
        elif "human_review" in nxt:
            decision = on_approval(values) if on_approval else {"approved": True}
            graph.update_state(config, apply_decision(values, decision))
    raise RuntimeError("Too many interrupts")


def apply_decision(values: Dict, decision: Dict) -> Dict:
    """Translate a human decision into a state update (and record who decided what in the audit log)."""
    from . import runstore
    rid = values.get("experiment_id") or values.get("run_id")
    runstore.audit("plan_rejected" if not decision.get("approved", True) else "plan_approved", rid,
                   {"drop_columns": decision.get("drop_columns"), "proposed": [i["column"] for i in values.get("analysis_plan", {}).get("column_issues", []) if i.get("destructive")]})
    if not decision.get("approved", True):
        return {"approval_status": "rejected"}
    plan = dict(values["analysis_plan"])
    if "drop_columns" in decision:
        keep = set(decision["drop_columns"])
        plan["drop_columns"] = [c for c in plan["drop_columns"] if c in keep]
    return {"approval_status": "approved", "analysis_plan": plan}
