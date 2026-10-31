"""Graph assembly (v6): a crew, not a pipeline. See agents/stages.py and agents/crew.py.

    START -> intake -> frame --(clear)--> plan --(needs approval)--> human_review -+--> prepare --> brief (Director: LLM)
                         \\--(unclear)--> await_clarification (loops until clear)                          |
                                       abort (rejected / invalid plan) -> END     +------------+------------+------------+
                                                                                  v            v            v            v
                                                                              sentinel      explore       scout        automl <-----+
                                                                              (leakage,    (stats, KPIs,  (feature     (goal-driven  | refine (LLM verdict,
                                                                               drift)       NLP / CV)      ideas)       AutoML agent) | guarded)
                                                                                  |            |            |            v            |
                                                                                  |            |            |          critic ---------+
                                                                                  |            |            |            | accept
                                                                                  |            |            |            v
                                                                                  |            |            |         explain
                                                                                  +------------+------+-----+------------+
                                                                                                      v
                                                                                                 synthesize -> publish -> END
    The four branches run concurrently and exchange messages on the blackboard. Deterministic routers guard validity, approval and budgets; the LLM (the Director) chooses
    the crew, the AutoML experiments and the accept / refine verdict inside those guards. Uses real LangGraph when installed (NEXUSML_BACKEND=builtin forces the offline engine).
"""
import os
from typing import Literal

try:
    if os.environ.get("NEXUSML_BACKEND") == "builtin":
        raise ImportError
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    BACKEND = "langgraph"
except ImportError:
    from .engine import END, START, MemorySaver, StateGraph
    BACKEND = "builtin"

from . import runlog
from .agents import stages as S
from .state import DataScientistState

INTERRUPTS = ["await_clarification", "human_review"]      # the only two places a human can be asked
NODES = ["intake", "frame", "await_clarification", "plan", "human_review", "prepare", "brief", "sentinel", "explore", "scout", "automl", "critic", "explain", "synthesize", "publish", "abort"]


# routers
def route_objective(state) -> Literal["plan", "clarify"]:
    return "plan" if state.get("is_objective_clear") else "clarify"


def route_plan(state) -> Literal["abort", "review", "go"]:
    if S.plan_errors(state):
        return "abort"
    return "review" if state.get("requires_approval") and state.get("approval_status") != "approved" else "go"


def route_review(state) -> Literal["abort", "go"]:
    return "abort" if S.plan_errors(state) else "go"


def route_critic(state) -> Literal["retry", "done"]:
    return S.route_critic(state)


def _logged(fn):
    """Every routing decision lands in the run log (audit + replay)."""
    def wrapper(state):
        res = fn(state)
        runlog.decision(fn.__name__, res)
        return res
    wrapper.__name__ = fn.__name__
    return wrapper


def _checked(name, fn):
    def node(state):
        out = fn(state)
        if os.environ.get("NEXUSML_VALIDATE") == "1":
            from .schemas import validate_update
            validate_update(name, out or {})
        return out
    node.__name__ = name
    return node


def build_graph(checkpointer=None):
    g = StateGraph(DataScientistState)
    fns = {"intake": S.intake_node, "frame": S.frame_node, "await_clarification": S.await_clarification_node, "plan": S.plan_node,
           "human_review": S.human_review_node, "prepare": S.prepare_node, "brief": S.brief_node, "sentinel": S.sentinel_node, "explore": S.explore_node,
           "scout": S.scout_node, "automl": S.automl_node, "critic": S.critic_node,
           "explain": S.explain_node, "synthesize": S.synthesize_node, "publish": S.publish_node, "abort": S.abort_node}
    for name in NODES:
        g.add_node(name, _checked(name, S.stage(name, fns[name])))

    g.add_edge(START, "intake")
    g.add_edge("intake", "frame")
    g.add_conditional_edges("frame", _logged(route_objective), {"plan": "plan", "clarify": "await_clarification"})
    g.add_conditional_edges("await_clarification", _logged(route_objective), {"plan": "plan", "clarify": "await_clarification"})   # pauses before each pass
    g.add_conditional_edges("plan", _logged(route_plan), {"abort": "abort", "review": "human_review", "go": "prepare"})
    g.add_conditional_edges("human_review", _logged(route_review), {"abort": "abort", "go": "prepare"})                              # pauses before
    g.add_edge("abort", END)
    # ---- the Director briefs the crew, then four agents work CONCURRENTLY and talk on the blackboard
    g.add_edge("prepare", "brief")
    for branch in ("sentinel", "explore", "scout", "automl"):
        g.add_edge("brief", branch)
    # ---- modeling branch: the AutoML agent <-> critic loop. The critic's verdict is an LLM decision bounded by guards (see crew.critic_node).
    #      `explain` is reached only when the loop is done, so it is the safe join source on real LangGraph.
    g.add_edge("automl", "critic")
    g.add_conditional_edges("critic", _logged(route_critic), {"retry": "automl", "done": "explain"})
    # ---- join: the business reasoning needs every branch
    g.add_edge(["sentinel", "explore", "scout", "explain"], "synthesize")
    g.add_edge("synthesize", "publish")
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer or MemorySaver(), interrupt_before=INTERRUPTS)
