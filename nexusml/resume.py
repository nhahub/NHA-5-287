"""Incremental analysis: build on a finished analysis instead of starting from nothing.

  run 1  "describe the data and the main KPIs"      -> profile, statistics, KPIs, trends, no model
  run 2  "now build a predictive model"             -> reuses what is still valid, does only the new work, writes a complete report

A resumed analysis is a NEW analysis (own id, own report) that remembers its parent. What it may reuse is deliberately narrow, and each rule is about whether the old
result is still TRUE for the new question:

  profile     always (the dataset file must be unchanged since the parent finished)
  kpis        yes: KPI definitions are pinned to the parent's, so the numbers and the definitions stay consistent
  trends      yes, when the date column is the same
  statistics  only when the target and the kind of problem (predictive or not) are unchanged: tests are chosen relative to the target
  everything else (plan, split, model search, explanation, charts, business reasoning, report) is computed fresh, because the new question changes it

Reused evidence keeps its original ids and is added only for reused phases, so ids can never collide with the new run's own.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import catalog

PHASES = ("profile", "statistics", "kpis", "trends")                   # what can be reused
EXPLORE_PHASES = ("statistics", "kpis", "trends")                      # reused inside the explore stage
LABEL = {"profile": "data profile", "statistics": "statistical tests", "kpis": "KPIs", "trends": "trends and outliers"}
KEYS = {"profile": ["dataset_profile"], "statistics": ["stat_results", "stat_skipped"], "kpis": ["kpi_results", "kpi_specs", "segment_columns"], "trends": ["trend_results", "outlier_results"]}
PREFIX = {"profile": "Q", "statistics": "S", "kpis": "K", "trends": "T"}
SUBTASK = {"statistical_analysis": "statistics", "kpi_analysis": "kpis", "trend_analysis": "trends"}
SUBTASK_OF = {v: k for k, v in SUBTASK.items()}


class ResumeError(ValueError):
    """The earlier analysis cannot be built on. The message is fit to show the person."""


def _same_file(a: Optional[str], b: Optional[str]) -> bool:
    try:
        return bool(a and b and Path(a).resolve() == Path(b).resolve())
    except OSError:
        return False


def dataset_unchanged(parent: Dict[str, Any], dataset_path: str) -> bool:
    """The dataset is the same file and has not been modified since the parent was saved."""
    from . import lineage
    if not _same_file(parent.get("dataset_path"), dataset_path) or not Path(dataset_path).exists():
        return False
    try:
        return Path(dataset_path).stat().st_mtime <= lineage.state_path(parent["experiment_id"]).stat().st_mtime + 1.0
    except (OSError, KeyError):
        return False


def _ran(parent: Dict[str, Any], phase: str) -> bool:
    """Did the parent actually compute this phase (not skip it, not fail in it)?"""
    if phase == "profile":
        return bool(parent.get("dataset_profile"))
    if phase in (parent.get("skip_phases") or []):
        return False
    failed = {e.get("task") for e in (parent.get("stage_errors") or []) if isinstance(e, dict)}
    if any(SUBTASK_OF[phase] == t for t in failed):
        return False
    main = {"statistics": "stat_results", "kpis": "kpi_results", "trends": "trend_results"}[phase]
    return main in parent if phase == "statistics" else bool(parent.get(main))      # an empty list of tests is still a valid result


def context_of(parent: Dict[str, Any]) -> Dict[str, Any]:
    """A short, aggregate-only description of the earlier analysis, given to the framing step so a follow-up like 'now build a model' means the right thing."""
    plan = parent.get("analysis_plan") or {}
    facts = [e["text"][:220] for e in (parent.get("evidence") or []) if e.get("id", " ")[0] in "SKTQ"][:8]
    best = next((c for c in (parent.get("candidate_metrics") or []) if c.get("name") == parent.get("best_model_name")), {})
    return {"request": (parent.get("user_request") or "")[:300], "business_question": (parent.get("business_question") or "")[:300], "analysis_type": parent.get("analysis_type"),
            "problem_type": plan.get("problem_type"), "target": plan.get("target"), "date_column": plan.get("date_column"), "modeling_done": bool(parent.get("best_model_name")),
            "best_model": parent.get("best_model_name"), "metric": parent.get("primary_metric"), "metric_value": best.get(parent.get("primary_metric") or ""),
            "headline": (parent.get("insights") or {}).get("headline"), "findings": facts, "skipped_phases": parent.get("skip_phases") or [],
            "columns": list((parent.get("dataset_profile") or {}).get("columns", {}))[:60]}


def seed(parent_exp: str, dataset_path: str, reuse: Optional[List[str]] = None) -> Dict[str, Any]:
    """-> {"state": keys for the new run's initial state, "resume": the bookkeeping the stages read}. `reuse=None` means every phase that is valid; a list restricts it ([] = reuse nothing)."""
    from . import lineage
    parent = lineage.load_state(parent_exp)
    if not parent:
        raise ResumeError("The analysis to build on could not be found. It may have been deleted.")
    if not _same_file(parent.get("dataset_path"), dataset_path):
        raise ResumeError("A follow-up must use the same dataset as the analysis it builds on. Start a new analysis for a different dataset.")
    unchanged = dataset_unchanged(parent, dataset_path)
    wanted = [p for p in (PHASES if reuse is None else reuse) if p in PHASES]
    notes: List[str] = []
    if wanted and not unchanged:
        notes.append("The dataset was changed after that analysis finished, so nothing could be reused and everything is recomputed.")
        wanted = []
    plan = parent.get("analysis_plan") or {}
    carry: Dict[str, Any] = {}
    state: Dict[str, Any] = {"parent_experiment_id": parent_exp}
    for ph in wanted:
        if not _ran(parent, ph):
            notes.append(f"The earlier analysis did not compute the {LABEL[ph]}, so they are computed now.")
            continue
        vals = {k: parent[k] for k in KEYS[ph] if k in parent}
        ev = [e for e in (parent.get("evidence") or []) if str(e.get("id", ""))[:1] == PREFIX[ph]]
        if ph == "profile":
            state.update(vals)
            state["evidence"] = ev
        else:
            carry[ph] = {"keys": vals, "evidence": ev}
    resume = {"parent": parent_exp, "parent_request": (parent.get("user_request") or "")[:200], "requested": wanted, "profile_reused": "dataset_profile" in state,
              "carry": carry, "parent_target": plan.get("target"), "parent_problem": plan.get("problem_type"), "parent_date_column": plan.get("date_column"),
              "context": context_of(parent), "notes": notes}
    return {"state": state, "resume": resume}


def pinned_frame(state: Dict[str, Any]) -> Dict[str, Any]:
    """After the question is framed: KPI definitions come from the parent, so reused KPI numbers and their definitions agree."""
    kp = ((state.get("resume") or {}).get("carry") or {}).get("kpis")
    return dict(kp["keys"]) if kp else {}


def applicable(state: Dict[str, Any]) -> List[str]:
    """Which explore-stage phases can really be reused for THIS plan (decided once the plan exists)."""
    r = state.get("resume") or {}
    plan = state.get("analysis_plan") or {}
    sup = set(catalog.SUPERVISED)
    out = []
    for ph in EXPLORE_PHASES:
        if ph not in (r.get("carry") or {}):
            continue
        if ph == "statistics":
            if (plan.get("target") or None) != (r.get("parent_target") or None) or ((plan.get("problem_type") in sup) != (r.get("parent_problem") in sup)):
                continue
        if ph == "trends" and (plan.get("date_column") or None) != (r.get("parent_date_column") or None):
            continue
        out.append(ph)
    return out


def carried_update(state: Dict[str, Any], phases: List[str]) -> Dict[str, Any]:
    """The state update that puts reused results back (plain keys + their evidence)."""
    carry = (state.get("resume") or {}).get("carry") or {}
    upd: Dict[str, Any] = {}
    ev: List[Dict[str, Any]] = []
    for ph in phases:
        upd.update(carry[ph]["keys"])
        ev += list(carry[ph]["evidence"])
    if ev:
        upd["evidence"] = ev
    return upd


def explain(state: Dict[str, Any]) -> str:
    """One sentence for the activity log: what was reused and what was recomputed."""
    r = state.get("resume") or {}
    done = (["data profile"] if r.get("profile_reused") else []) + [LABEL[p] for p in applicable(state)]
    redo = [LABEL[p] for p in (r.get("carry") or {}) if p not in applicable(state)]
    msg = f"Building on analysis {r.get('parent')}: " + (f"reused the {', '.join(done)}" if done else "nothing could be reused")
    return msg + (f"; recomputed the {', '.join(redo)} because the question changed." if redo else ".")


def slim(resume: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The bookkeeping that is worth keeping in the saved run (the carried payload is already in the run's own results)."""
    r = dict(resume or {})
    r.pop("carry", None)
    return r
