"""Planner agent = the BUSINESS ANALYST's framing step.

It turns a plain-English request into (a) a business problem (analysis_type), (b) an analytical
problem (problem_type), (c) a target / date column / segments, (d) candidate KPIs and hypotheses,
and (e) an analysis plan whose destructive steps need human approval.

The LLM decides; deterministic code only VALIDATES every column, enum and KPI against the real data.
There is no rule-based fallback: if the LLM cannot answer, the run stops with a clear error.
"""
from typing import Any, Dict, List, Optional

import pandas as pd

from .. import catalog, config, fairness, kpi, profiling
from . import supervisor
from ._util import ANALYST_PERSONA, ask, compact_profile, jsonable, log, make_evidence

AGENT = "Planner"


def _image_root(dataset_path: str, df) -> str:
    """Folder the image paths resolve against (the dataset's folder unless the files live in a stored upload)."""
    from pathlib import Path
    from .. import featurizers, media
    here = str(Path(dataset_path).resolve().parent)
    for c in df.columns:
        if featurizers.is_image_path(df[c]):
            vals = [str(v) for v in df[c].dropna().head(25)]
            if all(Path(v).is_absolute() for v in vals):
                return here
            found = media.locate_root(vals, dataset_path)
            return str(found) if found else here
    return here


def _date_candidates(profile, exclude: Optional[str]) -> List[str]:
    return [c for c, i in profile["columns"].items() if i["kind"] == "datetime" and c != exclude]


def _resolve(state, target: Optional[str], problem: Optional[str], date_col: Optional[str]) -> Dict[str, Any]:
    """Deterministic VALIDATION of the decided target/problem against the real data (no guessing)."""
    profile = state["dataset_profile"]
    cols = list(profile["columns"])
    if target and target not in cols:
        target = None
    if date_col and (date_col not in cols or profile["columns"][date_col]["kind"] != "datetime"):
        date_col = None
    problem = catalog.norm_problem_type(problem)
    warns: List[str] = []
    base = {"target_column": target, "problem_type": problem, "date_column": date_col}
    if problem is None or (problem in catalog.SUPERVISED and not target):
        return {**base, "is_objective_clear": False}
    if problem in ("clustering", "descriptive"):
        base["target_column"] = target if problem == "descriptive" else None
    else:
        info = profile["columns"][target]
        if problem in ("regression", "time_series") and info["kind"] != "numeric":
            warns.append(f"Target '{target}' is not numeric; switched problem_type to classification.")
            problem = "classification"
        if problem == "classification" and info["kind"] == "numeric" and info["unique_count"] > 20:
            warns.append(f"Target '{target}' is continuous with {info['unique_count']} distinct values; classification may be a poor choice.")
        if info["unique_count"] < 2 or info["missing_pct"] >= 99:
            return {**base, "target_column": None, "is_objective_clear": False,
                    "clarification_question": f"Column '{target}' has no usable variation. Pick another target."}
        if problem == "time_series":
            date_col = date_col or (_date_candidates(profile, target) or [None])[0]
            if not date_col:
                warns.append("No date column found; time_series downgraded to regression.")
                problem = "regression"
        base.update(problem_type=problem, date_column=date_col)
    if not base.get("date_column"):
        base["date_column"] = date_col or ((_date_candidates(profile, base["target_column"]) or [None])[0])
    out = {**base, "is_objective_clear": True, "clarification_question": None}
    if warns:
        out["warnings"] = warns
    return out


def frame_with_llm(state, profile) -> Dict[str, Any]:
    """The LLM frames the business problem. Output is validated against the real columns; invalid
    output is sent back once with the reason, then the run stops (no rule-based fallback)."""
    df = pd.read_csv(state["dataset_path"])
    cols = list(profile["columns"])
    overrides = {k: state.get(k) for k in ("target_column", "problem_type", "analysis_type") if state.get(k)}
    payload = {
        "request": state.get("user_request", ""), "columns": compact_profile(profile), "n_rows": profile["n_rows"],
        "allowed_analysis_types": catalog.ANALYSIS_TYPES, "allowed_problem_types": catalog.PROBLEM_TYPES,
        "user_choices_you_must_respect": overrides,
        "rules": ["analysis_type is the BUSINESS problem; problem_type is the ANALYTICAL method; they are different fields.",
                  "Predict/forecast/classify requests need a target. Segmenting needs none (clustering). Pure overviews are descriptive.",
                  "Use exact column names only. Provide 2-5 KPIs that a business owner would actually track."],
        "output_schema": {
            "analysis_type": "one allowed_analysis_types key", "problem_type": "one allowed_problem_types value",
            "target": "exact column name or null", "date_column": "exact datetime column or null",
            "business_question": "one sentence restating the decision the user needs to make",
            "segment_columns": "up to 5 categorical columns useful to slice KPIs by",
            "kpis": [{"name": "", "agg": "sum|mean|median|count|nunique|min|max|rate|ratio", "column": "exact column (not for count/ratio)",
                      "numerator": "ratio only", "denominator": "ratio only", "positive": "rate only: the event value as it appears in the data",
                      "format": "currency|percent|number", "higher_is_better": True, "group_by": ["exact columns"]}],
            "hypotheses": "2-4 business hypotheses worth testing"}}

    def validate(d: Dict[str, Any]) -> Dict[str, Any]:
        at, pt = catalog.norm_analysis_type(d.get("analysis_type")), catalog.norm_problem_type(d.get("problem_type"))
        if not at:
            raise ValueError(f"analysis_type must be one of {list(catalog.ANALYSIS_TYPES)}")
        if not pt:
            raise ValueError(f"problem_type must be one of {catalog.PROBLEM_TYPES}")
        t = d.get("target")
        if t not in cols:
            t = None
        dc = d.get("date_column")
        dc = dc if dc in cols and profile["columns"][dc]["kind"] == "datetime" else None
        good, why = [], []
        for k in [k for k in (d.get("kpis") or []) if isinstance(k, dict)][:6]:
            v, reason = kpi.validate_spec(k, df)
            (good if v else why).append(v or f"KPI '{k.get('name')}': {reason}")
        if not good:
            raise ValueError("no valid KPI. Problems: " + "; ".join(why or ["none provided"]) + f". Columns available: {cols}")
        return {"analysis_type": at, "problem_type": pt, "target": t, "date_column": dc,
                "business_question": str(d.get("business_question") or "")[:300],
                "segments": [c for c in (d.get("segment_columns") or []) if c in cols][:5], "kpis": good,
                "hypotheses": [str(h)[:200] for h in (d.get("hypotheses") or [])][:4]}

    out, _, calls = ask(state, "frame_business_problem",
                        ANALYST_PERSONA + " Map the request to a business analysis; use exact column names from the list.",
                        payload, max_tokens=1600, validate=validate)
    out["_calls"] = calls
    return out


def _apply_frame(state, frame: Dict[str, Any], how: str) -> Dict[str, Any]:
    cols = list(state["dataset_profile"]["columns"])
    target = state.get("target_column") if state.get("target_column") in cols else frame["target"]
    problem = state.get("problem_type") or frame["problem_type"]
    res = _resolve(state, target, problem, frame["date_column"])
    res.update(analysis_type=state.get("analysis_type") or frame["analysis_type"], segment_columns=frame["segments"],
               kpi_specs=jsonable(frame["kpis"]), hypotheses=frame["hypotheses"],
               business_question=frame["business_question"] or state.get("user_request") or "Analyse the dataset.",
               current_status="Business problem framed.", active_agent=AGENT, llm_calls=frame["_calls"])
    res["warnings"] = res.get("warnings", [])
    if res["is_objective_clear"]:
        res["agent_log"] = [log(AGENT, f"Business problem: {res['analysis_type']} | analytical problem: {res['problem_type']}"
                                       + (f" on '{res['target_column']}'" if res.get("target_column") else "")
                                       + (f", date column '{res['date_column']}'" if res.get("date_column") else "")
                                       + f"; {len(frame['kpis'])} KPI(s) proposed ({how}).")]
    else:
        res["agent_log"] = [log(AGENT, "Objective is ambiguous (e.g. a predictive question without a usable target); need the human to choose.")]
    return res


def understand_request_node(state) -> Dict[str, Any]:
    return _apply_frame(state, frame_with_llm(state, state["dataset_profile"]), "LLM")


def ask_clarification_node(state) -> Dict[str, Any]:
    profile = state["dataset_profile"]
    cols = [c for c, i in profile["columns"].items() if i["unique_count"] > 1]
    q = state.get("clarification_question") or ("I could not tell which column you want to predict. Pick a target column, "
                                                "or choose 'descriptive' / 'clustering' if there is no single outcome.")
    q += f" Candidate columns: {cols[:15]}"
    return {"clarification_question": q, "current_status": "Awaiting objective clarification.",
            "active_agent": AGENT, "agent_log": [log(AGENT, "Asking the human for the target / problem type.")]}


def await_clarification_node(state) -> Dict[str, Any]:
    """Resume after the human supplies target_column / problem_type. The LLM re-frames with those choices
    (so KPIs and problem type fit the chosen target); the human's choices always win."""
    cols = list(state["dataset_profile"]["columns"])
    target = state.get("target_column")
    if target and target not in cols:
        return {"is_objective_clear": False, "target_column": None,
                "clarification_question": f"'{target}' is not a column in this dataset.",
                "agent_log": [log(AGENT, f"'{target}' is not a column; asking again.")]}
    res = _apply_frame(state, frame_with_llm(state, state["dataset_profile"]), "LLM, with your choices")
    res["agent_log"] = [log(AGENT, f"Human chose target '{target}', problem '{state.get('problem_type')}'.")] + res["agent_log"]
    return res


MODELS = {"classification": ["LogisticRegression", "DecisionTree", "KNN", "RandomForest", "HistGradientBoosting"],
          "regression": ["Ridge", "DecisionTree", "KNN", "RandomForest", "HistGradientBoosting"],
          "time_series": ["Ridge", "DecisionTree", "KNN", "RandomForest", "HistGradientBoosting"], "clustering": ["KMeans"], "descriptive": []}


def build_plan_node(state) -> Dict[str, Any]:
    df = pd.read_csv(state["dataset_path"])
    profile = state["dataset_profile"]
    target, pt = state.get("target_column"), state["problem_type"]
    issues = profiling.audit_columns(df, profile, target, pt, state.get("date_column"))
    drops = [i["column"] for i in issues if i["action"] == "drop"]
    dts = [i["column"] for i in issues if i["action"] == "expand_datetime"]
    date_col = state.get("date_column")
    if pt == "time_series" and not date_col:
        date_col = dts[0] if dts else None
    destructive = [i for i in issues if i["destructive"]]
    warns: List[str] = []
    calls: List[Dict[str, Any]] = []
    if destructive:   # LLM second opinion on destructive drops (advisory only: the human decides)
        payload = {"business_question": state.get("business_question"), "target": target, "problem_type": pt,
                   "proposed_drops": [{"column": a["column"], "issue": a["issue"], "reason": a["reason"],
                                       "unique": profile["columns"][a["column"]]["unique_count"],
                                       "missing_pct": profile["columns"][a["column"]]["missing_pct"]} for a in destructive],
                   "output_schema": {"reviews": [{"column": "", "verdict": "drop|keep", "why": "one short sentence tied to the business question"}]}}
        def v_reviews(d):
            ok = [r for r in (d.get("reviews") or []) if isinstance(r, dict) and r.get("verdict") in ("drop", "keep")
                  and r.get("column") in {a["column"] for a in destructive}]
            if not ok:
                raise ValueError("provide a review (verdict drop|keep) for each proposed column")
            return ok
        reviews, warns, calls = ask(state, "review_column_drops", ANALYST_PERSONA + " Review the proposed column drops.", payload, 800, v_reviews)
        for r in reviews:
            for a in destructive:
                if a["column"] == r["column"]:
                    a["llm_opinion"] = f"{r['verdict']}: {str(r.get('why', ''))[:160]}"
    needs = bool(destructive) and not state.get("auto_approve")
    # ---- sensitive-attribute guard (governance)
    sens = fairness.detect_sensitive([c for c in df.columns if c != target and c != date_col])
    sens_note: List[str] = []
    if sens:
        if state.get("exclude_sensitive"):
            for c in sens:
                if c not in drops:
                    drops.append(c)
                    issues.append({"column": c, "issue": "sensitive_attribute", "action": "drop", "destructive": False,
                                   "reason": "protected attribute excluded at your request (it is still audited for group differences)"})
            sens_note.append(f"Sensitive attribute(s) {sens} are EXCLUDED from the model features at your request; a per-group fairness screen still runs.")
        else:
            sens_note.append(f"Sensitive attribute(s) detected: {sens}. They are USED as model features. Using protected attributes to score individuals carries "
                             "fairness and legal risk; consider re-running with exclude_sensitive=True. A per-group fairness screen runs either way.")
    supervised = pt in catalog.SUPERVISED
    plan = {
        "analysis_type": state.get("analysis_type"), "problem_type": pt, "target": target, "drop_columns": drops,
        "datetime_columns": dts, "date_column": date_col, "actions": issues,
        "split": "chronological" if pt == "time_series" else ("random" if supervised else "none"),
        "target_agg": "sum" if pt == "time_series" else None, "models": MODELS[pt],
        "tuning": {"enabled": bool(state.get("enable_tuning")) and pt != "descriptive",
                   "n_iter": state.get("tune_n_iter", config.TUNE_N_ITER),
                   "method": "randomized search + nested CV" if supervised else "silhouette-based k search"},
        "image_root": _image_root(state["dataset_path"], df),
        "image_prep": state.get("image_prep") or [], "deep_mode": state.get("deep_mode") or "auto",
        "selection": "CV on the train split only (nested CV for tuned models); test split used once for reporting",
    }
    sup = supervisor.plan_phases(state, pt)
    plan["phases"] = [ph for ph in supervisor.MENU if ph not in sup["skip_phases"]]
    out: Dict[str, Any] = {
        "supervisor": jsonable(sup), "skip_phases": sup["skip_phases"],
        "analysis_plan": jsonable(plan), "requires_approval": needs, "approval_status": "pending" if needs else "approved",
        "current_status": "Analysis plan built.", "active_agent": AGENT, "warnings": warns, "llm_calls": calls,
        "agent_log": [log(AGENT, f"Plan: {len(drops)} column(s) to drop, {len(dts)} date column(s) to expand; "
                                 + ("human approval required." if needs else "no approval needed.")
                                 + (f" Supervisor skipped phases {sup['skip_phases']}: {sup['rationale']}" if sup["skip_phases"] else ""))],
    }
    if destructive and state.get("auto_approve"):
        out["warnings"] = warns + [f"Auto-approved dropping: {[i['column'] for i in destructive]}"]
    if sens:
        out["sensitive_columns"] = sens
        out["warnings"] = list(out["warnings"]) + sens_note
        out["evidence"] = make_evidence("G", [("governance", sens_note[0], {"sensitive_columns": sens, "excluded": bool(state.get("exclude_sensitive"))})])
    return out


def human_review_node(state) -> Dict[str, Any]:
    """Pass-through. The graph pauses *before* this node; the UI/CLI writes approval_status
    (and optionally an edited analysis_plan) into state, then resumes."""
    return {"current_status": f"Human review: {state.get('approval_status')}.", "active_agent": "Human",
            "agent_log": [log("Human", f"Plan {state.get('approval_status')}.")]}


def validate_plan_node(state) -> Dict[str, Any]:
    plan = state["analysis_plan"]
    if state.get("approval_status") == "rejected":
        return {"errors": ["Plan was rejected during human review."], "current_status": "Aborted.",
                "active_agent": AGENT, "agent_log": [log(AGENT, "Plan rejected; aborting.")]}
    profile, errs = state["dataset_profile"], []
    skip = {plan.get("target")} | set(plan["drop_columns"])
    n_feat = len([c for c in profile["columns"] if c not in skip])
    pt = plan["problem_type"]
    if pt != "descriptive" and n_feat < 1:
        errs.append("No feature columns remain after drops.")
    if profile["n_rows"] < config.MIN_ROWS:
        errs.append(f"Need at least {config.MIN_ROWS} rows (have {profile['n_rows']}).")
    if pt == "time_series" and not plan.get("date_column"):
        errs.append("Time-series analysis needs a date column.")
    if errs:
        return {"errors": errs, "current_status": "Aborted.", "active_agent": AGENT,
                "agent_log": [log(AGENT, "Plan validation failed: " + "; ".join(errs))]}
    return {"current_status": "Plan validated.", "active_agent": AGENT, "agent_log": [log(AGENT, "Plan validated.")]}


def abort_node(state) -> Dict[str, Any]:
    return {"current_status": "Aborted.", "agent_log": [log(AGENT, "Run ended without training.")]}
