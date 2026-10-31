"""The v6 crew: agents that run IN PARALLEL, talk through the blackboard, and make LLM decisions inside deterministic guards.

  brief     Director: reads the situation, decides which specialists to deploy and how to split the budget (and extracts a modelling goal stated in plain words)
  sentinel  Data auditor: scans for leakage / drift / conflicts, asks the Director which suspect columns to quarantine, tells the AutoML agent
  scout     Idea generator: proposes domain feature ideas from column names and statistics (never raw values); a safe DSL executes them
  automl    Research lead: goal-driven search (automl_run) with an LLM that picks the next experiments; reads the sentinel / scout / explore messages every round
  critic    Quality gate + Director: audits the result, then decides ACCEPT or REFINE (another goal-directed pass with a directive) inside guard rails
"""
import os
import time
from typing import Any, Dict, List

import pandas as pd

from .. import blackboard, config, director, llm as llm_mod, leaderboard, modeling, sentinel, targets
from ..automl_core import Goal, Session
from ..automl_recipes import IDEA_OPS, validate_ideas
from ..automl_run import run as run_automl
from ..state import merge_update
from ..storage import get_registry
from . import critic as critic_mod
from . import segmenter
from ._util import ANALYST_PERSONA, jsonable, log

SUPERVISED = modeling.SUPERVISED
_SESSIONS: Dict[str, Session] = {}
def _max_workers() -> int:
    return max(1, int(os.environ.get("NEXUSML_AUTOML_WORKERS", "4")))


MAX_WORKERS = _max_workers()


def isolated(name: str, report_key: str):
    """A helper agent (sentinel, scout) must never take the run down: any failure becomes a recorded warning and the crew carries on. LLMRequired is the one exception
    (strict mode / a missing key must still stop the run with its clear message)."""
    def deco(fn):
        def wrapped(state):
            try:
                return fn(state)
            except Exception as e:
                if isinstance(e, llm_mod.LLMRequired):
                    raise
                msg = f"{name} failed ({type(e).__name__}: {str(e)[:140]}); the run continued without it."
                return {report_key: {"failed": True, "error": f"{type(e).__name__}: {str(e)[:200]}"}, "stage_errors": [{"stage": name, "task": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}],
                        "warnings": [msg], "agent_log": [log(name.capitalize(), msg)]}
        wrapped.__name__ = fn.__name__
        return wrapped
    return deco


def _supervised(state) -> bool:
    return state["analysis_plan"]["problem_type"] in SUPERVISED and "modeling" not in (state.get("skip_phases") or [])


def _board(state):
    return blackboard.get_board(state["run_id"])


# brief (Director)
def _validate_brief(d: Any) -> Dict[str, Any]:
    if not isinstance(d, dict) or not str(d.get("rationale", "")).strip():
        raise ValueError("return an object with a non-empty 'rationale'")
    crew = d.get("crew") if isinstance(d.get("crew"), dict) else {}
    am = d.get("automl") if isinstance(d.get("automl"), dict) else {}
    out = {"crew": {k: bool(crew.get(k, True)) for k in ("sentinel", "scout", "explore")}, "rationale": str(d["rationale"])[:500]}
    try:
        ms = am.get("max_seconds")
        out["automl"] = {"max_seconds": None if ms in (None, "", 0) else float(min(3600, max(15, float(ms)))), "max_rounds": int(min(8, max(1, int(am.get("max_rounds", 4) or 4)))),
                         "workers": int(min(8, max(1, int(am.get("workers", 1) or 1)))), "style": am.get("style") if am.get("style") in ("quick", "balanced", "thorough") else "balanced"}
    except (TypeError, ValueError):
        raise ValueError("automl.max_seconds / max_rounds / workers must be numbers")
    g = d.get("goal")
    out["goal"] = g if isinstance(g, dict) and g.get("metric") and g.get("value") is not None else None
    return out


def brief_node(state) -> Dict[str, Any]:
    pt = state["analysis_plan"]["problem_type"]
    ds, prof = state.get("data_summary") or {}, state.get("dataset_profile") or {}
    payload = {"request": state.get("user_request"), "business_question": state.get("business_question"), "analysis_type": state.get("analysis_type"), "problem_type": pt,
               "n_rows": prof.get("n_rows"), "n_features": ds.get("n_features"), "minority_share": ds.get("minority_share"), "modality": state.get("modality"),
               "pipeline_kind": state.get("pipeline_kind"), "goal_already_set": state.get("target_spec"), "cpu_count": os.cpu_count(), "max_workers_allowed": _max_workers(),
               "crew": {"sentinel": "audits leakage / drift / label conflicts before models are trusted", "scout": "proposes domain feature ideas for the AutoML agent to test",
                        "explore": "statistics, KPIs, trends, text / image analysis"},
               "guidance": ["Deploy the sentinel unless the data is tiny and clean; skip the scout for text or image-only data.", "Use max_seconds=null unless the user gave a time limit.",
                            "workers should not exceed cpu_count. 'quick' for a rough first look, 'thorough' when a demanding goal is stated.",
                            "If the user's request states a modelling goal in words (e.g. 'AUC of at least 0.9', 'recall above 70%') and goal_already_set is null, return it as "
                            "goal={metric, bound: 'min'|'max', value, subject_to: {metric, value} or omitted}; e.g. 'recall of at least 70% with precision above 50%' -> "
                            "{metric: 'recall', bound: 'min', value: 0.7, subject_to: {metric: 'precision', value: 0.5}}. Fractions, not percentages. Otherwise goal=null."],
               "output_schema": {"crew": {"sentinel": True, "scout": True, "explore": True}, "automl": {"max_seconds": None, "max_rounds": 4, "workers": 1, "style": "balanced"},
                                 "goal": None, "rationale": "2 sentences"}}

    def default():
        return {"crew": {"sentinel": True, "scout": True, "explore": True}, "automl": {"max_seconds": None, "max_rounds": 4, "workers": 1, "style": "balanced"}, "goal": None,
                "rationale": "Default crew: the whole team, balanced budget."}
    d, src, calls, warns = director.decide(state, "brief_crew", ANALYST_PERSONA + " You are also the team's director: you decide which specialists to deploy.", payload, _validate_brief, default, 900)
    crew, am = dict(d["crew"]), dict(d["automl"])
    guard = []
    if pt not in SUPERVISED or "modeling" in (state.get("skip_phases") or []):
        crew["scout"], am = False, {**am, "max_rounds": 0}
        guard.append("no supervised model: scout and AutoML are off")
    if state.get("pipeline_kind") in ("text", "image"):
        crew["scout"] = False
    am["workers"] = int(min(am["workers"], _max_workers(), max(1, os.cpu_count() or 1)))
    spec = state.get("target_spec")
    new_spec = None
    if not spec and d.get("goal"):
        g = d["goal"]
        try:
            m = targets.ALIASES.get(str(g["metric"]).lower(), str(g["metric"]).lower())
            cand = {"metric": m, ("max" if m in targets.LOWER_BETTER else "min"): float(g["value"])}      # the metric's own direction, not the LLM's word for it
            sub = g.get("subject_to")
            if isinstance(sub, dict) and sub.get("metric") and sub.get("value") is not None:
                sm = targets.ALIASES.get(str(sub["metric"]).lower(), str(sub["metric"]).lower())
                cand["subject_to"] = {"metric": sm, ("max" if sm in targets.LOWER_BETTER else "min"): float(sub["value"])}
            targets.parse(cand, pt)
            new_spec, spec = cand, cand
            guard.append(f"goal extracted from the request and validated: {cand}")
        except (ValueError, KeyError, TypeError) as e:
            warns.append(f"A goal the Director extracted was ignored ({str(e)[:100]}).")
    if state.get("time_budget"):
        am["max_seconds"] = float(state["time_budget"])                 # asked for on this run (CLI --time-budget or the form)
    elif os.environ.get("NEXUSML_AUTOML_BUDGET"):
        am["max_seconds"] = float(os.environ["NEXUSML_AUTOML_BUDGET"])  # deployment-wide default
    if state.get("workers"):
        am["workers"] = int(min(max(1, int(state["workers"])), _max_workers(), max(1, os.cpu_count() or 1)))      # a request is still bounded by the guard
    if spec and spec.get("max_seconds"):
        am["max_seconds"] = float(spec["max_seconds"])                  # the user's time limit always wins
    rec = director.record("brief_crew", src, {"crew": crew, "automl": am, "goal": spec}, d["rationale"], guard="; ".join(guard))
    b = _board(state)
    b.post("director", "crew", {"crew": crew, "automl": am}, text=f"Crew deployed: {', '.join(k for k, v in crew.items() if v) or 'none'}; AutoML {am['style']}, {am['workers']} worker(s). {d['rationale']}")
    for k, v in crew.items():
        if not v:
            b.mark_done(k)
    return {"crew": crew, "automl_plan": am, **({"target_spec": new_spec} if new_spec else {}), "decisions": [rec], "llm_calls": calls, "warnings": warns,
            "current_status": "Crew briefed.", "active_agent": "Director", "agent_log": [log("Director", f"[{src}] {d['rationale']} Crew: {crew}. AutoML: {am}." + (f" Goal: {new_spec}." if new_spec else ""))]}


# sentinel
def _validate_sentinel(suspects: List[Dict[str, Any]]):
    cols = {s["column"] for s in suspects}

    def v(d):
        if not isinstance(d, dict) or not isinstance(d.get("decisions"), list) or not str(d.get("rationale", "")).strip():
            raise ValueError("return {decisions: [{column, action: quarantine|keep, why}], rationale}")
        out = []
        for x in d["decisions"]:
            if not isinstance(x, dict) or x.get("column") not in cols or x.get("action") not in ("quarantine", "keep"):
                raise ValueError(f"each decision needs a column from {sorted(cols)} and action quarantine|keep")
            out.append({"column": x["column"], "action": x["action"], "why": str(x.get("why", ""))[:200]})
        return {"decisions": out, "rationale": str(d["rationale"])[:400]}
    return v


@isolated("sentinel", "sentinel_report")
def sentinel_node(state) -> Dict[str, Any]:
    b = _board(state)
    try:
        if not state.get("crew", {}).get("sentinel", True) or not _supervised(state):
            return {"sentinel_report": {"skipped": True}}
        reg = get_registry()
        splits = reg.load(state["splits_key"])
        pt, plan = state["analysis_plan"]["problem_type"], state["analysis_plan"]
        rep = sentinel.scan(splits, pt, plan.get("target") or "")
        suspects = [s for s in rep["leakage_suspects"] if s["severity"] in ("high", "medium")]
        calls: List[Dict[str, Any]] = []
        warns: List[str] = []
        decisions: List[Dict[str, Any]] = []
        quarantine: List[str] = []
        if suspects:
            payload = {"problem_type": pt, "target": plan.get("target"), "suspects": [{"column": s["column"], "single_column_score": s["score"], "metric": s["metric"], "severity": s["severity"],
                                                                                          "why_flagged": s["kind"]} for s in suspects],
                       "n_features": rep["n_features"], "id_like_columns": rep["id_like"][:6],
                       "guidance": ["Quarantine a column only if it is very likely a leak (an outcome, an ID, a proxy for the label) and could not be known when the prediction is made.",
                                    "A genuinely strong legitimate predictor should be kept: say so and warn instead.", "Only 'high' severity columns can be quarantined."],
                       "output_schema": {"decisions": [{"column": "name", "action": "quarantine|keep", "why": "one sentence"}], "rationale": "2 sentences"}}
            default = lambda: {"decisions": [{"column": s["column"], "action": "quarantine" if s["severity"] == "high" else "keep", "why": "default: near-perfect single-column predictor"} for s in suspects],
                               "rationale": "Default policy: quarantine near-perfect single-column predictors."}
            d, src, calls, warns = director.decide(state, "sentinel_review", ANALYST_PERSONA + " You are the team's data-leakage reviewer.", payload, _validate_sentinel(suspects), default, 800)
            highs = {s["column"] for s in suspects if s["severity"] == "high"}
            quarantine = [x["column"] for x in d["decisions"] if x["action"] == "quarantine" and x["column"] in highs][:max(1, int(0.3 * rep["n_features"]))]
            decisions.append(director.record("sentinel_review", src, {"quarantine": quarantine, "keep": [x["column"] for x in d["decisions"] if x["action"] == "keep"]}, d["rationale"],
                                             guard="only high-severity columns may be quarantined; at most 30% of the features"))
            for s in suspects:
                b.post("sentinel", "leakage_suspect", {"column": s["column"], "score": s["score"]}, severity="high" if s["severity"] == "high" else "warning",
                       text=f"Column '{s['column']}' predicts the target with {s['metric']} {s['score']} on its own ({s['kind']}).")
            if quarantine:
                b.post("sentinel", "quarantine_proposal", {"columns": quarantine}, severity="high", text=f"Proposal: drop {quarantine} (likely leakage). Use the drop_features action.")
        if rep["drift"].get("shifted"):
            b.post("sentinel", "drift", rep["drift"], severity="warning", text=f"Train and test rows are distinguishable (adversarial AUC {rep['drift']['auc']}): expect a performance gap on new data.")
        c = rep["conflicts"]
        if c.get("available"):
            b.post("sentinel", "noise_ceiling", c, severity="info", text=f"Identical feature rows with different outcomes cap what any model can reach: {c}.")
        if rep["id_like"]:
            b.post("sentinel", "id_like", {"columns": rep["id_like"]}, severity="warning", text=f"ID-like columns (unique per row): {rep['id_like'][:5]}.")
        rep["quarantine_proposed"] = quarantine
        return {"sentinel_report": jsonable(rep), "decisions": decisions, "llm_calls": calls, "warnings": warns + ([f"Possible leakage: {[s['column'] for s in suspects if s['severity'] == 'high']}"] if quarantine else []),
                "agent_log": [log("Sentinel", f"Scanned {rep['n_features']} features: {len(suspects)} leakage suspect(s), drift AUC {rep['drift'].get('auc')}, "
                                              f"{'label conflicts found' if c.get('available') else 'no label conflicts'}; quarantine proposal: {quarantine or 'none'}.")]}
    finally:
        b.mark_done("sentinel")


# scout
def _column_facts(X) -> List[Dict[str, Any]]:
    out = []
    for c in list(X.columns)[:60]:
        s = X[c]
        f = {"name": c, "kind": "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical", "n_unique": int(s.nunique()), "missing_share": round(float(s.isna().mean()), 3)}
        if f["kind"] == "numeric" and s.notna().any():
            f.update(min=round(float(s.min()), 3), median=round(float(s.median()), 3), max=round(float(s.max()), 3), skew=round(float(s.skew()), 2))
        out.append(f)
    return out


def _validate_scout(X):
    def v(d):
        if not isinstance(d, dict) or not isinstance(d.get("ideas", []), list) or not str(d.get("rationale", "")).strip():
            raise ValueError("return {ideas: [...], hypotheses: [...], rationale}")
        ideas, why = validate_ideas(d.get("ideas", []), X)
        return {"ideas": ideas, "rejected": why, "hypotheses": [str(h)[:200] for h in d.get("hypotheses", [])][:6], "rationale": str(d["rationale"])[:400]}
    return v


@isolated("scout", "scout_report")
def scout_node(state) -> Dict[str, Any]:
    b = _board(state)
    try:
        if not state.get("crew", {}).get("scout", True) or not _supervised(state) or state.get("pipeline_kind") in ("text", "image"):
            return {"scout_report": {"skipped": True}}
        splits = get_registry().load(state["splits_key"])
        X, plan = splits["X_train"], state["analysis_plan"]
        payload = {"business_question": state.get("business_question"), "analysis_type": state.get("analysis_type"), "target": plan.get("target"), "problem_type": plan["problem_type"],
                   "columns": _column_facts(X), "allowed_ops": {op: f"{n} column(s)" for op, n in IDEA_OPS.items()},
                   "guidance": ["Propose at most 8 features a domain expert would expect to matter: ratios (debt/income), differences, products, log transforms of skewed columns, group means (a numeric column averaged by a category), count encodings.",
                                "Use ONLY columns listed above. Each idea: {op, cols, name, why}. group_mean takes cols [categorical, numeric]; bin_quantile takes param 3-10.",
                                "You see column names and summary statistics only, never rows. Do not propose anything that uses the target."],
                   "output_schema": {"ideas": [{"op": "ratio", "cols": ["a", "b"], "name": "a_over_b", "why": "domain reason"}], "hypotheses": ["one sentence each"], "rationale": "2 sentences"}}
        d, src, calls, warns = director.decide(state, "scout_ideas", ANALYST_PERSONA + " You are also a domain-savvy feature-engineering scout.", payload, _validate_scout(X),
                                              lambda: {"ideas": [], "rejected": [], "hypotheses": [], "rationale": "Default: no ideas (LLM unavailable)."}, 1200)
        for i in d["ideas"]:
            b.post("scout", "feature_idea", i, text=f"Idea {i['name']}: {i['op']}({', '.join(i['cols'])}) - {i.get('why', '')}")
        if d["hypotheses"]:
            b.post("scout", "hypothesis", {"hypotheses": d["hypotheses"]}, text="Hypotheses: " + " | ".join(d["hypotheses"][:3]))
        rep = {"ideas": d["ideas"], "rejected": d["rejected"], "hypotheses": d["hypotheses"], "source": src}
        return {"scout_report": jsonable(rep), "decisions": [director.record("scout_ideas", src, [i["name"] for i in d["ideas"]], d["rationale"], guard="every idea validated against the real columns")],
                "llm_calls": calls, "warnings": warns, "agent_log": [log("Scout", f"[{src}] {len(d['ideas'])} valid feature idea(s), {len(d['rejected'])} rejected. {d['rationale']}")]}
    finally:
        b.mark_done("scout")


# automl
def _candidate_rows(state, out: Dict[str, Any], splits: Dict[str, Any], attempt: int) -> Dict[str, Any]:
    pt, pm = state["analysis_plan"]["problem_type"], state["primary_metric"]
    rows = []
    for c in out["candidates"]:
        est = out["models"][c["name"]]
        rows.append(leaderboard.make_row(c, attempt, leaderboard.candidate_extras(pt, est, splits, pm), leaderboard.diagnostics(pt, est, splits["X_test"], splits["y_test"], splits.get("classes")), pm))
    sess = out["sess"]
    levers = sorted({r for e in sess.effects for r in e.get("recipe", [])} | {f.split("=")[0] for e in sess.effects for f in e.get("flags", [])} | ({"ideas"} if sess.ideas else set()))
    return {"attempt": attempt, "levers": levers, "metric": pm, "rows": rows}


def automl_node(state) -> Dict[str, Any]:
    pt = state["analysis_plan"]["problem_type"]
    if "modeling" in (state.get("skip_phases") or []):
        return {"agent_log": [log("Supervisor", "Skipped 'modeling': not needed for this request.")]}
    if pt == "clustering":
        return segmenter.cluster_segments_node(state)
    if pt not in SUPERVISED:
        return {}
    reg = get_registry()
    splits = reg.load(state["splits_key"])
    plan = state.get("automl_plan") or {"max_seconds": None, "max_rounds": 4, "workers": 1, "style": "balanced"}
    attempt = int(state.get("automl_attempt", 0)) + 1
    goal = Goal.from_spec(state.get("target_spec"), pt, plan.get("max_seconds"), plan.get("max_rounds"))
    if not goal.active:
        goal.max_rounds = min(goal.max_rounds, 3)
    strategy = {**modeling.DEFAULT_STRATEGY, **(state.get("strategy") or {})}
    strategy["tune"] = bool(strategy.get("tune", True)) and bool(state.get("enable_tuning", True))      # the user's switch: no hyper-parameter search when off
    notes: List[str] = []
    calls: List[Dict[str, Any]] = []
    warns: List[str] = []

    def ask_fn(purpose, system, payload, validate, max_tokens, default):
        d, src, c, w = director.decide(state, purpose, system, payload, validate, default, max_tokens)
        calls.extend(c)
        warns.extend(w)
        return d, src, c, w
    sess = _SESSIONS.get(state["run_id"]) if attempt > 1 else None
    out = run_automl(pt, splits, strategy, goal, _board(state), ask_fn, seed=int(state.get("seed", config.RANDOM_STATE)), workers=int(plan.get("workers", 1)),
                     budget=plan.get("max_seconds") or (goal.spec or {}).get("max_seconds"), session=sess, directive=state.get("directive") or "", log=notes.append)
    sess = out["sess"]
    _SESSIONS[state["run_id"]] = sess
    key = reg.save({"models": out["models"]}, "models")
    pm = state["primary_metric"]
    best = next(c for c in out["candidates"] if c["name"] == out["best"])
    rec = _candidate_rows(state, out, splits, attempt)
    gr = out["goal_report"]
    report = {"attempt": attempt, "goal": gr, "stop_reason": out["stop_reason"], "stop_why": out["stop_why"], "seconds": out["seconds"], "n_trials": len(sess.trials), "workers": sess.n_workers,
              "budget_seconds": round(sess.budget, 1), "style": plan.get("style"), "rounds": jsonable(sess.rounds), "trials": [t.brief() for t in sess.trials], "ensemble": out["ensemble"],
              "learning_curve": out.get("lc"), "ceiling": sess.ceiling(), "effects": jsonable(sess.effects), "dropped_columns": list(sess.dropped), "ideas": list(sess.ideas.values()),
              "search_metric": sess.pm, "big_data_halving": sess.big, "best_config": jsonable(out["configs"].get(out["best"])), "stop_text": f"{out['stop_reason']}: {out['stop_why']}",
              "protocol": "search = CV on the training split; finalists re-scored on fresh folds; held-out split scored once at the end"}
    update: Dict[str, Any] = {"strategy": {**strategy, "pipeline_kind": state.get("pipeline_kind")}, "baseline_metrics": out["baseline"], "candidate_metrics": jsonable(out["candidates"]),
                              "best_model_name": out["best"], "models_key": key, "tuning_report": {"enabled": bool(strategy.get("tune", True)), "engine": "automl", "n_iter": len(sess.trials), "scoring": sess.pm, "models": {}, "space_source": "AutoML agent (LLM-guided, validated)"}, "automl_report": jsonable(report), "automl_attempt": attempt,
                              "candidate_history": [rec], "decisions": jsonable(sess.decisions[getattr(sess, "reported", 0):]),
                              "llm_calls": calls, "warnings": warns + [f"Candidate {f['name']} was excluded: {f['error']}" for f in out.get("failed", [])],
                              "strategies_applied": sorted(set((rec["levers"]) + (["drop_features"] if sess.dropped else []))), "quarantined": list(sess.dropped),
                              "current_status": "AutoML finished.", "active_agent": "AutoML Agent",
                              "agent_log": [log("AutoML Agent", n) for n in notes] + [log("AutoML Agent", f"Attempt {attempt}: {len(sess.trials)} trials, {len(sess.rounds)} LLM-directed round(s); "
                                                                                          f"best {out['best']} (fresh-fold {pm} {best['select_score']}, held-out {best.get(pm)}). Goal: {gr.get('status')}. Stopped: {out['stop_reason']}.")]}
    sess.reported = len(sess.decisions)
    if state.get("target_spec") and gr.get("confirmed_on_fresh_folds"):
        spec = targets.parse(state["target_spec"], pt)
        prev = dict(state.get("target_report") or {})
        update["target_report"] = {**prev, "spec": spec, "t0": prev.get("t0", time.time()), "attempts": list(prev.get("attempts", [])) + [{"attempt": attempt, "model": out["best"], **gr["confirmed_on_fresh_folds"]}],
                                  "validation": {**gr["confirmed_on_fresh_folds"], "source": "fresh-fold CV after the AutoML search"}}
    return update


# critic + verdict (Director)
def _validate_verdict(allowed: List[str]):
    def v(d):
        if not isinstance(d, dict) or d.get("route") not in allowed or not str(d.get("rationale", "")).strip():
            raise ValueError(f"return {{route: one of {allowed}, directive: str, rationale: str}}")
        return {"route": d["route"], "directive": str(d.get("directive", ""))[:400], "rationale": str(d["rationale"])[:500]}
    return v


def critic_node(state) -> Dict[str, Any]:
    """Deterministic audit first, then the Director chooses ACCEPT or REFINE among the routes the guards allow."""
    if state["analysis_plan"]["problem_type"] not in SUPERVISED or not state.get("candidate_metrics"):
        return {"retry_pending": False}
    upd = critic_mod.audit_models_node(state)
    st = merge_update(state, upd)
    ar = st.get("automl_report") or {}
    gr = ar.get("goal") or {}
    issues = st["audit"]["issues"]
    blockers = [i for i in issues if i["severity"] == "blocker"]
    attempt, max_refines = int(st.get("automl_attempt", 1)), min(2, critic_mod.retry_budget(st))
    refines_left = max(0, max_refines - (attempt - 1))
    stop = ar.get("stop_reason")
    futile = stop in ("plateau", "infeasible", "no_valid_actions") and (attempt >= 2 or stop == "infeasible")        # a flat search that already had one refine will not move; 'infeasible' never needs one
    unmet = gr.get("status") == "not_met" or bool(blockers)
    allowed = ["accept"] + (["refine"] if (unmet and refines_left > 0 and not futile and _SESSIONS.get(state["run_id"]) is not None) else [])
    guard = f"attempt {attempt}; refines left {refines_left}; stop reason {stop}; allowed {allowed}"
    payload = {"goal": gr.get("text"), "goal_status": gr.get("status"), "confirmed": (gr.get("confirmed_on_fresh_folds") or {}).get("value"), "heldout": (gr.get("heldout") or {}).get("value"),
               "heldout_ci": (gr.get("heldout") or {}).get("ci"), "within_uncertainty": gr.get("within_uncertainty"), "why_unmet": gr.get("evidence"), "audit_issues": [{"severity": i["severity"], "code": i["code"], "message": i["message"][:200]} for i in issues],
               "automl": {"stop_reason": stop, "rounds": len(ar.get("rounds", [])), "trials": ar.get("n_trials"), "seconds": ar.get("seconds"), "ensemble": (ar.get("ensemble") or {}).get("method"),
                          "learning_curve": (ar.get("learning_curve") or {}).get("verdict"), "what_worked": [e for e in ar.get("effects", []) if (e.get("mean_delta") or 0) > 0.001][:4]},
               "allowed_routes": allowed, "attempt": attempt, "refines_left": refines_left,
               "guidance": ["accept: deliver the current model with an account of what it does and does not achieve.",
                            "refine: run another goal-directed AutoML pass that keeps every trial so far. Write a directive saying WHAT to focus on (e.g. 'the learning curve is flat: concentrate on feature ideas and ensembling')."],
               "output_schema": {"route": "|".join(allowed), "directive": "what the AutoML agent should focus on if refining", "rationale": "2 sentences citing the numbers above"}}

    def default():
        return {"route": "refine" if ("refine" in allowed and stop in ("budget", "rounds", "goal_estimated_met")) else "accept", "directive": "Use the remaining budget on the most promising families and an ensemble.",
                "rationale": "Default policy: refine only when the budget, not the data, ended the search."}
    d, src, calls, warns = director.decide(state, "route_verdict", ANALYST_PERSONA + " You are the team's director, deciding whether the result is good enough to deliver.", payload, _validate_verdict(allowed), default, 900)
    route = d["route"] if d["route"] in allowed else "accept"
    rec = director.record("route_verdict", src, route, d["rationale"], options=allowed, guard=guard)
    out = {**upd, "retry_pending": route == "refine", "route_decision": route, "directive": d["directive"] if route == "refine" else "", "retry_count": int(st.get("retry_count", 0)) + (1 if route == "refine" else 0),
           "decisions": [rec], "llm_calls": calls, "warnings": warns}
    out["agent_log"] = list(upd.get("agent_log", [])) + [log("Director", f"[{src}] verdict: {route.upper()}. {d['rationale']}")]
    _board(state).post("director", "verdict", {"route": route}, text=f"Verdict: {route}. {d['rationale']}")
    return out
