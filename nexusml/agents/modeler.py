"""Modeler (default models) and Tuner (LLM-advised hyper-parameter search with nested CV)."""
from typing import Any, Dict

from .. import config, modeling
from ..storage import get_registry
from ._util import ANALYST_PERSONA, ask, jsonable, log

A_MODEL, A_TUNE = "Modeler", "Tuner"


def current_strategy(state) -> Dict[str, Any]:
    env = {"imbalance": config.DEFAULT_IMBALANCE} if config.DEFAULT_IMBALANCE in ("smote", "adasyn", "random") else {}
    return {**modeling.DEFAULT_STRATEGY, "tune": bool(state.get("enable_tuning", True)),
            "tune_n_iter": state.get("tune_n_iter") or config.TUNE_N_ITER, "tune_threshold": config.TUNE_THRESHOLD,
            "threshold_objective": config.THRESHOLD_OBJECTIVE, **env, **(state.get("strategy") or {})}


def train_models_node(state) -> Dict[str, Any]:
    reg = get_registry()
    plan = state["analysis_plan"]
    splits = reg.load(state["splits_key"])
    strategy = current_strategy(state)
    out = modeling.fit_default(plan["problem_type"], splits, strategy)
    key = reg.save({"models": out["models"]}, "models")
    pm = state["primary_metric"]
    lever = [k for k, v in strategy.items() if v and k in ("class_weight", "imbalance", "log_target", "delta_target", "extra_models", "regularize")]
    best = next(c for c in out["candidates"] if c["name"] == out["best"])
    return {"strategy": strategy, "baseline_metrics": out["baseline"], "candidate_metrics": jsonable(out["candidates"]),
            "best_model_name": out["best"], "models_key": key, "current_status": "Default models trained.", "active_agent": A_MODEL,
            **({"warnings": [f"Candidate {f['name']} was excluded: {f['error']}" for f in out["failed"]]} if out.get("failed") else {}),
            "agent_log": [log(A_MODEL, f"Attempt {state.get('retry_count', 0) + 1}: trained {len(out['candidates'])} model(s)"
                                       f"{' with ' + ', '.join(lever) if lever else ''}. Baseline ({out['baseline']['name']}) CV {pm} "
                                       f"{out['baseline']['cv_score']}; best default: {out['best']} {best['cv_mean']}.")]}


def _thr_sentence(rep: Dict[str, Any]) -> str:
    if not rep or not rep.get("enabled") or rep.get("error"):
        return ""
    return (f" Decision threshold tuned to {rep['threshold']:.3f} (was 0.5; {rep['objective']} on out-of-fold training predictions): {rep['reason']}." if rep.get("adopted")
            else f" Decision threshold: {rep.get('reason', 'kept 0.5')}.")


def _threshold_update(state, reg, strategy, log_list) -> Dict[str, Any]:
    """Tuning is off: still choose the decision threshold for a binary classifier (it needs no search) and persist the wrapped model."""
    pt = state["analysis_plan"]["problem_type"]
    if pt != "classification" or not strategy.get("tune_threshold"):
        return {}
    splits = reg.load(state["splits_key"])
    fitted = reg.load(state["models_key"])["models"]
    cands = [dict(c) for c in state["candidate_metrics"]]
    rep = modeling.finalize_threshold(pt, splits, fitted, cands, strategy, state.get("cost_fn"), state.get("cost_fp"))
    if not rep.get("enabled"):
        return {}
    log_list.append(log(A_TUNE, _thr_sentence(rep).strip()))
    return {"candidate_metrics": jsonable(cands), "best_model_name": modeling.pick_best(cands), "models_key": reg.save({"models": fitted}, "models"),
            "threshold_report": jsonable(rep)}


def tune_models_node(state) -> Dict[str, Any]:
    reg = get_registry()
    plan = state["analysis_plan"]
    pt = plan["problem_type"]
    strategy = current_strategy(state)
    if not strategy.get("tune"):
        out = {"tuning_report": {"enabled": False}, "current_status": "Tuning skipped.", "active_agent": A_TUNE,
               "agent_log": [log(A_TUNE, "Hyper-parameter tuning disabled for this attempt.")]}
        return {**out, **_threshold_update(state, reg, strategy, out["agent_log"])}
    splits = reg.load(state["splits_key"])
    fitted = reg.load(state["models_key"])["models"]
    cands = [dict(c) for c in state["candidate_metrics"]]
    n_iter = int(strategy.get("tune_n_iter") or config.TUNE_N_ITER)

    # ---- LLM advisor proposes search ranges; code clips them into hard bounds
    s = splits["summary"]
    payload = {"problem_type": pt, "business_question": state.get("business_question"),
               "n_train": s["n_train"], "n_features": s["n_features"], "class_distribution": s.get("class_distribution"),
               "metric": state["primary_metric"], "n_iter_budget": n_iter,
               "default_results": [{"model": c["name"], "cv_mean": c["cv_mean"], "cv_std": c["cv_std"],
                                   "heldout": c.get(state["primary_metric"])} for c in cands],
               "allowed_spaces": {c["name"]: modeling.describe_space(modeling.DEFAULT_SPACES[modeling.space_key(c["name"])])
                                  for c in cands if modeling.space_key(c["name"]) in modeling.DEFAULT_SPACES},
               "param_bounds": {k: list(v) for k, v in modeling.PARAM_BOUNDS.items()},
               "spec_format": {"loguniform|randint|uniform": {"type": "", "low": 0, "high": 0}, "choice": {"type": "choice", "values": []}},
               "guidance": ["Small data (<2000 rows): prefer shallower trees / stronger regularisation.",
                            "If CV std is high or train>>CV is likely, regularise more.", "Stay inside param_bounds."],
               "output_schema": {"spaces": {"ModelName": {"param": {"type": "loguniform", "low": 0.01, "high": 10}}},
                                 "rationale": "2-3 sentences on why these ranges suit this dataset"}}
    def v_adv(d):
        sp = d.get("spaces")
        if not isinstance(sp, dict) or not str(d.get("rationale", "")).strip():
            raise ValueError("return 'spaces' (object keyed by model name) and a non-empty 'rationale'")
        return {"spaces": sp, "rationale": str(d["rationale"])[:500]}

    adv, warns, calls = ask(state, "advise_hyperparameter_ranges",
                            ANALYST_PERSONA + " You are also an expert in hyper-parameter optimisation.", payload, 1400, v_adv)
    spaces, rationale = adv["spaces"], adv["rationale"]

    rep = modeling.tune_candidates(pt, splits, strategy, cands, fitted, spaces, n_iter)
    rep.update(space_source="LLM-advised (validated & clipped to safe bounds)", advisor_rationale=rationale)
    thr_rep = modeling.finalize_threshold(pt, splits, fitted, cands, strategy, state.get("cost_fn"), state.get("cost_fp"))
    key = reg.save({"models": fitted}, "models")
    best = modeling.pick_best(cands)
    used = [m for m, e in rep["models"].items() if e.get("used_tuned")]
    summary = ", ".join(f"{m}: {e['default_cv']}->{e['nested_mean']} nested" for m, e in rep["models"].items() if e.get("nested_mean") is not None)
    return {"candidate_metrics": jsonable(cands), "best_model_name": best, "models_key": key, "tuning_report": jsonable(rep),
            "threshold_report": jsonable(thr_rep),
            "warnings": warns + rep.get("notes", []), "llm_calls": calls, "current_status": "Models tuned.", "active_agent": A_TUNE,
            "agent_log": [log(A_TUNE, f"Randomized search ({n_iter} iter, {rep['inner_cv']}-fold inner / {rep['outer_cv']}-fold outer nested CV) "
                                      f"on {len(rep['models'])} model(s), ranges {rep['space_source']}. {summary}. "
                                      f"Tuned model adopted for: {used or 'none (defaults were as good out-of-fold)'}. Best: {best}." + _thr_sentence(thr_rep))]}
