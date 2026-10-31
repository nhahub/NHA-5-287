"""Critic: an independent quality gate. It never trains anything; it only judges the evidence.
All comparisons use the selection score (default CV, or nested CV for tuned models)."""
import time
from typing import Any, Dict, List

from .. import config, targets
from ..storage import get_registry
from ._util import log

AGENT = "Critic"


def audit_models_node(state) -> Dict[str, Any]:
    pm = state["primary_metric"]
    cands, base = state["candidate_metrics"], state["baseline_metrics"]
    best = next(c for c in cands if c["name"] == state["best_model_name"])
    issues: List[Dict[str, str]] = []
    sel, test = best["select_score"], best.get(pm)
    gain_cv = sel - base["cv_score"]
    gain_test = (test - base.get(pm, 0)) if test is not None else None
    summary = state.get("data_summary", {})

    def add(sev, code, msg):
        issues.append({"severity": sev, "code": code, "message": msg})

    if gain_cv < state.get("min_improvement", config.MIN_IMPROVEMENT):
        add("blocker", "no_gain", f"Best CV {pm} {sel} beats baseline ({base.get('name', 'baseline')}) {base['cv_score']} by only {gain_cv:.3f} "
                                  f"(need >= {state.get('min_improvement', config.MIN_IMPROVEMENT)}).")
    if gain_test is not None and gain_test < 0:
        add("blocker", "worse_than_baseline_test", f"Held-out {pm} {test} is below the baseline {base.get(pm)}.")
    if test is not None and sel - test > config.OVERFIT_GAP:
        add("warning", "overfit", f"Selection score {sel} vs held-out {test}: gap {sel - test:.3f} suggests overfitting or a shifted test set.")
    if best["cv_std"] > config.HIGH_CV_STD:
        add("warning", "high_variance", f"CV std {best['cv_std']} is high; the estimate is unstable.")
    if test is not None and test >= config.PERFECT_SCORE:
        add("warning", "too_good", f"Held-out {pm} {test} is suspiciously perfect: check for leakage.")
    if state["analysis_plan"]["problem_type"] == "classification":
        ms = summary.get("minority_share", 1)
        if ms < config.IMBALANCE_SHARE and not state.get("strategy", {}).get("class_weight"):
            add("warning", "imbalance", f"Minority class is only {ms:.0%} of training rows and no class weighting was used.")
    elif summary.get("target_skew") is not None and abs(summary["target_skew"]) > config.SKEW_THRESHOLD \
            and summary.get("target_min", -1) >= 0 and not state.get("strategy", {}).get("log_target") \
            and state["analysis_plan"]["problem_type"] == "regression":
        add("warning", "skewed_target", f"Target skew {summary['target_skew']:.1f}: a log transform may help.")
    if summary.get("n_train", 1e9) < config.SMALL_TRAIN:
        add("warning", "small_data", f"Only {summary['n_train']} training rows; results are fragile.")
    tr = state.get("tuning_report") or {}
    if tr.get("enabled") and tr.get("models") and not any(e.get("used_tuned") for e in tr["models"].values()):
        add("info", "tuning_no_help", "Hyper-parameter tuning did not beat the defaults out-of-fold, so defaults were kept.")
    tgt_update: Dict[str, Any] = {}
    sr = state.get("sentinel_report") or {}
    still = [x["column"] for x in sr.get("leakage_suspects", []) if x.get("severity") == "high" and x["column"] not in (state.get("quarantined") or [])]
    if still:
        add("warning", "leakage_suspect_kept", f"The sentinel flagged {still} as a possible leak and it is still a model feature: treat the scores with suspicion until a domain expert confirms the column is known at prediction time.")
    spec = state.get("target_spec")
    tgt_update: Dict[str, Any] = {}
    ar = state.get("automl_report") or {}
    if spec and ar.get("goal"):                                         # v6: the AutoML engine already judged the goal on fresh folds + held-out; a clone-based re-check of an ensemble would be leaky
        gr = ar["goal"]
        if gr.get("status") == "not_met":
            ev = ("; ".join(gr.get("evidence") or [])) or "see the AutoML report"
            add("blocker", "target_not_met", f"Target ({gr.get('text')}) not met: fresh-fold {gr.get('confirmed_on_fresh_folds', {}).get('value')}, held-out {gr.get('heldout', {}).get('value')} ({ev}).")
        elif gr.get("status") == "met_on_test_only":
            add("warning", "target_borderline", f"Target ({gr.get('text')}) met on the held-out set but not on the fresh training folds ({gr.get('confirmed_on_fresh_folds', {}).get('value')}): borderline.")
    elif spec:
        try:
            ps = targets.parse(spec, state["analysis_plan"]["problem_type"])
            reg = get_registry()
            splits = reg.load(state["splits_key"])
            est = reg.load(state["models_key"])["models"][best["name"]]
            vr = targets.validation_report(est, splits["X_train"], splits["y_train"], ps, state["analysis_plan"]["problem_type"])
            prev = dict(state.get("target_report") or {})
            att = list(prev.get("attempts", [])) + [{"attempt": state.get("retry_count", 0) + 1, "model": best["name"], **vr}]
            tgt_update = {"target_report": {**prev, "spec": ps, "t0": prev.get("t0", time.time()), "attempts": att, "validation": vr}}
            sec = ps.get("secondary")
            goal = f"{ps['metric']} {'>=' if ps['bound'] == 'min' else '<='} {ps['value']}" + (f" with {sec['metric']} {'>=' if sec['bound'] == 'min' else '<='} {sec['value']}" if sec else "")
            if not vr["met"]:
                add("blocker", "target_not_met", f"Target ({goal}) not met on out-of-fold training predictions: {vr['metric']} = {vr['value']}"
                    + (f", {vr['secondary']['metric']} = {vr['secondary']['value']}" if vr.get("secondary") else "") + ".")
        except ValueError as e:
            add("warning", "target_invalid", f"Target ignored: {e}")
    passed = not any(i["severity"] == "blocker" for i in issues)
    audit = {"passed": passed, "issues": issues, "gain_cv": round(gain_cv, 4),
             "gain_test": None if gain_test is None else round(gain_test, 4), "best": best["name"]}
    attempt = {"attempt": state.get("retry_count", 0) + 1, "levers": [k for k, v in (state.get("strategy") or {}).items()
                                                                       if v and k in ("class_weight", "log_target", "delta_target", "extra_models", "regularize")],
               "tuned": bool(tr.get("enabled")), "best": best["name"], "select_score": sel, "heldout": test, "passed": passed}
    return {**tgt_update, "audit": audit, "quality_passed": passed, "attempts": [attempt], "current_status": "Quality audit complete.",
            "active_agent": AGENT,
            "agent_log": [log(AGENT, f"{'PASS' if passed else 'FAIL'}: " + ("; ".join(i['code'] for i in issues) or "no issues")
                              + f". Gain over baseline: CV {gain_cv:+.3f}" + (f", test {gain_test:+.3f}." if gain_test is not None else "."))]}


def retry_budget(state) -> int:
    spec = state.get("target_spec")
    return max(int(state.get("max_retries", 2)), int(spec.get("max_retries", 4))) if spec else int(state.get("max_retries", 2))


def stop_reason(state) -> str:
    """Why the model loop ended (used in the target report)."""
    if (state.get("automl_report") or {}).get("stop_text"):
        return state["automl_report"]["stop_text"]
    tr = state.get("target_report") or {}
    if state.get("quality_passed"):
        return "target met" if tr.get("spec") else "quality gate passed"
    spec = tr.get("spec") or {}
    if tr.get("t0") and time.time() - tr["t0"] > float(spec.get("max_seconds", config.TARGET_MAX_SECONDS)):
        return "time budget reached"
    if state.get("strategy_exhausted"):
        return "no untried strategy lever left"
    return "retry budget reached"


def route_quality(state) -> str:
    if state.get("quality_passed"):
        return "explain"
    tr = state.get("target_report") or {}
    spec = tr.get("spec") or {}
    if tr.get("t0") and time.time() - tr["t0"] > float(spec.get("max_seconds", config.TARGET_MAX_SECONDS)):
        return "explain"
    if state.get("retry_count", 0) < retry_budget(state) and not state.get("strategy_exhausted"):
        return "revise"
    return "explain"
