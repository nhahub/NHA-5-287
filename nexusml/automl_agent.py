"""The AutoML research lead: an LLM that decides WHAT to try next, inside a deterministic safety envelope.

  The LLM sees  a compact DIGEST (goal gap, leaderboard, learning curve, noise ceiling, what earlier actions achieved, messages from the other agents, budget).
  The LLM picks from a WHITELISTED action menu (search_family, try_recipe, try_ideas, class_weight, regularize, log_target, deepen, ensemble, drop_features, stop).
  It never writes code, never sees raw rows, and never sees the held-out split.
  Every action is validated against the real data (columns exist, the recipe is applicable, it was not already tried, the budget allows it); a rejected action is
  reported back to the LLM in the next digest. `stop` is accepted only when the evidence supports it.
  If the LLM is unavailable the same loop runs on a deterministic default policy (flagged as degraded), so a run never dies for lack of a decision.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import targets
from .automl_core import Session, goal_value
from .automl_recipes import IDEA_OPS, RECIPES, validate_ideas

ACTION_MENU = {
    "search_family": {"params": {"family": "one of families_available", "n_trials": "1-12", "focus": "'exploit' (refine the best config) | 'explore' (sample widely)",
                                 "space_hint": "optional {param: {type, low, high}} narrowing the search range (validated and clipped to safe bounds)"},
                      "use_when": "a family looks promising but is under-explored, or the best family has not converged"},
    "try_recipe": {"params": {"recipe": "one of recipe_catalog", "families": "optional list; default = the 3 best families"},
                   "use_when": "the learning curve is flat (a feature problem, not a data problem) or the data has skew / missing values / high-cardinality categoricals"},
    "try_ideas": {"params": {"ideas": "optional inline ideas [{op, cols, name, why}]; omit to use the ideas proposed by the scout agent"},
                  "use_when": "domain ratios / differences / group statistics could expose structure the models cannot build themselves"},
    "class_weight": {"params": {}, "use_when": "classification with an imbalanced target, or recall/F1 goals"},
    "regularize": {"params": {}, "use_when": "large train-vs-CV gap (over-fitting) or very small data"},
    "log_target": {"params": {}, "use_when": "regression with a right-skewed, non-negative target"},
    "deepen": {"params": {}, "use_when": "the top configs are within noise of each other; re-score them on other folds before trusting the order"},
    "ensemble": {"params": {"method": "'greedy' | 'stack'", "size": "5-25"}, "use_when": "several different families score within a few points: averaging them usually helps; choose before stopping"},
    "drop_features": {"params": {"columns": "only columns the sentinel agent flagged for quarantine"}, "use_when": "the sentinel found leakage"},
    "stop": {"params": {"reason": "goal_met | infeasible | plateau | satisfied", "why": "evidence from the digest"},
             "use_when": "the goal is met on the confirmed estimate, or the evidence says it cannot be reached (noise ceiling, plateau), or more search would only chase noise"},
}
MAX_ACTIONS = 4
MAX_TRIALS_PER_ACTION = 12
RECIPE_ORDER = ["power", "missing_ind", "interactions", "target_enc", "select"]


# goal estimate during the search
def goal_estimate(sess: Session, trial=None) -> Dict[str, Any]:
    """The goal metric on the best model's SEARCH out-of-fold predictions. Optimistic after many trials: used only to steer, never to declare victory."""
    g = sess.goal
    if not g.active:
        return {"active": False}
    t = trial or sess.best()
    if t is None or t.oof is None:
        return {"active": True, "value": None}
    m = sess.oof_mask
    y, S = sess.y[m], t.oof[m]
    try:
        if sess.task == "classification" and S.ndim == 2 and S.shape[1] == 2:
            thr = targets.choose_threshold(g.spec, y, S[:, 1])
            r = goal_value(g, sess.task, y, S, thr)
        else:
            r = goal_value(g, sess.task, y, S, None)
    except Exception as e:
        return {"active": True, "value": None, "error": f"{type(e).__name__}: {str(e)[:80]}"}
    r["gap"] = g.gap(r.get("value"))
    r["source"] = "search CV (optimistic)"
    return r


# digest
def _data_facts(sess: Session) -> Dict[str, Any]:
    X = sess.X
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    facts = {"n_train": int(len(sess.y)), "n_features": int(X.shape[1]), "n_numeric": len(num), "n_categorical": len(cat), "task": sess.task, "n_classes": sess.k or None,
             "high_cardinality_categoricals": [c for c in cat if X[c].nunique() > 12][:6], "columns_with_missing": [c for c in X.columns if X[c].isna().any()][:8],
             "skewed_numeric_columns": [c for c in num if abs(float(X[c].skew())) > 2][:8] if num else [], "large_data_uses_successive_halving": sess.big, "dropped_columns": list(sess.dropped)}
    if sess.task == "classification":
        vc = np.bincount(sess.y) / len(sess.y)
        facts["class_balance"] = [round(float(v), 3) for v in vc]
        facts["minority_share"] = round(float(vc.min()), 3)
    else:
        yy = sess.y.astype(float)
        facts["target_skew"] = round(float(np.nan_to_num(((yy - yy.mean()) ** 3).mean() / (yy.std() ** 3 + 1e-12))), 2)
        facts["target_non_negative"] = bool((yy >= 0).all())
    return facts


def applicable_recipes(sess: Session) -> List[str]:
    f = _data_facts(sess)
    ok = []
    if f["n_numeric"] >= 1:
        ok.append("power")
    if f["columns_with_missing"]:
        ok.append("missing_ind")
    if f["n_numeric"] >= 2:
        ok.append("interactions")
    if f["high_cardinality_categoricals"]:
        ok.append("target_enc")
    if f["n_features"] >= 8:
        ok.append("select")
    return ok


def tried_recipes(sess: Session) -> List[str]:
    return sorted({r for e in sess.effects for r in e.get("recipe", [])})


def digest(sess: Session, rounds: List[Dict[str, Any]], inbox: List[Dict[str, Any]], rejected: List[str], lc: Optional[Dict[str, Any]], directive: str = "") -> Dict[str, Any]:
    g = sess.goal
    fb = sorted(sess.family_best().values(), key=lambda t: -t.mean)
    est = goal_estimate(sess)
    untried = [r for r in applicable_recipes(sess) if r not in tried_recipes(sess)]
    best = sess.best()
    return {
        "goal": {"text": g.describe(), "active": g.active, "metric": g.metric, "target": g.value, "direction": "higher is better" if g.bound == "min" else "lower is better",
                 "estimate_on_search_cv": est.get("value"), "estimate_meets_goal": est.get("met"), "gap_to_goal": est.get("gap"),
                 "note": "this estimate is optimistic (many configs were tried); the goal is judged later on fresh folds and then once on held-out data"},
        "search_metric": sess.pm, "best_so_far": None if best is None else {"model": best.cfg.label(), "cv_mean": round(best.mean, 4), "cv_std": round(best.std, 4)},
        "budget": {"seconds_left": round(max(0.0, sess.time_left()), 1), "rounds_left": max(0, g.max_rounds - len(rounds)), "trials_run": len(sess.trials), "workers": sess.n_workers},
        "data": _data_facts(sess),
        "leaderboard": [{"family": t.cfg.family, "best_config": t.cfg.label(), "cv_mean": round(t.mean, 4), "cv_std": round(t.std, 4),
                         "train_minus_cv": None if t.train_mean is None else round(t.train_mean - t.mean, 4),
                         "n_trials": sum(1 for x in sess.full() if x.cfg.family == t.cfg.family), "avg_seconds": round(float(np.mean([x.seconds for x in sess.full() if x.cfg.family == t.cfg.family])), 2)}
                        for t in fb[:6]],
        "what_earlier_actions_did": [{k: v for k, v in e.items() if k != "families"} | {"families": e.get("families", [])[:3]} for e in sess.effects[-6:]],
        "learning_curve_of_best": lc, "noise_ceiling": sess.ceiling(),
        "messages_from_other_agents": inbox, "your_previous_rounds": rounds[-3:], "rejected_last_time": rejected[-6:],
        "recipes_untried_and_applicable": untried, "recipe_catalog": {k: v for k, v in RECIPES.items() if k in applicable_recipes(sess)},
        "families_available": sess.families(),
        "families_left_out_by_advisor": [{"family": e["family"], "why": e["why"]} for e in ((sess.plan or {}).get("excluded") or [])][:8],
        "idea_ops": list(IDEA_OPS), "plateau_rounds": plateau_rounds(rounds), "directive_from_reviewer": directive or None,
        "action_menu": ACTION_MENU, "max_actions": MAX_ACTIONS,
        "output_schema": {"diagnosis": "2-3 sentences: what limits the score (bias / variance / noise / features / data size), citing numbers from this digest",
                          "actions": [{"type": "one of action_menu", "params": {}, "why": "one sentence"}], "stop": "null, or {\"reason\": \"goal_met|infeasible|plateau|satisfied\", \"why\": \"evidence\"}"}}


def plateau_rounds(rounds: List[Dict[str, Any]]) -> int:
    n = 0
    for r in reversed(rounds):
        if (r.get("gain") or 0) < 0.0005:
            n += 1
        else:
            break
    return n


# validation (the guards)
def validate_decision(d: Any) -> Dict[str, Any]:
    """Schema check for the raw LLM answer (raises ValueError so the model gets one corrective retry)."""
    if not isinstance(d, dict):
        raise ValueError("answer must be a JSON object")
    if not str(d.get("diagnosis", "")).strip():
        raise ValueError("'diagnosis' must be a non-empty string")
    acts = d.get("actions", [])
    if not isinstance(acts, list):
        raise ValueError("'actions' must be a list")
    clean = []
    for a in acts:
        if not isinstance(a, dict) or a.get("type") not in ACTION_MENU:
            raise ValueError(f"unknown action {a!r}; allowed types: {sorted(ACTION_MENU)}")
        clean.append({"type": a["type"], "params": a.get("params") if isinstance(a.get("params"), dict) else {}, "why": str(a.get("why", ""))[:200]})
    stop = d.get("stop")
    if stop is not None and not (isinstance(stop, dict) and stop.get("reason") in ("goal_met", "infeasible", "plateau", "satisfied")):
        raise ValueError("'stop' must be null or {reason: goal_met|infeasible|plateau|satisfied, why}")
    if not clean and not stop:
        raise ValueError("give at least one action, or a stop")
    return {"diagnosis": str(d["diagnosis"])[:700], "actions": clean[:MAX_ACTIONS + 2], "stop": stop}


def _stop_allowed(sess: Session, reason: str, rounds: List[Dict[str, Any]]) -> Tuple[bool, str]:
    pend = sorted(c for c in getattr(sess, "quarantine_ok", set()) if c in sess.X.columns)
    if pend and reason in ("goal_met", "satisfied", "plateau"):
        return False, f"a possible leak is still in the data ({pend}): review the sentinel's proposal before stopping"
    est = goal_estimate(sess)
    if reason == "goal_met":
        return (bool(est.get("met")) or not sess.goal.active), "the search estimate does not meet the goal yet"
    if reason == "infeasible":
        c = sess.ceiling()
        g = sess.goal
        beyond = False
        if g.active and c.get("available"):
            cap = c.get("max_accuracy") if g.metric in ("accuracy", "balanced_accuracy") else c.get("max_r2") if g.metric == "r2" else None
            beyond = cap is not None and g.bound == "min" and g.value > cap + 1e-6
        best = sess.best()
        flat = len(rounds) >= 2 and plateau_rounds(rounds) >= 2 and best is not None and (est.get("gap") or 0) > 3 * max(best.std, 0.01)
        return (beyond or flat), "infeasible needs a noise ceiling below the goal, or two flat rounds with a gap larger than 3 standard deviations"
    if reason in ("plateau", "satisfied"):
        return (plateau_rounds(rounds) >= 1 or not sess.goal.active or len(rounds) >= sess.goal.max_rounds - 1), "no plateau yet: the last round still improved the score"
    return True, ""


def validate_actions(sess: Session, actions: List[Dict[str, Any]], inbox_all: List[Dict[str, Any]], quarantine_ok: set) -> Tuple[List[Dict[str, Any]], List[str]]:
    ok: List[Dict[str, Any]] = []
    why: List[str] = []
    fams = {f.lower(): f for f in (getattr(sess, "_pool", None) or sess.families())}      # the agent may ask for ANY family, including ones the advisor left out
    seen_eff = {(tuple(e.get("recipe", [])), tuple(e.get("ideas", [])), tuple(e.get("flags", []))) for e in sess.effects}
    app = applicable_recipes(sess)
    for a in actions:
        t, p = a["type"], a.get("params") or {}
        if len(ok) >= MAX_ACTIONS:
            why.append(f"{t}: more than {MAX_ACTIONS} actions in one round")
            continue
        if t == "search_family":
            if not sess.allow_search:
                why.append("search_family: hyper-parameter tuning is switched off for this run")
                continue
            f = fams.get(str(p.get("family", "")).lower())
            if not f:
                why.append(f"search_family: unknown family {p.get('family')!r}")
                continue
            if not sess.space(f):
                why.append(f"search_family: {f} has no tunable parameters")
                continue
            n = int(min(MAX_TRIALS_PER_ACTION, max(1, int(p.get("n_trials", 6) or 6))))
            ok.append({**a, "params": {"family": f, "n_trials": n, "focus": p.get("focus") if p.get("focus") in ("explore", "exploit") else None, "space_hint": p.get("space_hint")}})
        elif t == "try_recipe":
            r = str(p.get("recipe", "")).lower()
            if r not in RECIPES:
                why.append(f"try_recipe: unknown recipe {r!r}")
            elif r not in app:
                why.append(f"try_recipe: {r} does not apply to this data")
            elif ((r,), (), ()) in seen_eff:
                why.append(f"try_recipe: {r} was already tried")
            else:
                fl = [fams[x.lower()] for x in (p.get("families") or []) if str(x).lower() in fams] or None
                ok.append({**a, "params": {"recipe": r, "families": fl}})
        elif t == "try_ideas":
            raw = p.get("ideas") or [i["payload"] for m in inbox_all if m.get("kind") == "feature_idea" for i in [m]]
            ideas, rej = validate_ideas(raw, sess.X)
            ideas = [i for i in ideas if i["name"] not in sess.ideas]
            if not ideas:
                why.append("try_ideas: no valid new ideas" + (f" ({'; '.join(rej[:2])})" if rej else ""))
            else:
                ok.append({**a, "params": {"ideas": ideas}})
        elif t == "class_weight":
            if sess.task != "classification" or ((), (), ("class_weight=balanced",)) in seen_eff:
                why.append("class_weight: not a classification task, or already tried")
            else:
                ok.append({**a, "params": {}})
        elif t == "regularize":
            if ((), (), ("regularize",)) in seen_eff:
                why.append("regularize: already tried")
            else:
                ok.append({**a, "params": {}})
        elif t == "log_target":
            if sess.task == "classification" or not (sess.y.astype(float) >= 0).all() or ((), (), ("log_target",)) in seen_eff:
                why.append("log_target: needs a non-negative regression target and must not have been tried")
            else:
                ok.append({**a, "params": {}})
        elif t == "deepen":
            if not any(e.get("deepen") for e in sess.effects[-2:]):
                ok.append({**a, "params": {}})
            else:
                why.append("deepen: done very recently")
        elif t == "ensemble":
            m = p.get("method") if p.get("method") in ("greedy", "stack") else "greedy"
            ok.append({**a, "params": {"method": m, "size": int(min(25, max(5, int(p.get("size", 15) or 15))))}})
        elif t == "drop_features":
            cols = [c for c in (p.get("columns") or []) if c in quarantine_ok and c in sess.X.columns]
            if cols:
                ok.append({**a, "params": {"columns": cols}})
            else:
                why.append("drop_features: only columns the sentinel flagged for quarantine can be dropped")
        elif t == "stop":
            ok.append({**a, "params": p})
    return ok, why


# default policy (used when the LLM cannot answer)
def default_actions(sess: Session, rounds: List[Dict[str, Any]], inbox_all: List[Dict[str, Any]]) -> Dict[str, Any]:
    acts: List[Dict[str, Any]] = []
    q = [c for c in sorted(getattr(sess, "quarantine_ok", set())) if c in sess.X.columns]
    if q:
        acts.append({"type": "drop_features", "params": {"columns": q}, "why": "the sentinel proposed quarantining likely leakage"})
    untried = [r for r in RECIPE_ORDER if r in applicable_recipes(sess) and r not in tried_recipes(sess)]
    fb = sorted(sess.family_best().values(), key=lambda t: -t.mean)
    n = len(rounds)
    if n == 0 and sess.task == "classification" and _data_facts(sess).get("minority_share", 1) < 0.2:
        acts.append({"type": "class_weight", "params": {}, "why": "imbalanced classes"})
    if fb:
        acts.append({"type": "search_family", "params": {"family": fb[0].cfg.family, "n_trials": 6, "focus": "exploit"}, "why": "refine the leader"})
    for r in untried[:2]:
        acts.append({"type": "try_recipe", "params": {"recipe": r}, "why": f"default order: {r}"})
    if any(m.get("kind") == "feature_idea" for m in inbox_all) and n >= 1:
        acts.append({"type": "try_ideas", "params": {}, "why": "scout proposed ideas"})
    if n >= 2:
        acts.append({"type": "deepen", "params": {}, "why": "confirm the order of the top configs"})
        acts.append({"type": "ensemble", "params": {"method": "greedy", "size": 15}, "why": "average the strong models"})
    stop = {"reason": "plateau", "why": "default policy: nothing left in the standard list"} if (n >= 3 or (not untried and n >= 1)) else None
    return {"diagnosis": "Default policy (no LLM decision available): refine the leader, then try the standard feature recipes in a fixed order.", "actions": acts[:MAX_ACTIONS], "stop": stop}


# applying actions
def apply_action(sess: Session, a: Dict[str, Any]) -> Dict[str, Any]:
    t, p = a["type"], a["params"]
    before = sess.best().mean if sess.best() else None
    res: Dict[str, Any] = {"type": t, "why": a.get("why", ""), "params": {k: v for k, v in p.items() if k not in ("ideas",)}}
    n0 = len(sess.trials)
    try:
        if t == "search_family":
            if p.get("space_hint"):
                res["notes"] = sess.set_hint(p["family"], p["space_hint"])
            sess.run_search(p["n_trials"], [p["family"]], explore=None if not p.get("focus") else p["focus"] == "explore")
        elif t == "try_recipe":
            res["effect"] = sess.variant_trials(recipe=(p["recipe"],), families=p.get("families"))
        elif t == "try_ideas":
            for i in p["ideas"]:
                sess.ideas[i["name"]] = i
            res["ideas"] = [i["name"] for i in p["ideas"]]
            res["effect"] = sess.variant_trials(ideas=tuple(i["name"] for i in p["ideas"]))
        elif t == "class_weight":
            res["effect"] = sess.variant_trials(flags=("class_weight=balanced",))
        elif t == "regularize":
            res["effect"] = sess.variant_trials(flags=("regularize",))
        elif t == "log_target":
            res["effect"] = sess.variant_trials(flags=("log_target",))
        elif t == "deepen":
            res["effect"] = sess.deepen()
        elif t == "ensemble":
            sess.ens_method, sess.ens_size = p["method"], p["size"]
            res["note"] = f"final ensemble will use {p['method']} selection (size {p['size']})"
        elif t == "drop_features":
            res["effect"] = sess.drop_features(p["columns"])
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:140]}"
    after = sess.best().mean if sess.best() else None
    res["trials"] = len(sess.trials) - n0
    res["best_before"], res["best_after"] = (None if before is None else round(before, 4)), (None if after is None else round(after, 4))
    res["gain"] = None if before is None or after is None else round(after - before, 4)
    return res
