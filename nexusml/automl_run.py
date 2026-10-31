"""Runs one goal-driven AutoML session: portfolio -> LLM-directed rounds -> ensemble -> fresh-fold confirmation -> held-out verdict.

Goal verdict (never decided by the search itself):
  met                  confirmed on fresh training folds AND met once on the held-out split
  met_on_test_only     held-out meets it but the fresh-fold estimate does not: borderline, treat with care
  not_met              held-out misses it; `evidence` says whether the data allowed more (noise ceiling, plateau, budget)
  no_goal              best-effort run
"""
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import targets
from .automl_agent import apply_action, default_actions, digest, goal_estimate, plateau_rounds, validate_actions, validate_decision, _stop_allowed
from .automl_core import Goal, Session, goal_value
from .automl_ensemble import finalize

PERSONA = ("You are the AutoML research lead of a data-science team. Your job is to reach the user's modelling goal with the least wasted compute, and to say so honestly when the data "
           "cannot support it. Read the digest like an experienced Kaggle grandmaster: compare train-vs-CV gaps (variance) against absolute level (bias), use the learning curve to "
           "decide between 'more capacity / better features' and 'more regularisation / ensembling', and use what earlier actions achieved (what_earlier_actions_did) so you never "
           "repeat a failed idea. Prefer cheap, high-yield actions first. Never propose an action listed in rejected_last_time again. Search scores are optimistic once many configs "
           "were tried, so do not chase differences smaller than the CV standard deviation. Propose 'stop' with evidence when the goal is met, when the noise ceiling is below the "
           "goal, or when rounds have stopped improving. Answer with JSON only.")


def default_budget(n_train: int, n_features: int) -> float:
    return float(min(240, max(45, 30 + n_train * max(1, n_features) / 2000)))


def build_session(task: str, splits: Dict[str, Any], strategy: Dict[str, Any], goal: Goal, seed: int, workers: int, budget: Optional[float], log: Callable[[str], None]) -> Session:
    if budget is None and not (goal.spec and goal.spec.get("max_seconds")):
        budget = default_budget(len(splits["y_train"]), splits["X_train"].shape[1])
    return Session(task, splits, strategy, goal, seed=seed, n_workers=workers, time_budget=budget, log=log)


def _inbox(board, sess: Session, seen: Dict[str, int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set]:
    """New messages for the AutoML agent (compact for the LLM), all idea messages so far, and the columns the sentinel proposed to quarantine."""
    msgs, seen["cursor"] = board.read("automl", seen.get("cursor", 0))
    compact = [{"from": m["sender"], "kind": m["kind"], "severity": m["severity"], "text": m["text"][:240], **({"columns": m["payload"].get("columns")} if m["kind"] == "quarantine_proposal" else {})}
               for m in msgs if m["kind"] != "feature_idea"]
    ideas = [m for m in msgs if m["kind"] == "feature_idea"]
    if ideas:
        compact.append({"from": ideas[0]["sender"], "kind": "feature_ideas", "severity": "info", "text": f"{len(ideas)} feature idea(s) available: use the try_ideas action (no params needed)"})
    seen.setdefault("ideas", []).extend(ideas)
    for m in msgs:
        if m["kind"] == "quarantine_proposal":
            seen.setdefault("quarantine", set()).update(m["payload"].get("columns", []))
    return compact, seen["ideas"], seen.get("quarantine", set())


def run(task: str, splits: Dict[str, Any], strategy: Dict[str, Any], goal: Goal, board, ask_fn: Callable, seed: int = 42, workers: int = 1,
        budget: Optional[float] = None, session: Optional[Session] = None, directive: str = "", log: Optional[Callable[[str], None]] = None,
        wait_for_peers: float = 20.0) -> Dict[str, Any]:
    """-> everything the graph stage needs: fit-style result (+ report). `ask_fn(purpose, system, payload, validate, max_tokens) -> (decision, source, calls, warnings)`."""
    log = log or (lambda m: None)
    t_start = time.time()
    resume = session is not None
    sess = session or build_session(task, splits, strategy, goal, seed, workers, budget, log)
    if resume:                                                          # a refine pass: keep every trial, extend the budget and the round allowance
        sess.budget += float(budget or sess.budget * 0.5)
        goal.max_rounds = len(getattr(sess, "rounds", [])) + max(2, goal.max_rounds // 2)
    sess.rounds = getattr(sess, "rounds", [])
    sess.decisions = getattr(sess, "decisions", [])
    sess.llm_calls = getattr(sess, "llm_calls", [])
    sess.warnings = getattr(sess, "warnings", [])
    sess.ens_method, sess.ens_size = getattr(sess, "ens_method", "greedy"), getattr(sess, "ens_size", 15)
    seen: Dict[str, Any] = getattr(sess, "inbox_state", {})
    sess.inbox_state = seen
    rejected: List[str] = []
    lc: Optional[Dict[str, Any]] = None
    stop_reason, stop_why = None, ""

    if not sess.trials:
        log(f"Portfolio: {len(sess.families())} model families, {'successive halving (large data)' if sess.big else 'full CV (small data)'}, {sess.n_workers} worker(s).")
        sess.round = 0
        sess.portfolio()
        if sess.best() is None:
            errs = sorted({t.error for t in sess.trials if t.status == 'failed' and t.error})[:3]
            raise RuntimeError("AutoML: no model could be trained. " + ("Errors: " + "; ".join(errs) if errs else "No trial finished inside the time budget; raise the budget or reduce the data."))
        board.post("automl", "progress", {"best": sess.best().cfg.label()}, text=f"Portfolio done: best {sess.best().cfg.label()} {sess.best().mean:.4f} {sess.pm} (search CV).")
        board.wait_for(["sentinel"], timeout=min(wait_for_peers, max(1.0, 0.15 * sess.budget)))

    while True:
        n_done = len(sess.rounds)
        if n_done >= goal.max_rounds:
            stop_reason, stop_why = "rounds", "round limit reached"
            break
        if sess.time_left() < 3:
            stop_reason, stop_why = "budget", "time budget used"
            break
        inbox, all_ideas, quarantine = _inbox(board, sess, seen)
        sess.quarantine_ok = set(quarantine)
        pending_leaks = sorted(c for c in quarantine if c in sess.X.columns)
        est = goal_estimate(sess)
        b = sess.best()
        # A goal "met" while a flagged leak is still in the data is not a success: the sentinel's proposal must be reviewed first, whatever the score says.
        if goal.active and est.get("met") and b is not None and not pending_leaks and (est.get("gap") is not None and -est["gap"] >= 0.5 * max(b.std, 0.002)):
            stop_reason, stop_why = "goal_estimated_met", f"search estimate {est['value']} clears the target with a margin; confirmation follows"
            break
        sess.round = n_done + 1
        if lc is None or n_done in (1, 3):
            lc = sess.learning_curve(sess.best().cfg)
        dg = digest(sess, sess.rounds, inbox, rejected, lc, directive)
        dec, source, calls, warns = ask_fn("automl_round", PERSONA, dg, validate_decision, 1800, lambda: default_actions(sess, sess.rounds, all_ideas))
        sess.llm_calls += calls
        sess.warnings += warns
        rejected = []
        stop_accepted = False
        if dec.get("stop"):
            ok, why = _stop_allowed(sess, dec["stop"]["reason"], sess.rounds)
            if ok:
                stop_accepted = True
            else:
                rejected.append(f"stop({dec['stop']['reason']}): {why}")
        actions, rej = validate_actions(sess, [a for a in dec["actions"] if a["type"] != "stop"], all_ideas, quarantine)
        rejected += rej
        if actions:                                                     # the round's actions run FIRST, even when the same answer also asks to stop
            before = sess.best().mean
            results = [apply_action(sess, a) for a in actions]
            gain = round(sess.best().mean - before, 4)
            log(f"Round {sess.round} ({source}): " + ", ".join(r["type"] + (f" {r['params'].get('recipe') or r['params'].get('family') or ''}" if r["params"] else "") for r in results)
                + f" -> best {sess.best().cfg.label()} {sess.best().mean:.4f} ({gain:+.4f})")
            sess.rounds.append({"round": sess.round, "diagnosis": dec["diagnosis"], "source": source, "actions": results, "rejected": rejected, "gain": gain, "best": sess.best().cfg.label(),
                                "best_cv": round(sess.best().mean, 4)})
            sess.decisions.append({"junction": "automl_round", "round": sess.round, "source": source, "diagnosis": dec["diagnosis"], "decision": [r["type"] for r in results], "accepted": True})
            board.post("automl", "progress", {"round": sess.round}, text=f"Round {sess.round}: best {sess.best().cfg.label()} {sess.best().mean:.4f} ({gain:+.4f}).")
        else:
            rounds_idle = sum(1 for r in sess.rounds[-2:] if not r["actions"])
            sess.rounds.append({"round": sess.round, "diagnosis": dec["diagnosis"], "source": source, "actions": [], "rejected": rejected, "gain": 0.0, "best": b.cfg.label()})
            if stop_accepted:
                sess.decisions.append({"junction": "automl_round", "round": sess.round, "source": source, "diagnosis": dec["diagnosis"], "decision": "stop: " + dec["stop"]["reason"], "accepted": True})
        if stop_accepted:
            stop_reason, stop_why = dec["stop"]["reason"], str(dec["stop"].get("why", ""))[:300]
            break
        if not actions:
            if rounds_idle >= 1:
                stop_reason, stop_why = "no_valid_actions", "two rounds without a valid action"
                break
            continue
        if plateau_rounds(sess.rounds) >= 3:
            stop_reason, stop_why = "plateau", "three rounds in a row improved the CV score by less than 0.0005"
            break

    # ---------------------------------------------------------------- finalise: ensemble, fresh-fold confirmation, held-out
    out = finalize(sess, sess.ens_method, sess.ens_size)
    best = out["best"]
    est_obj = out["models"][best]
    X_te = splits["X_test"].drop(columns=[c for c in sess.dropped if c in splits["X_test"].columns])
    y_te = np.asarray(splits["y_test"])
    report_goal = goal_verdict(sess, out, best, est_obj, X_te, y_te, stop_reason)
    seconds = round(time.time() - t_start, 1)
    out.update(sess=sess, goal_report=report_goal, stop_reason=stop_reason, stop_why=stop_why, seconds=seconds, lc=lc)
    return out


def goal_verdict(sess: Session, out: Dict[str, Any], best: str, est, X_te, y_te, stop_reason: Optional[str]) -> Dict[str, Any]:
    g = sess.goal
    if not g.active:
        return {"status": "no_goal", "text": g.describe()}
    S = out["ens_fresh_oof"] if best.startswith(("Ensemble", "Stacked")) and out.get("ens_fresh_oof") is not None else out["fresh_oof"].get(best)
    y = sess.y
    mask = np.ones(len(y), bool) if sess.task != "time_series" else sess.oof_mask
    thr = None
    conf: Dict[str, Any] = {}
    if S is not None:
        s2 = S[mask]
        if sess.task == "classification" and s2.ndim == 2 and s2.shape[1] == 2:
            thr = targets.choose_threshold(g.spec, y[mask], s2[:, 1])
        conf = goal_value(g, sess.task, y[mask], s2, thr)
    try:
        if sess.task == "classification" and sess.k > 2:
            held = goal_value(g, sess.task, y_te, np.asarray(est.predict_proba(X_te)), None)
        else:
            held = targets.heldout_report(est, X_te, y_te, g.spec, sess.task, thr)
    except Exception as e:
        held = {"met": False, "value": None, "error": f"{type(e).__name__}: {str(e)[:100]}"}
    c_ok, h_ok = bool(conf.get("met")), bool(held.get("met"))
    ci = held.get("ci")
    uncertain = bool(ci and ci[0] <= g.value <= ci[1] and not h_ok)
    status = "met" if (c_ok and h_ok) else "met_on_test_only" if h_ok else "not_met"
    ceiling = sess.ceiling()
    ev = []
    if status == "not_met":
        if ceiling.get("available"):
            cap = ceiling.get("max_accuracy") if g.metric in ("accuracy", "balanced_accuracy") else ceiling.get("max_r2") if g.metric == "r2" else None
            if cap is not None and g.bound == "min":
                ev.append(f"rows with identical features but different outcomes cap {g.metric} at about {cap}" + (" (below the goal)" if g.value > cap else ""))
        if stop_reason in ("plateau", "infeasible"):
            ev.append(f"the search plateaued after {len(sess.rounds)} round(s)")
        if stop_reason in ("budget", "rounds"):
            ev.append("the time/round budget ended the search; more budget might help")
    if uncertain:
        ev.append(f"the held-out 95% interval {ci} still contains the target, so the true performance may be at the goal; a larger test set would settle it")
    return {"status": status, "within_uncertainty": uncertain, "text": g.describe(), "metric": g.metric, "target": g.value, "bound": g.bound, "confirmed_on_fresh_folds": conf, "heldout": held, "threshold": thr,
            "search_estimate": goal_estimate(sess), "ceiling": ceiling, "evidence": ev, "stop_reason": stop_reason,
            "protocol": "searched with CV on the training split; re-scored on fresh folds; held-out split scored once at the end"}
