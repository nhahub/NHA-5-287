"""Explainer: permutation importance on RAW features, diagnostics, lift tables, model charts."""
from typing import Any, Dict, List

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score,
                             roc_curve)

from .. import config, diagnostics, fairness, interactive, modeling, targets, uncertainty, viz
from .critic import stop_reason
from ..storage import get_registry
from ._util import jsonable, log, make_evidence

AGENT = "Explainer"


def lift_table(y: np.ndarray, p: np.ndarray) -> List[Dict[str, float]]:
    order = np.argsort(-p)
    y = y[order]
    n, total = len(y), max(1, int(y.sum()))
    rows, cum = [], 0
    for d in range(10):
        part = y[int(d * n / 10): int((d + 1) * n / 10)]
        cum += int(part.sum())
        rows.append({"decile": d + 1, "n": int(len(part)), "event_rate": float(part.mean()) if len(part) else 0.0,
                     "cum_capture": cum / total})
    return rows


def explain_results_node(state) -> Dict[str, Any]:
    reg = get_registry()
    splits = reg.load(state["splits_key"])
    models = reg.load(state["models_key"])["models"]
    name, pt, pm = state["best_model_name"], state["analysis_plan"]["problem_type"], state["primary_metric"]
    est = models[name]
    X_te, y_te = splits["X_test"], splits["y_test"]
    n = min(len(X_te), config.MAX_PERM_ROWS)
    idx = np.random.RandomState(config.RANDOM_STATE).choice(len(X_te), n, replace=False)
    Xs, ys = X_te.iloc[idx], y_te[idx]
    pi = permutation_importance(est, Xs, ys, scoring=pm if pm != "roc_auc" or len(np.unique(ys)) == 2 else "accuracy",
                                n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=1)
    imp = sorted([{"feature": f, "importance": round(float(m), 5), "std": round(float(s), 5)}
                  for f, m, s in zip(X_te.columns, pi.importances_mean, pi.importances_std)], key=lambda r: -r["importance"])
    diag: Dict[str, Any] = {}
    items: List[Any] = []
    best = next(c for c in state["candidate_metrics"] if c["name"] == name)
    base = state["baseline_metrics"]
    pred = est.predict(X_te)
    ci: Dict[str, Any] = {}
    try:
        proba_te = est.predict_proba(X_te)[:, 1] if (pt == "classification" and len(splits["classes"]) == 2 and hasattr(est, "predict_proba")
                                                        and len(np.unique(y_te)) == 2) else None
        ci_metric = pm if pm in ("roc_auc", "r2", "f1_macro") else {"neg_root_mean_squared_error": "rmse"}.get(pm, pm)
        ci = uncertainty.bootstrap_ci(ci_metric, y_te, pred, proba_te)
        n_pos = int((y_te == 1).sum()) if pt == "classification" and len(splits["classes"]) == 2 else None
        ci["n_test"] = int(len(y_te))
        ci["n_test_positives"] = n_pos
        rng_txt = f" ({int(ci['level'] * 100)}% bootstrap CI {ci['lo']}-{ci['hi']})" if ci.get("lo") is not None else ""
        noisy = (f" Only {n_pos} positive cases are in the test set, so this score is noisy." if n_pos is not None and n_pos < config.MIN_TEST_POSITIVES_WARN else "")
        items.append(("model", f"Headline (held-out test set, {len(y_te)} rows): {pm} = {ci['value']}{rng_txt}.{noisy}", ci))
    except Exception as e:
        items.append(("model", f"Held-out confidence interval skipped ({type(e).__name__}).", {}))
    items.append(("model", f"Best model {name} ({best['select_source']}): cross-validated {pm} on training data {best['select_score']} "
                           f"vs baseline {base['cv_score']}. This is the model-selection estimate; the held-out figure above is the headline.", {}))
    stab = {}
    try:
        stab = uncertainty.seed_stability(est, splits["X_train"], splits["y_train"], pt, pm)
        if stab:
            items.append(("model", f"Stability: cross-validated {pm} across {len(stab['seeds'])} fold seeds ranged {min(stab['scores'])}-{max(stab['scores'])} "
                                   f"(sd {stab['sd']}).", stab))
    except Exception:
        pass
    if pt == "classification":
        classes = splits["classes"]
        cm = confusion_matrix(y_te, pred, labels=list(range(len(classes))))
        per = {}
        for i, c in enumerate(classes):
            tp = cm[i, i]
            prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
            rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
            per[c] = {"precision": round(float(prec), 3), "recall": round(float(rec), 3), "support": int(cm[i, :].sum())}
        diag.update(confusion_matrix=cm.tolist(), classes=classes, per_class=per)
        if len(classes) == 2 and hasattr(est, "predict_proba") and len(np.unique(y_te)) == 2:
            p = est.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, p)
            pr, rc, _ = precision_recall_curve(y_te, p)
            step = max(1, len(fpr) // 200)
            diag["roc"] = {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist(), "auc": float(roc_auc_score(y_te, p))}
            diag["pr"] = {"precision": pr[::step].tolist(), "recall": rc[::step].tolist(),
                          "ap": float(average_precision_score(y_te, p)), "base_rate": float(y_te.mean())}
            lt = lift_table(y_te, p)
            diag["lift"], diag["base_rate"] = lt, float(y_te.mean())
            items.append(("model", f"Targeting the top 10% highest-risk records captures {lt[0]['cum_capture']:.0%} of all '{splits['positive_class']}' cases "
                                   f"(lift {lt[0]['event_rate'] / max(1e-9, y_te.mean()):.1f}x); the top 30% captures {lt[2]['cum_capture']:.0%}.", {"lift": lt[:3]}))
    else:
        k = min(len(y_te), 400)
        diag.update(actual=y_te[:k].tolist(), predicted=pred[:k].tolist(),
                    residual_std=round(float(np.std(y_te - pred)), 4))
        items.append(("model", f"Typical held-out error: RMSE {best.get('rmse')}, MAE {best.get('mae')} (R2 {best.get('r2')}).", {}))
    # ---- extra diagnostics (each optional: a failure must not kill the run)
    try:
        if pt == "classification" and len(splits["classes"]) == 2 and hasattr(est, "predict_proba") and len(np.unique(y_te)) == 2:
            p_all = est.predict_proba(X_te)[:, 1]
            diag["calibration"] = diagnostics.calibration_data(y_te, p_all)
            diag["threshold_table"] = diagnostics.threshold_table(y_te, p_all)
            tt = max(diag["threshold_table"], key=lambda r: r["f1"])
            items.append(("model", f"F1 is highest at a probability threshold of {tt['threshold']:.2f} (precision {tt['precision']:.2f}, recall {tt['recall']:.2f}) "
                                   "versus the default 0.50: choose the threshold from the cost of a missed case vs a false alarm.", tt))
            ct = uncertainty.cost_threshold(est, splits["X_train"], splits["y_train"], X_te, y_te, state.get("cost_fn"), state.get("cost_fp"))
            diag["cost_threshold"] = ct
            src = "your stated costs" if ct["costs_from_user"] else f"DEFAULT costs (a miss = {ct['cost_fn']:g}x a false alarm; replace with your real costs)"
            t = ct["test"]
            items.append(("model", f"Cost-based cutoff {ct['threshold']:.2f} (chosen on training out-of-fold predictions using {src}): on the test set it flags "
                                   f"{t['flagged']} records, precision {t['precision']:.2f}, recall {t['recall']:.2f}; expected cost {t['expected_cost']:g} vs {t['cost_at_0.5']:g} at 0.50 "
                                   f"and {t['cost_flag_nobody']:g} if nobody is flagged.", ct))
            cal = diag["calibration"]
            if cal and abs(cal["mean_pred"] - cal["base_rate"]) > 0.05:
                items.append(("model", f"Predicted probabilities are not well calibrated (average predicted {cal['mean_pred']:.1%} vs actual {cal['base_rate']:.1%}); "
                                       "use them for ranking, not as literal risk percentages.", cal))
        lc = diagnostics.learning_curve_data(est, splits["X_train"], splits["y_train"], pm, modeling._cv(pt, splits["y_train"], 3))
        if lc:
            diag["learning"] = lc
            gap, rise = lc["train_mean"][-1] - lc["val_mean"][-1], lc["val_mean"][-1] - lc["val_mean"][-2]
            if gap > 0.1 or rise > 0.01:
                items.append(("model", f"Learning curve: validation {pm} is still improving with more rows (+{rise:.3f} at the last step; train-validation gap {gap:.2f}): "
                                       "collecting more data is likely to help.", lc))
        diag["pdp"] = diagnostics.pdp_curves(est, splits["X_test"], [i["feature"] for i in imp if i["importance"] > 0][:6])
    except Exception as e:
        items.append(("model", f"Some extra diagnostics were skipped ({type(e).__name__}).", {}))
    tgt_out: Dict[str, Any] = {}
    try:
        tr = state.get("target_report") or {}
        if tr.get("spec") and tr.get("validation"):
            spec, vr = tr["spec"], tr["validation"]
            hr = targets.heldout_report(est, X_te, y_te, spec, pt, vr.get("threshold"))
            sec = spec.get("secondary")
            goal = f"{spec['metric']} {'>=' if spec['bound'] == 'min' else '<='} {spec['value']}" + (f" with {sec['metric']} {'>=' if sec['bound'] == 'min' else '<='} {sec['value']}" if sec else "")
            why = stop_reason(state)
            ci_txt = f" (95% bootstrap CI {hr['ci'][0]}-{hr['ci'][1]})" if hr.get("ci") else ""
            sec_txt = lambda r: (f", {r['secondary']['metric']} {r['secondary']['value']}" if r.get("secondary") else "")
            items.append(("model", f"Target [{goal}]: {'MET' if vr['met'] else 'NOT MET'} on out-of-fold training predictions ({vr['metric']} {vr['value']}{sec_txt(vr)}"
                                   f"{', threshold ' + str(vr['threshold']) if vr.get('threshold') is not None else ''}). Held-out test: {hr['metric']} {hr['value']}{ci_txt}{sec_txt(hr)} "
                                   f"({'meets' if hr['met'] else 'does not meet'} the target). Search stopped: {why} after {len(tr.get('attempts', []))} attempt(s). "
                                   "The test set was never used to choose the model or the threshold.", {"spec": spec, "validation": vr, "heldout": hr}))
            tgt_out = {"target_report": {**tr, "heldout": hr, "met_validation": vr["met"], "met_heldout": hr["met"], "stopped_because": why}}
    except Exception as ex:
        items.append(("model", f"Target report skipped ({type(ex).__name__}: {ex}).", {}))
    fair: List[Any] = []
    try:
        sens = state.get("sensitive_columns") or []
        if sens and pt == "classification" and len(splits["classes"]) == 2:
            import pandas as pd
            raw = pd.read_csv(state["dataset_path"])
            idx = X_te.index
            if idx.max() < len(raw) and idx.is_unique:
                thr = (diag.get("cost_threshold") or {}).get("threshold")
                p_te = est.predict_proba(X_te)[:, 1] if hasattr(est, "predict_proba") else None
                pr_use = (p_te >= thr).astype(int) if (thr is not None and p_te is not None) else pred
                used = {c: c in X_te.columns for c in sens}
                fair = fairness.group_audit(raw.loc[idx], y_te, pr_use, sens, used)
                for a in fair:
                    items.append(("fairness", fairness.describe(a), a))
    except Exception as ex:
        items.append(("model", f"Fairness screen skipped ({type(ex).__name__}).", {}))
    top = [i for i in imp if i["importance"] > 0][:5]
    if top:
        items.append(("model", "Strongest predictive drivers (permutation importance, held-out): "
                               + ", ".join(f"{i['feature']} ({i['importance']:.3f})" for i in top) + ". Predictive, not causal.", {"top": top}))
    g = viz.Gallery(reg, state["run_id"], "model")
    warns = []
    try:
        viz.model_charts(g, state, diag, imp)
        interactive.model(g, state, diag, imp)
    except Exception as e:
        warns.append(f"Model charts partly failed: {type(e).__name__}: {e}")
    expl = {"importances": imp[:20], "diagnostics": diag, "model": name}
    return {"explanations": jsonable(expl), "fairness": jsonable({"groups": fair}), "metric_ci": jsonable(ci), "seed_stability": jsonable(stab),
            "cost_threshold": jsonable(diag.get("cost_threshold", {})), **jsonable(tgt_out), "charts": g.items, "evidence": make_evidence("M", items), "warnings": warns,
            "current_status": "Explanations ready.", "active_agent": AGENT,
            "agent_log": [log(AGENT, f"Explained {name}; top driver: {top[0]['feature'] if top else 'none above zero'}; {len(g.items)} model chart(s).")]}
