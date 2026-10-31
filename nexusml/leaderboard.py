"""Model leaderboard: EVERY candidate of EVERY attempt, with the analytics needed to judge them, not only the winner.

Why this exists
  The v4 flow kept one candidate list and overwrote it on every retry, logged only scalar metrics for the non-winners, and explained
  only the winner. Here each candidate becomes a self-contained *row* that carries:
    identity      name, attempt, tuned?, hyper-parameters
    CV score  selection score, CV std, the fold scores behind it, which CV scheme produced them
    held-out      every metric on the test split (scored once, never used to choose)
    fit quality   train score and the train - held-out gap, fit time, predict time, model size
    diagnostics   confusion matrix / per-class P-R-F1 / ROC / PR / calibration (classification), residual profile (regression), top features
    comparison    rank, gap to the winner, a Nadeau-Bengio corrected paired t-test against the winner, a plain-language verdict
  plus a one-standard-error "parsimony" note: the simplest model that is statistically tied with the winner.

Nothing here chooses the model: selection stays in modeling.pick_best (CV, never the test set). This module explains the choice.
"""
import html
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config

# rough "how simple is it" ordering for the one-SE parsimony note (lower = simpler / cheaper to explain)
COMPLEXITY = {"LogisticRegression": 1, "Ridge": 1, "ElasticNet": 1, "ComplementNB": 1, "MultinomialNB": 1, "LinearSVC": 1, "SGD": 1,
              "DecisionTree": 2, "KNN": 2, "AdaBoost": 3, "ExtraTrees": 4, "RandomForest": 4, "HistGradientBoosting": 5,
              "XGBoost": 5, "LightGBM": 5}


def complexity_of(name: str) -> int:
    base = str(name).split(" · ")[0].split(" (")[0]
    return COMPLEXITY.get(base, 3)


def _f(v, nd=4):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x or x in (float("inf"), float("-inf")) else round(x, nd)


# model introspection
def final_model(est):
    """The last estimator inside a Pipeline / target wrapper / calibrator (what actually holds the learned parameters)."""
    seen = 0
    while seen < 6:
        seen += 1
        if hasattr(est, "regressor_"):
            est = est.regressor_
        elif hasattr(est, "named_steps"):
            est = est.named_steps.get("model", list(est.named_steps.values())[-1])
        elif hasattr(est, "calibrated_classifiers_") and est.calibrated_classifiers_:
            est = getattr(est.calibrated_classifiers_[0], "estimator", None) or getattr(est.calibrated_classifiers_[0], "base_estimator", est)
            break
        else:
            break
    return est


def _prep_of(est):
    seen = 0
    while seen < 4:
        seen += 1
        if hasattr(est, "regressor_"):
            est = est.regressor_
        elif hasattr(est, "named_steps"):
            return est.named_steps.get("prep")
        else:
            return None
    return None


def hyperparams(est) -> Dict[str, Any]:
    """Scalar hyper-parameters of the final estimator (what the MLflow run and the leaderboard show)."""
    m = final_model(est)
    out: Dict[str, Any] = {}
    try:
        for k, v in m.get_params(deep=False).items():
            if v is None or isinstance(v, (bool, int, float, str)):
                if isinstance(v, float) and v != v:
                    continue
                out[k] = v
    except Exception:
        pass
    return out


def feature_names(est) -> Optional[List[str]]:
    prep = _prep_of(est)
    if prep is None:
        return None
    try:
        names = [str(n) for n in prep.get_feature_names_out()]
    except Exception:
        return None
    clean = []
    for n in names:
        for pre in ("num__", "cat__", "remainder__"):
            if n.startswith(pre):
                n = n[len(pre):]
        if n.startswith(("text_", "img_")) and "__" in n:      # 'text_0__word:cache' -> 'word:cache'
            n = n.split("__", 1)[1]
        clean.append(n)
    return clean


def native_importances(est, top: int = 15) -> List[Dict[str, Any]]:
    """Feature weights straight from the fitted model (coef_ or feature_importances_). Cheap; permutation importance is computed for the winner only."""
    m = final_model(est)
    vals = None
    if hasattr(m, "feature_importances_"):
        vals = np.asarray(m.feature_importances_, dtype=float)
    elif hasattr(m, "coef_"):
        c = np.abs(np.asarray(m.coef_, dtype=float))
        vals = c.mean(axis=0) if c.ndim > 1 else c
    if vals is None or vals.ndim != 1 or not len(vals):
        return []
    names = feature_names(est)
    if not names or len(names) != len(vals):
        names = [f"feature_{i}" for i in range(len(vals))]
    order = np.argsort(-vals)[:top]
    return [{"feature": names[i], "importance": _f(vals[i], 5)} for i in order if vals[i] > 0]


# diagnostics
def _thin(a, k: int = 60):
    a = np.asarray(a)
    if len(a) <= k:
        return [round(float(x), 4) for x in a]
    idx = np.unique(np.linspace(0, len(a) - 1, k).astype(int))
    return [round(float(a[i]), 4) for i in idx]


def diagnostics(task: str, est, X, y, classes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compact, JSON-friendly held-out diagnostics for ONE candidate (a failure returns {'error': ...}, never raises)."""
    from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    out: Dict[str, Any] = {}
    try:
        pred = np.asarray(est.predict(X))
        y = np.asarray(y)
        if task == "classification":
            classes = [str(c) for c in (classes or [])]
            k = len(classes)
            cm = confusion_matrix(y, pred, labels=list(range(k)))
            per = {}
            for i, c in enumerate(classes):
                tp = int(cm[i, i])
                fp, fn = int(cm[:, i].sum()) - tp, int(cm[i, :].sum()) - tp
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                per[c] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, "support": tp + fn}
            conf = sorted(((int(cm[i, j]), i, j) for i in range(k) for j in range(k) if i != j and cm[i, j] > 0), reverse=True)[:5]
            out.update(classes=classes, confusion_matrix=cm.tolist(), per_class=per,
                       top_confusions=[{"true": classes[i], "predicted": classes[j], "count": n} for n, i, j in conf])
            proba = None
            if hasattr(est, "predict_proba"):
                try:
                    proba = np.asarray(est.predict_proba(X))
                except Exception:
                    proba = None
            if proba is not None and proba.ndim == 2 and proba.shape[1] == k:
                right = pred == y
                out["confidence"] = {"mean_when_right": _f(proba.max(axis=1)[right].mean()) if right.any() else None,
                                     "mean_when_wrong": _f(proba.max(axis=1)[~right].mean()) if (~right).any() else None}
                if k == 2 and len(np.unique(y)) == 2:
                    p1 = proba[:, 1]
                    fpr, tpr, _ = roc_curve(y, p1)
                    pr, rc, _ = precision_recall_curve(y, p1)
                    out["roc"] = {"fpr": _thin(fpr), "tpr": _thin(tpr), "auc": _f(roc_auc_score(y, p1))}
                    out["pr"] = {"precision": _thin(pr), "recall": _thin(rc), "ap": _f(average_precision_score(y, p1)), "base_rate": _f(y.mean())}
                    out["brier"] = _f(brier_score_loss(y, p1))
                elif k > 2:
                    try:
                        out["auc_per_class"] = {classes[i]: _f(roc_auc_score((y == i).astype(int), proba[:, i])) for i in range(k) if 0 < (y == i).sum() < len(y)}
                    except Exception:
                        pass
                try:
                    out["log_loss"] = _f(log_loss(y, proba, labels=list(range(k))))
                except Exception:
                    pass
        else:
            y = y.astype(float)
            res = y - pred
            ae = np.abs(res)
            out.update(residual_mean=_f(res.mean()), residual_std=_f(res.std()),
                       abs_error={"p50": _f(np.percentile(ae, 50)), "p90": _f(np.percentile(ae, 90)), "p95": _f(np.percentile(ae, 95))})
            order = np.argsort(pred)
            bins = np.array_split(order, min(5, max(1, len(order) // 5)))
            out["bias_by_predicted_band"] = [{"band": i + 1, "mean_residual": _f(res[b].mean()), "n": int(len(b))} for i, b in enumerate(bins) if len(b)]
            k = min(len(y), 150)
            idx = np.linspace(0, len(y) - 1, k).astype(int)
            out["actual_vs_predicted"] = {"actual": [round(float(y[i]), 4) for i in idx], "predicted": [round(float(pred[i]), 4) for i in idx]}
        out["top_features"] = native_importances(est)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return out


def candidate_extras(task: str, est, splits: Dict[str, Any], metric: str) -> Dict[str, Any]:
    """Train score (-> overfit gap), predict time per 1,000 rows, model size, hyper-parameters. Never raises."""
    from . import modeling
    X_tr, y_tr, X_te = splits["X_train"], splits["y_train"], splits["X_test"]
    n_classes = len(splits.get("classes") or [])
    out: Dict[str, Any] = {"train_score": None, "predict_ms_per_1k": None, "model_kb": None, "hyperparams": hyperparams(est)}
    try:
        out["train_score"] = modeling.evaluate(task, est, X_tr, y_tr, n_classes).get(metric)
    except Exception:
        pass
    try:
        sample = X_te.iloc[: min(len(X_te), 1000)]
        t0 = time.perf_counter()
        est.predict(sample)
        out["predict_ms_per_1k"] = round((time.perf_counter() - t0) / max(1, len(sample)) * 1000 * 1000, 2)
    except Exception:
        pass
    try:
        import pickle
        out["model_kb"] = round(len(pickle.dumps(est, protocol=4)) / 1024, 1)
    except Exception:
        pass
    return out


# significance
def corrected_paired_t(a: List[float], b: List[float], test_over_train: float) -> Dict[str, Any]:
    """Nadeau & Bengio (2003) corrected resampled t-test on paired per-fold scores.

    Fold scores are NOT independent (training sets overlap), so the plain paired t-test is far too optimistic. The correction inflates the
    variance by (1/k + n_test/n_train). With k-fold CV, n_test/n_train = 1/(k-1). Few folds means low power: a non-significant result
    means 'cannot tell them apart', not 'they are equal'."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    k = len(d)
    if k < 2 or np.isnan(d).any():
        return {"t": None, "p": None, "k": k}
    mean, var = float(d.mean()), float(d.var(ddof=1))
    if var <= 1e-18:
        return {"t": None, "p": 1.0 if abs(mean) < 1e-12 else 0.0, "k": k, "mean_diff": round(mean, 5)}
    t = mean / np.sqrt((1.0 / k + test_over_train) * var)
    return {"t": round(float(t), 3), "p": round(float(2 * sps.t.sf(abs(t), k - 1)), 4), "k": k, "mean_diff": round(mean, 5)}


def _folds_and_scheme(row: Dict[str, Any]) -> Tuple[List[float], str]:
    t = row.get("tuning") or {}
    if row.get("tuned") and t.get("nested_scores"):
        return list(t["nested_scores"]), f"nested{len(t['nested_scores'])}"
    f = list(row.get("cv_scores") or [])
    return f, f"cv{len(f)}"


def annotate(rows: List[Dict[str, Any]], metric: str, winner: Optional[str] = None) -> Dict[str, Any]:
    """Rank rows (higher select_score is better), compare each to the winner, write verdicts. Mutates rows; returns summary facts."""
    if not rows:
        return {}
    ranked = sorted(rows, key=lambda r: -(r.get("select_score") if r.get("select_score") is not None else -1e9))
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    top = next((r for r in ranked if r["name"] == winner), ranked[0])
    top_folds, top_scheme = _folds_and_scheme(top)
    ties: List[str] = []
    for r in ranked:
        gap = (r.get("select_score") or 0) - (top.get("select_score") or 0)
        r["delta_vs_best"] = round(gap, 4)
        folds, scheme = _folds_and_scheme(r)
        r["fold_scheme"] = scheme
        r["fold_scores_used"] = [round(float(x), 4) for x in folds]
        ratio = 1.0 / max(1, len(folds) - 1) if len(folds) > 1 else 1.0
        if r is top:
            r["p_vs_best"], r["comparable"] = None, True
            r["verdict"] = f"Selected: highest cross-validated {metric} ({r.get('select_score')})."
        elif folds and scheme == top_scheme and len(folds) == len(top_folds):
            tt = corrected_paired_t(folds, top_folds, ratio)
            r["p_vs_best"], r["comparable"] = tt["p"], True
            if tt["p"] is not None and tt["p"] >= 0.05:
                r["verdict"] = f"Statistically tied with the winner (diff {gap:+.3f}, corrected p={tt['p']:.2f}); the winner was kept on score."
                ties.append(r["name"])
            elif tt["p"] is not None:
                r["verdict"] = f"Reliably worse than the winner ({gap:+.3f}, corrected p={tt['p']:.3f})."
            else:
                r["verdict"] = f"Behind the winner by {abs(gap):.3f}."
        else:
            sd = max(top.get("cv_std") or 0, r.get("cv_std") or 0, 1e-9)
            r["p_vs_best"], r["comparable"] = None, False
            r["verdict"] = (f"Behind the winner by {abs(gap):.3f} (about {abs(gap) / sd:.1f} fold-SD); different CV scheme "
                            f"({scheme} vs {top_scheme}), so no formal test.")
            if abs(gap) <= sd:
                ties.append(r["name"])
        tr, te = r.get("train_score"), (r.get("heldout") or {}).get(metric)
        if tr is not None and te is not None and tr - te > config.OVERFIT_GAP:
            r["verdict"] += f" Overfits: train {tr} vs held-out {te}."
        r["overfit_gap"] = None if tr is None or te is None else round(tr - te, 4)
    # one-standard-error rule: the simplest model whose score is within one SE of the best
    se = (top.get("cv_std") or 0) / np.sqrt(max(1, len(top_folds)))
    near = [r for r in ranked if (top.get("select_score") or 0) - (r.get("select_score") or -1e9) <= se + 1e-12]
    simplest = min(near, key=lambda r: (complexity_of(r["name"]), r.get("fit_seconds") or 0, r["rank"]))
    return {"winner": top["name"], "tied_with_winner": ties, "one_se": round(float(se), 4),
            "parsimony_pick": simplest["name"], "parsimony_differs": simplest["name"] != top["name"]}


# assembling rows
def make_row(c: Dict[str, Any], attempt: int, extras: Optional[Dict[str, Any]], diag: Optional[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    """One leaderboard row from a modeling candidate dict (+ extras + diagnostics)."""
    skip = {"name", "cv_mean", "cv_std", "default_cv", "select_score", "select_source", "tuned", "tuning", "fit_seconds", "cv_scores", "n_trials"}
    held = {k: v for k, v in c.items() if k not in skip and isinstance(v, (int, float)) and not isinstance(v, bool)}
    t = c.get("tuning") or {}
    row = {"attempt": attempt, "name": c["name"], "tuned": bool(c.get("tuned")), "score_source": c.get("select_source"),
           "select_score": c.get("select_score"), "cv_std": c.get("cv_std"), "default_cv": c.get("default_cv"),
           "cv_scores": list(c.get("cv_scores") or []), "fit_seconds": c.get("fit_seconds"), "heldout": held,
           "tuning": {k: t.get(k) for k in ("nested_mean", "nested_std", "nested_scores", "best_params", "delta_vs_default", "inner_best", "seconds")} if t else None,
           "n_trials": len(t.get("trials") or []) if t else 0}
    row.update(extras or {})
    row["best_params"] = (t.get("best_params") if c.get("tuned") and t else None) or None
    row["diagnostics"] = diag or {}
    return row


def attempt_record(state_like: Dict[str, Any], rows: List[Dict[str, Any]], metric: str, attempt: int, levers: List[str]) -> Dict[str, Any]:
    return {"attempt": attempt, "levers": levers, "metric": metric, "rows": rows}


def build(state: Dict[str, Any]) -> Dict[str, Any]:
    """Final leaderboard from state['candidate_history'] (all attempts). The LAST attempt is the one that produced the delivered model."""
    hist = [h for h in (state.get("candidate_history") or []) if h.get("rows")]
    metric = state.get("primary_metric") or ""
    if not hist:
        return {}
    final = hist[-1]
    winner = state.get("best_model_name")
    summary = annotate(final["rows"], metric, winner)
    for r in final["rows"]:
        r["selected"] = r["name"] == winner
    past = []
    for h in hist[:-1]:
        annotate(h["rows"], metric)                       # rank within that attempt (its winner = its top score)
        past.append({"attempt": h["attempt"], "levers": h.get("levers", []), "rows": h["rows"]})
    base = state.get("baseline_metrics") or {}
    rows = sorted(final["rows"], key=lambda r: r["rank"])
    gains = [{"attempt": h["attempt"], "levers": h.get("levers", []), "best": max(h["rows"], key=lambda r: r.get("select_score") or -1e9)["name"],
              "best_score": max((r.get("select_score") or -1e9) for r in h["rows"])} for h in hist]
    txt = (f"{len(rows)} candidate model(s) compared in the final attempt ({len(hist)} attempt(s), {sum(len(h['rows']) for h in hist)} candidate fits in total). "
           f"Winner {summary.get("winner")} on CV {metric}")
    if summary.get("tied_with_winner"):
        txt += f"; statistically tied or within noise: {', '.join(summary['tied_with_winner'])}"
    if summary.get("parsimony_differs"):
        txt += f". One-standard-error rule: {summary['parsimony_pick']} is the simplest model within noise of the winner"
    return {"metric": metric, "baseline": base, "winner": winner, "rows": rows, "past_attempts": past, "attempt_progress": gains,
            "n_attempts": len(hist), "summary": txt + ".", **summary}


# views / exports
def to_dataframe(lb: Dict[str, Any], include_past: bool = False) -> pd.DataFrame:
    metric = lb.get("metric", "")
    groups = [lb.get("rows", [])] + ([h["rows"] for h in lb.get("past_attempts", [])] if include_past else [])
    out = []
    for rows in groups:
        for r in rows:
            d = {"rank": r.get("rank"), "attempt": r.get("attempt"), "model": r["name"], "selected": bool(r.get("selected")), "tuned": r.get("tuned"),
                 f"cv_{metric}": r.get("select_score"), "cv_std": r.get("cv_std"), f"heldout_{metric}": (r.get("heldout") or {}).get(metric),
                 "train_score": r.get("train_score"), "overfit_gap": r.get("overfit_gap"), "delta_vs_best": r.get("delta_vs_best"),
                 "p_vs_best": r.get("p_vs_best"), "fit_s": r.get("fit_seconds"), "predict_ms_per_1k": r.get("predict_ms_per_1k"),
                 "model_kb": r.get("model_kb"), "verdict": r.get("verdict")}
            for k, v in (r.get("heldout") or {}).items():
                if k != metric:
                    d[f"heldout_{k}"] = v
            out.append(d)
    return pd.DataFrame(out)


def to_csv(lb: Dict[str, Any]) -> str:
    return to_dataframe(lb, include_past=True).to_csv(index=False)


def to_html(lb: Dict[str, Any]) -> str:
    """A self-contained leaderboard page (no external assets): score bars, tie / overfit flags, verdicts."""
    rows, m = lb.get("rows", []), lb.get("metric", "")
    if not rows:
        return "<p>No candidates.</p>"
    lo = min((r.get("select_score") or 0) for r in rows + [{"select_score": (lb.get("baseline") or {}).get("cv_score", 0)}])
    hi = max((r.get("select_score") or 0) for r in rows)
    span = max(hi - lo, 1e-9)
    body = []
    for r in rows:
        w = 100 * ((r.get("select_score") or lo) - lo) / span
        flag = "★ " if r.get("selected") else ""
        body.append(f"<tr class='{'sel' if r.get('selected') else ''}'><td>{r.get('rank')}</td><td>{flag}{html.escape(r['name'])}</td>"
                    f"<td><div class='bar'><span style='width:{w:.0f}%'></span></div>{r.get('select_score')} ± {r.get('cv_std')}</td>"
                    f"<td>{(r.get('heldout') or {}).get(m, '')}</td><td>{r.get('train_score', '')}</td><td>{r.get('fit_seconds', '')}</td>"
                    f"<td>{'' if r.get('p_vs_best') is None else r['p_vs_best']}</td><td>{html.escape(r.get('verdict') or '')}</td></tr>")
    return ("<style>.lb{border-collapse:collapse;font:13px system-ui}.lb td,.lb th{border-bottom:1px solid #ddd;padding:6px 10px;text-align:left;vertical-align:top}"
            ".lb .sel{background:#eef6ff}.bar{height:6px;background:#e5e9f0;border-radius:3px;width:140px;margin-bottom:3px}.bar span{display:block;height:6px;background:#2f6fed;border-radius:3px}</style>"
            f"<h3>Model leaderboard ({html.escape(m)})</h3><p>{html.escape(lb.get('summary', ''))}</p><table class='lb'><tr><th>#</th><th>Model</th><th>Honest {html.escape(m)} ± SD</th>"
            "<th>Held-out</th><th>Train</th><th>Fit s</th><th>p vs winner</th><th>Verdict</th></tr>" + "".join(body) + "</table>")


def evidence_items(lb: Dict[str, Any]) -> List[Tuple[str, str, Any]]:
    """Facts for the narrative layer (so the LLM and the claims validator see the comparison, not only the winner)."""
    if not lb:
        return []
    items = [("leaderboard", lb["summary"], {k: lb.get(k) for k in ("winner", "tied_with_winner", "parsimony_pick", "n_attempts")})]
    worst = [r for r in lb["rows"] if r.get("overfit_gap") is not None and r["overfit_gap"] > config.OVERFIT_GAP]
    if worst:
        items.append(("leaderboard", "Candidates that overfit (train score far above held-out): "
                      + ", ".join(f"{r['name']} (train {r.get('train_score')} vs held-out {(r.get('heldout') or {}).get(lb['metric'])})" for r in worst[:4]) + ".", {}))
    if len(lb.get("attempt_progress", [])) > 1:
        ap = lb["attempt_progress"]
        items.append(("leaderboard", "Retry progress (best CV score per attempt): " + "; ".join(f"attempt {a['attempt']} {a['best']} {a['best_score']}" for a in ap) + ".", ap))
    return items


def model_card(state: Dict[str, Any]) -> str:
    """A short model card assembled only from facts already in state."""
    lb = state.get("leaderboard") or {}
    plan = state.get("analysis_plan") or {}
    name, pm = state.get("best_model_name"), state.get("primary_metric")
    best = next((r for r in lb.get("rows", []) if r["name"] == name), {})
    ci = state.get("metric_ci") or {}
    ds, prof = state.get("data_summary") or {}, state.get("dataset_profile") or {}
    audit = state.get("audit") or {}
    L = [f"# Model card: {name or 'n/a'}", "", f"*Experiment `{state.get('experiment_id')}` · generated from the run record; nothing here is hand-written.*", "",
         "## Intended use", state.get("business_question") or "Not stated.", "",
         "## Data", f"- {prof.get('n_rows', '?')} rows; {ds.get('n_features', '?')} model features after preparation; split: {ds.get('split', '?')} "
         f"({ds.get('n_train', '?')} train / {ds.get('n_test', '?')} test).",
         f"- Target: `{plan.get('target')}` ({plan.get('problem_type')}). Dropped columns: {', '.join(plan.get('drop_columns') or []) or 'none'}.",
         f"- Modalities: {', '.join(k for k, v in (state.get('modality') or {}).items() if v) or 'tabular'}.", "",
         "## Model", f"- {name}; tuned: {'yes' if best.get('tuned') else 'no'}; hyper-parameters: `{best.get('best_params') or best.get('hyperparams') or {}}`.",
         f"- Compared {len(lb.get('rows', []))} candidate(s) over {lb.get('n_attempts', 1)} attempt(s). {lb.get('summary', '')}", "",
         "## Performance", f"- Cross-validated {pm}: **{best.get('select_score')}** ± {best.get('cv_std')} (baseline: {(state.get('baseline_metrics') or {}).get('cv_score')}).",
         f"- Held-out {pm}: **{(best.get('heldout') or {}).get(pm)}**" + (f" (95% bootstrap CI {ci.get('lo')}-{ci.get('hi')})" if ci.get("lo") is not None else "") + ".",
         f"- Train {pm}: {best.get('train_score')} (gap to held-out {best.get('overfit_gap')}).", ""]
    d = best.get("diagnostics") or {}
    if d.get("per_class"):
        L += ["### Per-class (held-out)", "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
        L += [f"| {c} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |" for c, v in d["per_class"].items()]
        L.append("")
    L += ["## Limitations and risks"]
    L += [f"- **{i['severity']}** `{i['code']}`: {i['message']}" for i in audit.get("issues", [])] or ["- No critic issues were raised."]
    for w in (state.get("warnings") or [])[:6]:
        L.append(f"- {w}")
    if state.get("sensitive_columns"):
        L.append(f"- Sensitive attribute(s) present: {state['sensitive_columns']}; see the fairness screen in the report.")
    L += ["", "## Reproducibility", f"- Seed {state.get('seed')}; LLM `{config.LLM_MODEL}` (used for framing and narrative only, never for scoring).",
          "- Selection used cross-validation on the training split; the test split was scored once and never used to choose.", ""]
    return "\n".join(L)
