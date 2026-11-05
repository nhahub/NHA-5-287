"""Ensembles and the finalisation step of the AutoML engine.

  caruana()        greedy forward selection WITH replacement on out-of-fold predictions (Caruana et al. 2004; used by auto-sklearn and AutoGluon): resists over-fitting
                   because it only ever averages, and many different weak winners average out.
  Ensemble*/Stack* picklable, sklearn-compatible models (predict / predict_proba) so the bundle, explanations, export and the Predict tab work unchanged.
  finalize()       refit members on all training rows, RE-SCORE every finalist on fresh folds, score the held-out split once, build the candidate table.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import modeling
from .automl_core import Config, Session, Trial
from .automl_models import EnsembleClassifier, EnsembleRegressor, StackClassifier, StackRegressor, _stack_features  # noqa: F401  (re-exported)


# selection
def caruana(oofs: List[np.ndarray], y: np.ndarray, score: Callable[[np.ndarray, np.ndarray], float], size: int = 15, mask: Optional[np.ndarray] = None) -> Tuple[List[int], float]:
    """Greedy ensemble selection with replacement. -> (member index per pick, best OOF score). Stops growing when the score no longer improves."""
    if not oofs:
        return [], float("nan")
    mask = np.ones(len(y), bool) if mask is None else mask
    yv, O = y[mask], [o[mask] for o in oofs]
    picks: List[int] = []
    total = np.zeros_like(O[0])
    best_score, best_n = -np.inf, 0
    for it in range(size):
        cand = []
        for i, o in enumerate(O):
            try:
                cand.append((score(yv, (total + o) / (len(picks) + 1)), i))
            except Exception:
                cand.append((-np.inf, i))
        s, i = max(cand, key=lambda t: (t[0], -t[1]))
        picks.append(i)
        total = total + O[i]
        if s > best_score + 1e-9:
            best_score, best_n = s, len(picks)
    return picks[:best_n], float(best_score)


# finalisation
def finalize(sess: Session, method: str = "greedy", size: int = 15, max_members: int = 8, top_families: int = 8) -> Dict[str, Any]:
    """Turn the search history into the deliverable. -> dict shaped like modeling.fit_default's result, plus the ensemble / confirmation / goal facts."""
    task, k, y = sess.task, sess.k, sess.y
    sp = sess.splits
    X_tr, y_te = sess.X, np.asarray(sp["y_test"])
    X_te = sp["X_test"].drop(columns=[c for c in sess.dropped if c in sp["X_test"].columns])
    fb = sess.family_best()
    if not fb:
        raise RuntimeError("AutoML found no working model: every trial failed. First errors: " + "; ".join(sorted({t.error for t in sess.trials if t.status == 'failed'})[:3]))
    classification = task == "classification"

    # ---- 1. ensemble selection on the SEARCH out-of-fold predictions
    pool = sorted({t.cfg.key(): t for t in sess.full()}.values(), key=lambda t: -t.mean)[:20]
    ens_info: Dict[str, Any] = {"method": None}
    picks: List[int] = []
    meta = None
    if len(pool) >= 2:
        picks, oof_score = caruana([t.oof for t in pool], y, sess.metric, size, sess.oof_mask)
        counts = {i: picks.count(i) for i in set(picks)}
        top = sorted(counts, key=lambda i: -counts[i])[:max_members]
        if method == "stack" and len(pool) >= 3:
            from sklearn.linear_model import LogisticRegression, RidgeCV
            sel = pool[:min(8, len(pool))]
            Z = _stack_features([t.oof[sess.oof_mask] for t in sel], classification)
            meta = (LogisticRegression(max_iter=2000, C=1.0) if classification else RidgeCV()).fit(Z, y[sess.oof_mask])
            ens_info = {"method": "stack", "members": [t.cfg.label() for t in sel], "search_oof_score": round(float(sess.metric(y[sess.oof_mask], meta.predict_proba(Z) if classification else meta.predict(Z))), 4)}
            ens_trials, ens_w = sel, None
        else:
            ens_trials = [pool[i] for i in top]
            ens_w = np.array([counts[i] for i in top], float)
            ens_w = ens_w / ens_w.sum()
            ens_info = {"method": "greedy", "members": [{"name": t.cfg.label(), "weight": round(float(w), 3)} for t, w in zip(ens_trials, ens_w)], "search_oof_score": round(oof_score, 4)}
    else:
        ens_trials, ens_w = [], None

    # ---- 2. re-score every finalist on FRESH folds (the winner's-curse correction)
    fresh_folds = sess._fresh_folds(sess.seed + 1000)
    need: Dict[str, Trial] = {t.cfg.key(): t for t in sorted(fb.values(), key=lambda t: -t.mean)[:top_families]}
    for t in ens_trials:
        need.setdefault(t.cfg.key(), t)
    keys = list(need)
    fresh = dict(zip(keys, sess._map(lambda kk: sess.evaluate(need[kk].cfg, 1.0, folds=fresh_folds, rnd=sess.round), keys)))

    # ---- 3. refit members on all training rows
    def refit(cfg: Config):
        return sess.make_estimator(cfg).fit(X_tr, y)
    fitted_by_key = dict(zip(keys, sess._map(lambda kk: refit(need[kk].cfg) if fresh[kk].status == "ok" else None, keys)))

    def fold_scores(P: np.ndarray) -> List[float]:
        return [float(sess.metric(y[va], P[va])) for _, va in fresh_folds]

    n_cls = k
    def cfg_dict(t: Trial, weight: Optional[float] = None) -> Dict[str, Any]:
        c = t.cfg
        return {"family": c.family, "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in c.params.items()}, "recipe": list(c.recipe),
                "ideas": [sess.ideas[n] for n in c.ideas if n in sess.ideas], "flags": list(c.flags), **({"weight": weight} if weight is not None else {})}
    configs: Dict[str, Any] = {}
    candidates, models, failed = [], {}, []
    trials_by_family: Dict[str, List[Trial]] = {}
    for t in sess.full():
        trials_by_family.setdefault(t.cfg.family, []).append(t)
    default_by_family = {t.cfg.family: t for t in sess.trials if t.round == 0 and t.fidelity >= 0.999 and t.status == "ok" and not t.cfg.params}

    def cand_from(name: str, est, scores: List[float], search_mean: float, secs: float, extra: Dict[str, Any], default_cv: Optional[float], tuning=None) -> Dict[str, Any]:
        test = modeling.evaluate(task if task != "time_series" else "regression", est, X_te, y_te, n_cls)
        mean = float(np.mean(scores))
        c = {"name": name, "cv_mean": round(default_cv if default_cv is not None else search_mean, 4), "cv_std": round(float(np.std(scores)), 4), "default_cv": round(default_cv if default_cv is not None else search_mean, 4),
             "select_score": round(mean, 4), "select_source": "fresh-fold CV (re-scored after the search)", "tuned": tuning is not None, "tuning": tuning, "fit_seconds": round(secs, 2),
             "cv_scores": [round(s, 4) for s in scores], **test, **extra}
        return c

    for fam, t in sorted(fb.items(), key=lambda kv: -kv[1].mean)[:top_families]:
        kk = t.cfg.key()
        if fresh[kk].status != "ok" or fitted_by_key.get(kk) is None:
            failed.append({"name": fam, "error": fresh[kk].error or "refit failed"})
            continue
        ft = fresh[kk]
        fam_trials = sorted(trials_by_family.get(fam, []), key=lambda x: -x.mean)
        dflt = default_by_family.get(fam)
        tuning = None
        if t.cfg.params:
            tuning = {"best_params": {k_: v for k_, v in t.brief()["params"].items()}, "nested_mean": None, "nested_std": None, "nested_scores": [],
                      "delta_vs_default": None if dflt is None else round(t.mean - dflt.mean, 4), "inner_best": round(t.mean, 4), "seconds": round(sum(x.seconds for x in fam_trials), 2),
                      "trials": [{"rank": i + 1, "params": x.brief()["params"], "mean_score": round(x.mean, 4), "std_score": round(x.std, 4), "fit_seconds": round(x.seconds, 2)} for i, x in enumerate(fam_trials[:12])]}
        name = t.cfg.label()
        c = cand_from(name, fitted_by_key[kk], ft.scores, t.mean, sum(x.seconds for x in fam_trials), {"n_trials": len(fam_trials), "recipe": list(t.cfg.recipe), "ideas_used": list(t.cfg.ideas), "flags": list(t.cfg.flags)},
                      None if dflt is None else dflt.mean, tuning)
        candidates.append(c)
        models[name] = fitted_by_key[kk]
        configs[name] = {"kind": "single", "members": [cfg_dict(t)]}

    if ens_trials and all(fresh[t.cfg.key()].status == "ok" for t in ens_trials):
        members = [fitted_by_key[t.cfg.key()] for t in ens_trials]
        if ens_info["method"] == "stack":
            Pf = [fresh[t.cfg.key()].oof for t in ens_trials]
            Pfresh = meta.predict_proba(_stack_features(Pf, True)) if classification else meta.predict(_stack_features(Pf, False))
            est = (StackClassifier if classification else StackRegressor)(members=members, weights=None, classes=list(range(k)) if classification else None, member_names=[t.cfg.label() for t in ens_trials], meta=meta)
            label = f"Stacked ensemble ({len(members)} models)"
        else:
            Pfresh = sum(w * fresh[t.cfg.key()].oof for t, w in zip(ens_trials, ens_w))
            est = (EnsembleClassifier if classification else EnsembleRegressor)(members=members, weights=list(map(float, ens_w)), classes=list(range(k)) if classification else None,
                                                                              member_names=[t.cfg.label() for t in ens_trials])
            label = f"Ensemble ({len(members)} models, greedy)"
        sc = fold_scores(Pfresh)
        c = cand_from(label, est, sc, ens_info["search_oof_score"], sum(fresh[t.cfg.key()].seconds for t in ens_trials), {"ensemble": ens_info,
                      "ensemble_note": "weights were chosen on the training rows, so this score is mildly optimistic; the held-out score is the honest one"}, ens_info["search_oof_score"])
        candidates.append(c)
        models[label] = est
        w_ = list(ens_w) if ens_w is not None else [1.0 / len(ens_trials)] * len(ens_trials)       # a stack is exported as its members with uniform weights (documented)
        configs[label] = {"kind": "ensemble", "method": ens_info["method"], "members": [cfg_dict(t, float(w)) for t, w in zip(ens_trials, w_)]}
    if not candidates:
        raise RuntimeError("AutoML could not finalise any model: " + "; ".join(f["error"] for f in failed))
    if task == "time_series" and len(candidates) > 1:       # ensemble weights were tuned on these same windows: demand a clear win over the best single model
        ens = [c for c in candidates if c["name"].startswith(("Ensemble", "Stacked"))]
        singles = [c for c in candidates if c not in ens]
        if ens and singles:
            b1 = max(singles, key=lambda c: c["select_score"])
            if ens[0]["select_score"] - b1["select_score"] < max(0.01, b1["cv_std"]):
                ens_info["dropped"] = f"no clear win over {b1['name']} on a time series (optimistic weights); using the single model"
                for c in ens:
                    candidates.remove(c)
                    models.pop(c["name"], None)
    best = modeling.pick_best(candidates)
    # ---- 4. goal evaluation on the confirmed (fresh-fold) out-of-fold predictions of the winner
    return {"baseline": modeling.make_baseline(task, sess.splits, sess.folds, modeling._scoring(task, k)), "candidates": candidates, "models": models, "best": best, "failed": failed, "ensemble": ens_info,
            "fresh_oof": {t.cfg.label(): fresh[t.cfg.key()].oof for t in need.values() if fresh[t.cfg.key()].status == "ok"}, "fresh_folds": fresh_folds,
            "ens_fresh_oof": None if not ens_trials else Pfresh, "configs": configs}
