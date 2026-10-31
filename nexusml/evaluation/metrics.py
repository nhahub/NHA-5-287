"""Scoring a finished run against the ground truth of a dataset (all computed by code)."""
from typing import Any, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import config
from ..storage import get_registry


def framing(case, values) -> Dict[str, Any]:
    t = case["truth"]
    plan = values.get("analysis_plan", {})
    return {"target_ok": plan.get("target") == t.get("target"), "problem_type_ok": plan.get("problem_type") == t.get("problem_type")}


def driver_recovery(case, values) -> Dict[str, Any]:
    drivers = case["truth"].get("drivers")
    imps = [i for i in (values.get("explanations") or {}).get("importances", []) if i["importance"] > 0]
    if not drivers or not imps:
        return {}
    k = len(drivers)
    top = [i["feature"] for i in imps[:k]]
    hit = [d for d in drivers if any(f == d or (d.endswith("__") and f.startswith(d)) for f in top)]
    return {"top": top, "recall": round(len(hit) / k, 3), "precision": round(len(hit) / max(1, len(top)), 3)}


def interaction_found(case, values) -> Dict[str, Any]:
    it = case["truth"].get("interaction")
    if not it:
        return {}
    ev = [e["text"] for e in values.get("evidence", []) if e.get("source") == "interaction_analysis"]
    return {"found": any(it[0] in t and it[1] in t for t in ev), "n_interactions": len(ev)}


def cleaning(case, values) -> Dict[str, Any]:
    must = case["truth"].get("must_drop")
    if not must:
        return {}
    dropped = set(values.get("analysis_plan", {}).get("drop_columns", []))
    return {"dropped_all": all(m in dropped for m in must), "missing": [m for m in must if m not in dropped]}


def claims_quality(values) -> Dict[str, Any]:
    items = list((values.get("insights") or {}).get("key_insights", [])) + list(values.get("recommendations", [])) + list(values.get("root_causes", []))
    n = len(items)
    bad = sum(1 for i in items if i.get("unverified"))
    return {"claims": n, "unverified": bad, "unverified_share": round(bad / n, 3) if n else 0.0}


def _baseline_pipe(X: pd.DataFrame, clf):
    num = [c for c in X if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X if c not in num and X[c].nunique() <= 30]
    return Pipeline([("p", ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
                                              ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])),
                     ("m", clf)])


def model_quality(case, values, seed: int) -> Dict[str, Any]:
    """Held-out score of the system vs a plain tabular sklearn baseline (best of linear / boosted) on the SAME kind of split."""
    pt = case["truth"].get("problem_type")
    if pt not in ("classification", "regression"):
        return {}
    cands = values.get("candidate_metrics") or []
    best = next((c for c in cands if c["name"] == values.get("best_model_name")), None)
    pm = values.get("primary_metric")
    if not best or pm not in best:
        return {}
    df = pd.read_csv(case["path"])
    target = case["truth"]["target"]
    drop = set(values["analysis_plan"].get("drop_columns", [])) | {target}
    X = df.drop(columns=[c for c in df.columns if c in drop or c == target])
    if pt == "classification":
        classes = df[target].astype(str).value_counts()
        y = (df[target].astype(str) == classes.index[-1]).astype(int)
        mods = [LogisticRegression(max_iter=2000), HistGradientBoostingClassifier(random_state=seed)]
        score = lambda m, Xt, yt: roc_auc_score(yt, m.predict_proba(Xt)[:, 1])
        strat = y
    else:
        y = df[target].astype(float)
        mods = [Ridge(), HistGradientBoostingRegressor(random_state=seed)]
        score = lambda m, Xt, yt: r2_score(yt, m.predict(Xt))
        strat = None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=seed, stratify=strat)
    base = max(score(_baseline_pipe(X, m).fit(Xtr, ytr), Xte, yte) for m in mods)
    return {"metric": pm, "system": best[pm], "baseline": round(float(base), 4), "gap": round(float(best[pm] - base), 4)}


def clustering_ari(case, values) -> Dict[str, Any]:
    if case["truth"].get("problem_type") != "clustering":
        return {}
    try:
        labels = get_registry().load(values["models_key"]).get("labels")
        truth = case["truth"].get("labels")
        if truth is None and case["truth"].get("labels_by_name"):                       # media folders: truth keyed by file name
            col = next(c for c in ("image", "document") if c in pd.read_csv(values["dataset_path"], nrows=1).columns)
            names = pd.read_csv(values["dataset_path"])[col].map(lambda p: str(p).split("/")[-1])
            truth = [case["truth"]["labels_by_name"][n] for n in names]
        return {"ari": round(float(adjusted_rand_score(truth, list(labels))), 3)}
    except Exception as e:
        return {"error": str(e)}


def ci_sane(values) -> Dict[str, Any]:
    ci = values.get("metric_ci") or {}
    if ci.get("lo") is None:
        return {}
    return {"contains_point": bool(ci["lo"] <= ci["value"] <= ci["hi"]), "width": round(ci["hi"] - ci["lo"], 3)}


def score_case(case, values, seed: int) -> Dict[str, Any]:
    return {"framing": framing(case, values), "drivers": driver_recovery(case, values), "interaction": interaction_found(case, values),
            "cleaning": cleaning(case, values), "claims": claims_quality(values), "model": model_quality(case, values, seed),
            "clustering": clustering_ari(case, values), "ci": ci_sane(values), "errors": values.get("errors", [])}
