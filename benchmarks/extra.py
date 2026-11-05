import sys, json, time, tempfile, warnings, shutil
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/NexusML_v0")
import numpy as np, pandas as pd
from sklearn import datasets as skd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from nexusml.evaluation import harness

W = Path("/home/claude/work/extra"); W.mkdir(exist_ok=True)
KPI = [{"name": "Records", "agg": "count"}]
def frame(pt, target): return {"analysis_type": "general_exploration", "problem_type": pt, "target": target, "date_column": None, "segment_columns": [], "kpis": KPI}

def build():
    cases = {}
    d = skd.load_breast_cancer(as_frame=True); df = d.frame.copy(); df["diagnosis"] = d.target_names[df.pop("target")]; p = W/"breast_cancer.csv"; df.to_csv(p, index=False)
    cases["breast_cancer"] = {"name": "breast_cancer", "path": str(p), "frame": frame("classification", "diagnosis"), "truth": {"target": "diagnosis", "problem_type": "classification"}}
    d = skd.load_wine(as_frame=True); df = d.frame.copy(); df["cultivar"] = df.pop("target").map(dict(enumerate(d.target_names))); p = W/"wine.csv"; df.to_csv(p, index=False)
    cases["wine"] = {"name": "wine", "path": str(p), "frame": frame("classification", "cultivar"), "truth": {"target": "cultivar", "problem_type": "classification"}}
    d = skd.load_diabetes(as_frame=True); df = d.frame.copy().rename(columns={"target": "progression"}); p = W/"diabetes.csv"; df.to_csv(p, index=False)
    cases["diabetes"] = {"name": "diabetes", "path": str(p), "frame": frame("regression", "progression"), "truth": {"target": "progression", "problem_type": "regression"}}
    r = np.random.RandomState(5); n = 1500
    tenure, spend, calls = r.exponential(20, n), r.gamma(2, 40, n), r.poisson(2, n)
    logit = -1.0 - 0.05 * tenure + 0.01 * spend + 0.35 * calls
    churn = (r.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(n)], "tenure": tenure, "monthly_spend": spend, "support_calls": calls, "country": r.choice([f"country_{i}" for i in range(120)], n),
                       "plan": r.choice(["basic", "plus", "pro"], n), "constant_flag": 1, "refund_issued_after_cancel": churn * r.choice([1, 1, 1, 0], n), "churned": np.where(churn == 1, "Yes", "No")})
    df.loc[r.choice(n, int(.2 * n), replace=False), "monthly_spend"] = np.nan
    p = W/"messy_telco.csv"; df.to_csv(p, index=False)
    cases["messy_telco"] = {"name": "messy_telco", "path": str(p), "frame": frame("classification", "churned"),
                            "truth": {"target": "churned", "problem_type": "classification", "must_drop": ["customer_id", "constant_flag", "refund_issued_after_cancel"], "drivers": ["tenure", "support_calls"]}}
    from PIL import Image
    d = skd.load_digits(); root = W/"digits_images"
    if root.exists(): shutil.rmtree(root)
    cnt = {}
    for img, y in zip(d.images, d.target):
        cnt[y] = cnt.get(y, 0) + 1
        if cnt[y] > 30: continue
        (root/f"digit_{y}").mkdir(parents=True, exist_ok=True)
        Image.fromarray((img / 16 * 255).astype("uint8")).resize((32, 32), Image.BICUBIC).save(root/f"digit_{y}"/f"d{cnt[y]:02d}.png")
    cases["digits_images"] = {"name": "digits_images", "kind": "media", "folder": str(root), "frame": frame("classification", "label"), "truth": {"target": "label", "problem_type": "classification", "classes": 10, "n": 300}}
    return cases

def my_baseline(case):
    """What I (Claude) would do by hand: a few standard models, 5-fold CV on all the data, take the best."""
    t = case["truth"]
    if case.get("folder"):
        from PIL import Image
        X, y = [], []
        for f in sorted(Path(case["folder"]).glob("*/*.png")): X.append(np.asarray(Image.open(f).convert("L"), float).ravel() / 255); y.append(f.parent.name)
        X, y = np.array(X), np.array(y)
        models = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), "hgb": HistGradientBoostingClassifier(random_state=0), "rf": RandomForestClassifier(300, random_state=0)}
        cv, metric = StratifiedKFold(5, shuffle=True, random_state=0), "f1_macro"
    else:
        df = pd.read_csv(case["path"]); y = df.pop(t["target"])
        drop = [c for c in df.columns if df[c].nunique() <= 1 or (df[c].dtype == object and df[c].nunique() > .5 * len(df))] + t.get("must_drop", [])
        X = pd.get_dummies(df.drop(columns=list(set(drop)), errors="ignore"), drop_first=True).astype(float)
        if t["problem_type"] == "classification":
            models = {"logreg": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=3000)), "hgb": HistGradientBoostingClassifier(random_state=0), "rf": make_pipeline(SimpleImputer(), RandomForestClassifier(300, random_state=0))}
            cv, metric = StratifiedKFold(5, shuffle=True, random_state=0), "f1_macro"
        else:
            models = {"ridge": make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV()), "hgb": HistGradientBoostingRegressor(random_state=0)}
            cv, metric = KFold(5, shuffle=True, random_state=0), "r2"
    res = {k: round(float(cross_val_score(m, X, y, cv=cv, scoring=metric).mean()), 4) for k, m in models.items()}
    return {"metric": metric, "scores": res, "best": max(res, key=res.get), "best_score": max(res.values())}

if __name__ == "__main__":
    names = sys.argv[1:] or ["breast_cancer", "wine", "diabetes", "messy_telco", "digits_images"]
    cases = build(); wd = Path(tempfile.mkdtemp())
    for nm in names:
        c = cases[nm]; t = time.time()
        try:
            r = harness.run_case(c, wd); v = r["values"]; lb = v.get("leaderboard") or {}
            rows = lb.get("rows") or []; top = rows[0] if rows else {}
            ev = {e["id"]: e["text"] for e in v["evidence"]}
            out = {"name": nm, "seconds": r["seconds"], "llm_calls": r["llm_calls"], "errors": v.get("errors"), "metric": lb.get("metric"), "winner": lb.get("winner"),
                   "cv": top.get("select_score"), "heldout": (top.get("heldout") or {}), "baseline_cv": (lb.get("baseline") or {}).get("cv_score"),
                   "n_candidates": len(rows), "scores": r["scores"], "key_evidence": {k: x for k, x in ev.items() if k[0] in "LMBQ" or k[:2] in ("X0",)}}
        except Exception as e:
            import traceback; out = {"name": nm, "crashed": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
        out["me"] = my_baseline(c)
        json.dump(out, open(W/f"res_{nm}.json", "w"), default=str, indent=1); print(nm, "done", round(time.time()-t), "s", flush=True)
