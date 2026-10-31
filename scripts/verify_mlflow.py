#!/usr/bin/env python3
"""Checks the MLflow integration against a REAL MLflow install (it cannot be done in a sandbox without network).

    pip install mlflow
    python scripts/verify_mlflow.py

Each step prints PASS / FAIL with the reason. Exit code 0 only if everything passed. Uses a temporary directory; nothing of yours is touched."""
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
RESULTS = []


def step(name):
    def deco(fn):
        try:
            detail = fn()
            RESULTS.append((name, True, detail or ""))
            print(f"PASS  {name}" + (f"  ({detail})" if detail else ""))
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"FAIL  {name}\n      {type(e).__name__}: {e}")
            if os.environ.get("VERIFY_VERBOSE"):
                traceback.print_exc()
        return fn
    return deco


def main() -> int:
    try:
        import mlflow
    except ImportError:
        print("mlflow is not installed. Run: pip install mlflow   (then re-run this script)")
        return 2
    print(f"MLflow {mlflow.__version__}")
    tmp = tempfile.mkdtemp()
    os.environ["NEXUSML_ARTIFACTS"] = tmp
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("NEXUSML_TRACKING", None)
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from nexusml import modelio, tracking
    from mlflow.tracking import MlflowClient

    ctx = {}

    @step("1. integration reports MLflow as active")
    def _():
        st = tracking.mlflow_status()
        assert st["active"], st
        ctx["uri"] = st["uri"]
        return st["uri"]

    X = pd.DataFrame({"a": np.random.RandomState(0).normal(size=60)})
    y = (X.a > 0).astype(int)
    bundle = Path(tmp) / "bundle.joblib"
    modelio.dump({"pipeline": LogisticRegression().fit(X, y)}, bundle)
    state = {"analysis_plan": {"problem_type": "classification", "target": "t", "drop_columns": []}, "analysis_type": "churn",
             "dataset_profile": {"n_rows": 60, "quality": {"score": 90}}, "candidate_metrics": [
                 {"name": "LogisticRegression", "tuned": False, "select_source": "cv", "select_score": 0.9, "cv_scores": [0.9, 0.91], "roc_auc": 0.9}],
             "best_model_name": "LogisticRegression", "primary_metric": "roc_auc", "experiment_id": "verify_run", "user_request": "verify", "evidence": [],
             "charts": [], "llm_calls": [], "quality_passed": True}

    @step("2. log run + nested candidate runs + model + registry version")
    def _():
        info = tracking.log_analysis(state, "models/x.joblib")
        out = tracking.mirror_mlflow(info["run_id"], bundle, info["registered"]["model_name"], "Staging")
        assert out["ok"], out["message"]
        ctx.update(info=info, out=out, name=info["registered"]["model_name"])
        return out["message"][:90]

    @step("3. the run, its parameters and metrics exist in MLflow")
    def _():
        c = MlflowClient(ctx["uri"])
        run = c.get_run(ctx["out"]["mlflow_run_id"])
        assert run.data.metrics, "no metrics logged"
        assert run.data.params, "no params logged"
        return f"{len(run.data.params)} params, {len(run.data.metrics)} metrics"

    @step("4. nested child runs were logged")
    def _():
        c = MlflowClient(ctx["uri"])
        exp = c.get_experiment_by_name(tracking.get_run(ctx["info"]["run_id"])["experiment"])
        runs = c.search_runs([exp.experiment_id])
        kids = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == ctx["out"]["mlflow_run_id"]]
        assert kids, "no child runs found"
        return f"{len(kids)} child run(s)"

    @step("5. registry version has the 'staging' alias and a stage tag")
    def _():
        c = MlflowClient(ctx["uri"])
        mv = c.get_model_version_by_alias(ctx["name"], "staging")
        assert mv.tags.get("stage") == "Staging", mv.tags
        ctx["version"] = str(mv.version)
        return f"version {mv.version}"

    @step("6. the registered model loads back and predicts")
    def _():
        m = tracking.load_registered_model(ctx["name"], "staging")
        p = m.predict(X.head(5))
        assert len(p) == 5
        return "predicted 5 rows"

    @step("7. promoting to Production moves the alias")
    def _():
        reg = tracking.registry()
        ver = reg["models"][ctx["name"]]["versions"][-1]["version"]
        tracking.set_stage(ctx["name"], ver, "Production")
        c = MlflowClient(ctx["uri"])
        mv = c.get_model_version_by_alias(ctx["name"], "production")
        assert str(mv.version) == ctx["version"], (mv.version, ctx["version"])

    bad = [r for r in RESULTS if not r[1]]
    print("\nALL STEPS PASSED" if not bad else f"\n{len(bad)} step(s) FAILED. Send me the lines above (run with VERIFY_VERBOSE=1 for tracebacks).")
    print(f'To browse the data:  mlflow ui --backend-store-uri "{ctx.get("uri", "")}"')
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
