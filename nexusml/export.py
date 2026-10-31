"""Deployable package export (inspired by PromptML Studio's Developer Mode).

build_package() returns a ZIP containing everything needed to use the model WITHOUT this project:
  model_bundle.joblib   the trained pipeline + metadata
  nexusml/             the exact preprocessing/model code the bundle needs (vendored copy)
  predict.py            score a CSV from the command line
  serve_app.py          a Streamlit prediction form generated from the data schema (+ Dockerfile)
  train.py              retrains the SAME model (cleaning plan, model, tuned parameters) on a CSV
  schema.json / input_template.csv / requirements.txt (pinned to this environment) / README.md / report.html
"""
import io
import json
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .storage import get_registry

VENDOR = ["config.py", "profiling.py", "trends.py", "modeling.py", "featurizers.py", "modelio.py", "automl_models.py", "automl_recipes.py", "nlp.py", "cv.py", "imageprep.py"]   # everything a pickled model may reference


def schema_for(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Raw input fields a user must supply for one prediction (typed, with defaults/choices)."""
    plan, prof = state["analysis_plan"], state["dataset_profile"]["columns"]
    df = pd.read_csv(state["dataset_path"])
    skip = set(plan["drop_columns"]) | {plan.get("target")}
    out = []
    for c, i in prof.items():
        if c in skip or c not in df.columns:
            continue
        s = df[c].dropna()
        if i["kind"] == "numeric":
            integer = bool(i.get("is_integer_like"))
            d = float(s.median()) if len(s) else 0.0
            out.append({"name": c, "kind": "numeric", "integer": integer, "min": float(s.min()) if len(s) else 0.0,
                        "max": float(s.max()) if len(s) else 1.0, "default": round(d) if integer else round(d, 4)})
        elif i["kind"] == "datetime":
            dt = pd.to_datetime(s, errors="coerce").dropna()
            out.append({"name": c, "kind": "datetime", "default": str(dt.max().date()) if len(dt) else "2024-01-01"})
        elif i["kind"] in ("text", "image_path"):
            out.append({"name": c, "kind": i["kind"], "default": ""})
        elif i["kind"] in ("categorical", "boolean"):
            vc = s.astype(str).value_counts()
            out.append({"name": c, "kind": "categorical", "choices": vc.index[:30].tolist(), "default": str(vc.index[0]) if len(vc) else ""})
    return out


PREDICT = '''"""Score a CSV with the saved model:  python predict.py input.csv [output.csv]"""
import sys
import pandas as pd
from nexusml import modelio
from nexusml.modeling import predict_with_bundle

bundle = modelio.load("model_bundle.joblib")          # verifies model_bundle.joblib.sha256 first
data = pd.read_csv(sys.argv[1])
out = predict_with_bundle(bundle, data)
res = pd.concat([data.reset_index(drop=True), out.reset_index(drop=True)], axis=1) if bundle["problem_type"] != "time_series" else out
res.to_csv(sys.argv[2] if len(sys.argv) > 2 else "predictions.csv", index=False)
print(res.head(10).to_string())
'''

SERVE = '''"""Prediction web app generated from the trained model.  Run:  streamlit run serve_app.py"""
import json
from datetime import date
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from nexusml.modeling import predict_with_bundle

HERE = Path(__file__).parent
st.set_page_config(page_title="__TITLE__", page_icon="🔮", layout="centered")


@st.cache_resource
def load():
    from nexusml import modelio
    return modelio.load(HERE / "model_bundle.joblib"), json.loads((HERE / "schema.json").read_text())


bundle, schema = load()
st.title("🔮 __TITLE__")
st.caption(f"Model: {bundle['best_model']} · problem: {bundle['problem_type']} · target: {bundle.get('target')}")
if bundle["problem_type"] == "time_series":
    st.info("Time-series models need recent history of the target; use predict.py with a CSV instead.")
    st.stop()
with st.form("inputs"):
    row = {}
    cols = st.columns(2)
    for i, f in enumerate(schema):
        with cols[i % 2]:
            if f["kind"] == "numeric":
                row[f["name"]] = st.number_input(f["name"], value=float(f["default"]), step=1.0 if f["integer"] else 0.01,
                                                 format="%d" if f["integer"] else "%.4f")
            elif f["kind"] == "categorical":
                row[f["name"]] = st.selectbox(f["name"], f["choices"], index=f["choices"].index(f["default"]) if f["default"] in f["choices"] else 0)
            elif f["kind"] in ("text", "image_path"):
                row[f["name"]] = st.text_input(f["name"] + (" (image file path)" if f["kind"] == "image_path" else " (text)"), value=f["default"])
            else:
                row[f["name"]] = str(st.date_input(f["name"], value=pd.to_datetime(f["default"]).date()))
    go = st.form_submit_button("Predict", use_container_width=True)
if go:
    out = predict_with_bundle(bundle, pd.DataFrame([row])).iloc[0]
    st.success(f"Prediction: **{out.get('prediction', out.get('cluster'))}**")
    proba = {k[6:]: float(v) for k, v in out.items() if str(k).startswith("proba_")}
    if proba:
        st.bar_chart(pd.Series(proba))
'''

TRAIN_AUTOML = '''"""Retrain the SAME model on a CSV: the cleaning plan and the exact winning configuration found by the AutoML agent.   python train.py data.csv"""
import copy
import sys
import joblib
import pandas as pd
from nexusml import modeling as M

PLAN = __PLAN__
STRATEGY = __STRATEGY__
CONFIG = __CONFIG__          # {"kind": "single" | "ensemble", "members": [{family, params, recipe, ideas, flags, weight?}]}

df = pd.read_csv(sys.argv[1])
task = PLAN["problem_type"]
splits = M.make_splits(df, PLAN)
n_classes = len(splits["classes"] or [])
X, y = splits["X_train"], splits["y_train"]


def build(cfg):
    strat = dict(STRATEGY)
    for f in cfg["flags"]:
        k, _, v = f.partition("=")
        strat[k] = v or True
    m, spec = M._unspec(copy.deepcopy(M._candidates(task, strat)[cfg["family"]]))
    if cfg["params"]:
        m.set_params(**cfg["params"])
    sp = M.Spec(m, text_rep=getattr(spec, "text_rep", None), image_rep=getattr(spec, "image_rep", None), meta=getattr(spec, "meta", True), recipe=tuple(cfg["recipe"]), ideas=cfg["ideas"] or None)
    return M._build(sp, X, task, strat)[0]


members = [build(c).fit(X, y) for c in CONFIG["members"]]
if CONFIG["kind"] == "ensemble":
    from nexusml.automl_models import EnsembleClassifier, EnsembleRegressor
    w = [c["weight"] for c in CONFIG["members"]]
    est = (EnsembleClassifier if task == "classification" else EnsembleRegressor)(members=members, weights=w, classes=list(range(n_classes)) if task == "classification" else None,
                                                                                 member_names=[c["family"] for c in CONFIG["members"]])
    print("ensemble of", len(members), "models (weights as found by the AutoML agent)")
else:
    est = members[0]
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(build(CONFIG["members"][0]), X, y, cv=M._cv(task, y), scoring=M.primary_metric_for(task, n_classes))
    print("cross-validated", M.primary_metric_for(task, n_classes), round(float(cv.mean()), 4))
print("held-out", M.evaluate(task, est, splits["X_test"], splits["y_test"], n_classes))
from nexusml import modelio
bundle = modelio.load("model_bundle.joblib")
bundle["pipeline"] = est
joblib.dump(bundle, "model_bundle_retrained.joblib")
print("saved model_bundle_retrained.joblib")
'''

TRAIN = '''"""Retrain the SAME model (cleaning plan, model type, tuned parameters) on a CSV:  python train.py data.csv"""
import sys
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score
from nexusml import modeling as M

PLAN = __PLAN__
STRATEGY = __STRATEGY__
MODEL = "__MODEL__"
BEST_PARAMS = __PARAMS__

df = pd.read_csv(sys.argv[1])
task = PLAN["problem_type"]
splits = M.make_splits(df, PLAN)
n_classes = len(splits["classes"] or [])
est, prefix = M._build(M._candidates(task, STRATEGY)[MODEL], splits["X_train"], task, STRATEGY)
est.set_params(**{prefix + k: v for k, v in BEST_PARAMS.items()})
cv = cross_val_score(est, splits["X_train"], splits["y_train"], cv=M._cv(task, splits["y_train"]), scoring=M.primary_metric_for(task, n_classes))
est.fit(splits["X_train"], splits["y_train"])
print("cross-validated", M.primary_metric_for(task, n_classes), round(float(cv.mean()), 4))
print("held-out", M.evaluate(task, est, splits["X_test"], splits["y_test"], n_classes))
from nexusml import modelio
bundle = modelio.load("model_bundle.joblib")
bundle["pipeline"] = est
joblib.dump(bundle, "model_bundle_retrained.joblib")
print("saved model_bundle_retrained.joblib")
'''

DOCKER = '''FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["sh", "-c", "streamlit run serve_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
'''


def _lit(obj: Any) -> str:
    """Python literal (True/None, not JSON's true/null) for embedding data in generated scripts."""
    return repr(json.loads(json.dumps(obj, default=str)))


def _pins() -> str:
    pk = ["pandas", "numpy", "scipy", "scikit-learn", "joblib", "streamlit"]
    out = []
    for p in pk:
        try:
            out.append(f"{p}=={metadata.version(p)}")        # the pickle needs the same scikit-learn it was trained with
        except metadata.PackageNotFoundError:
            out.append(p)
    return "\n".join(out) + "\n"


def build_package(state: Dict[str, Any]) -> bytes:
    reg = get_registry()
    if not state.get("model_key"):
        raise ValueError("This analysis has no deployable model (e.g. descriptive analysis).")
    plan, pt = state["analysis_plan"], state["analysis_plan"]["problem_type"]
    root = f"{state['experiment_id']}_package/"
    schema = schema_for(state)
    title = f"{(state.get('analysis_type') or 'model').replace('_', ' ').title()}: {plan.get('target') or pt}"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        mb = reg.path_of(state["model_key"]).read_bytes()
        z.writestr(root + "model_bundle.joblib", mb)
        z.writestr(root + "model_bundle.joblib.sha256", __import__("hashlib").sha256(mb).hexdigest() + "  joblib\n")
        z.writestr(root + "nexusml/__init__.py", "")
        for f in VENDOR:
            z.writestr(root + f"nexusml/{f}", (Path(__file__).parent / f).read_text(encoding="utf-8"))
        z.writestr(root + "predict.py", PREDICT)
        z.writestr(root + "serve_app.py", SERVE.replace("__TITLE__", title))
        z.writestr(root + "schema.json", json.dumps(schema, indent=1))
        z.writestr(root + "input_template.csv", pd.DataFrame([{f["name"]: f["default"] for f in schema}]).to_csv(index=False))
        z.writestr(root + "requirements.txt", _pins())
        z.writestr(root + "Dockerfile", DOCKER)
        best = next((c for c in state.get("candidate_metrics", []) if c["name"] == state.get("best_model_name")), {})
        if pt in ("classification", "regression", "time_series") and best:
            strategy = {k: v for k, v in (state.get("strategy") or {}).items()}
            params = (best.get("tuning") or {}).get("best_params", {}) if best.get("tuned") else {}
            bc = (state.get("automl_report") or {}).get("best_config")
            if bc:
                z.writestr(root + "train.py", TRAIN_AUTOML.replace("__PLAN__", _lit(plan)).replace("__STRATEGY__", _lit(strategy)).replace("__CONFIG__", _lit(bc)))
            else:
                z.writestr(root + "train.py", TRAIN.replace("__PLAN__", _lit(plan)).replace("__STRATEGY__", _lit(strategy))
                           .replace("__MODEL__", best["name"]).replace("__PARAMS__", _lit(params)))
        if state.get("report_html_key"):
            z.writestr(root + "report.html", reg.path_of(state["report_html_key"]).read_bytes())
        z.writestr(root + "README.md", f"""# {title}

Generated by Business Analyst agents (experiment `{state['experiment_id']}`). Model: **{state.get('best_model_name')}**.

```bash
pip install -r requirements.txt          # pins match the training environment (the model file is a pickle)
python predict.py input_template.csv     # score a CSV -> predictions.csv
streamlit run serve_app.py               # prediction web app (form generated from schema.json)
python train.py your_training_data.csv   # retrain the same model on new data
docker build -t model-app . && docker run -p 8501:8501 model-app
```
**Security:** `model_bundle.joblib` is a pickle (it can run code when loaded). The scripts verify `model_bundle.joblib.sha256` first and refuse a modified or unverified file. Only use model files you created or can verify.
""")
    return buf.getvalue()



# reproducibility kit
REPRODUCE = '''"""Re-runs this analysis exactly as it was run (same seed, same options).

Offline replay: the LLM answers recorded in runlog.jsonl are reused, so no API key or network is needed and the evidence is regenerated
identically (a prompt that was not recorded fails loudly). Delete the NEXUSML_REPLAY line below to call the live LLM instead.
Run from the nexusml project root:   python reproduce.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent) if (HERE.parent / "nexusml").is_dir() else str(Path.cwd()))
os.environ.setdefault("NEXUSML_REPLAY", str(HERE / "runlog.jsonl"))
os.environ.setdefault("NEXUSML_ARTIFACTS", str(HERE / "reproduced_artifacts"))
os.environ.setdefault("GROQ_API_KEY", "replay-only")

from nexusml import llm
from nexusml.runtime import run_to_completion

args = json.loads((HERE / "run_args.json").read_text())
dataset = str(HERE / args.pop("dataset_path"))
request = args.pop("request", "")
llm.available = lambda: True                      # replay mode never contacts the provider
values = run_to_completion(dataset, request, **args)
print("done. errors:", values.get("errors"), "| report:", values.get("report_key"))
'''


def build_reproducibility_kit(state, reg) -> bytes:
    """ZIP with reproduce.py, run_args.json, the run log (for offline replay), a copy of the dataset and a dependency lock."""
    import io
    import shutil
    import zipfile
    from pathlib import Path
    from . import runlog
    args = dict(state.get("run_args") or {})
    ds = Path(str(state.get("dataset_path", "")))
    args["dataset_path"] = "dataset.csv"
    args["seed"] = state.get("seed")
    args = {k: v for k, v in args.items() if k not in ("on_event", "on_clarify", "on_approval") and v is not None}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("reproduce.py", REPRODUCE)
        z.writestr("run_args.json", json.dumps(args, indent=1, default=str))
        rl = runlog.path(state.get("run_id", "run"))
        if rl.is_file():
            z.write(rl, "runlog.jsonl")
        if ds.is_file() and ds.stat().st_size < 50_000_000:
            z.write(ds, "dataset.csv")
        else:
            z.writestr("DATASET_MISSING.txt", f"Copy the original dataset ({ds.name}) next to reproduce.py as dataset.csv")
        lock = Path(__file__).resolve().parent.parent / "requirements-lock.txt"
        if lock.is_file():
            z.write(lock, "requirements-lock.txt")
        z.writestr("README.txt", "Reproducibility kit.\nseed: %s\nrun id: %s\nLLM model: %s\n\nRun `python reproduce.py` from the project root. See comments inside.\n"
                   % (state.get("seed"), state.get("run_id"), __import__("nexusml.config", fromlist=["x"]).LLM_MODEL))
    return buf.getvalue()
