"""Experiment tracking + model registry (MLflow data model, stored as plain JSON on disk).

Concepts mirror MLflow so the UI feels familiar and the data can be mirrored to a real MLflow server:
  experiment  - grouping name (here: the business problem)
  run         - one analysis; has params, metrics (+ per-step history), tags, artifacts
  child run   - one per candidate model; grand-child runs = hyper-parameter search trials
  registry    - named models with numbered versions and stages (None / Staging / Production / Archived)
v4: when `mlflow` is installed, MLflow is the STORE OF RECORD: runs (with nested candidate / trial runs), params, metrics with per-step
history, tags, artifacts, the model, and the real Model Registry (versions + aliases `staging` / `production` + a `stage` tag) are written
to MLflow (tracking URI from MLFLOW_TRACKING_URI, default: a SQLite DB under the artifact dir, which supports the registry).
The JSON store below stays as a local index for the app's Experiments page and as the fallback when mlflow is missing
(set NEXUSML_TRACKING=json to force it). Every JSON run records its `mlflow_run_id`."""
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .storage import get_registry
from .util import jsonable

STAGES = ["None", "Staging", "Production", "Archived"]


def _root() -> Path:
    p = get_registry().path_of("tracking")
    (p / "runs").mkdir(parents=True, exist_ok=True)
    return p


def _write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1), encoding="utf-8")
    os.replace(tmp, path)


def new_run(experiment: str, name: str, params: Optional[Dict[str, Any]] = None, metrics: Optional[Dict[str, Any]] = None,
            tags: Optional[Dict[str, Any]] = None, parent: Optional[str] = None, history: Optional[Dict[str, List[float]]] = None,
            artifacts: Optional[Dict[str, str]] = None, status: str = "FINISHED", started: Optional[float] = None) -> Dict[str, Any]:
    now = time.time()
    run = {"run_id": uuid.uuid4().hex[:16], "experiment": experiment, "name": name, "parent": parent, "status": status,
           "start": started or now, "end": now, "params": {k: v for k, v in (params or {}).items() if v is not None},
           "metrics": {k: v for k, v in (metrics or {}).items() if isinstance(v, (int, float)) and v == v},
           "history": history or {}, "tags": tags or {}, "artifacts": artifacts or {}}
    _write(_root() / "runs" / f"{run['run_id']}.json", run)
    return run


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    f = _root() / "runs" / f"{re.sub(r'[^a-zA-Z0-9]', '', run_id)}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def list_runs(experiment: Optional[str] = None, parent: Any = "any") -> List[Dict[str, Any]]:
    """parent: 'any' = all; None = top-level analysis runs only; <run_id> = children of that run."""
    out = []
    for f in (_root() / "runs").glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(r, dict) or "run_id" not in r or "experiment" not in r:
            continue                                          # not a run file (or damaged): skip instead of crashing the page
        r.setdefault("start", 0.0)
        for k in ("params", "metrics", "tags", "history", "artifacts"):
            r.setdefault(k, {})
        r.setdefault("name", r["run_id"])
        if experiment and r["experiment"] != experiment:
            continue
        if parent != "any" and r.get("parent") != parent:
            continue
        out.append(r)
    return sorted(out, key=lambda r: -r["start"])


def experiments() -> List[str]:
    return sorted({r["experiment"] for r in list_runs(parent=None)})


# model registry
def _reg_path() -> Path:
    return _root() / "registry.json"


def registry() -> Dict[str, Any]:
    f = _reg_path()
    reg = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    reg.setdefault("models", {})
    return reg


def register_model(name: str, run_id: str, metrics: Dict[str, Any], model_key: Optional[str], description: str = "", stage: str = "Staging") -> Dict[str, Any]:
    reg = registry()
    m = reg["models"].setdefault(name, {"versions": []})
    v = {"version": len(m["versions"]) + 1, "run_id": run_id, "stage": stage if stage in STAGES else "None", "created": time.time(),
         "metrics": {k: x for k, x in metrics.items() if isinstance(x, (int, float))}, "model_key": model_key, "description": description}
    m["versions"].append(v)
    _write(_reg_path(), reg)
    return v


def attach_mlflow_version(name: str, version: int, mlflow_version: str) -> None:
    reg = registry()
    for v in reg["models"][name]["versions"]:
        if v["version"] == version:
            v["mlflow_version"] = mlflow_version
    _write(_reg_path(), reg)


def set_stage(name: str, version: int, stage: str) -> None:
    """Promote/demote a version. Only one Production version per model (the previous one is archived), as in MLflow."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    reg = registry()
    for v in reg["models"][name]["versions"]:
        if stage == "Production" and v["stage"] == "Production" and v["version"] != version:
            v["stage"] = "Archived"
        if v["version"] == version:
            v["stage"] = stage
    _write(_reg_path(), reg)
    st = mlflow_status()
    if st["active"]:
        try:
            from mlflow.tracking import MlflowClient
            import mlflow
            mlflow.set_tracking_uri(tracking_uri())
            mv = next(v for v in reg["models"][name]["versions"] if v["version"] == version)
            if mv.get("mlflow_version"):
                _mlflow_stage(MlflowClient(), name, str(mv["mlflow_version"]), stage)
        except Exception:
            pass                                                     # JSON registry already updated; MLflow sync is best-effort here


# log one analysis
def _hp(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in (d or {}).items()}


def jsonable_safe(x):
    from .agents._util import jsonable
    return jsonable(x)


def _candidate_rows(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every candidate of every attempt (leaderboard rows). Falls back to the plain candidate list when no leaderboard exists (older states, descriptive runs)."""
    lb = state.get("leaderboard") or {}
    if lb.get("rows"):
        out = [dict(r) for r in lb["rows"]]
        for h in lb.get("past_attempts", []):
            out += [dict(r) for r in h["rows"]]
        return out
    rows = []
    for c in state.get("candidate_metrics") or []:
        skip = {"name", "cv_mean", "cv_std", "default_cv", "select_score", "select_source", "tuned", "tuning", "fit_seconds", "cv_scores"}
        rows.append({"attempt": 1, "name": c["name"], "tuned": c.get("tuned"), "score_source": c.get("select_source"), "select_score": c.get("select_score"),
                     "cv_std": c.get("cv_std"), "default_cv": c.get("default_cv"), "cv_scores": c.get("cv_scores", []), "fit_seconds": c.get("fit_seconds"),
                     "heldout": {k: v for k, v in c.items() if k not in skip and isinstance(v, (int, float)) and not isinstance(v, bool)},
                     "tuning": c.get("tuning"), "best_params": (c.get("tuning") or {}).get("best_params") if c.get("tuned") else None,
                     "selected": c["name"] == state.get("best_model_name")})
    return rows


def candidate_metrics_of(r: Dict[str, Any]) -> Dict[str, Any]:
    """Flat numeric metrics for one candidate run (names kept compatible with v4: select_score, cv_std, default_cv, cv_mean, fit_seconds + held-out metrics)."""
    m: Dict[str, Any] = {"select_score": r.get("select_score"), "cv_mean": r.get("default_cv"), "default_cv": r.get("default_cv"), "cv_std": r.get("cv_std"),
                         "fit_seconds": r.get("fit_seconds"), "train_score": r.get("train_score"), "overfit_gap": r.get("overfit_gap"),
                         "predict_ms_per_1k": r.get("predict_ms_per_1k"), "model_kb": r.get("model_kb"), "rank": r.get("rank"),
                         "delta_vs_best": r.get("delta_vs_best"), "p_vs_best": r.get("p_vs_best")}
    m.update({k: v for k, v in (r.get("heldout") or {}).items()})
    return {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def log_analysis(state: Dict[str, Any], model_key: Optional[str]) -> Dict[str, Any]:
    plan, pt = state["analysis_plan"], state["analysis_plan"]["problem_type"]
    exp = f"{state.get('analysis_type', 'analysis')} | {plan.get('target') or pt}"
    cands = state.get("candidate_metrics") or []
    rows = _candidate_rows(state)
    best = next((c for c in cands if c["name"] == state.get("best_model_name")), None)
    lb = state.get("leaderboard") or {}
    ds = state.get("data_summary") or {}
    tu = state.get("tuning_report") or {}
    params = {"analysis_type": state.get("analysis_type"), "problem_type": pt, "target": plan.get("target"), "date_column": plan.get("date_column"),
              "n_rows": state["dataset_profile"]["n_rows"], "n_features": ds.get("n_features"), "split": ds.get("split"),
              "primary_metric": state.get("primary_metric"), "pipeline_kind": state.get("pipeline_kind"), "tuning_enabled": bool(tu.get("enabled")),
              "tuning_iterations": tu.get("n_iter"), "tuning_inner_cv": tu.get("inner_cv"), "tuning_outer_cv": tu.get("outer_cv"),
              "search_space_source": tu.get("space_source"), "retry_levers": ",".join(state.get("strategies_applied", [])) or "none",
              "dropped_columns": ",".join(plan.get("drop_columns", [])) or "none", "n_attempts": lb.get("n_attempts") or (1 if cands else None),
              "n_candidates_total": len(rows) or None}
    ar = state.get("automl_report") or {}
    gr = ar.get("goal") or {}
    if ar:
        params.update({"search_engine": "automl-agent", "automl_trials": ar.get("n_trials"), "automl_rounds": len(ar.get("rounds", [])), "automl_attempts": ar.get("attempt"),
                       "automl_workers": ar.get("workers"), "automl_budget_seconds": ar.get("budget_seconds"), "automl_stop": ar.get("stop_reason"), "goal": gr.get("text"),
                       "crew": ",".join(k for k, v in (state.get("crew") or {}).items() if v) or "none", "quarantined_columns": ",".join(state.get("quarantined") or []) or "none"})
    metrics: Dict[str, Any] = {"data_quality_score": state["dataset_profile"]["quality"]["score"], "n_evidence": len(state.get("evidence", [])),
                               "n_charts": len(state.get("charts", [])), "n_llm_calls": len(state.get("llm_calls", []))}
    if best:
        metrics.update({"best_select_score": best["select_score"], "baseline_cv": (state.get("baseline_metrics") or {}).get("cv_score"),
                        **{f"best_{k}": v for k, v in best.items() if isinstance(v, float) and k not in ("cv_mean", "cv_std", "default_cv", "select_score", "fit_seconds")}})
    final = sorted([r for r in rows if r.get("attempt") == (lb.get("n_attempts") or 1)], key=lambda r: -(r.get("select_score") or -1e9))
    if len(final) >= 2:
        metrics["winner_margin"] = round((final[0].get("select_score") or 0) - (final[1].get("select_score") or 0), 4)
        metrics["n_tied_with_winner"] = len(lb.get("tied_with_winner") or [])
    for k, v in (state.get("timings") or {}).items():
        if "." not in k and isinstance(v, dict):
            metrics[f"seconds_{k}"] = v.get("seconds")
    if ar:
        metrics.update({"automl_trials": ar.get("n_trials"), "automl_seconds": ar.get("seconds"), "n_decisions": len(state.get("decisions") or []), "n_board_messages": len(state.get("board") or [])})
        if gr.get("status") not in (None, "no_goal"):
            metrics.update({"goal_met": 1.0 if gr["status"] == "met" else 0.0, "goal_confirmed_value": (gr.get("confirmed_on_fresh_folds") or {}).get("value"),
                            "goal_heldout_value": (gr.get("heldout") or {}).get("value"), "goal_target": gr.get("target")})
    if state.get("cluster_results"):
        metrics["silhouette"] = state["cluster_results"]["silhouette"]
        params["k"] = state["cluster_results"]["k"]
    extra_arts: Dict[str, Any] = {}
    if ar:
        try:
            import pandas as pd
            reg_ = get_registry()
            tdf = pd.DataFrame([{**{k: v for k, v in t.items() if k != "params"}, **{f"hp.{k}": v for k, v in (t.get("params") or {}).items()}} for t in ar.get("trials", [])])
            extra_arts["automl_trials"] = reg_.save_text(tdf.to_csv(index=False), f"{state['experiment_id']}/automl_trials.csv")
            extra_arts["decisions"] = reg_.save_text(json.dumps(jsonable_safe(state.get("decisions") or []), indent=2, default=str), f"{state['experiment_id']}/decisions.json")
            extra_arts["blackboard"] = reg_.save_text(json.dumps(jsonable_safe(state.get("board") or []), indent=2, default=str), f"{state['experiment_id']}/blackboard.json")
        except Exception:
            pass
    arts = {k: v for k, v in {**extra_arts, "report_markdown": state.get("report_key"), "report_html": state.get("report_html_key"), "model": model_key,
                              "leaderboard_html": state.get("leaderboard_key"), "model_card": state.get("model_card_key")}.items() if v}
    tags = {"best_model": state.get("best_model_name"), "quality_passed": state.get("quality_passed"), "request": (state.get("user_request") or "")[:200],
            "experiment_id": state["experiment_id"], "llm_model": __import__("nexusml.config", fromlist=["config"]).LLM_MODEL, "ds.role": "analysis",
            "leaderboard_summary": (lb.get("summary") or "")[:400], "goal_status": gr.get("status"), "automl_stop": ar.get("stop_reason")}
    parent = new_run(exp, state["experiment_id"], params, metrics, tags, artifacts=arts)
    children: List[Dict[str, Any]] = []
    for r in rows:
        t = r.get("tuning") or {}
        hist = {"cv_fold_score": r.get("cv_scores", [])}
        if t.get("nested_scores"):
            hist["nested_fold_score"] = t["nested_scores"]
        hp = (r.get("best_params") if r.get("tuned") else None) or r.get("hyperparams") or {}
        label = r["name"] if r.get("attempt", 1) == (lb.get("n_attempts") or 1) else f"{r['name']} [attempt {r.get('attempt')}]"
        ctags = {"selected": bool(r.get("selected")), "ds.role": "candidate", "attempt": r.get("attempt", 1), "fold_scheme": r.get("fold_scheme"),
                 "verdict": (r.get("verdict") or "")[:400]}
        cr = new_run(exp, label, {"model": r["name"], "tuned": r.get("tuned"), "score_source": r.get("score_source"), "attempt": r.get("attempt", 1), **_hp("hp.", hp)},
                     candidate_metrics_of(r), ctags, parent=parent["run_id"], history=hist)
        if r.get("diagnostics"):
            try:
                (_root() / "diag").mkdir(exist_ok=True)
                _write(_root() / "diag" / f"{cr['run_id']}.json", r["diagnostics"])
            except Exception:
                pass
        children.append(cr)
        for tr in (t.get("trials") or []):
            new_run(exp, f"{r['name']} trial {tr['rank']}", {"model": r["name"], **_hp("hp.", tr["params"])},
                    {"mean_cv_score": tr["mean_score"], "std_cv_score": tr["std_score"], "fit_seconds": tr["fit_seconds"]}, {"trial": True}, parent=cr["run_id"])
    for r in (state.get("cluster_results") or {}).get("k_scores", []):
        children.append(new_run(exp, f"KMeans k={r['k']}", {"k": r["k"]}, {k: v for k, v in r.items() if k != "k"}, {"chosen": r["k"] == state["cluster_results"]["k"], "ds.role": "candidate"}, parent=parent["run_id"]))
    reg = None
    if model_key and best:
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"{state.get('analysis_type')}-{plan.get('target') or pt}").strip("-")
        reg = register_model(name, parent["run_id"], {k: v for k, v in best.items() if isinstance(v, float)}, model_key,
                             f"{best['name']} ({best['select_source']}); {state.get('primary_metric')} {best['select_score']}",
                             "Staging" if state.get("quality_passed") else "None")
        reg["model_name"] = name
    return {"run_id": parent["run_id"], "experiment": exp, "child_runs": len(children), "registered": reg}


def candidate_diagnostics(run_id: str) -> Dict[str, Any]:
    """Held-out diagnostics saved next to a candidate's run (confusion matrix, per-class metrics, curves, top features)."""
    f = _root() / "diag" / f"{re.sub(r'[^a-zA-Z0-9]', '', run_id)}.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def all_candidate_runs(experiment: Optional[str] = None) -> List[Dict[str, Any]]:
    """Flat list of every candidate run across analyses (for the Experiments page candidate table / cross-run comparison)."""
    out = []
    for r in list_runs(experiment):
        if r["tags"].get("ds.role") == "candidate" or (r.get("parent") and not r["tags"].get("trial")):
            if r["tags"].get("trial"):
                continue
            out.append(r)
    return out


def mlflow_status() -> Dict[str, Any]:
    """Why MLflow is / is not in use (shown in the report so a missing or failing MLflow is never silent)."""
    if os.environ.get("NEXUSML_TRACKING", "").lower() == "json":
        return {"active": False, "reason": "NEXUSML_TRACKING=json (JSON store only)"}
    try:
        import mlflow
        return {"active": True, "version": getattr(mlflow, "__version__", "?"), "uri": tracking_uri()}
    except ImportError:
        return {"active": False, "reason": "mlflow is not installed (pip install mlflow); runs are stored in the JSON store only"}


def tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    return "sqlite:///" + str((get_registry().path_of("mlflow.db")).resolve()).replace("\\", "/")


def _clean(k: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-. /]", "_", k)[:240]


def _ensure_experiment(client, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    art = Path(get_registry().path_of("mlartifacts")).resolve()
    art.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(name, artifact_location=art.as_uri())


def _skops_trusted_types(est):
    """The exact types inside OUR pipeline that skops would refuse to load without being told they are trusted (sklearn classes, nexusml featurizers)."""
    import skops.io as sio
    return list(sio.get_untrusted_types(data=sio.dumps(est)))


def _log_sklearn(mlflow, est, name: str = "model"):
    """Log the model and return MLflow's ModelInfo (register `info.model_uri`: MLflow 3 no longer keeps models at runs:/<id>/model).

    MLflow 3 saves scikit-learn models with the 'skops' format by default, which REFUSES any type it does not know (our own text / image featurizers),
    so the logged model ends with status 'Failed'. We therefore choose the format explicitly: cloudpickle first (always works; pickle-style, so only
    load models from stores you control), then skops with exactly the types found in this pipeline marked as trusted. MLflow 2 uses `artifact_path`,
    MLflow 3 uses `name`."""
    import mlflow.sklearn

    def call(**extra):
        try:
            return mlflow.sklearn.log_model(sk_model=est, name=name, **extra)                   # MLflow >= 3
        except TypeError:
            return mlflow.sklearn.log_model(est, artifact_path=name, **extra)                   # MLflow 2.x
    errors = []
    try:
        return call(serialization_format="cloudpickle")
    except Exception as e:
        errors.append(f"cloudpickle: {type(e).__name__}: {str(e)[:160]}")
    try:
        return call(serialization_format="skops", skops_trusted_types=_skops_trusted_types(est))
    except Exception as e:
        errors.append(f"skops: {type(e).__name__}: {str(e)[:160]}")
    raise RuntimeError("; ".join(errors))


def _finite(d: Dict[str, Any]) -> Dict[str, float]:
    """Only real numbers can be MLflow metrics (None / NaN / inf / text would make the whole logging call fail)."""
    out = {}
    for k, v in (d or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if isinstance(v, bool) or f != f or f in (float("inf"), float("-inf")):
            continue
        out[_clean(k)] = f
    return out


def _save_mlflow_fields(run_id: str, **fields: Any) -> None:
    """Remember on the run's own record whether / why it reached MLflow (shown on the Experiments page)."""
    try:
        r = get_run(run_id)
        if r:
            r.update(fields)
            _write(_root() / "runs" / f"{run_id}.json", r)
    except Exception:
        pass


def registered_name_for(run_id: str) -> Optional[str]:
    for name, m in (registry().get("models") or {}).items():
        if any(v.get("run_id") == run_id for v in m.get("versions", [])):
            return name
    return None


def _try(fn, problems: List[str], what: str) -> bool:
    """Optional MLflow features (log_dict / log_table / log_text) vary by version; a missing one is noted, never fatal."""
    try:
        fn()
        return True
    except Exception as e:
        if what not in "".join(problems):
            problems.append(f"{what} not logged ({type(e).__name__}: {str(e)[:70]})")
        return False


def _log_candidate(mlflow, ch: Dict[str, Any], candidates: Optional[Dict[str, Any]], counts: Dict[str, int], problems: List[str]) -> None:
    """Params, metrics, tags, fold-score history, diagnostics artifacts, the fitted model and tuning trials of ONE candidate (inside its active MLflow run)."""
    mlflow.log_params({_clean(k): str(v)[:500] for k, v in ch["params"].items() if v is not None})
    mlflow.log_metrics(_finite(ch["metrics"]))
    mlflow.set_tags({_clean(k): str(v)[:500] for k, v in ch["tags"].items() if v is not None and not str(k).startswith("_")})
    diag = candidate_diagnostics(ch["run_id"])
    if diag:
        _try(lambda: mlflow.log_dict(diag, "diagnostics.json"), problems, "diagnostics artifacts")
        if diag.get("confusion_matrix") and diag.get("classes"):
            import pandas as pd
            cm = pd.DataFrame(diag["confusion_matrix"], index=[f"true:{c}" for c in diag["classes"]], columns=[f"pred:{c}" for c in diag["classes"]])
            _try(lambda: mlflow.log_text(cm.to_csv(), "confusion_matrix.csv"), problems, "confusion-matrix artifacts")
        if diag.get("per_class"):
            import pandas as pd
            _try(lambda: mlflow.log_text(pd.DataFrame(diag["per_class"]).T.to_csv(), "per_class_metrics.csv"), problems, "per-class artifacts")
        if diag.get("top_features"):
            import pandas as pd
            _try(lambda: mlflow.log_text(pd.DataFrame(diag["top_features"]).to_csv(index=False), "top_features.csv"), problems, "feature-importance artifacts")
    if candidates and ch["params"].get("model") in candidates and ch["tags"].get("selected") is not None and os.environ.get("NEXUSML_MLFLOW_ALL_MODELS", "1") == "1" \
            and str(ch["tags"].get("attempt", 1)) == str(ch["tags"].get("_final_attempt", ch["tags"].get("attempt", 1))):
        name = ch["params"]["model"]
        try:                                                      # every compared model becomes a real MLflow model (Models tab), not only the winner
            import pickle
            if len(pickle.dumps(candidates[name], protocol=4)) > 80_000_000:
                raise MemoryError("larger than 80 MB")
            cinfo = _log_sklearn(mlflow, candidates[name], name=_clean(name).replace(" ", "_"))
            counts["models"] += 1
            try:
                mlflow.log_metrics(_finite(ch["metrics"]), model_id=getattr(cinfo, "model_id", None))      # MLflow 3: metrics shown next to the model
            except Exception:
                pass
        except Exception as e:
            problems.append(f"model {name} not logged ({type(e).__name__}: {str(e)[:90]})")
    for key, series in ch["history"].items():
        for i, v in enumerate(series):
            fv = _finite({key: v})
            if fv:
                mlflow.log_metric(_clean(key), list(fv.values())[0], step=i)
    for tr in list_runs(parent=ch["run_id"]):
        counts["tuning_trials"] += 1
        with mlflow.start_run(run_name=tr["name"], nested=True):
            mlflow.log_params({_clean(k): str(v)[:500] for k, v in tr["params"].items() if v is not None})
            mlflow.log_metrics(_finite(tr["metrics"]))


def mirror_mlflow(run_id: str, bundle_path: Optional[Path], registered_name: Optional[str] = None, stage: str = "Staging",
                  candidates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write one analysis to MLflow: a SUMMARY run (params, metrics, report + leaderboard + model-card artifacts, the registered model) and ONE RUN PER CANDIDATE
    of every attempt (hyper-parameters, held-out + CV metrics, train score, overfit gap, p-value vs the winner, verdict, diagnostics artifacts, the fitted model).

    Layout: candidate runs are TOP-LEVEL runs tagged ds.parent_run_id / ds.role=candidate, so the MLflow runs table lists every model as its own sortable,
    comparable row (the usual AutoML layout). Nested runs hide behind a collapsed parent row, which is why v4 looked as if only the best model was tracked.
    Set NEXUSML_MLFLOW_NESTED=1 to nest them under the summary run instead. Tuning trials always nest under their candidate.
    Never raises: returns {"ok": bool, "message": str, ...}; a failure names the STEP that failed and is saved on the run record."""
    st = mlflow_status()
    if not st["active"]:
        return {"ok": False, "message": st["reason"]}
    step, problems = "starting", []
    nested_mode = os.environ.get("NEXUSML_MLFLOW_NESTED", "0") == "1"
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        run = get_run(run_id)
        if not run:
            return {"ok": False, "message": f"run record {run_id} was not found"}
        step = "connecting to the MLflow store"
        uri = tracking_uri()
        mlflow.set_tracking_uri(uri)
        client = MlflowClient()
        step = "creating the experiment"
        exp_id = _ensure_experiment(client, run["experiment"])
        out: Dict[str, Any] = {"ok": True, "uri": uri}
        model_uri = None
        counts = {"candidate_runs": 0, "tuning_trials": 0, "models": 0}
        kids = list_runs(parent=run_id)
        finals = [int(k["tags"].get("attempt", 1)) for k in kids if k["tags"].get("ds.role") == "candidate"]
        final_attempt = max(finals) if finals else 1
        for k in kids:
            k["tags"]["_final_attempt"] = final_attempt
        with mlflow.start_run(experiment_id=exp_id, run_name=run["name"]) as parent:
            step = "logging parameters, metrics and tags"
            mlflow.log_params({_clean(k): str(v)[:500] for k, v in run["params"].items() if v is not None})
            mlflow.log_metrics(_finite(run["metrics"]))
            mlflow.set_tags({_clean(k): str(v)[:500] for k, v in run["tags"].items() if v is not None})
            step = "logging report files"
            for key in run["artifacts"].values():
                p = get_registry().path_of(key)
                if p.exists() and p.suffix not in (".joblib", ".sha256"):
                    mlflow.log_artifact(str(p))
            step = "logging the leaderboard"
            if kids:
                import pandas as pd
                table = pd.DataFrame([{"model": k["params"].get("model", k["name"]), "attempt": k["tags"].get("attempt", 1), "selected": k["tags"].get("selected"), **k["metrics"],
                                       "verdict": k["tags"].get("verdict")} for k in kids if k["tags"].get("ds.role") == "candidate"])
                if len(table):
                    table = table.sort_values(["attempt", "rank"] if "rank" in table else ["attempt"], ascending=[False, True] if "rank" in table else [False])
                    _try(lambda: mlflow.log_table(table, "leaderboard.json"), problems, "the MLflow evaluation table")
                    _try(lambda: mlflow.log_text(table.to_csv(index=False), "leaderboard_all_attempts.csv"), problems, "leaderboard csv")
            if nested_mode:
                step = "logging the candidate models' runs (nested)"
                for ch in kids:
                    counts["candidate_runs"] += 1
                    with mlflow.start_run(run_name=ch["name"], nested=True):
                        _log_candidate(mlflow, ch, candidates, counts, problems)
            out["mlflow_run_id"] = parent.info.run_id
            _save_mlflow_fields(run_id, mlflow_run_id=out["mlflow_run_id"], mlflow_error="", mlflow_notes="")      # the run itself is safely in MLflow now
            step = "logging the model"
            if bundle_path and Path(bundle_path).exists():
                try:
                    from . import modelio
                    est = modelio.load(Path(bundle_path), trusted=True).get("pipeline")
                    if est is not None:
                        info = _log_sklearn(mlflow, est)
                        model_uri = getattr(info, "model_uri", None) or f"runs:/{out['mlflow_run_id']}/model"
                        counts["models"] += 1
                except Exception as e:
                    problems.append(f"the model could not be logged ({type(e).__name__}: {str(e)[:200]})")
        if not nested_mode:
            step = "logging one run per candidate model"
            for ch in kids:
                counts["candidate_runs"] += 1
                with mlflow.start_run(experiment_id=exp_id, run_name=ch["name"]):
                    mlflow.set_tags({"ds.parent_run_id": out["mlflow_run_id"], "ds.analysis": run["name"]})
                    _log_candidate(mlflow, ch, candidates, counts, problems)
        if model_uri and registered_name:
            step = "registering the model"
            try:
                mv = mlflow.register_model(model_uri, registered_name)
                out["registered"] = {"name": registered_name, "version": str(mv.version)}
                _mlflow_stage(client, registered_name, str(mv.version), stage)
            except Exception as e:
                problems.append(f"the model could not be registered ({type(e).__name__}: {str(e)[:200]})")
        layout = "nested under the summary run" if nested_mode else "each as its own top-level run, sortable in the runs table"
        summary = (f"logged 1 summary run + {counts['candidate_runs']} compared-model run(s) across {final_attempt} attempt(s) ({layout}), "
                   f"{counts['tuning_trials']} tuning trial(s), {counts['models']} model(s) (the best one is registered)")
        out["counts"] = counts
        out["message"] = f"stored in MLflow ({uri}): {summary}"
        _save_mlflow_fields(run_id, mlflow_summary=summary)
        if problems:
            out["message"] = f"the run is in MLflow ({uri}) ({summary}), but " + "; ".join(problems)
            _save_mlflow_fields(run_id, mlflow_notes=out["message"])
        return out
    except Exception as e:
        msg = f"MLflow logging failed while {step} ({type(e).__name__}: {str(e)[:300]}). The JSON store still has this run."
        _save_mlflow_fields(run_id, mlflow_error=msg)
        return {"ok": False, "message": msg}


def mlflow_inventory() -> Dict[str, Any]:
    """What is physically inside the MLflow store right now (experiments, run counts, logged models with their status, registered models + versions)."""
    st = mlflow_status()
    if not st["active"]:
        return {"active": False, "reason": st["reason"]}
    try:
        from mlflow.tracking import MlflowClient
        c = MlflowClient(tracking_uri=tracking_uri())
        out: Dict[str, Any] = {"active": True, "uri": tracking_uri(), "experiments": [], "registered": []}
        for e in c.search_experiments():
            runs = c.search_runs([e.experiment_id], max_results=1000)
            models = None
            try:
                models = [{"name": m.name, "status": str(getattr(m.status, "name", m.status)), "id": m.model_id} for m in c.search_logged_models(experiment_ids=[e.experiment_id], max_results=200)]
            except Exception:
                pass                                                          # MLflow 2.x has no 'logged models'
            out["experiments"].append({"name": e.name, "runs": len(runs), "models": models})
        for rm in c.search_registered_models(max_results=200):
            vers = c.search_model_versions(f"name='{rm.name}'")
            out["registered"].append({"name": rm.name, "versions": [{"version": str(v.version), "status": str(getattr(v, "status", "")),
                                                                    "aliases": list(getattr(v, "aliases", []) or []), "run_id": v.run_id} for v in vers]})
        return out
    except Exception as e:
        return {"active": True, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def _mlflow_stage(client, name: str, version: str, stage: str) -> None:
    """Portable staging: a `stage` tag on the version + aliases for Staging/Production (MLflow stages are deprecated in favour of aliases)."""
    client.set_model_version_tag(name, version, "stage", stage)
    alias = {"Staging": "staging", "Production": "production"}.get(stage)
    if alias:
        client.set_registered_model_alias(name, alias, version)
    try:                                                               # an older alias holder that is no longer the holder becomes Archived
        if alias == "production":
            for mv in client.search_model_versions(f"name='{name}'"):
                if str(mv.version) != version and (mv.tags or {}).get("stage") == "Production":
                    client.set_model_version_tag(name, str(mv.version), "stage", "Archived")
    except Exception:
        pass


def load_registered_model(name: str, alias: str = "production"):
    """Load a model from the real MLflow registry (needs mlflow)."""
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(tracking_uri())
    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")


# views for the Experiments page
def runs_table(experiment: Optional[str] = None) -> List[Dict[str, Any]]:
    """Flat rows (one per analysis run) for a sortable/selectable table."""
    rows = []
    for r in list_runs(experiment, parent=None):
        m, p, t = r["metrics"], r["params"], r["tags"]
        rows.append({"run_id": r["run_id"], "name": r["name"], "experiment": r["experiment"],
                     "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["start"] or 0)), "best_model": t.get("best_model"),
                     "passed": t.get("quality_passed"), "primary_metric": p.get("primary_metric"), "score": m.get("best_select_score"),
                     "baseline": m.get("baseline_cv"), "tuning_iter": p.get("tuning_iterations"), "levers": p.get("retry_levers"),
                     "quality": m.get("data_quality_score")})
    return rows


def trial_models(run_id: str) -> Dict[str, Any]:
    """Rebuild a tuning-report-like structure from a run's child/grand-child runs (for parallel-coordinates etc.)."""
    out: Dict[str, Any] = {}
    for ch in list_runs(parent=run_id):
        trials = [{"rank": i + 1, "params": {k[3:]: v for k, v in tr["params"].items() if k.startswith("hp.")},
                   "mean_score": tr["metrics"].get("mean_cv_score"), "std_score": tr["metrics"].get("std_cv_score", 0.0)}
                  for i, tr in enumerate(sorted(list_runs(parent=ch["run_id"]), key=lambda r: -(r["metrics"].get("mean_cv_score") or -1e9)))]
        if trials:
            out[ch["name"]] = {"trials": trials, "default_cv": ch["metrics"].get("default_cv"), "nested_mean": ch["metrics"].get("select_score") if ch["params"].get("tuned") else None}
    return out


def param_diff(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = sorted({k for r in runs for k in r["params"]})
    rows = []
    for k in keys:
        vals = [str(r["params"].get(k, "")) for r in runs]
        if len(set(vals)) > 1:
            rows.append({"param": k, **{r["name"]: v for r, v in zip(runs, vals)}})
    return rows
