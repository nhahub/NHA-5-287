"""Power BI export (optional), report generation and run finalisation."""
from typing import Any, Dict

import pandas as pd

from .. import powerbi, reporting
from ..storage import get_registry
from ._util import ANALYST_PERSONA, ask, jsonable, log

A_PBI, A_REP = "Power BI Exporter", "Reporter"


def route_powerbi(state) -> str:
    return "powerbi" if state.get("powerbi_enabled") else "report"


def powerbi_export_node(state) -> Dict[str, Any]:
    reg = get_registry()
    df = pd.read_csv(state["dataset_path"])
    labels = reg.load(state["models_key"]).get("labels") if state.get("analysis_plan", {}).get("problem_type") == "clustering" else None
    tables = powerbi.build_tables(state, df, labels)
    base = f"powerbi/{state['experiment_id']}"
    files = []
    for name, t in tables.items():
        reg.save_bytes(t.to_csv(index=False).encode("utf-8"), f"{base}/{name}.csv")
        files.append(f"{name}.csv")
    measures = [{"name": n, "dax": powerbi.dax_for(r["spec"]), "format": r["spec"]["format"]} for n, r in (state.get("kpi_results") or {}).items()]
    # LLM proposes a report layout from METADATA only; code keeps only visuals whose fields exist
    meta = {n: list(t.columns) for n, t in tables.items() if n != "Fact"} | {"Fact": list(tables["Fact"].columns)[:40]}
    payload = {"business_question": state.get("business_question"), "tables": meta, "measures": [m["name"] for m in measures],
               "output_schema": {"pages": [{"title": "", "visuals": [{"type": "card|bar|line|table|scatter", "table": "", "fields": [""], "title": ""}]}]}}
    names = {m["name"] for m in measures}

    def v_layout(d):
        pages = []
        for p in (d.get("pages") or [])[:3]:
            vis = []
            for v in (p.get("visuals", []) if isinstance(p, dict) else []):
                t = tables.get(v.get("table")) if isinstance(v, dict) else None
                if t is not None and v.get("fields") and all(f in t.columns or f in names for f in v["fields"]):
                    vis.append({k: v[k] for k in ("type", "table", "fields", "title") if k in v})
            if vis:
                pages.append({"title": str(p.get("title", "Page"))[:60], "visuals": vis})
        if not pages:
            raise ValueError(f"each visual must use an existing table and fields. Tables/columns: {meta}")
        return pages

    pages, warns, calls = ask(state, "powerbi_report_layout", ANALYST_PERSONA + " Design a concise Power BI report (3 pages max).", payload, 1200, v_layout)
    reg.save_bytes(__import__("json").dumps({"measures": measures, "pages": pages, "relationships": [], "data_dictionary": powerbi.data_dictionary(tables)}, indent=2).encode(),
                   f"{base}/report_spec.json")
    reg.save_bytes(("\n".join(f"{m['name']} = {m['dax']}" for m in measures)).encode(), f"{base}/measures.dax")
    reg.save_bytes(b"Load every CSV here via Power BI Desktop > Get Data > Text/CSV (or Folder). Create the measures in measures.dax on the 'Fact' table, "
                   b"then build pages following report_spec.json.\n", f"{base}/README_powerbi.txt")
    push = powerbi.push(tables)
    res = {"package_dir": str(reg.path_of(base)), "files": files + ["report_spec.json", "measures.dax", "README_powerbi.txt"],
           "n_measures": len(measures), "pages": pages, "push": push, "layout_source": "LLM"}
    return {"powerbi_result": jsonable(res), "warnings": warns + ([f"Power BI push: {push.get('status')} {push.get('errors', '')}"] if push.get("status") in ("failed", "partial") else []),
            "llm_calls": calls, "current_status": "Power BI package written.", "active_agent": A_PBI,
            "agent_log": [log(A_PBI, f"Wrote Power BI package ({len(files)} tables, {len(measures)} DAX measures); push status: {push['status']}.")]}


def generate_report_node(state) -> Dict[str, Any]:
    reg = get_registry()
    md, html = reporting.build(state, reg)
    base = state["experiment_id"]
    key = reg.save_text(md, f"{base}/report.md")
    hkey = reg.save_text(html, f"{base}/report.html")
    extra: Dict[str, Any] = {}
    try:                                                      # role-based output: one-page executive brief next to the analyst report
        from .. import reporting_roles
        emd, ehtml = reporting_roles.build_executive(state, reg)
        reg.save_text(emd, f"{base}/executive_summary.md")
        extra["exec_summary_key"] = reg.save_text(ehtml, f"{base}/executive_summary.html")
    except Exception as e:
        extra["warnings"] = [f"Executive brief skipped ({type(e).__name__}: {e})"]
    return {**extra, "report_markdown": md, "report_key": key, "report_html_key": hkey, "current_status": "Report generated.", "active_agent": A_REP,
            "agent_log": [log(A_REP, f"Report written: {len(md.split()):,} words, {len(state.get('charts', []))} chart(s), HTML with embedded charts.")]}


def save_bundle(state):
    """Persist the deployable model bundle (None when there is nothing to deploy, e.g. descriptive analysis)."""
    reg = get_registry()
    plan, pt = state["analysis_plan"], state["analysis_plan"]["problem_type"]
    models = reg.load(state["models_key"])["models"] if state.get("models_key") else {}
    name = state.get("best_model_name")
    if not (models and name in models):
        return None
    splits = reg.load(state["splits_key"])
    bundle = {"experiment_id": state["experiment_id"], "problem_type": pt, "analysis_type": state.get("analysis_type"),
              "target": plan.get("target"), "date_column": plan.get("date_column"), "drop_columns": plan["drop_columns"],
              "datetime_columns": plan["datetime_columns"], "feature_columns": list(splits["X_train"].columns),
              "classes": splits.get("classes"), "positive_class": splits.get("positive_class"), "ts_info": splits.get("ts_info"),
              "best_model": name, "pipeline": models[name]}
    if state.get("candidate_metrics"):
        bundle["metrics"] = next((c for c in state["candidate_metrics"] if c["name"] == name), {})
    from .. import modelio
    key = f"models/{state['experiment_id']}.joblib"
    p = reg.path_of(key)
    modelio.dump(bundle, p)                                   # + .sha256 sidecar
    return key


def track_experiment_node(state) -> Dict[str, Any]:
    """Save the model, then log the analysis as MLflow-style runs (+ registry version); mirror to MLflow if installed."""
    from .. import tracking
    reg = get_registry()
    model_key = save_bundle(state)
    snap = {**state, "model_key": model_key}
    info = tracking.log_analysis(snap, model_key)
    reg_name = (info.get("registered") or {}).get("model_name")
    cands = None
    try:
        cands = reg.load(state["models_key"]).get("models") if state.get("models_key") else None
    except Exception:
        cands = None
    ml = tracking.mirror_mlflow(info["run_id"], reg.path_of(model_key) if model_key else None, reg_name,
                                (info.get("registered") or {}).get("stage", "Staging"), candidates=cands)
    info["mlflow"] = ml.get("message", "")
    info["mlflow_detail"] = {k: v for k, v in ml.items() if k != "message"}
    if ml.get("registered") and info.get("registered"):
        tracking.attach_mlflow_version(reg_name, info["registered"]["version"], ml["registered"]["version"])
    reg_v = info.get("registered") or {}
    return {"model_key": model_key, "tracking": jsonable(info), "current_status": "Experiment tracked.", "active_agent": "Experiment Tracker",
            "agent_log": [log("Experiment Tracker", f"Logged run {info['run_id']} with {info['child_runs']} child run(s)"
                              + (f"; registered {reg_v.get('model_name')} v{reg_v.get('version')} ({reg_v.get('stage')})" if reg_v else "")
                              + f". MLflow: {info['mlflow'][:70]}")]}


def finalize_node(state) -> Dict[str, Any]:
    import json
    reg = get_registry()
    plan = state["analysis_plan"]
    pt = plan["problem_type"]
    from .. import resume as _resume
    saved = {k: v for k, v in state.items()}
    if saved.get("resume"):
        saved["resume"] = _resume.slim(saved["resume"])                  # the carried payload is already inside this run's own results
    skey = reg.save_bytes(json.dumps(jsonable(saved), default=str).encode("utf-8"), f"runs/{state['experiment_id']}/state.json")
    reg.log_run({"experiment_id": state["experiment_id"], "created": pd.Timestamp.now().isoformat(timespec="seconds"),
                 "request": state.get("user_request"), "analysis_type": state.get("analysis_type"), "problem_type": pt,
                 "target": plan.get("target"), "best_model": state.get("best_model_name"), "quality_passed": state.get("quality_passed"),
                 "primary_metric": state.get("primary_metric"), "report": state.get("report_key"), "state_key": skey,
                 "headline": (state.get("insights") or {}).get("headline")})
    try:
        from .. import runstore
        runstore.record_run({**state, "state_key": skey})
    except Exception:
        pass
    try:
        from .. import runlog as _rl
        _rl.end_run()
    except Exception:
        pass
    repro = None
    try:
        from .. import export
        repro = reg.save_bytes(export.build_reproducibility_kit({**state}, reg), f"{state['experiment_id']}/reproducibility_kit.zip")
    except Exception:
        pass
    return {"saved_experiment_id": state["experiment_id"], **({"repro_key": repro} if repro else {}), "current_status": "Complete.", "active_agent": A_REP,
            "agent_log": [log(A_REP, f"Saved experiment {state['experiment_id']} (run state, report, tracking).")]}
