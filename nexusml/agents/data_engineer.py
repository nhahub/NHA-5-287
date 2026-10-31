"""Data Engineer agent: profiles the dataset, then prepares leak-free splits."""
from typing import Any, Dict

import pandas as pd

from .. import config, featurizers, llm, modeling, profiling, runlog
from ..storage import get_registry
from ._util import log, make_evidence

AGENT = "Data Engineer"


def profile_dataset_node(state) -> Dict[str, Any]:
    runlog.start(state.get("run_id", "run"), state.get("seed", config.RANDOM_STATE),
                 {"dataset": str(state.get("dataset_path")), "request": state.get("user_request", "")[:200], "model": config.LLM_MODEL})
    llm.preflight()      # fail fast: the LLM is mandatory (no rule-based fallback)
    df = pd.read_csv(state["dataset_path"])
    profile = profiling.profile_dataset(df)
    profile["quality"] = profiling.data_quality(df, profile)
    q = profile["quality"]
    kinds: Dict[str, int] = {}
    for info in profile["columns"].values():
        kinds[info["kind"]] = kinds.get(info["kind"], 0) + 1
    items = [("data_quality", f"Data quality score {q['score']}/100 ({q['label']}): {q['missing_cell_pct']}% of cells missing, "
                              f"{q['duplicate_row_pct']}% duplicate rows, {len(q['constant_columns'])} constant column(s). {q['explanation']}", q)]
    if q["outlier_columns_pct"]:
        worst = max(q["outlier_columns_pct"], key=q["outlier_columns_pct"].get)
        items.append(("data_quality", f"{len(q['outlier_columns_pct'])} numeric column(s) have >1% IQR outliers "
                                      f"(highest: {worst} at {q['outlier_columns_pct'][worst]}%).", q["outlier_columns_pct"]))
    return {
        "dataset_profile": profile, "warnings": profile["warnings"], "evidence": make_evidence("Q", items),
        "current_status": "Dataset profiled.", "active_agent": AGENT,
        "agent_log": [log(AGENT, f"Profiled {profile['n_rows']:,} rows x {profile['n_cols']} columns "
                                 f"({', '.join(f'{v} {k}' for k, v in kinds.items())}); quality {q['score']}/100.")],
    }


def prepare_data_node(state) -> Dict[str, Any]:
    plan = state["analysis_plan"]
    df = pd.read_csv(state["dataset_path"])
    splits = modeling.make_splits(df, plan)
    key = get_registry().save(splits, "splits")
    s = splits["summary"]
    pt = plan["problem_type"]
    if pt in modeling.SUPERVISED:
        extra = ""
        if s.get("time_series"):
            ts = s["time_series"]
            extra = f" Lags {ts['lags']}, rolling {ts['rolls']}" + (f", aggregated to {ts['freq']}" if ts["aggregated"] else "") + "."
        msg = (f"{s['split'].capitalize()} split: {s['n_train']:,} train / {s['n_test']:,} test, "
               f"{s['n_features']} features. Preprocessing is fit inside each CV fold (no leakage).{extra}")
        metric = modeling.primary_metric_for(pt, len(splits["classes"] or []))
    else:
        msg = f"No train/test split needed for {pt} analysis; {s['n_rows_used']:,} rows, {s['n_features']} features."
        metric = ""
    out = {"splits_key": key, "data_summary": s, "primary_metric": metric,
           "current_status": "Data prepared.", "active_agent": AGENT, "agent_log": [log(AGENT, msg)]}
    try:                                                     # text / image columns: describe them and check train/test duplicates (leakage)
        X_tr = splits.get("X_train")
        mods = featurizers.modalities(X_tr) if X_tr is not None else {"text": [], "image": []}
        if mods["text"] or mods["image"]:
            out["modality"] = mods
            items = [("modality", f"Non-tabular inputs detected: text column(s) {mods['text'] or 'none'}, image-path column(s) {mods['image'] or 'none'}. "
                                  "They are converted to TF-IDF / image features inside the same leak-free pipeline; images are never sent to the LLM.", mods)]
            if splits.get("X_test") is not None:
                dup = featurizers.duplicate_report(X_tr, splits["X_test"], mods)
                for c, r in dup.items():
                    if r["near_dup_share"] >= 0.01:
                        items.append(("leakage", f"{r['near_dup_share']:.1%} of held-out '{c}' {r['kind']} values are exact or near duplicates of training rows "
                                                 "(this inflates the held-out score; consider de-duplicating before splitting).", r))
            out["evidence"] = make_evidence("L", items)
    except Exception as e:
        out["warnings"] = [f"Text/image checks skipped ({type(e).__name__}: {e})"]
    return out
