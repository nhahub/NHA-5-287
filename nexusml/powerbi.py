"""OPTIONAL Power BI integration (downstream only).

  * Always works offline: writes a Power BI-ready package (CSV tables, data dictionary, DAX
    measures, report-page spec) that can be loaded with Get Data > Folder/Text.
  * Optionally pushes the summary tables to a Power BI push/streaming dataset when credentials
    are present in the ENVIRONMENT:
        POWERBI_PUSH_URL                       (streaming/push dataset URL, key included)   -- or --
        POWERBI_DATASET_ID + POWERBI_ACCESS_TOKEN (REST push dataset; token from your own auth flow)
  * Credentials are read here only: never put in graph state, prompts, logs or reports.
    The LLM only ever sees metadata (table/column names, KPI specs) to propose a report layout.
"""
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

import pandas as pd

MAX_FACT_ROWS = 50000


def configured() -> bool:
    return bool(os.environ.get("POWERBI_PUSH_URL") or (os.environ.get("POWERBI_DATASET_ID") and os.environ.get("POWERBI_ACCESS_TOKEN")))


def dax_for(spec: Dict[str, Any], table: str = "Fact") -> str:
    a, c = spec["agg"], spec.get("column")
    t = f"'{table}'"
    if a == "sum":
        return f"SUM({t}[{c}])"
    if a == "mean":
        return f"AVERAGE({t}[{c}])"
    if a == "median":
        return f"MEDIAN({t}[{c}])"
    if a == "min":
        return f"MIN({t}[{c}])"
    if a == "max":
        return f"MAX({t}[{c}])"
    if a == "count":
        return f"COUNTROWS({t})"
    if a == "nunique":
        return f"DISTINCTCOUNT({t}[{c}])"
    if a == "rate":
        pos = str(spec["positive"]).replace('"', '""')
        return f'DIVIDE(CALCULATE(COUNTROWS({t}), {t}[{c}] = "{pos}"), COUNTROWS({t}))'
    return f"DIVIDE(SUM({t}[{spec['numerator']}]), SUM({t}[{spec['denominator']}]))"


def build_tables(state: Dict[str, Any], df: pd.DataFrame, labels=None) -> Dict[str, pd.DataFrame]:
    t: Dict[str, pd.DataFrame] = {}
    fact = df.head(MAX_FACT_ROWS).copy()
    if labels is not None and len(labels) == len(df):
        fact["segment_cluster"] = list(labels)[: len(fact)]
    t["Fact"] = fact
    rows, seg_rows, time_rows = [], [], []
    for name, r in (state.get("kpi_results") or {}).items():
        rows.append({"kpi": name, "value": r["overall"], "formatted": r["overall_fmt"], "format": r["spec"]["format"],
                     "higher_is_better": r["spec"]["higher_is_better"], "dax": dax_for(r["spec"])})
        for col, segs in r["segments"].items():
            for s in segs:
                seg_rows.append({"kpi": name, "dimension": col, **{k: s[k] for k in ("segment", "value", "n", "vs_overall_pct")}})
        tm = r.get("time")
        if tm:
            time_rows += [{"kpi": name, "period": p, "value": v} for p, v in zip(tm["periods"], tm["values"])]
    t["KPI_Summary"] = pd.DataFrame(rows)
    t["KPI_By_Segment"] = pd.DataFrame(seg_rows)
    t["KPI_Trend"] = pd.DataFrame(time_rows)
    an = (state.get("trend_results") or {}).get("anomalies", [])
    t["Anomalies"] = pd.DataFrame(an)
    t["Statistical_Tests"] = pd.DataFrame([{"id": r["id"], "test": r["test"], "method": r.get("method"), "p_adjusted": r.get("p_adjusted", r.get("p_value")),
                                           "effect": r.get("effect_size"), "effect_label": r.get("effect_label"), "significant": r.get("significant"),
                                           "interpretation": r.get("interpretation")} for r in state.get("stat_results", [])])
    t["Root_Causes"] = pd.DataFrame([{**{k: v for k, v in rc.items() if k != "evidence_ids"}, "evidence_ids": ",".join(rc["evidence_ids"])} for rc in state.get("root_causes", [])])
    t["Recommendations"] = pd.DataFrame([{**{k: v for k, v in r.items() if k != "evidence_ids"}, "evidence_ids": ",".join(r["evidence_ids"])} for r in state.get("recommendations", [])])
    if state.get("candidate_metrics"):
        t["Model_Metrics"] = pd.DataFrame([{k: v for k, v in c.items() if k not in ("tuning",)} for c in state["candidate_metrics"]])
    imp = (state.get("explanations") or {}).get("importances")
    if imp:
        t["Feature_Importance"] = pd.DataFrame(imp)
    return {k: v for k, v in t.items() if v is not None and len(v)}


def push(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Push the SMALL summary tables (never the fact table). Returns a status dict without secrets."""
    url, ds, tok = os.environ.get("POWERBI_PUSH_URL"), os.environ.get("POWERBI_DATASET_ID"), os.environ.get("POWERBI_ACCESS_TOKEN")
    if not (url or (ds and tok)):
        return {"status": "not_configured", "detail": "Set POWERBI_PUSH_URL or POWERBI_DATASET_ID + POWERBI_ACCESS_TOKEN to enable pushing."}
    sent, errors = [], []
    for name in ("KPI_Summary", "KPI_By_Segment", "KPI_Trend", "Anomalies", "Recommendations"):
        df = tables.get(name)
        if df is None or df.empty:
            continue
        rows = json.loads(df.to_json(orient="records", date_format="iso"))
        target = url or f"https://api.powerbi.com/v1.0/myorg/datasets/{ds}/tables/{name}/rows"
        for i in range(0, len(rows), 5000):
            req = urllib.request.Request(target, data=json.dumps({"rows": rows[i:i + 5000]}).encode(), method="POST",
                                         headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {tok}"} if not url else {})})
            try:
                urllib.request.urlopen(req, timeout=30).read()
            except urllib.error.HTTPError as e:
                errors.append(f"{name}: HTTP {e.code}")
                break
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}")
                break
        else:
            sent.append(name)
    return {"status": "pushed" if sent and not errors else ("partial" if sent else "failed"), "tables_sent": sent, "errors": errors}


def data_dictionary(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    return {n: [{"column": c, "type": str(d)} for c, d in df.dtypes.items()] for n, df in tables.items()}
