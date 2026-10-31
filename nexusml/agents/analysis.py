"""Analysis-phase agents (run before any modeling):

  Statistician   - selects and runs only the statistical tests that fit the data and question
  KPI Analyst    - validates/computes business KPIs and segment breakdowns
  Trend Analyst  - trend, seasonality, level shifts, anomalies, outliers
  Visualizer     - EDA, KPI and trend charts (box plots, correlation maps, ...)
"""
from typing import Any, Dict, List, Optional

import pandas as pd

from .. import config, interactive, kpi, llm, modeling, stats_tests, trends, viz
from ..storage import get_registry
from ._util import ANALYST_PERSONA, ask, jsonable, log, make_evidence


def _load(state):
    return pd.read_csv(state["dataset_path"])


def _features(state) -> List[str]:
    plan, prof = state["analysis_plan"], state["dataset_profile"]["columns"]
    skip = {plan.get("target")} | set(plan["drop_columns"])
    return [c for c, i in prof.items() if c not in skip and i["kind"] != "datetime"]


def _kpi_measure(state, df) -> Optional[str]:
    for s in state.get("kpi_specs") or []:
        v, _ = kpi.validate_spec(s, df)
        if v and v.get("column") and v["agg"] in ("sum", "mean", "median"):
            return v["column"]
    nums = [c for c, i in state["dataset_profile"]["columns"].items() if i["kind"] == "numeric" and i["unique_count"] > 5]
    return nums[0] if nums else None


# Statistician
def statistical_analysis_node(state) -> Dict[str, Any]:
    A = "Statistician"
    plan, profile = state["analysis_plan"], state["dataset_profile"]
    df = _load(state)
    pt, target = plan["problem_type"], plan.get("target")
    measure = target or _kpi_measure(state, df)
    feats = _features(state)

    ref = cur = None
    have_holdout = have_ref = False
    date_derived = {f"{d}_{s}" for d in plan["datetime_columns"] for s in ("year", "month", "day", "dayofweek", "ordinal")}
    if state.get("reference_path"):
        try:
            r = pd.read_csv(state["reference_path"])
            ref = modeling.clean_frame(r, plan["drop_columns"], plan["datetime_columns"])
            cur = modeling.clean_frame(df, plan["drop_columns"], plan["datetime_columns"])
            have_ref = True
        except Exception as e:
            ref = None
    if not have_ref and state.get("splits_key") and pt in modeling.SUPERVISED:
        sp = get_registry().load(state["splits_key"])
        if sp.get("X_test") is not None and len(sp["X_test"]) >= 30:
            ref, cur, have_holdout = sp["X_train"], sp["X_test"], True

    cands, skipped = stats_tests.enumerate_candidates(
        df, profile, measure, pt, feats, state.get("segment_columns", []), have_holdout, have_ref)
    for c in cands:
        if c["test"] == "ks_drift":
            c["params"]["columns"] = [x for x in c["params"]["columns"] if x not in date_derived]

    # ---- selection: the LLM chooses from the VALID menu (mandatory; no heuristic fallback)
    chosen: Dict[str, str] = {}
    warns: List[str] = []
    calls: List[Dict[str, Any]] = []
    if cands:
        ids = {c["id"] for c in cands}
        payload = {"business_question": state.get("business_question"), "analysis_type": state.get("analysis_type"),
                   "problem_type": pt, "measure": measure, "max_tests": config.MAX_TESTS,
                   "candidates": [{"id": c["id"], "test": c["test"],
                                   "params": {k: v for k, v in c["params"].items() if k != "columns"} if c["test"] in ("vif", "ks_drift") else c["params"],
                                   "why": c["why"]} for c in cands],
                   "rules": ["Choose only tests that help answer the business question.",
                             "Do not pick redundant tests on near-identical variables.",
                             "Include drift/VIF only if they matter for trusting a model.",
                             "Prefer fewer, meaningful tests over many; never more than max_tests."],
                   "output_schema": {"selected": [{"id": "T001", "reason": "why this test serves the business question"}]}}

        def v_sel(d):
            out = {}
            for x in d.get("selected") or []:
                if isinstance(x, dict) and x.get("id") in ids and x["id"] not in out:
                    out[x["id"]] = str(x.get("reason", ""))[:200]
            if not out:
                raise ValueError(f"select at least one id from {sorted(ids)[:60]}")
            return out

        chosen, warns, calls = ask(state, "select_statistical_tests",
                                   ANALYST_PERSONA + " You choose which statistical tests to run from a pre-validated menu.", payload, 1200, v_sel)
        chosen = dict(list(chosen.items())[: config.MAX_TESTS])
    by = "LLM"

    results = []
    for c in cands:
        if c["id"] in chosen:
            r = stats_tests.execute(c, df, ref, cur)
            r["selection_reason"] = chosen[c["id"]]
            results.append(r)
    results = stats_tests.finalize(results)
    ran = [r for r in results if "skipped" not in r]
    for r in results:
        if "skipped" in r:
            skipped.append({"test": f"{r['test']} {r['params'].get('measure') or r['params'].get('a') or ''}".strip(), "reason": r["skipped"]})
    n_unsel = len(cands) - len(chosen)
    if n_unsel > 0:
        skipped.append({"test": f"{n_unsel} other applicable test(s)", "reason": "Not selected: lower relevance to the business question (not run to avoid multiple-testing noise)."})

    items = []
    for r in ran:
        if r.get("significant") and r.get("interpretation"):
            items.append(("statistical_test", f"[{r['method']}] {r['interpretation']}", {"test_id": r["id"]}))
    if not items and ran:
        items.append(("statistical_test", f"None of {len(ran)} selected tests showed a significant, non-negligible effect after multiple-comparison correction.", {}))
    ev = make_evidence("S", items)
    return {"stat_results": jsonable(ran), "stat_skipped": skipped, "evidence": ev, "warnings": warns, "llm_calls": calls,
            "current_status": "Statistical analysis done.", "active_agent": A,
            "agent_log": [log(A, f"Ran {len(ran)} of {len(cands)} applicable tests (selected by {by}); "
                                 f"{sum(1 for r in ran if r.get('significant'))} significant after BH correction.")]}


# KPI analyst
def kpi_analysis_node(state) -> Dict[str, Any]:
    A = "KPI Analyst"
    df = _load(state)
    plan = state["analysis_plan"]
    date_col = plan.get("date_column")
    warns, specs, rejected = [], [], []
    for s_ in state.get("kpi_specs") or []:
        v, why = kpi.validate_spec(s_, df)
        if v:
            specs.append(v)
        else:
            rejected.append(f"KPI '{(s_ or {}).get('name', '?')}' rejected: {why}")
    src = "LLM"
    if not specs:
        raise llm.LLMRequired("The LLM supplied no valid KPI definitions (re-run; check the business question and column names).")
    specs = specs[:6]
    results = {}
    for s in specs:
        s = dict(s)
        if not s["group_by"]:
            s["group_by"] = state.get("segment_columns", [])[:3]
        try:
            results[s["name"]] = kpi.compute(df, s, date_col)
        except Exception as e:
            warns.append(f"KPI '{s['name']}' failed: {e}")
    items = []
    for name, r in results.items():
        sp = r["spec"]
        items.append(("kpi", f"KPI '{name}' = {r['overall_fmt']} overall.", {"kpi": name}))
        t = r.get("time")
        if t and t.get("change_pct") is not None:
            items.append(("kpi", f"'{name}' changed {t['change_pct']:+.1f}% in the latest complete period vs the previous one ({t['freq']}).", {"kpi": name}))
        for col, rows in r["segments"].items():
            rows = [x for x in rows if x["n"] >= 30 and x["vs_overall_pct"] is not None]
            if len(rows) >= 2:
                top, low = rows[0], rows[-1]
                items.append(("kpi", f"'{name}' by {col}: best '{top['segment']}' ({top['value_fmt']}, n={top['n']:,}) vs worst '{low['segment']}' ({low['value_fmt']}, n={low['n']:,}).", {"kpi": name, "col": col}))
                break
    return {"kpi_results": jsonable(results), "evidence": make_evidence("K", items), "warnings": warns + rejected,
            "current_status": "KPIs computed.", "active_agent": A,
            "agent_log": [log(A, f"Computed {len(results)} KPI(s) ({src}-defined): {', '.join(results)}"
                                 + (f"; rejected {len(rejected)} invalid spec(s)" if rejected else ""))]}


# Trend analyst
def primary_measure_spec(state, df) -> Dict[str, Any]:
    plan = state["analysis_plan"]
    target, pt = plan.get("target"), plan["problem_type"]
    prof = state["dataset_profile"]["columns"]
    if target and pt == "classification":
        counts = df[target].astype(str).value_counts()
        return {"name": f"{target} rate", "agg": "rate", "column": target, "positive": counts.index[-1], "format": "percent"}
    if target and prof[target]["kind"] == "numeric":
        return {"name": f"{'Total' if pt == 'time_series' else 'Average'} {target}",
                "agg": "sum" if pt == "time_series" else "mean", "column": target, "format": "number"}
    for r in (state.get("kpi_results") or {}).values():
        if r["spec"]["agg"] != "count":
            return r["spec"]
    return {"name": "Records", "agg": "count", "format": "number"}


def trend_analysis_node(state) -> Dict[str, Any]:
    A = "Trend Analyst"
    df = _load(state)
    plan, prof = state["analysis_plan"], state["dataset_profile"]
    date_col = plan.get("date_column")
    items, trend = [], {}
    if date_col:
        spec = primary_measure_spec(state, df)
        try:
            s = trends.period_series(df, date_col, spec)
            res = trends.analyze_series(s, spec["name"])
            trend = {"spec": spec, "freq_code": trends.choose_freq(pd.to_datetime(df[date_col], errors="coerce")), **res}
            if "skipped" not in res:
                items.append(("trend", f"'{spec['name']}' shows {res['trend']} over {res['start']} to {res['end']} "
                                       f"({res['slope_pct_of_mean']:+.2f}% of the mean per {res['freq']} period; Mann-Kendall p={res['mann_kendall_p']:.2g}; "
                                       f"volatility CV {res['volatility_cv']}).", {}))
                se = res.get("seasonality")
                if se and se["present"]:
                    items.append(("trend", f"Seasonality by {se['by']} explains {se['strength']:.0%} of the variance in '{spec['name']}'.", se))
                if res.get("change_point"):
                    cp = res["change_point"]
                    items.append(("trend", f"Level shift around {cp['period']}: the series sits {cp['shift_vs_trend']:+,.2f} away from its trend line after that point (trend removed first).", cp))
                for a in res["anomalies"][:3]:
                    items.append(("trend", f"Anomalous {a['direction']} in {a['period']}: {a['value']:,.2f} vs expected ~{a['expected']:,.2f} (robust z={a['robust_z']}).", a))
        except Exception as e:
            trend = {"skipped": f"Trend analysis failed: {e}"}
    else:
        trend = {"skipped": "No date column: trend/seasonality analysis not applicable."}
    nums = [c for c in _features(state) if prof["columns"][c]["kind"] == "numeric"]
    out = trends.numeric_outliers(df, nums)
    if out["columns"]:
        w = out["columns"][0]
        items.append(("outlier", f"{out['n_columns_with_outliers']} numeric column(s) contain IQR outliers; most: {w['column']} ({w['pct']}% beyond [{w['lower_fence']:,.2f}, {w['upper_fence']:,.2f}]).", out))
    return {"trend_results": jsonable(trend), "outlier_results": jsonable(out), "evidence": make_evidence("T", items),
            "current_status": "Trend analysis done.", "active_agent": A,
            "agent_log": [log(A, (f"{trend.get('trend', 'n/a')}; {trend.get('n_anomalies', 0)} anomaly(ies)" if "skipped" not in trend else trend["skipped"])
                              + f"; outliers in {out['n_columns_with_outliers']} column(s).")]}


# Visualizer
def visualize_eda_node(state) -> Dict[str, Any]:
    A = "Visualizer"
    df = _load(state)
    plan, prof = state["analysis_plan"], state["dataset_profile"]
    reg = get_registry()
    g = viz.Gallery(reg, state["run_id"], "eda")
    warns = []
    pos = None
    if plan.get("target") and plan["problem_type"] == "classification":
        pos = df[plan["target"]].astype(str).value_counts().index[-1]
    for label, fn in (
        ("EDA", lambda: viz.eda_charts(g, df, prof, _features(state), plan.get("target"), plan["problem_type"], pos)),
        ("interactive EDA", lambda: interactive.eda(g, df, prof, _features(state), plan.get("target"), pos)),
        ("KPI", lambda: viz.kpi_charts(g, state.get("kpi_results") or {})),
        ("interactive KPI", lambda: interactive.kpi(g, state.get("kpi_results") or {})),
        ("trend", lambda: _trend_chart(g, state, df)),
    ):
        try:
            fn()
        except Exception as e:     # one broken chart family must not kill the run
            warns.append(f"{label} charts partly failed: {type(e).__name__}: {e}")
    return {"charts": g.items, "warnings": warns, "current_status": "Visualizations created.", "active_agent": A,
            "agent_log": [log(A, f"Created {len(g.items)} chart(s): " + ", ".join(sorted({c['section'] for c in g.items})))]}


def _trend_chart(g, state, df):
    t = state.get("trend_results") or {}
    if "skipped" in t or not t.get("spec"):
        return
    s = trends.period_series(df, state["analysis_plan"]["date_column"], t["spec"])
    viz.trend_charts(g, s, t, t["spec"]["name"])
    interactive.trend(g, s, t, t["spec"]["name"])
