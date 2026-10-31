"""Detailed report builder. One document structure renders to Markdown (relative chart links)
and to a self-contained HTML file (charts embedded as base64), so the report opens anywhere."""
import base64
import html as _html
from typing import Any, Dict, List

from . import config


PLOTLY = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'


class Doc:
    def __init__(self):
        self.blocks: List[tuple] = []

    def h(self, level: int, text: str):
        self.blocks.append(("h", level, text))

    def p(self, text: str):
        if text:
            self.blocks.append(("p", text))

    def bullets(self, items: List[str]):
        items = [i for i in items if i]
        if items:
            self.blocks.append(("ul", items))

    def table(self, headers: List[str], rows: List[List[Any]]):
        if rows:
            self.blocks.append(("table", headers, [[("" if c is None else str(c)) for c in r] for r in rows]))

    def img(self, chart: Dict[str, str]):
        self.blocks.append(("img", chart))

    # ---------------------------------------------------------------- renderers
    def markdown(self, run_id: str) -> str:
        out = []
        for b in self.blocks:
            if b[0] == "h":
                out.append(f"\n{'#' * b[1]} {b[2]}\n")
            elif b[0] == "p":
                out.append(b[1] + "\n")
            elif b[0] == "ul":
                out.append("\n".join(f"- {i}" for i in b[1]) + "\n")
            elif b[0] == "table":
                esc = lambda s: s.replace("|", "\\|").replace("\n", " ")
                out.append("| " + " | ".join(esc(h) for h in b[1]) + " |\n|" + "---|" * len(b[1]) + "\n"
                           + "\n".join("| " + " | ".join(esc(c) for c in r) + " |" for r in b[2]) + "\n")
            elif b[0] == "img":
                c = b[1]
                if c.get("key"):
                    out.append(f"![{c['title']}](../../{c['key']})\n*{c['title']}: {c['caption']}*\n")
        return "\n".join(out)

    def html(self, title: str, reg) -> str:
        e = _html.escape
        parts = []
        n_pl = [0]
        for b in self.blocks:
            if b[0] == "h":
                parts.append(f"<h{b[1]}>{e(b[2])}</h{b[1]}>")
            elif b[0] == "p":
                parts.append(f"<p>{e(b[1])}</p>")
            elif b[0] == "ul":
                parts.append("<ul>" + "".join(f"<li>{e(i)}</li>" for i in b[1]) + "</ul>")
            elif b[0] == "table":
                parts.append("<div class='tw'><table><tr>" + "".join(f"<th>{e(h)}</th>" for h in b[1]) + "</tr>"
                             + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in b[2]) + "</table></div>")
            elif b[0] == "img":
                c = b[1]
                if c.get("dup"):
                    continue
                try:
                    if c.get("spec_key"):
                        n_pl[0] += 1
                        spec = reg.path_of(c["spec_key"]).read_text(encoding="utf-8").replace("</", "<\\/")
                        h = 460 if "heatmap" in spec[:400] or "parcoords" in spec[:400] else 400
                        parts.append(f"<figure><div id='pl{n_pl[0]}' style='height:{h}px'></div><figcaption><b>{e(c['title'])}.</b> {e(c['caption'])}</figcaption></figure>"
                                     f"<script>(function(){{var s={spec};Plotly.newPlot('pl{n_pl[0]}',s.data,s.layout,{{responsive:true,displaylogo:false}});}})();</script>")
                    elif c.get("key"):
                        data = base64.b64encode(reg.path_of(c["key"]).read_bytes()).decode()
                        parts.append(f"<figure><img src='data:image/png;base64,{data}' alt='{e(c['title'])}'>"
                                     f"<figcaption><b>{e(c['title'])}.</b> {e(c['caption'])}</figcaption></figure>")
                except Exception:
                    parts.append(f"<p><i>[chart unavailable: {e(c['title'])}]</i></p>")
        css = ("body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#1f2937}"
               "h1{background:linear-gradient(135deg,#071738 0%,#0d2f7d 55%,#1f5fe0 100%);color:#fff;padding:1.3rem 1.5rem;border-radius:14px;font-size:1.7rem}"
               "h2{margin-top:2.2rem;color:#0a2158;border-left:5px solid #1f5fe0;padding-left:.7rem}"
               "table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #d1d5db;padding:4px 8px;text-align:left;vertical-align:top}"
               "th{background:#0a2158;color:#fff;border-color:#0a2158}tr:nth-child(even) td{background:#f4f7ff}.tw{overflow-x:auto;margin:.6rem 0}figure{margin:1rem 0}img{max-width:100%;border:1px solid #e5e7eb}"
               "figcaption{font-size:12.5px;color:#4b5563}@media print{body{max-width:none;margin:0}figure,table{break-inside:avoid}h2{break-before:auto}}")
        return f"<!doctype html><html><head><meta charset='utf-8'><title>{e(title)}</title><style>{css}</style>{PLOTLY if n_pl[0] else ''}</head><body>{''.join(parts)}</body></html>"


def _n(v, d: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{d}g}" if abs(v) < 1e4 else f"{v:,.0f}"
    return str(v)


def _charts(doc: Doc, state, *sections: str, limit: int = 99):
    n = 0
    for c in state.get("charts", []):
        if c["section"] in sections and n < limit:
            doc.img(c)
            n += 1


def build(state: Dict[str, Any], reg) -> tuple:
    d = Doc()
    plan, prof = state["analysis_plan"], state["dataset_profile"]
    pt = plan["problem_type"]
    ins = state.get("insights") or {}
    pm = state.get("primary_metric")

    d.h(1, "Business Analysis Report")
    d.table(["Item", "Value"], [
        ["Experiment", state["experiment_id"]], ["Request", state.get("user_request") or "(none)"],
        ["Business problem (analysis_type)", state.get("analysis_type")], ["Analytical problem (problem_type)", pt],
        ["Target", plan.get("target") or "none"], ["Date column", plan.get("date_column") or "none"],
        ["Dataset", f"{prof['n_rows']:,} rows x {prof['n_cols']} columns"],
        ["Data quality score", f"{prof['quality']['score']}/100 ({prof['quality'].get('label', '')}). {prof['quality'].get('explanation', '')}"],
        ["Random seed", state.get("seed")],
        ["Reasoning engine", "LLM-driven (" + config.LLM_PROVIDER_NAME + ": " + config.LLM_MODEL + ")"]])

    # ---------------------------------------------------------------- 1 executive summary
    d.h(2, "1. Executive summary")
    d.p(f"**{ins.get('headline', '')}**".replace("**", ""))
    d.p(ins.get("executive_summary", ""))
    ci = state.get("metric_ci") or {}
    if ci.get("value") is not None:
        rng = f" (95% CI {ci['lo']}-{ci['hi']})" if ci.get("lo") is not None else ""
        pos = f", {ci['n_test_positives']} positive cases" if ci.get("n_test_positives") is not None else ""
        d.p(f"Model performance (held-out test set, {ci.get('n_test')} rows{pos}): {ci['metric']} = {ci['value']}{rng}. "
            "This is the headline figure; cross-validated scores on training data are shown in the modeling section only for model selection.")
    tr_ = state.get("target_report") or {}
    if tr_.get("spec"):
        d.p(f"Performance target: {'MET' if tr_.get('met_heldout') else 'NOT met'} on the held-out set; search stopped because: {tr_.get('stopped_because')}.")
    if ins.get("unverified_summary"):
        d.p("Note: some figures in the summary could not be matched to the evidence register: " + "; ".join(ins["unverified_summary"][:3]))
    kp = state.get("kpi_results") or {}
    if kp:
        d.table(["KPI", "Value", "Latest period change"], [[n, r["overall_fmt"], (f"{r['time']['change_pct']:+.1f}%" if r.get("time") and r["time"].get("change_pct") is not None else "n/a")] for n, r in kp.items()])
    if state.get("recommendations"):
        d.p("Top recommended actions:")
        d.bullets([f"[{r['priority'].upper()}] {r['action']}" for r in state["recommendations"][:3]])
    if ins.get("caveats"):
        d.p("Caveats:")
        d.bullets(ins["caveats"])

    # ---------------------------------------------------------------- 2 approach
    d.h(2, "2. Business question and approach")
    d.p(state.get("business_question", ""))
    if state.get("hypotheses"):
        d.p("Hypotheses considered:")
        d.bullets(state["hypotheses"])
    d.p("Pipeline: Data Engineer (profile) > Business Analyst (frame problem, KPIs) > Statistician (selected tests) > KPI Analyst > "
        "Trend Analyst > Visualizer > Modeler/Tuner/Critic/Strategist (if predictive) > Root Cause Analyst > Insight Analyst > "
        "Recommender > optional Power BI export > Reporter.")

    # ---------------------------------------------------------------- 3 data
    d.h(2, "3. Dataset profile and data quality")
    q = prof["quality"]
    d.bullets([f"Missing cells: {q['missing_cell_pct']}%; duplicate rows: {q['duplicate_row_pct']}% ({prof['duplicates']} rows)",
               f"Constant columns: {q['constant_columns'] or 'none'}", f"High-cardinality text/categorical: {q['high_cardinality_columns'] or 'none'}",
               f"Columns with >1% IQR outliers: {q['outlier_columns_pct'] or 'none'}"])
    d.table(["Column", "Kind", "Missing %", "Unique", "Mean / top"],
            [[c, i["kind"], i["missing_pct"], i["unique_count"], _n(i.get("mean")) if i["kind"] == "numeric" else (i.get("sample") or [""])[0] if config.LLM_SEND_SAMPLES else ""]
             for c, i in list(prof["columns"].items())[:60]])
    acts = [a for a in plan["actions"]]
    if acts:
        d.h(3, "Cleaning decisions")
        d.table(["Column", "Issue", "Action", "Needs approval", "Reason", "LLM second opinion"],
                [[a["column"], a["issue"], a["action"], "yes" if a["destructive"] else "no", a["reason"], a.get("llm_opinion", "")] for a in acts])
    _charts(d, state, "data_quality", "distributions", "outliers")

    # ---------------------------------------------------------------- 4 statistics
    d.h(2, "4. Statistical analysis")
    d.p("Tests were chosen for fit to the data types and the business question, not run exhaustively. p-values are Benjamini-Hochberg adjusted. "
        "Correlation and association do not imply causation.")
    sr = state.get("stat_results") or []
    d.table(["ID", "Test", "Variables", "Method used", "Statistic", "Adj. p", "Effect", "Significant", "Why selected"],
            [[r["id"], r["test"], ", ".join(f"{v}" for k, v in r["params"].items() if k != "columns") or "numeric features", r.get("method"),
              _n(r.get("statistic")), _n(r.get("p_adjusted", r.get("p_value"))), f"{r.get('effect_name', '')} {_n(r.get('effect_size'), 3)} ({r.get('effect_label', '')})",
              "yes" if r.get("significant") else "no", r.get("selection_reason", "")] for r in sr])
    for r in sr:
        if r.get("interpretation"):
            d.p(f"**{r['id']}** {r['interpretation']}".replace("**", ""))
            if r.get("assumptions"):
                d.bullets([f"Assumption check: {a}" for a in r["assumptions"]])
            if r.get("table"):
                d.table(list(r["table"][0].keys()), [list(x.values()) for x in r["table"]])
    if state.get("stat_skipped"):
        d.h(3, "Tests deliberately not run")
        d.table(["Test", "Reason"], [[s["test"], s["reason"]] for s in state["stat_skipped"]])
    _charts(d, state, "correlations")

    # ---------------------------------------------------------------- 5 KPIs
    d.h(2, "5. Business KPIs")
    for name, r in kp.items():
        sp = r["spec"]
        d.h(3, f"{name}: {r['overall_fmt']}")
        d.p(f"Definition: {sp['agg']}" + (f" of {sp['column']}" if sp.get("column") else "") + f"; {'higher' if sp['higher_is_better'] else 'lower'} is better.")
        for col, rows in r["segments"].items():
            d.table([col, "Value", "n", "Share of rows", "vs overall"], [[x["segment"], x["value_fmt"], f"{x['n']:,}", f"{x['share_of_rows']:.1%}",
                                                                          "n/a" if x["vs_overall_pct"] is None else f"{x['vs_overall_pct']:+.1f}%"] for x in rows])
    _charts(d, state, "kpis", "segments")

    # ---------------------------------------------------------------- 6 trends
    d.h(2, "6. Trends, seasonality and anomalies")
    tr = state.get("trend_results") or {}
    if "skipped" in tr:
        d.p(tr["skipped"])
    else:
        d.bullets([f"Measure: {tr['spec']['name']} ({tr['start']} to {tr['end']}, {tr['n_periods']} periods)",
                   f"Trend: {tr['trend']} (Mann-Kendall tau {tr['mann_kendall_tau']}, p {_n(tr['mann_kendall_p'], 3)}); {tr['slope_pct_of_mean']:+.2f}% of the mean per period",
                   f"Volatility (CV): {tr['volatility_cv']}",
                   (f"Seasonality by {tr['seasonality']['by']}: strength {tr['seasonality']['strength']}" if tr.get("seasonality") else "No seasonality tested (insufficient history)"),
                   (f"Level shift near {tr['change_point']['period']}: {tr['change_point']['shift_vs_trend']:+,.2f} relative to the trend line (detrended test)" if tr.get("change_point") else "No clear level shift")])
        if tr.get("anomalies"):
            d.table(["Period", "Direction", "Value", "Expected", "Robust z"], [[a["period"], a["direction"], a["value"], a["expected"], a["robust_z"]] for a in tr["anomalies"]])
    ol = state.get("outlier_results") or {}
    if ol.get("columns"):
        d.h(3, "Cross-sectional outliers (IQR fences)")
        d.table(["Column", "Outliers", "% of values", "Lower fence", "Upper fence", "Min", "Max"],
                [[o["column"], o["n_outliers"], o["pct"], o["lower_fence"], o["upper_fence"], o["min"], o["max"]] for o in ol["columns"]])
    _charts(d, state, "trends")

    # ---------------------------------------------------------------- 7 root cause
    d.h(2, "7. Root-cause analysis (why is it happening?)")
    d.p("Hypotheses below are grounded in cited evidence (see appendix). They are associations to validate, not proven causes.")
    d.table(["Finding", "Hypothesis", "Confidence", "Evidence", "How to validate"],
            [[r["finding"], r["hypothesis"], r["confidence"], ", ".join(r["evidence_ids"]), r["validation_step"]] for r in state.get("root_causes", [])])
    rt = state.get("rca_tables") or {}
    if rt.get("change"):
        ch = rt["change"]
        d.p(f"Change decomposition ({rt['measure']}): {ch['windows']['base'][0]}..{ch['windows']['base'][1]} vs {ch['windows']['current'][0]}..{ch['windows']['current'][1]}.")
        for col, rows in ch["by_column"].items():
            d.table([col, "Before", "After", "Contribution", "Share of change"], [[x["level"], _n(x["base"]), _n(x["current"]), _n(x["delta"]),
                                                                                    "n/a" if x.get("share_of_change") is None else f"{x['share_of_change']:.0%}"] for x in rows[:4]])

    xc = state.get("external_context") or {}
    if xc:
        d.h(3, "External web context (optional, unverified)")
        if xc.get("status") == "ok":
            d.p("Generic public-context queries were sent to the Tavily search API (no dataset values). Treat the results as background, not proof: " + "; ".join(xc["queries"]))
            d.table(["Source", "Query"], [[x["url"], x["query"]] for x in xc["sources"]])
        else:
            d.p(xc.get("reason") or "No relevant external sources were found.")

    # ---------------------------------------------------------------- 8 modeling
    if pt in ("classification", "regression", "time_series") and state.get("candidate_metrics"):
        d.h(2, f"8. Predictive modeling ({pt})")
        s = state.get("data_summary", {})
        d.bullets([f"Split: {s.get('split')} ({s.get('n_train'):,} train / {s.get('n_test'):,} test), {s.get('n_features')} features",
                   f"Primary metric: {pm} (selection uses cross-validation on the train split only; the test split is scored once)",
                   (f"Time-series setup: lags {s['time_series']['lags']}, rolling windows {s['time_series']['rolls']}; one-step-ahead evaluation" if s.get("time_series") else ""),
                   (f"Class distribution (train): {s.get('class_distribution')}; positive/event class: {s.get('positive_class')}" if pt == "classification" else "")])
        b = state["baseline_metrics"]
        d.p(f"Baseline ({b.get('name')}): CV {pm} {b['cv_score']}; held-out {pm} {_n(b.get(pm))}.")
        cols = ["Model", "Default CV", "Selection score", "Score source", "CV std", "Tuned?"] + [k for k in (pm, "f1_macro", "balanced_accuracy", "accuracy", "rmse", "mae") if k and any(k in c for c in state["candidate_metrics"])]
        extra = cols[6:]
        d.table(cols, [[c["name"] + (" *" if c["name"] == state.get("best_model_name") else ""), c["default_cv"], c["select_score"], c["select_source"], c["cv_std"],
                        "yes" if c["tuned"] else "no"] + [_n(c.get(k)) for k in extra] for c in state["candidate_metrics"]])
        d.p("* = selected. Held-out columns are reporting only.")
        _leaderboard_section(d, state, pm)
        if state.get("attempts"):
            d.h(3, "Attempts and strategy changes")
            d.table(["Attempt", "Levers", "Tuning", "Best", "Selection score", "Held-out", "Passed"],
                    [[a["attempt"], ", ".join(a["levers"]) or "defaults", "on" if a["tuned"] else "off", a["best"], a["select_score"], _n(a["heldout"]), "yes" if a["passed"] else "no"] for a in state["attempts"]])
        au = state.get("audit") or {}
        d.h(3, "Quality gate (Critic)")
        d.p(f"Result: {'PASSED' if au.get('passed') else 'DID NOT PASS'}. Gain over baseline: CV {au.get('gain_cv')}, held-out {au.get('gain_test')}.")
        d.bullets([f"[{i['severity']}] {i['code']}: {i['message']}" for i in au.get("issues", [])])

        # ---------------------------------------------------------------- 9 tuning
        tu = state.get("tuning_report") or {}
        if state.get("automl_report"):
            _automl_section(d, state)
        else:
            d.h(2, "9. Hyper-parameter tuning (detailed)")
        if state.get("automl_report"):
            pass
        elif not tu.get("enabled"):
            d.p("Tuning was disabled for this run.")
        else:
            d.bullets([f"Method: randomized search ({tu['n_iter']} sampled configurations per model) inside nested cross-validation "
                       f"({tu['inner_cv']}-fold inner for the search, {tu['outer_cv']}-fold outer for the unbiased estimate); metric {tu['scoring']}",
                       f"Search ranges: {tu.get('space_source')}",
                       ("LLM rationale: " + tu["advisor_rationale"]) if tu.get("advisor_rationale") else "",
                       "Comparison: the tuned model is adopted only if its NESTED-CV score beats the default model's CV score. "
                       "The search's own best score is optimistic (it is the maximum over many trials) and is shown for reference only. "
                       "Nested folds train on less data, so this comparison is slightly conservative."])
            d.table(["Model", "Default CV", "Search best (optimistic)", "Nested CV mean", "Nested std", "Delta vs default", "Adopted tuned?", "Seconds"],
                    [[m, e.get("default_cv"), e.get("inner_best"), e.get("nested_mean"), e.get("nested_std"), e.get("delta_vs_default"),
                      "yes" if e.get("used_tuned") else "no", e.get("seconds")] for m, e in tu["models"].items() if "error" not in e or e.get("default_cv")])
            for m, e in tu["models"].items():
                if "trials" not in e:
                    d.p(f"{m}: tuning failed ({e.get('error')}).")
                    continue
                d.h(3, f"{m}")
                d.p(f"Best parameters found: {e['best_params']}")
                d.p("Search space: " + "; ".join(f"{k}: {v}" for k, v in e["space"].items()))
                d.p(f"Outer-fold scores: {e['nested_scores']}")
                d.table(["Rank", "Parameters", "CV mean", "CV std", "Fit s"], [[t["rank"], t["params"], t["mean_score"], t["std_score"], t["fit_seconds"]] for t in e["trials"][:config.TUNE_MAX_TRIALS_REPORTED]])
            if tu.get("notes"):
                d.p("Notes: " + "; ".join(tu["notes"]))
        _charts(d, state, "tuning")

        # ---------------------------------------------------------------- 10 explainability
        d.h(2, "10. Model explanation and diagnostics")
        ex = state.get("explanations") or {}
        d.table(["Feature", "Importance", "Std"], [[i["feature"], i["importance"], i["std"]] for i in ex.get("importances", [])[:15]])
        dg = ex.get("diagnostics", {})
        if dg.get("per_class"):
            d.table(["Class", "Precision", "Recall", "Support"], [[c, v["precision"], v["recall"], v["support"]] for c, v in dg["per_class"].items()])
        if dg.get("lift"):
            d.table(["Risk decile", "Rows", "Event rate", "Cumulative capture"], [[x["decile"], x["n"], f"{x['event_rate']:.1%}", f"{x['cum_capture']:.1%}"] for x in dg["lift"]])
        if "residual_std" in dg:
            d.p(f"Residual standard deviation on held-out data: {dg['residual_std']}.")
        _charts(d, state, "modeling", "explain")
        best_c = next((c for c in state["candidate_metrics"] if c["name"] == state.get("best_model_name")), {})
        d.h(3, "Model card")
        d.bullets([f"Model: {state.get('best_model_name')} ({best_c.get('select_source')}); purpose: {state.get('business_question')}",
                   f"Trained on {s.get('n_train'):,} rows, evaluated on {s.get('n_test'):,} held-out rows; metric {pm}: {best_c.get('select_score')}, held-out {_n(best_c.get(pm))}",
                   "Top drivers (predictive, not causal): " + ", ".join(i["feature"] for i in ex.get("importances", [])[:5]),
                   "Known limitations: " + ("; ".join(i["message"] for i in au.get("issues", []) if i["severity"] != "info") or "none flagged by the quality gate"),
                   "Not suitable for: causal conclusions, populations unlike the training data, or decisions without monitoring for drift."])
    elif pt == "clustering" and state.get("cluster_results"):
        cr = state["cluster_results"]
        d.h(2, "8. Segmentation (clustering)")
        d.p(f"{cr['k']} segments chosen by silhouette search (silhouette {cr['silhouette']}, structure: {cr['quality']}).")
        d.table(["k", "Silhouette", "Davies-Bouldin", "Inertia"], [[r["k"], r["silhouette"], r["davies_bouldin"], r["inertia"]] for r in cr["k_scores"]])
        d.table(["Segment", "Name", "Share", "Most distinctive features", "Description"],
                [[p["cluster"], p.get("name"), f"{p['share']:.1%}", ", ".join(f"{f['feature']} ({f['z_vs_overall']:+.1f}sd)" for f in p["defining_features"]), p.get("description", "")] for p in cr["profiles"]])
        _charts(d, state, "tuning", "segments")

    _nlp_section(d, state)
    _vision_section(d, state)
    # ---------------------------------------------------------------- insights & recs
    d.h(2, "11. Business insights (what does it mean?)")
    for k in ins.get("key_insights", []):
        d.p(f"{k['title']}: {k['insight']}" + (f" So what: {k['so_what']}" if k.get("so_what") else "") + f" [evidence: {', '.join(k['evidence_ids'])}]"
            + (f" (UNVERIFIED: {'; '.join(k['unverified'][:2])})" if k.get("unverified") else ""))
    d.h(2, "12. Recommendations (what should we do?)")
    d.table(["Priority", "Action", "Rationale", "Effort", "Expected impact", "KPI to track", "Risk", "Evidence"],
            [[r["priority"], r["action"] + (" (UNVERIFIED figure: " + "; ".join(r["unverified"][:1]) + ")" if r.get("unverified") else ""), r["rationale"], r["effort"], r["expected_impact"], r["kpi_to_track"], r["risk"], ", ".join(r["evidence_ids"])] for r in state.get("recommendations", [])])

    _crew_section(d, state)
    _timeline_section(d, state)
    n = 13
    tk = state.get("tracking")
    if tk:
        d.h(2, f"{n}. Experiment tracking and model registry (MLflow-style)")
        rv = tk.get("registered") or {}
        d.bullets([f"Experiment: {tk['experiment']}; run id {tk['run_id']} with {tk['child_runs']} child run(s) (one per candidate model; tuned models also log every search trial)",
                   (f"Registered model: {rv.get('model_name')} version {rv.get('version')}, stage {rv.get('stage')} (promote in the Experiments tab)" if rv else "No model registered (nothing deployable)"),
                   f"MLflow: {tk.get('mlflow')}"])
        n += 1
    if state.get("powerbi_result"):
        pb = state["powerbi_result"]
        d.h(2, f"{n}. Power BI integration (optional)")
        d.bullets([f"Package: {pb['package_dir']}", f"Files: {', '.join(pb['files'])}", f"DAX measures: {pb['n_measures']}; report layout from {pb['layout_source']}",
                   f"Push to Power BI service: {pb['push']['status']}"])
        n += 1

    # ---------------------------------------------------------------- images / documents
    xs = [e for e in state.get("evidence", []) if e["id"].startswith("X")]
    if xs:
        d.h(2, f"{n}. Images and documents")
        n += 1
        d.bullets([f"{e['text']} [{e['id']}]" for e in xs])
        _charts(d, state, "media")
        if (state.get("media_report") or {}).get("themes"):
            d.table(["Theme", "Documents", "Top terms"], [[t["name"], t["size"], ", ".join(t["terms"][:6])] for t in state["media_report"]["themes"]])

    # ---------------------------------------------------------------- governance, reproducibility, diagnostics
    d.h(2, f"{n}. Governance, reproducibility and run diagnostics")
    n += 1
    gov = []
    if state.get("sensitive_columns"):
        gov.append(f"Sensitive attributes detected: {state['sensitive_columns']} ({'excluded from' if state.get('exclude_sensitive') else 'USED in'} the model).")
    for g_ in (state.get("fairness") or {}).get("groups", []):
        gov.append(f"Fairness screen '{g_['column']}': lowest/highest selection-rate ratio {g_['selection_ratio']}"
                   + (" (below the 0.8 screen)" if g_.get("flag") else "") + f"; recall gap {g_.get('recall_gap')}.")
    tp = state.get("table_plan") or {}
    if tp.get("joins") is not None and tp.get("base"):
        gov.append(f"Multi-table input: base table '{tp['base']}' ({tp.get('rows_base'):,} rows) joined with " + (", ".join(j["table"] for j in tp["joins"]) or "nothing")
                   + f"; {len(tp.get('rejected', []))} candidate join(s) rejected by guards. The plan was approved before it was applied.")
    sup = state.get("supervisor") or {}
    if sup:
        gov.append(f"Phase planning: {sup.get('source')} ({sup.get('rationale')}); skipped phases: {state.get('skip_phases') or 'none'}.")
    ct = state.get("cost_threshold") or {}
    if ct:
        gov.append(f"Decision cutoff {ct['threshold']} chosen from {'your stated' if ct.get('costs_from_user') else 'DEFAULT (replace with real)'} costs "
                   f"(miss = {ct['cost_fn']:g}, false alarm = {ct['cost_fp']:g}) on out-of-fold training predictions.")
    ss = state.get("seed_stability") or {}
    if ss:
        gov.append(f"Stability across {len(ss['seeds'])} fold seeds: {ss['mean']} +/- {ss['sd']} (range {ss['range']}).")
    gov.append(f"Random seed {state.get('seed')}; every LLM call and routing decision is in the run log (runs/{state.get('run_id')}/runlog.jsonl) and can be replayed.")
    d.bullets(gov)
    try:
        from . import runlog as _rl
        sm = _rl.summary(state.get("run_id", ""))
        if sm.get("llm_calls"):
            d.p(f"Run diagnostics: {sm['llm_calls']} LLM call(s) ({sm['failed_calls']} failed), ~{sm['tokens']:,} tokens, {sm['llm_seconds']} s of LLM time"
                + (f", estimated cost ${sm['est_cost_usd']}" if sm.get("est_cost_usd") is not None else "") + ".")
            d.table(["Purpose", "Calls", "Failed", "Seconds", "Tokens"], [[k, int(v["calls"]), int(v["failed"]), v["seconds"], int(v["tokens"])] for k, v in sm["by_purpose"].items()])
            d.p("Routing decisions: " + ", ".join(f"{a}->{b}" for a, b in sm["routes"][:40]))
    except Exception:
        pass
    if state.get("messages"):
        d.p("Agent-to-agent requests (bounded, logged):")
        d.bullets([f"{m['from']} -> {m['to']} [{m['kind']}] {m['topic']}: {str(m['payload'])[:140]}" for m in state["messages"][:12]])

    # ---------------------------------------------------------------- appendix
    d.h(2, f"{n}. Evidence register")
    d.p("Every insight, root cause and recommendation cites these ids.")
    d.table(["ID", "Source", "Evidence"], [[e["id"], e["source"], e["text"]] for e in sorted(state.get("evidence", []), key=lambda e: e["id"])])
    d.h(2, f"{n + 1}. Methodology, guardrails and limitations")
    n_guard = n + 1
    d.bullets(["Leak-free evaluation: preprocessing is fit inside each CV fold; the test split never influences selection or tuning.",
               "Nested CV makes tuned-model scores unbiased; selection never uses the test set.",
               "Statistical tests are selected from a pre-validated menu; assumptions are checked; p-values are BH-corrected.",
               "Shapiro-Wilk is only used inside group comparisons with small groups; K-S drift only with a real hold-out/reference.",
               "LLM outputs are validated: unknown columns, tests, ranges and ungrounded claims are discarded. The LLM sees aggregated metadata and evidence, not raw rows.",
               "Root causes are observational hypotheses; confirm with experiments before committing budget.",
               "Time-series models are evaluated one step ahead using true previous values; multi-step forecasts will be less accurate.",
               "Time-series CV uses few long expanding windows; with short histories treat scores as indicative.",
               "Numbers and wording written by the LLM are checked against the evidence register; unmatched figures are flagged UNVERIFIED and ROI/certainty language is rejected.",
               "Confidence intervals are bootstrap intervals over the held-out set; with few positive cases they are wide and the score is noisy.",
               "Metric targets and decision cutoffs are chosen on out-of-fold training predictions only; the test set is used once for the final report."])
    if state.get("warnings"):
        d.h(3, "Warnings raised during the run")
        d.bullets(state["warnings"][:40])
    calls = state.get("llm_calls", [])
    if calls:
        d.h(3, "LLM usage")
        d.p(f"{len(calls)} call(s), {sum(1 for c in calls if c.get('ok'))} successful, ~{sum(c.get('tokens', 0) for c in calls):,} tokens, model {config.LLM_MODEL}.")
        d.table(["Purpose", "OK", "Seconds", "Tokens"], [[c.get("purpose"), "yes" if c.get("ok") else "no", c.get("seconds"), c.get("tokens", 0)] for c in calls])
    d.h(3, "Agent trace")
    d.table(["Time", "Agent", "Message"], [[e["ts"], e["agent"], e["message"]] for e in state.get("agent_log", [])])
    return d.markdown(state["run_id"]), d.html(f"Business Analysis Report {state['experiment_id']}", reg)


# v5 sections
def _leaderboard_section(d: "Doc", state, pm) -> None:
    lb = state.get("leaderboard") or {}
    if not lb.get("rows"):
        return
    d.h(3, "Model leaderboard: every candidate compared")
    d.p(lb.get("summary", ""))
    d.table(["#", "Model", f"CV {pm} (CV)", "CV std", f"Held-out {pm}", "Train", "Gap", "p vs winner", "Fit s", "Verdict"],
            [[r.get("rank"), r["name"] + (" *" if r.get("selected") else "") + (" (tuned)" if r.get("tuned") else ""), _n(r.get("select_score")), _n(r.get("cv_std")),
              _n((r.get("heldout") or {}).get(pm)), _n(r.get("train_score")), _n(r.get("overfit_gap")), _n(r.get("p_vs_best")) if r.get("p_vs_best") is not None else "-",
              _n(r.get("fit_seconds")), r.get("verdict", "")] for r in lb["rows"]])
    d.p("p vs winner: Nadeau-Bengio corrected paired t-test on the per-fold scores (small p = reliably different; large p = cannot be told apart with this much data). "
        "Gap = train minus held-out score: a large gap means the model memorises.")
    if lb.get("parsimony_differs"):
        d.p(f"Parsimony note (one-standard-error rule): {lb['parsimony_pick']} is the simplest model whose score is within one standard error of the winner. "
            "If you value explainability or speed, it is a defensible alternative.")
    win = next((r for r in lb["rows"] if r.get("selected")), None)
    dg = (win or {}).get("diagnostics") or {}
    if dg.get("per_class"):
        d.h(3, f"Winner: per-class results (held-out)")
        d.table(["Class", "Precision", "Recall", "F1", "Support"], [[c, v["precision"], v["recall"], v["f1"], v["support"]] for c, v in dg["per_class"].items()])
        if dg.get("top_confusions"):
            d.p("Most frequent confusions: " + "; ".join(f"{c['true']} mistaken for {c['predicted']} ({c['count']}x)" for c in dg["top_confusions"][:4]) + ".")
    if lb.get("past_attempts"):
        d.h(3, "Earlier attempts (kept, not discarded)")
        for h in lb["past_attempts"]:
            d.p(f"Attempt {h['attempt']} ({', '.join(h.get('levers') or []) or 'defaults'}):")
            d.table(["Model", "CV score", "CV std", "Tuned?"], [[r["name"], _n(r.get("select_score")), _n(r.get("cv_std")), "yes" if r.get("tuned") else "no"] for r in h["rows"]])
    if state.get("model_card_key"):
        d.p(f"A model card was generated: `{state['model_card_key']}`.")


def _nlp_section(d: "Doc", state) -> None:
    r = state.get("nlp_report") or {}
    if not r:
        return
    d.h(2, "Text (NLP) pipeline")
    c, lg = r.get("corpus") or {}, r.get("language") or {}
    if c:
        d.bullets([f"{c['n_docs']:,} documents, {c['n_tokens']:,} tokens, vocabulary {c['vocab_size']:,}; median {c['tokens_median']:.0f} tokens per document",
                   f"Language: {lg.get('language')} ({lg.get('confidence')}); {c['n_empty']} empty and {c['short_docs']} very short documents"])
    if r.get("class_terms"):
        d.h(3, "Terms that distinguish each class")
        d.table(["Class", "Distinguishing terms (log-odds z)"], [[k, ", ".join(f"{t['term']} ({t['z']})" for t in v[:6])] for k, v in r["class_terms"].items()])
    t = r.get("topics")
    if t:
        d.h(3, f"Topics (NMF, k = {t['k']}, coherence {t['coherence']})")
        d.table(["Topic", "Share", "Top terms", "Dominant label", "Purity"], [[x["topic"], f"{x['share']:.0%}", ", ".join(x["terms"][:6]), x.get("dominant_label", ""), x.get("label_purity", "")] for x in t["topics"]])
    if r.get("label_issues", {}).get("suspects"):
        li = r["label_issues"]
        d.h(3, f"Possible label errors ({li['n_suspects']})")
        d.table(["Row", "Given", "Model suggests", "P(given)", "P(suggested)", "Text"], [[x["index"], x["given"], x["suggested"], x["p_given"], x["p_suggested"], x.get("text", "")[:120]] for x in li["suspects"][:10]])
    if r.get("misclassified"):
        d.h(3, "Most confident mistakes on held-out documents")
        d.table(["True", "Predicted", "Confidence", "Words that drove it", "Text"],
                [[m["true"], m["predicted"], m["confidence"], ", ".join(w["word"] for w in (m.get("word_effects") or {}).get("supporting", [])[:4]), m["text"][:140]] for m in r["misclassified"][:8]])
    if r.get("pii", {}).get("docs_affected"):
        d.p(f"Possible personal data in {r['pii']['docs_affected']} document(s): " + ", ".join(f"{k} x{v}" for k, v in r["pii"]["counts"].items()) + " (counted, never stored).")
    st = r.get("stress_test")
    if st:
        d.h(3, "Text stress test")
        d.table(["Perturbation", st["metric"], "Drop"], [[x["perturbation"], x.get("score", ""), x.get("drop", x.get("error", ""))] for x in st["results"]])
    _charts(d, state, "nlp")


def _vision_section(d: "Doc", state) -> None:
    r = state.get("vision_report") or {}
    if not r:
        return
    d.h(2, "Computer-vision pipeline")
    d.bullets([f"{r.get('n_images', 0):,} images analysed; {r.get('n_features', 0)} hand-crafted features per image (HOG + LBP + HSV + colour/edge statistics)",
               "A pretrained ResNet candidate was " + ("available." if r.get("resnet_available") else "not available (torch is not installed); hand-crafted features capture colour, edge, texture and shape, not objects.")])
    if r.get("label_issues", {}).get("suspects"):
        li = r["label_issues"]
        d.p(f"{li['n_suspects']} image(s) look mislabelled; the most likely: " + ", ".join(f"{x['file']} ({x['given']}→{x['suggested']})" for x in li["suspects"][:5]) + ".")
    rob = r.get("robustness")
    if rob:
        d.h(3, "Robustness stress test")
        d.table(["Degradation", rob["metric"], "Drop"], [[x["perturbation"], x.get("score", ""), x.get("drop", x.get("error", ""))] for x in rob["results"]])
    if r.get("misclassified"):
        d.h(3, "Most confident mistakes")
        d.table(["File", "True", "Predicted", "Confidence"], [[m["file"], m["true"], m["predicted"], m["confidence"]] for m in r["misclassified"][:8]])
    _charts(d, state, "vision")


def _timeline_section(d: "Doc", state) -> None:
    tm = {k: v for k, v in (state.get("timings") or {}).items() if isinstance(v, dict)}
    if not tm:
        return
    top = [(k, v) for k, v in tm.items() if "." not in k]
    total = sum(v.get("seconds", 0) for _, v in top) or 1
    d.h(2, "Run timeline")
    d.table(["Stage", "Seconds", "Share", "Runs", "LLM calls", "Tokens"], [[k, v.get("seconds"), f"{v.get('seconds', 0) / total:.0%}", v.get("runs", ""), v.get("llm_calls", ""), v.get("tokens", "")] for k, v in top])
    if state.get("stage_errors"):
        d.p("Sub-tasks that failed and were skipped (the run continued): " + "; ".join(f"{e['task']} ({e['error'][:80]})" for e in state["stage_errors"]))


def _automl_section(d: "Doc", state) -> None:
    ar = state.get("automl_report") or {}
    gr = ar.get("goal") or {}
    d.h(2, "9. Goal-driven AutoML (what was tried, why, and what it achieved)")
    d.bullets([f"Goal: {gr.get('text', 'best effort')}" + (f": **{gr['status'].replace('_', ' ').upper()}**" if gr.get("status") not in (None, "no_goal") else ""),
               f"Search: {ar.get('n_trials')} trials in {ar.get('seconds')} s over {len(ar.get('rounds', []))} LLM-directed round(s) (attempt {ar.get('attempt')}); {ar.get('workers')} worker(s); stopped: {ar.get('stop_text')}",
               f"Search metric: {ar.get('search_metric')}; " + ("large data: successive halving (25% → 50% → 100% of the rows)" if ar.get("big_data_halving") else "small data: full cross-validation for every trial"),
               "Protocol: " + str(ar.get("protocol")), (f"Quarantined columns (possible leakage): {', '.join(ar['dropped_columns'])}" if ar.get("dropped_columns") else "")])
    if gr.get("status") not in (None, "no_goal"):
        c, h = gr.get("confirmed_on_fresh_folds") or {}, gr.get("heldout") or {}
        d.table(["Check", "Value", "Meets goal?"], [["Search estimate (optimistic)", (gr.get("search_estimate") or {}).get("value"), (gr.get("search_estimate") or {}).get("met")],
                                                    ["Fresh-fold CV (after the search)", c.get("value"), c.get("met")], ["Held-out, scored once", h.get("value"), h.get("met")]])
        if h.get("ci"):
            d.p(f"Held-out 95% bootstrap interval: {h['ci']}.")
        for e in gr.get("evidence") or []:
            d.p("Evidence: " + e)
    if ar.get("rounds"):
        d.h(3, "How the AutoML agent reasoned, round by round")
        for r in ar["rounds"]:
            acts = ", ".join(f"{a['type']}" + (f" ({a['params'].get('recipe') or a['params'].get('family')})" if a.get("params", {}).get("recipe") or a.get("params", {}).get("family") else "")
                             + (f" → {a['effect']['mean_delta']:+.4f}" if (a.get("effect") or {}).get("mean_delta") is not None else "") for a in r.get("actions", [])) or "no valid action"
            d.p(f"**Round {r['round']}** [{r.get('source')}] {r.get('diagnosis', '')} Actions: {acts}. Gain in best CV score: {r.get('gain')}." + (f" Rejected by the guards: {'; '.join(r['rejected'])}." if r.get("rejected") else ""))
    lc = ar.get("learning_curve")
    if lc and lc.get("points"):
        d.p("Learning curve of the incumbent: " + ", ".join(f"{p['fraction']:.0%} → {p['cv']}" for p in lc["points"]) + f". {lc.get('verdict')}.")
    if (ar.get("ceiling") or {}).get("available"):
        d.p(f"Noise ceiling: {ar['ceiling']}")
    if ar.get("ensemble", {}).get("method"):
        e = ar["ensemble"]
        d.p(f"Ensemble ({e['method']}): " + (", ".join(f"{m['name']} ×{m['weight']}" for m in e["members"]) if e["method"] == "greedy" else ", ".join(e["members"])) + f". Out-of-fold score on the search folds {e.get('search_oof_score')}.")
    fam = {}
    for t in ar.get("trials", []):
        if t.get("mean") is not None and t.get("fidelity", 1) >= 0.999:
            f = fam.setdefault(t["family"], {"n": 0, "best": -9, "sec": 0.0})
            f["n"] += 1
            f["best"], f["sec"] = max(f["best"], t["mean"]), f["sec"] + t["seconds"]
    if fam:
        d.table(["Family", "Trials", "Best CV", "Seconds"], [[k, v["n"], v["best"], round(v["sec"], 1)] for k, v in sorted(fam.items(), key=lambda kv: -kv[1]["best"])])


def _crew_section(d: "Doc", state) -> None:
    dec, board = state.get("decisions") or [], state.get("board") or []
    if not dec and not board:
        return
    d.h(2, "The crew: decisions and conversation")
    if dec:
        d.p("Every decision that needed judgement was made by the LLM inside deterministic guards. `source` says whether the LLM answered or the deterministic default was used.")
        d.table(["Junction", "Source", "Decision", "Rationale", "Guard"], [[x.get("junction"), x.get("source"), str(x.get("decision"))[:90], str(x.get("rationale") or x.get("diagnosis") or "")[:200], str(x.get("guard") or "")[:120]] for x in dec])
    if board:
        d.h(3, "Blackboard (what the parallel agents told each other)")
        d.table(["t (s)", "From", "Kind", "Message"], [[m["t"], m["sender"], m["kind"], m["text"][:200]] for m in board])
