"""The analysis result viewer (Insights, Explore, Data, Statistics, KPIs, Charts, Root causes, Models, Actions, Report, Predict, What-if ...) and the experiment registry.

Lifted unchanged from the original single-page app, which had no other dependency on the chat flow; it is now rendered inside the Analyses and Models pages.
"""
import contextlib
import json
import os
import time
import traceback
import uuid
import webbrowser
from pathlib import Path

import pandas as pd

from .. import config, export, media, media_search, mlflow_ui, plotly_viz as pv, runlog, tracking, ui_panels
from ..modeling import predict_with_bundle
from ..storage import get_registry
from ._st import ss, st


class _Registry:
    def __getattr__(self, name):
        return getattr(get_registry(), name)


reg = _Registry()


def log_error(where, e):
    """Append the full traceback to <artifacts>/app_errors.log and return the text."""
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    try:
        p = Path(config.artifacts_dir()) / "app_errors.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {where}\n{tb}")
    except OSError:
        pass
    return tb


@contextlib.contextmanager
def guarded(where):
    """Show a readable error panel (what failed, which key, full traceback) instead of a bare crash. Streamlit's own rerun / stop signals pass through."""
    try:
        yield
    except Exception as e:
        if any(w in type(e).__name__.lower() for w in ("rerun", "stop", "script")):
            raise
        tb = log_error(where, e)
        what = f"missing key {e.args[0]!r}" if isinstance(e, KeyError) and e.args else f"{type(e).__name__}: {e}"
        st.error(f"Something went wrong in **{where}** ({what}). Your data and chats are safe.")
        with st.expander("Technical details (copy this if you ask for help)"):
            st.code(tb)
            st.caption(f"Also saved to {Path(config.artifacts_dir()) / 'app_errors.log'}")


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def chart(c, where: str):
    """Render one chart: interactive Plotly when available, else the PNG."""
    if c.get("dup"):
        return
    key = f"{where}-{c['id']}-{ss.uid}"
    if c.get("spec_key"):
        try:
            spec = json.loads(reg.path_of(c["spec_key"]).read_text(encoding="utf-8"))
            st.plotly_chart(spec, use_container_width=True, key=key, config={"displaylogo": False}, theme=None)
            st.caption(c["caption"])
            return
        except Exception as e:
            st.caption(f"(interactive view unavailable: {type(e).__name__}; showing static image)")
    if c.get("key"):
        st.image(str(reg.path_of(c["key"])), caption=f"{c['title']}: {c['caption']}", use_container_width=True)


def charts(values, sections, where, pick=False):
    got = [c for c in values.get("charts", []) if c["section"] in sections and not c.get("dup")]
    if not got:
        st.caption("No charts for this section.")
        return
    if pick and len(got) > 1:
        titles = [c["title"] for c in got]
        sel = st.selectbox("Chart", titles, key=f"{where}-pick-{ss.uid}")
        chart(got[titles.index(sel)], where)
    else:
        for c in got:
            chart(c, where)


def explore_tab(values, kp):
    df = load_df(values["dataset_path"])
    cols = list(df.columns)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    opt = ["(none)"] + cols
    pick = lambda v: None if v == "(none)" else v
    st.caption("Build your own chart: pick fields, aggregate, colour and slice with filters. Hover, zoom and pan on the chart.")
    c1, c2, c3 = st.columns(3)
    ctype = c1.selectbox("Chart type", pv.EXPLORE_TYPES, key=f"{kp}-ex-type")
    x = pick(c2.selectbox("X / category", opt, index=1 if len(opt) > 1 else 0, key=f"{kp}-ex-x"))
    y = pick(c3.selectbox("Y / measure", opt, key=f"{kp}-ex-y"))
    c4, c5, c6 = st.columns(3)
    color = pick(c4.selectbox("Colour / split", opt, key=f"{kp}-ex-c"))
    agg = c5.selectbox("Aggregation", pv.AGGS, key=f"{kp}-ex-agg")
    top = c6.slider("Top N categories", 5, 40, 20, key=f"{kp}-ex-top")
    filters = {}
    with st.expander("Filters (slicers)"):
        for col in st.multiselect("Filter on columns", cols, key=f"{kp}-ex-fcols"):
            if col in num and df[col].nunique() > 8:
                lo, hi = float(df[col].min()), float(df[col].max())
                r = st.slider(col, lo, hi, (lo, hi), key=f"{kp}-ex-f-{col}")
                filters[col] = ("range", r[0], r[1])
            else:
                vals = st.multiselect(col, sorted(df[col].dropna().astype(str).unique())[:200], key=f"{kp}-ex-f-{col}")
                filters[col] = ("in", vals)
    spec = {"type": ctype, "x": x, "y": y, "color": color, "agg": agg, "top_n": top, "filters": filters}
    fig = pv.explore(df, spec)
    st.plotly_chart(fig, use_container_width=True, key=f"{kp}-ex-fig", config={"displaylogo": False}, theme=None)
    sub = pv.apply_filters(df, filters)
    st.caption(f"{len(sub):,} of {len(df):,} rows after filters")
    st.download_button("Download filtered rows (CSV)", sub.to_csv(index=False), file_name="filtered.csv", key=f"{kp}-ex-dl")


def models_tab(values, kp):
    pt = values["analysis_plan"]["problem_type"]
    if pt == "clustering":
        cr = values["cluster_results"]
        st.write(f"**{cr['k']} segments** · silhouette {cr['silhouette']} ({cr['quality']} structure)")
        st.dataframe(pd.DataFrame([{k: v for k, v in p.items() if k not in ("numeric_means", "defining_features")} for p in cr["profiles"]]), use_container_width=True)
        charts(values, ["tuning", "segments"], f"{kp}-cl", pick=True)
        return
    if not values.get("candidate_metrics"):
        st.info("Descriptive analysis: no predictive model was needed.")
        return
    pm = values["primary_metric"]
    best = next(c for c in values["candidate_metrics"] if c["name"] == values["best_model_name"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best model", values["best_model_name"])
    m2.metric(f"CV {pm}", best["select_score"], delta=f"{values['audit']['gain_cv']:+.3f} vs baseline")
    m3.metric(f"Held-out {pm}", best.get(pm))
    m4.metric("Quality gate", "PASSED" if values["audit"]["passed"] else "NOT PASSED")
    if values.get("leaderboard", {}).get("rows"):
        st.subheader("Model leaderboard: every candidate, every attempt")
        ui_panels.leaderboard_panel(st, values, kp)
    else:
        st.dataframe(pd.DataFrame([{k: v for k, v in c.items() if k not in ("tuning", "cv_scores")} for c in values["candidate_metrics"]]), use_container_width=True)
    for i in values["audit"]["issues"]:
        st.write(f"**{i['severity']}** · `{i['code']}`: {i['message']}")
    tu = values.get("tuning_report", {})
    if tu.get("engine") == "automl":
        st.caption(f"Hyper-parameters and features were searched automatically ({tu.get('n_iter')} trials): see the **Model search** tab for every trial and the reasoning behind it.")
    elif tu.get("enabled"):
        st.subheader("Hyper-parameter tuning")
        st.caption(f"Randomized search ({tu['n_iter']} iterations) inside nested CV · {tu.get('space_source')}")
        if tu.get("advisor_rationale"):
            st.info("Reasoning: " + tu["advisor_rationale"])
        for m, e in tu["models"].items():
            with st.expander(f"{m}: default {e.get('default_cv')} → nested {e.get('nested_mean')} ({'tuned adopted' if e.get('used_tuned') else 'default kept'})"):
                st.write("Best parameters:", e.get("best_params"))
                st.write("Search space:", e.get("space"))
                st.dataframe(pd.DataFrame([{"rank": t["rank"], **t["params"], "cv_mean": t["mean_score"], "cv_std": t["std_score"]} for t in e.get("trials", [])]), use_container_width=True)
    tt = (values.get("explanations") or {}).get("diagnostics", {}).get("threshold_table")
    if tt:
        st.subheader("Decision threshold and business cost")
        st.caption("Set what a missed case and a false alarm cost you; the chart finds the cheapest probability threshold on the held-out data.")
        a, b = st.columns(2)
        c_fn = a.number_input("Cost of a missed case (false negative)", min_value=0.0, value=5.0, step=1.0, key=f"{kp}-cfn")
        c_fp = b.number_input("Cost of a false alarm (false positive)", min_value=0.0, value=1.0, step=1.0, key=f"{kp}-cfp")
        st.plotly_chart(pv.cost_curve(tt, c_fp, c_fn), use_container_width=True, key=f"{kp}-cost", config={"displaylogo": False}, theme=None)
        costs = [r["fp"] * c_fp + r["fn"] * c_fn for r in tt]
        r = tt[min(range(len(costs)), key=costs.__getitem__)]
        st.success(f"Best threshold **{r['threshold']:.2f}**: precision {r['precision']:.2f}, recall {r['recall']:.2f}, {r['fn']} missed, {r['fp']} false alarms.")
    st.subheader("Charts and diagnostics")
    charts(values, ["modeling", "tuning", "explain"], f"{kp}-md", pick=True)


def result_view(values, kp):
    ins, pt = values.get("insights", {}), values["analysis_plan"]["problem_type"]
    st.markdown(f"### {ins.get('headline') or 'Analysis complete'}")
    st.caption(f"`{values['analysis_type']}` · `{pt}` · {values['dataset_profile']['n_rows']:,} rows · experiment `{values['experiment_id']}`")
    kpis = list((values.get("kpi_results") or {}).items())[:4]
    if kpis:
        cols = st.columns(len(kpis))
        for col, (n, r) in zip(cols, kpis):
            d = r.get("time", {}).get("change_pct") if r.get("time") else None
            col.metric(n, r["overall_fmt"], delta=None if d is None else f"{d:+.1f}% vs prev. period")
    st.write(ins.get("executive_summary", ""))
    _mods = values.get("modality") or {}
    _has_files = bool(_mods.get("image") or _mods.get("text"))
    _extra = (["Text analysis"] if values.get("nlp_report") else []) + (["Image analysis"] if values.get("vision_report") else []) + (["Model search"] if values.get("automl_report") else []) + (["Activity"] if (values.get("decisions") or values.get("board")) else []) + (["Timing"] if values.get("timings") else [])
    tabs = st.tabs(["Insights", "Explore", "Data", "Statistics", "KPIs & trends", "Charts", "Root causes", "Models", "Actions", "Report", "Predict", "What-if"]
                   + (["Files"] if _has_files else []) + _extra)
    _xt = dict(zip(_extra, tabs[12 + (1 if _has_files else 0):]))
    if "Text analysis" in _xt:
        with _xt["Text analysis"]:
            with guarded("Text analysis tab"):
                ui_panels.nlp_panel(st, values, kp, charts, lambda: reg.load(values["model_key"]), ss)
    if "Image analysis" in _xt:
        with _xt["Image analysis"]:
            with guarded("Image analysis tab"):
                ui_panels.vision_panel(st, values, kp, charts, lambda: reg.load(values["model_key"]), None, reg.path_of("uploads_try"))
    if "Model search" in _xt:
        with _xt["Model search"]:
            with guarded("Model search tab"):
                ui_panels.automl_panel(st, values, kp)
    if "Activity" in _xt:
        with _xt["Activity"]:
            with guarded("Activity tab"):
                ui_panels.crew_panel(st, values, kp)
    if "Timing" in _xt:
        with _xt["Timing"]:
            with guarded("Timing tab"):
                ui_panels.timeline_panel(st, values, kp)
    if _has_files:
        with tabs[12]:
            try:
                dfm = pd.read_csv(values["dataset_path"])
                root_ = Path(values["analysis_plan"].get("image_root") or Path(values["dataset_path"]).parent)
            except Exception as ex:
                dfm, root_ = None, None
                st.warning(f"The collection could not be reopened: {ex}")
            if dfm is not None and "text" in dfm.columns and _mods.get("text"):
                st.markdown("**Search inside the documents**")
                if "_doc_idx" not in ss or ss._doc_idx[0] != values["dataset_path"]:
                    try:
                        ss._doc_idx = (values["dataset_path"], media_search.DocIndex(dfm))
                    except ValueError as ex:
                        ss._doc_idx = (values["dataset_path"], None)
                        st.info(str(ex))
                ix_ = ss._doc_idx[1]
                q_ = st.text_input("Search words or a question", key=f"{kp}-fq")
                hits_ = ix_.search(q_, 5) if (ix_ is not None and q_) else []
                if q_ and ix_ is not None and not hits_:
                    st.caption("No passage matches those words.")
                for h_ in hits_:
                    with st.expander(f"{h_['document']}  ·  match {h_['score']}" + (f"  ·  {h_['label']}" if h_.get("label") else "")):
                        st.write(h_["snippet"])
                if hits_:
                    ok_ = st.checkbox("Let the assistant read these retrieved passages to answer my question (the passages are sent to the language-model provider)", key=f"{kp}-fconsent")
                    if ok_ and st.button("Answer from these passages", key=f"{kp}-fans"):
                        try:
                            r_ = media_search.answer(values, q_, hits_)
                            st.write(r_["answer"])
                            st.caption("Sources: " + (", ".join(r_["sources"]) or "none cited"))
                        except Exception as ex:
                            st.warning(f"I could not get a grounded answer: {ex}")
                if ix_ is not None:
                    names_ = list(dfm.get("document", dfm.index.astype(str)))
                    pick_ = st.selectbox("Find documents similar to...", names_, key=f"{kp}-fsim")
                    if pick_:
                        sim_ = ix_.similar_to_text(str(dfm.loc[names_.index(pick_), "text"]), 5, exclude=pick_)
                        st.dataframe(pd.DataFrame(sim_), use_container_width=True)
            if dfm is not None and "image" in dfm.columns and _mods.get("image"):
                st.markdown("**Find similar images**")
                up_ = st.file_uploader("Upload an image to find the closest ones in this collection", type=sorted(e.lstrip(".") for e in media.IMAGE_EXT), key=f"{kp}-fimg")
                if up_:
                    qp_ = reg.path_of(f"predict_uploads/{uuid.uuid4().hex[:8]}_{os.path.basename(up_.name)}")
                    qp_.parent.mkdir(parents=True, exist_ok=True)
                    qp_.write_bytes(up_.getvalue())
                    try:
                        res_ = media_search.image_similarity(dfm, "image", root_, qp_, 6)
                        st.dataframe(pd.DataFrame(res_), use_container_width=True)
                        for r_ in res_[:4]:
                            st.image(str(root_ / r_["image"]), caption=f"{r_['image']} ({r_['similarity']})", width=140)
                    except Exception as ex:
                        st.warning(str(ex))
    with tabs[0]:
        for k in ins.get("key_insights", []):
            st.markdown(f"**{k['title']}**: {k['insight']}  \n*So what:* {k.get('so_what', '')}  `{', '.join(k['evidence_ids'])}`")
        for c in ins.get("caveats", []):
            st.caption("Note: " + c)
        if values.get("warnings"):
            with st.expander(f"{len(values['warnings'])} warnings from the run"):
                for w in values["warnings"]:
                    st.write("- " + w)
    with tabs[1]:
        explore_tab(values, kp)
    with tabs[2]:
        q = values["dataset_profile"]["quality"]
        a, b, c = st.columns(3)
        a.metric("Quality score", f"{q['score']}/100")
        b.metric("Missing cells", f"{q['missing_cell_pct']}%")
        c.metric("Duplicate rows", f"{q['duplicate_row_pct']}%")
        st.dataframe(pd.DataFrame(values["dataset_profile"]["columns"]).T[["kind", "missing_pct", "unique_count"]], use_container_width=True)
        st.dataframe(pd.DataFrame(values["analysis_plan"]["actions"]), use_container_width=True)
        charts(values, ["data_quality", "distributions", "outliers"], f"{kp}-dq", pick=True)
    with tabs[3]:
        st.caption("Tests were selected for fit to the data and the question; p-values are Benjamini-Hochberg adjusted. Association is not causation.")
        sr = values.get("stat_results", [])
        st.dataframe(pd.DataFrame([{"id": r["id"], "test": r["test"], "method": r.get("method"), "adj_p": r.get("p_adjusted", r.get("p_value")),
                                    "effect": f"{r.get('effect_name', '')} {r.get('effect_size')}", "significant": r.get("significant"),
                                    "why": r.get("selection_reason"), "interpretation": r.get("interpretation")} for r in sr]), use_container_width=True)
        with st.expander("Tests deliberately not run"):
            st.dataframe(pd.DataFrame(values.get("stat_skipped", [])), use_container_width=True)
        charts(values, ["correlations"], f"{kp}-st", pick=True)
    with tabs[4]:
        for name, r in (values.get("kpi_results") or {}).items():
            for col, rows in r["segments"].items():
                with st.expander(f"{name} by {col}"):
                    st.dataframe(pd.DataFrame(rows).rename(columns={"segment": col}), use_container_width=True)
        tr = values.get("trend_results", {})
        if tr and "skipped" not in tr:
            st.write(f"**{tr['spec']['name']}**: {tr['trend']} · volatility CV {tr['volatility_cv']} · {tr['n_anomalies']} anomalies")
            if tr.get("anomalies"):
                st.dataframe(pd.DataFrame(tr["anomalies"]), use_container_width=True)
        charts(values, ["kpis", "trends", "segments"], f"{kp}-kp", pick=True)
    with tabs[5]:
        secs = sorted({c["section"] for c in values.get("charts", []) if not c.get("dup")})
        sec = st.selectbox("Section", secs, key=f"{kp}-sec") if secs else None
        if sec:
            for c in [c for c in values["charts"] if c["section"] == sec]:
                chart(c, f"{kp}-all")
    with tabs[6]:
        st.caption("Hypotheses grounded in cited evidence: associations to validate, not proven causes.")
        for r in values.get("root_causes", []):
            st.markdown(f"**{r['finding']}**  \n*Hypothesis:* {r['hypothesis']}  \n*Confidence:* {r['confidence']} · *Evidence:* `{', '.join(r['evidence_ids'])}`  \n*Validate by:* {r['validation_step']}")
            st.divider()
        xc = values.get("external_context") or {}
        if xc.get("status") == "ok":
            st.markdown("**External web context (unverified)**")
            for s_ in xc["sources"]:
                st.markdown(f"- [{s_['url']}]({s_['url']})  \n  _query: {s_['query']}_")
        with st.expander("Evidence register"):
            st.dataframe(pd.DataFrame(values.get("evidence", []))[["id", "source", "text"]], use_container_width=True)
    with tabs[7]:
        models_tab(values, kp)
    with tabs[8]:
        st.dataframe(pd.DataFrame(values.get("recommendations", [])), use_container_width=True)
        if values.get("powerbi_result"):
            pb = values["powerbi_result"]
            st.success(f"Power BI package: `{pb['package_dir']}` · push status: {pb['push']['status']}")
    with tabs[9]:
        if values.get("report_html_key"):
            st.download_button("Download report (HTML, interactive charts)", reg.path_of(values["report_html_key"]).read_bytes(), file_name="business_report.html", mime="text/html", key=f"{kp}-dl1")
            st.download_button("Download report (Markdown)", values["report_markdown"], file_name="business_report.md", key=f"{kp}-dl2")
        if values.get("exec_summary_key"):
            st.download_button("Download executive brief (1 page, HTML)", reg.path_of(values["exec_summary_key"]).read_bytes(), file_name="executive_brief.html", mime="text/html", key=f"{kp}-dlx")
        if values.get("repro_key"):
            st.download_button("Download reproducibility kit (ZIP)", reg.path_of(values["repro_key"]).read_bytes(), file_name="reproducibility_kit.zip", mime="application/zip", key=f"{kp}-dlr")
        _ci = values.get("metric_ci") or {}
        if _ci.get("value") is not None:
            st.info(f"Held-out {_ci['metric']}: {_ci['value']}" + (f" (95% CI {_ci['lo']}-{_ci['hi']})" if _ci.get("lo") is not None else ""))
        try:
            _sm = runlog.summary(values.get("run_id", ""))
            if _sm.get("llm_calls"):
                with st.expander("Run details (model calls, time, usage)"):
                    st.write(f"{_sm['llm_calls']} model calls ({_sm['failed_calls']} failed), ~{_sm['tokens']:,} tokens, {_sm['llm_seconds']} s"
                             + (f", est. cost ${_sm['est_cost_usd']}" if _sm.get("est_cost_usd") is not None else ""))
                    st.dataframe(pd.DataFrame([{"purpose": k, **v} for k, v in _sm["by_purpose"].items()]))
        except Exception:
            pass
        if values.get("model_key"):
            pk = f"pkg-{values['experiment_id']}"
            if st.button("Prepare deployable package (ZIP)", key=f"{kp}-pkgbtn"):
                try:
                    ss[pk] = export.build_package(values)
                except Exception as ex:
                    st.error(f"Could not build the package: {ex}")
            if ss.get(pk):
                st.download_button("Download deployable package", ss[pk], file_name=f"{values['experiment_id']}_package.zip", mime="application/zip", key=f"{kp}-pkgdl")
                st.caption("Contains the model, predict.py, a prediction web app, train.py to retrain, Dockerfile and README.")
            st.download_button("Download model bundle", reg.path_of(values["model_key"]).read_bytes(), file_name="model_bundle.joblib", key=f"{kp}-dl3")
            st.caption("The model file is a pickle: it can run code when loaded. A .sha256 file is stored beside it; only load files you created or can verify.")
        st.caption("The HTML report loads plotly.js from a CDN for interactive charts (needs internet when opened).")
    with tabs[10]:
        if not values.get("model_key"):
            st.info("No deployable model for this analysis type.")
        else:
            _m = values.get("modality") or {}
            if _m.get("image") or _m.get("text"):
                st.markdown("**Score new images or documents**")
                nf = st.file_uploader("Upload files to classify / assign to a group", type=[t for t in media.UPLOAD_TYPES if t not in ("csv", "xlsx", "xls", "zip")],
                                      accept_multiple_files=True, key=f"{kp}-media")
                if nf:
                    bundle = reg.load(values["model_key"])
                    rows_, names_, bad_ = [], [], []
                    for f_ in nf:
                        p_ = reg.path_of(f"predict_uploads/{uuid.uuid4().hex[:8]}_{os.path.basename(f_.name)}")
                        p_.parent.mkdir(parents=True, exist_ok=True)
                        p_.write_bytes(f_.getvalue())
                        try:
                            rows_.append(media.row_for_file(p_, list(bundle["feature_columns"]), p_.parent))
                            names_.append(f_.name)
                        except Exception as ex:
                            bad_.append(f"{f_.name}: {ex}")
                    if rows_:
                        out_ = predict_with_bundle(bundle, pd.DataFrame(rows_))
                        st.dataframe(pd.concat([pd.DataFrame({"file": names_}), out_.reset_index(drop=True)], axis=1), use_container_width=True)
                    for b_ in bad_:
                        st.warning(b_)
            new = st.file_uploader("CSV to score", type=["csv"], key=f"{kp}-score")
            if new is not None:
                try:
                    data = pd.read_csv(new)
                    out = predict_with_bundle(reg.load(values["model_key"]), data)
                    res = pd.concat([data.reset_index(drop=True), out.reset_index(drop=True)], axis=1) if pt != "time_series" else out
                    st.dataframe(res.head(200), use_container_width=True)
                    st.download_button("Download predictions", res.to_csv(index=False), file_name="predictions.csv", key=f"{kp}-dlp")
                except Exception as ex:
                    st.error(f"Could not score this file: {ex}")


    with tabs[11]:
        if not values.get("model_key") or pt == "time_series":
            st.info("The what-if form needs a deployable model for a non-time-series problem.")
        else:
            st.caption("Change one record's values and see what the model predicts: a quick way to test scenarios.")
            schema = export.schema_for(values)
            row = {}
            with st.form(f"{kp}-wi"):
                cols = st.columns(2)
                for j, f in enumerate(schema):
                    with cols[j % 2]:
                        k_ = f"{kp}-wi-{f['name']}"
                        if f["kind"] == "numeric":
                            row[f["name"]] = st.number_input(f["name"], value=float(f["default"]), step=1.0 if f["integer"] else 0.01, key=k_)
                        elif f["kind"] == "categorical":
                            row[f["name"]] = st.selectbox(f["name"], f["choices"], index=f["choices"].index(f["default"]) if f["default"] in f["choices"] else 0, key=k_)
                        elif f["kind"] in ("text", "image_path"):
                            row[f["name"]] = st.text_input(f["name"] + (" (image file path)" if f["kind"] == "image_path" else " (text)"), value=f["default"], key=k_)
                        else:
                            row[f["name"]] = str(st.date_input(f["name"], value=pd.to_datetime(f["default"]).date(), key=k_))
                go = st.form_submit_button("Predict")
            if go:
                out = predict_with_bundle(reg.load(values["model_key"]), pd.DataFrame([row])).iloc[0]
                st.success(f"Prediction: **{out.get('prediction', out.get('cluster'))}**")
                proba = {str(k)[6:]: float(v) for k, v in out.items() if str(k).startswith("proba_")}
                if proba:
                    st.plotly_chart(pv.bars(list(proba), list(proba.values()), "Class probabilities", "probability"), use_container_width=True, key=f"{kp}-wi-proba", theme=None)


def mlflow_controls(where):
    """One-click MLflow: start (or reuse) the MLflow web interface in the background and open it in the browser. No terminal needed."""
    ml = tracking.mlflow_status()
    if not ml["active"]:
        st.caption("MLflow is not available: " + ml["reason"])
        return
    if st.button("Open MLflow", use_container_width=True, key=f"mlflow_open_{where}"):
        with st.spinner("Starting MLflow (the first start takes a few seconds)..."):
            res = mlflow_ui.start()
        ss.mlflow_res = res
        if res["ok"] and os.environ.get("NEXUSML_OPEN_BROWSER", "1") == "1":
            try:
                webbrowser.open(res["url"])                         # opens on THIS machine; the link below works from any browser that can reach it
            except Exception:
                pass
        st.rerun()
    s = mlflow_ui.status()
    if s["running"]:
        st.link_button(f"Open MLflow  ·  {s['url']}", s["url"], use_container_width=True)
        if s["ours"] and st.button("Stop MLflow", key=f"mlflow_stop_{where}", use_container_width=True):
            mlflow_ui.stop()
            ss.pop("mlflow_res", None)
            st.rerun()
    elif ss.get("mlflow_res") and not ss.mlflow_res.get("ok"):
        st.error(ss.mlflow_res["message"])


def _send_run_to_mlflow(run):
    """Re-log one run (with the best model and, when available, every compared model) to MLflow."""
    mk = (run.get("artifacts") or {}).get("model")
    cands = None
    try:
        meta = next((r for r in reg.list_runs() if r.get("experiment_id") == run.get("name") and r.get("state_key")), None)
        if meta:
            mk_ = json.loads(reg.path_of(meta["state_key"]).read_text(encoding="utf-8")).get("models_key")
            cands = reg.load(mk_).get("models") if mk_ else None
    except Exception as e:
        log_error("loading the compared models", e)
    return tracking.mirror_mlflow(run["run_id"], reg.path_of(mk) if mk else None, tracking.registered_name_for(run["run_id"]), candidates=cands)


def experiments_view():
    st.markdown("## Experiments and model registry")
    _ml = tracking.mlflow_status()
    if _ml["active"]:
        st.success(f"MLflow {_ml['version']} is installed and is the store of record for new runs.")
        st.caption("This page shows the local index of those runs. MLflow's own interface opens with one click:")
        mlflow_controls("page")
        with st.expander("Prefer the terminal?"):
            st.code(f'mlflow ui --backend-store-uri "{_ml["uri"]}"')
    else:
        st.warning(f"MLflow is not being used: {_ml['reason']}.")
    if _ml["active"]:
        with st.expander("What is inside MLflow right now"):
            inv = tracking.mlflow_inventory()
            if inv.get("error"):
                st.error(inv["error"])
            else:
                st.markdown("**Registered models** (Models page in MLflow)")
                st.dataframe(pd.DataFrame([{"model": m["name"], "version": v["version"], "status": v["status"], "aliases": ", ".join(v["aliases"])}
                                           for m in inv.get("registered", []) for v in m["versions"]] or [{"model": "(none yet)"}]), use_container_width=True)
                st.markdown("**Experiments** (runs and the models logged in each)")
                for e_ in inv.get("experiments", []):
                    ms = e_["models"]
                    st.write(f"**{e_['name']}**: {e_['runs']} run(s); " + ("models: " + ", ".join(f"{m['name']} ({m['status']})" for m in ms) if ms else "no logged models"))
    exps = tracking.experiments()
    if not exps:
        st.info("No runs yet. Every analysis is logged here automatically (parameters, metrics, per-model child runs, search trials, artifacts).")
        return
    t1, t2, t3 = st.tabs(["Runs", "Model registry", "All candidates"])
    with t3:
        with guarded("candidates page"):
            ui_panels.candidates_page(st, tracking)
    with t1:
        exp = st.selectbox("Experiment", ["(all)"] + exps)
        rows = tracking.runs_table(None if exp == "(all)" else exp)
        if not rows:
            st.info("No runs for this experiment yet.")
            return
        ev = st.dataframe(pd.DataFrame(rows).drop(columns=["run_id"], errors="ignore"), on_select="rerun", selection_mode="multi-row", use_container_width=True, key="runs_tbl")
        sel = [rows[i] for i in (ev.selection.rows if ev and ev.selection else [])]
        if len(sel) >= 2:
            runs = [r_ for r_ in (tracking.get_run(r["run_id"]) for r in sel) if r_]
            st.subheader("Compare runs")
            keys = sorted({k for r in runs for k in r["metrics"]})
            mk = st.selectbox("Metric", keys, index=keys.index("best_select_score") if "best_select_score" in keys else 0)
            st.plotly_chart(pv.bars([r["name"][-14:] for r in runs], [r["metrics"].get(mk) for r in runs], f"{mk} by run", mk), use_container_width=True, key="cmp_bar", theme=None)
            st.markdown("**Parameters that differ**")
            st.dataframe(pd.DataFrame(tracking.param_diff(runs)), use_container_width=True)
        elif len(sel) == 1:
            run = tracking.get_run(sel[0]["run_id"])
            if not run:
                st.warning("That run's file could not be found.")
                return
            st.subheader(f"Run {run.get('name', '')}")
            if run.get("mlflow_run_id"):
                st.success(f"This run is in MLflow (run id {run['mlflow_run_id']})." + (f" It contains: {run['mlflow_summary']}." if run.get("mlflow_summary") else ""))
                if run.get("mlflow_notes"):
                    st.warning(run["mlflow_notes"])
                if st.button("Log this run to MLflow again (creates a new MLflow run with its models)", key=f"mlagain-{run['run_id']}"):
                    res_ = _send_run_to_mlflow(run)
                    (st.success if res_["ok"] else st.error)(res_["message"])
            else:
                if run.get("mlflow_error"):
                    st.error(run["mlflow_error"])
                elif tracking.mlflow_status()["active"]:
                    st.warning("This run is not in MLflow yet (MLflow may not have been installed when it ran).")
                else:
                    st.info("MLflow is not active, so this run is only in the local index.")
                if tracking.mlflow_status()["active"] and st.button("Send this run to MLflow now", key=f"mlretry-{run['run_id']}"):
                    res_ = _send_run_to_mlflow(run)
                    (st.success if res_["ok"] else st.error)(res_["message"])
                    st.rerun() if res_["ok"] else None
            a, b = st.columns(2)
            a.markdown("**Parameters**")
            a.dataframe(pd.DataFrame(run["params"].items(), columns=["param", "value"]).astype(str), use_container_width=True)
            b.markdown("**Metrics**")
            b.dataframe(pd.DataFrame(run["metrics"].items(), columns=["metric", "value"]), use_container_width=True)
            st.markdown("**Tags**: " + ", ".join(f"`{k}`={v}" for k, v in run["tags"].items()))
            kids = tracking.list_runs(parent=run["run_id"])
            if kids:
                st.markdown("**Candidate / child runs**")
                st.dataframe(pd.DataFrame([{"run": k["name"], **{m: v for m, v in k["metrics"].items() if m in ("cv_mean", "cv_std", "select_score", "roc_auc", "r2", "f1_macro", "silhouette")}, "tuned": k["params"].get("tuned")} for k in kids]), use_container_width=True)
                hist = [k for k in kids if k["history"].get("cv_fold_score")]
                if hist:
                    st.plotly_chart(pv.fig([{"type": "box", "y": k["history"]["cv_fold_score"], "name": k["name"], "boxpoints": "all"} for k in hist], pv._layout("Cross-validation fold scores by model")), use_container_width=True, key="cv_box", theme=None)
                tm = tracking.trial_models(run["run_id"])
                par = pv.parallel(tm)
                if par:
                    st.plotly_chart(par, use_container_width=True, key="par_runs", theme=None)
            for label, key in run["artifacts"].items():
                p = reg.path_of(key)
                if p.exists():
                    st.download_button(f"Download {label}", p.read_bytes(), file_name=p.name, key=f"art-{label}")
        else:
            st.caption("Select one run to inspect it, or several to compare them.")
    with t2:
        models = tracking.registry()["models"]
        if not models:
            st.info("No registered models yet.")
        for name, m in models.items():
            st.markdown(f"#### {name}")
            st.dataframe(pd.DataFrame([{"version": v.get("version"), "stage": v.get("stage", "None"), "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(v.get("created", 0))),
                                        "description": v.get("description", ""), **{k: x for k, x in (v.get("metrics") or {}).items() if k in ("select_score", "roc_auc", "r2", "f1_macro")}} for v in m.get("versions", [])]), use_container_width=True)
            c1, c2, c3 = st.columns([1, 1, 1])
            if not m.get("versions"):
                continue
            ver = c1.selectbox("Version", [v.get("version") for v in m["versions"]], key=f"ver-{name}")
            _cur_stage = next((v.get("stage", "None") for v in m["versions"] if v.get("version") == ver), "None")
            stage = c2.selectbox("Stage", tracking.STAGES, index=tracking.STAGES.index(_cur_stage) if _cur_stage in tracking.STAGES else 0, key=f"stg-{name}")
            if c3.button("Set stage", key=f"set-{name}"):
                tracking.set_stage(name, ver, stage)
                st.rerun()

