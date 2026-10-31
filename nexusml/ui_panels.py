"""Streamlit panels added in v5. They take the `st` module as a parameter (no import-time Streamlit dependency), so they can be exercised in tests with a
stand-in `st` and real pipeline output. app.py only wires them into tabs."""
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from . import cv, leaderboard, nlp, plotly_viz as pv

PCFG = {"displaylogo": False}


def _plot(st, spec, key):
    if spec:
        st.plotly_chart(spec, use_container_width=True, key=key, config=PCFG, theme=None)


# leaderboard
def candidate_detail(st, r: Dict[str, Any], pt: str, pm: str, kp: str) -> None:
    """Everything known about ONE candidate: verdict, scores, hyper-parameters, tuning, held-out diagnostics, features."""
    key = f"{kp}-{abs(hash(r['name'] + str(r.get('attempt')))) % 10**8}"
    st.markdown(f"**{r['name']}**: rank {r.get('rank')} · attempt {r.get('attempt', 1)}" + (" · selected" if r.get("selected") else ""))
    st.write(r.get("verdict", ""))
    c = st.columns(5)
    c[0].metric(f"CV {pm}", r.get("select_score"))
    c[1].metric(f"Held-out {pm}", (r.get("heldout") or {}).get(pm))
    c[2].metric("Train score", r.get("train_score"), delta=None if r.get("overfit_gap") is None else f"gap {r['overfit_gap']:+.3f}", delta_color="inverse")
    c[3].metric("Fit seconds", r.get("fit_seconds"))
    c[4].metric("Predict ms / 1k rows", r.get("predict_ms_per_1k"))
    hp = r.get("best_params") or r.get("hyperparams") or {}
    if hp:
        st.caption(("Tuned hyper-parameters: " if r.get("best_params") else "Hyper-parameters: ") + ", ".join(f"{k}={v}" for k, v in hp.items()))
    t = r.get("tuning")
    if t:
        st.caption(f"Nested-CV estimate {t.get('nested_mean')} ± {t.get('nested_std')} · change vs default {t.get('delta_vs_default')} · {r.get('n_trials', 0)} search trials")
    d = r.get("diagnostics") or {}
    if d.get("error"):
        st.warning("Diagnostics unavailable: " + d["error"])
        return
    if pt == "classification" and d.get("confusion_matrix"):
        a, b = st.columns(2)
        with a:
            _plot(st, pv.confusion(d["confusion_matrix"], d["classes"]), f"{key}-cm")
        with b:
            st.markdown("**Per class**")
            st.dataframe(pd.DataFrame(d["per_class"]).T, use_container_width=True)
            if d.get("top_confusions"):
                st.caption("Most confused: " + "; ".join(f"{x['true']}→{x['predicted']} ({x['count']}x)" for x in d["top_confusions"]))
            if d.get("confidence"):
                st.caption(f"Mean confidence when right {d['confidence']['mean_when_right']}, when wrong {d['confidence']['mean_when_wrong']}")
        if d.get("roc"):
            a, b = st.columns(2)
            with a:
                _plot(st, pv.roc(d["roc"]), f"{key}-roc")
            with b:
                _plot(st, pv.pr(d["pr"]), f"{key}-pr")
    elif d.get("actual_vs_predicted"):
        a, b = st.columns(2)
        with a:
            _plot(st, pv.residuals(d["actual_vs_predicted"]["actual"], d["actual_vs_predicted"]["predicted"]), f"{key}-res")
        with b:
            _plot(st, pv.residual_hist(d["actual_vs_predicted"]["actual"], d["actual_vs_predicted"]["predicted"]), f"{key}-rh")
        st.caption(f"Typical absolute error: median {d['abs_error']['p50']}, 90th percentile {d['abs_error']['p90']}, 95th {d['abs_error']['p95']}; mean residual {d['residual_mean']}")
        if d.get("bias_by_predicted_band"):
            st.dataframe(pd.DataFrame(d["bias_by_predicted_band"]), use_container_width=True)
    if d.get("top_features"):
        _plot(st, pv.hbar([f["feature"] for f in d["top_features"]], [f["importance"] for f in d["top_features"]], "Largest model weights", xtitle="weight / importance"), f"{key}-imp")


def leaderboard_panel(st, values: Dict[str, Any], kp: str) -> None:
    lb = values.get("leaderboard") or {}
    rows, pm, pt = lb.get("rows", []), lb.get("metric"), values["analysis_plan"]["problem_type"]
    st.info(lb.get("summary", ""))
    st.dataframe(leaderboard.to_dataframe(lb), use_container_width=True, hide_index=True)
    a, b = st.columns(2)
    with a:
        _plot(st, pv.leaderboard_scores(rows, pm, (lb.get("baseline") or {}).get("cv_score")), f"{kp}-lbs")
    with b:
        _plot(st, pv.leaderboard_folds(rows), f"{kp}-lbf")
        st.caption("Overlapping boxes mean the models cannot be reliably told apart with this much data.")
    if lb.get("parsimony_differs"):
        st.success(f"One-standard-error rule: **{lb['parsimony_pick']}** is the simplest model within noise of the winner: a defensible choice if you value speed or explainability.")
    _plot(st, pv.roc_overlay(rows), f"{kp}-rocall")
    st.markdown("#### Inspect a candidate")
    names = [r["name"] for r in rows]
    sel = st.selectbox("Candidate", names, key=f"{kp}-lbsel")
    candidate_detail(st, next(r for r in rows if r["name"] == sel), pt, pm, kp)
    for h in lb.get("past_attempts", []):
        with st.expander(f"Attempt {h['attempt']} ({', '.join(h.get('levers') or []) or 'defaults'}): kept for the record"):
            st.dataframe(pd.DataFrame([{"model": r["name"], "cv score": r.get("select_score"), "cv std": r.get("cv_std"), "tuned": r.get("tuned")} for r in h["rows"]]), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.download_button("Leaderboard (CSV, all attempts)", leaderboard.to_csv(lb), file_name="leaderboard.csv", key=f"{kp}-lbdl1")
    c2.download_button("Leaderboard (HTML)", leaderboard.to_html(lb), file_name="leaderboard.html", key=f"{kp}-lbdl2")


# NLP
def _blank_row(bundle: Dict[str, Any], **cols) -> pd.DataFrame:
    row = pd.DataFrame([{c: None for c in bundle["feature_columns"]}])
    for k, v in cols.items():
        row[k] = v
    return row


def nlp_panel(st, values: Dict[str, Any], kp: str, render_charts: Callable, load_bundle: Callable, ss: Any = None) -> None:
    r = values.get("nlp_report") or {}
    c, lg = r.get("corpus") or {}, r.get("language") or {}
    m = st.columns(4)
    m[0].metric("Documents", f"{c.get('n_docs', 0):,}")
    m[1].metric("Vocabulary", f"{c.get('vocab_size', 0):,}")
    m[2].metric("Median length (tokens)", c.get("tokens_median"))
    m[3].metric("Language", lg.get("language"))
    t_over, t_cls, t_qual, t_err, t_try = st.tabs(["Overview", "Classes & topics", "Data quality", "Errors & robustness", "Try it"])
    with t_over:
        if r.get("top_bigrams"):
            st.markdown("**Most frequent two-word phrases**")
            st.dataframe(pd.DataFrame(r["top_bigrams"]), use_container_width=True, hide_index=True)
        if r.get("sentiment"):
            st.caption(f"Lexicon sentiment (English heuristic): {r['sentiment']['share']}")
        render_charts(values, ["nlp"], f"{kp}-nlpc", pick=True)
    with t_cls:
        if r.get("class_terms"):
            st.markdown("**Terms that distinguish each class** (log-odds z; above 1.96 is reliable)")
            st.dataframe(pd.DataFrame([{"class": k, "terms": ", ".join(f"{t['term']} ({t['z']})" for t in v[:8])} for k, v in r["class_terms"].items()]), use_container_width=True, hide_index=True)
        if r.get("topics"):
            tp = r["topics"]
            st.markdown(f"**{tp['k']} topics** (NMF, chosen by coherence {tp['coherence']})")
            st.dataframe(pd.DataFrame([{"topic": x["topic"], "share": x["share"], "terms": ", ".join(x["terms"][:8]), "dominant label": x.get("dominant_label"), "purity": x.get("label_purity")} for x in tp["topics"]]),
                         use_container_width=True, hide_index=True)
        reps = r.get("representative_docs") or {}
        if reps:
            with st.expander("Representative documents per group"):
                for k, v in reps.items():
                    st.markdown(f"**{k}**")
                    for d in v:
                        st.caption(d["text"])
    with t_qual:
        li = r.get("label_issues") or {}
        if li.get("suspects"):
            st.warning(f"{li['n_suspects']} document(s) ({li['share']:.1%}) look mislabelled. Review these first.")
            st.dataframe(pd.DataFrame(li["suspects"]), use_container_width=True, hide_index=True)
        else:
            st.success("No likely label errors found.")
        pii = r.get("pii") or {}
        if pii.get("docs_affected"):
            st.error(f"Possible personal data in {pii['docs_affected']} document(s): " + ", ".join(f"{k} ×{v}" for k, v in pii["counts"].items()) + ". Matches are counted, never stored.")
        if (r.get("entities") or {}).get("counts"):
            st.caption("Entity-like patterns: " + ", ".join(f"{k} ×{v}" for k, v in r["entities"]["counts"].items()) + " (pattern-based, not a trained NER model)")
    with t_err:
        mis = r.get("misclassified") or []
        if mis:
            st.markdown("**Held-out documents the model got wrong with the highest confidence**")
            for x in mis[:6]:
                st.markdown(f"*true* **{x['true']}** · *predicted* **{x['predicted']}** ({x['confidence']})")
                eff = (x.get("word_effects") or {}).get("all_effects")
                if eff:
                    st.markdown(nlp.highlight_html(x["text"], eff), unsafe_allow_html=True)
                else:
                    st.caption(x["text"])
        stt = r.get("stress_test")
        if stt:
            st.markdown("**Stress test**: how much does the score fall when the text is degraded?")
            _plot(st, pv.stress_bars(stt, "text"), f"{kp}-nlpstress")
            if stt.get("fragile"):
                st.warning("The model is fragile to ordinary text noise.")
    with t_try:
        col = (values.get("modality") or {}).get("text", [None])[0]
        txt = st.text_area("Type or paste a text", key=f"{kp}-nlptry", height=110)
        if txt and txt.strip() and col and values.get("model_key"):
            try:
                b = load_bundle()
                est, classes = b["pipeline"], b.get("classes")
                row = _blank_row(b, **{col: txt})
                if b.get("problem_type") == "classification":
                    P = est.predict_proba(row)[0]
                    st.success(f"Prediction: **{classes[int(P.argmax())]}** ({P.max():.0%})")
                else:
                    st.success(f"Prediction: **{float(est.predict(row)[0]):.4g}**")
                ex = nlp.explain_document(est, row, col, classes if b.get("problem_type") == "classification" else None, top=8)
                st.markdown(nlp.highlight_html(txt, ex.get("all_effects", {})), unsafe_allow_html=True)
                st.caption("Green words support the prediction, red words argue against it (each word removed in turn; works for any model).")
                if len(nlp.sentences(txt)) > 3:
                    st.markdown("**Extractive summary**")
                    for s_ in nlp.extractive_summary(txt, 3):
                        st.write("• " + s_)
            except Exception as e:
                st.warning(f"Could not score this text: {type(e).__name__}: {str(e)[:160]}")
        q = st.text_input("Search the collection by meaning", key=f"{kp}-nlpsearch")
        if q and values.get("dataset_path") and col:
            try:
                if ss is not None and ss.get("_nlp_ix", (None,))[0] != values["dataset_path"]:
                    df_ = pd.read_csv(values["dataset_path"])
                    ss["_nlp_ix"] = (values["dataset_path"], nlp.SemanticIndex(df_[col].astype(str).tolist()), df_[col].astype(str).tolist())
                _, ix, docs = ss["_nlp_ix"]
                st.dataframe(pd.DataFrame([{"score": s_, "text": docs[i][:220]} for i, s_ in ix.search(q, 6)]), use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Search unavailable: {type(e).__name__}: {str(e)[:120]}")


# vision
def vision_panel(st, values: Dict[str, Any], kp: str, render_charts: Callable, load_bundle: Callable, root: Optional[Path], tmp_dir: Path) -> None:
    r = values.get("vision_report") or {}
    m = st.columns(3)
    m[0].metric("Images analysed", f"{r.get('n_images', 0):,}")
    m[1].metric("Features per image", r.get("n_features"))
    m[2].metric("Deep model", "ResNet-18 available" if r.get("resnet_available") else "not installed")
    if not r.get("resnet_available"):
        st.caption("Candidates use hand-crafted features (colour, edges, texture, shape). They cannot recognise objects the way a deep network does; install `torch` + `torchvision` and set NEXUSML_IMAGE_BACKEND=resnet to add a ResNet candidate.")
    t_over, t_qual, t_err, t_try = st.tabs(["Collection", "Data quality", "Errors & robustness", "Try it"])
    with t_over:
        render_charts(values, ["vision"], f"{kp}-viscoll", pick=True)
    with t_qual:
        li = r.get("label_issues") or {}
        if li.get("suspects"):
            st.warning(f"{li['n_suspects']} image(s) ({li['share']:.1%}) look mislabelled.")
            st.dataframe(pd.DataFrame(li["suspects"]), use_container_width=True, hide_index=True)
        else:
            st.success("No likely label errors found.")
        if r.get("outliers"):
            st.markdown("**Most unusual images**")
            st.dataframe(pd.DataFrame(r["outliers"]), use_container_width=True, hide_index=True)
    with t_err:
        rob = r.get("robustness")
        if rob:
            _plot(st, pv.stress_bars(rob, "image"), f"{kp}-visrob")
            st.dataframe(pd.DataFrame(rob["results"]), use_container_width=True, hide_index=True)
            if rob.get("fragile"):
                st.warning("The model is fragile to ordinary image changes: do not expect it to hold on photos taken differently from the training set.")
        if r.get("misclassified"):
            st.markdown("**Most confident mistakes**")
            st.dataframe(pd.DataFrame(r["misclassified"]), use_container_width=True, hide_index=True)
    with t_try:
        col = (values.get("modality") or {}).get("image", [None])[0]
        up = st.file_uploader("Upload an image to classify and explain", type=["png", "jpg", "jpeg", "bmp", "gif", "webp"], key=f"{kp}-vistry")
        if up is not None and col and values.get("model_key"):
            try:
                tmp_dir.mkdir(parents=True, exist_ok=True)
                f = tmp_dir / f"upload_{abs(hash(up.name)) % 10**8}{Path(up.name).suffix or '.png'}"
                f.write_bytes(up.getvalue())
                b = load_bundle()
                est, classes = b["pipeline"], b.get("classes")
                row = _blank_row(b, **{col: str(f)})
                cls = classes if b.get("problem_type") == "classification" else None
                if cls:
                    P = est.predict_proba(row)[0]
                    st.success(f"Prediction: **{cls[int(P.argmax())]}** ({P.max():.0%})")
                else:
                    st.success(f"Prediction: **{float(est.predict(row)[0]):.4g}**")
                occ = cv.occlusion_map(est, row, col, Path("."), cls)
                st.pyplot(occ["figure"])
                st.caption("Occlusion sensitivity: red cells supported the prediction, blue cells argued against it (each cell greyed out in turn; works for any model).")
            except Exception as e:
                st.warning(f"Could not score this image: {type(e).__name__}: {str(e)[:160]}")


# timeline
def timeline_panel(st, values: Dict[str, Any], kp: str) -> None:
    tm = {k: v for k, v in (values.get("timings") or {}).items() if isinstance(v, dict)}
    top = {k: v for k, v in tm.items() if "." not in k}
    if not top:
        st.info("No timing data for this run.")
        return
    total = sum(v.get("seconds", 0) for v in top.values()) or 1
    st.metric("Total stage time", f"{total:.1f} s")
    df = pd.DataFrame([{"stage": k, "seconds": v.get("seconds"), "share": round(v.get("seconds", 0) / total, 3), "runs": v.get("runs"), "model calls": v.get("llm_calls"), "tokens": v.get("tokens")} for k, v in top.items()])
    st.dataframe(df, use_container_width=True, hide_index=True)
    _plot(st, pv.hbar(list(top), [v.get("seconds", 0) for v in top.values()], "Seconds per stage", xtitle="seconds"), f"{kp}-tl")
    subs = {k: v for k, v in tm.items() if "." in k}
    if subs:
        with st.expander("Sub-tasks inside the parallel stages"):
            st.dataframe(pd.DataFrame([{"task": k, "seconds": v.get("seconds")} for k, v in subs.items()]), use_container_width=True, hide_index=True)
    for e in values.get("stage_errors") or []:
        st.warning(f"{e['task']} failed and was skipped: {e['error']}")


# experiments: all candidates
def candidates_page(st, tracking, kp: str = "cand") -> None:
    """Every candidate run of every analysis in one sortable table, with side-by-side comparison and held-out diagnostics."""
    exps = tracking.experiments()
    exp = st.selectbox("Experiment", ["(all)"] + exps, key=f"{kp}-exp")
    runs = tracking.all_candidate_runs(None if exp == "(all)" else exp)
    if not runs:
        st.info("No candidate runs yet. Every model compared in an analysis is logged here with its parameters, metrics and diagnostics.")
        return
    keep = ("select_score", "cv_std", "roc_auc", "f1_macro", "accuracy", "r2", "rmse", "mae", "train_score", "overfit_gap", "p_vs_best", "fit_seconds")
    rows = [{"candidate": r["name"], "experiment": r["experiment"], "attempt": r["tags"].get("attempt"), "selected": r["tags"].get("selected"),
             **{k: r["metrics"].get(k) for k in keep if k in r["metrics"]}} for r in runs]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    labels = [f"{r['name']} · {r['experiment']} · {r['run_id'][:6]}" for r in runs]
    chosen = st.multiselect("Compare candidates", labels, key=f"{kp}-sel")
    sel = [runs[labels.index(c)] for c in chosen]
    if len(sel) >= 2:
        mk = st.selectbox("Metric", sorted({k for r in sel for k in r["metrics"]}), key=f"{kp}-mk")
        _plot(st, pv.bars([r["name"][-18:] for r in sel], [r["metrics"].get(mk) for r in sel], f"{mk} by candidate", mk), f"{kp}-cmp")
        st.dataframe(pd.DataFrame(tracking.param_diff(sel)), use_container_width=True)
    if len(sel) == 1:
        r = sel[0]
        st.markdown(f"**{r['name']}**: {r['tags'].get('verdict', '')}")
        d = tracking.candidate_diagnostics(r["run_id"])
        if d.get("confusion_matrix"):
            _plot(st, pv.confusion(d["confusion_matrix"], d["classes"]), f"{kp}-cm")
            st.dataframe(pd.DataFrame(d["per_class"]).T, use_container_width=True)
        elif d.get("actual_vs_predicted"):
            _plot(st, pv.residuals(d["actual_vs_predicted"]["actual"], d["actual_vs_predicted"]["predicted"]), f"{kp}-res")
        if d.get("top_features"):
            _plot(st, pv.hbar([f["feature"] for f in d["top_features"]], [f["importance"] for f in d["top_features"]], "Largest model weights"), f"{kp}-imp")


# v6: AutoML + crew
_VERDICT = {"met": ("success", "Goal met: confirmed on fresh folds and on the held-out data."), "met_on_test_only": ("warning", "Borderline: met on the held-out data but not on the fresh training folds."),
            "not_met": ("error", "Goal not met."), "no_goal": ("info", "Best-effort run (no threshold given).")}


def automl_panel(st, values: Dict[str, Any], kp: str) -> None:
    ar = values.get("automl_report") or {}
    gr = ar.get("goal") or {}
    kind, msg = _VERDICT.get(gr.get("status"), ("info", ""))
    getattr(st, kind)(f"{gr.get('text', '')}: {msg}")
    for e in gr.get("evidence") or []:
        st.caption("• " + e)
    m = st.columns(4)
    m[0].metric("Trials", ar.get("n_trials"))
    m[1].metric("Search rounds", len(ar.get("rounds", [])))
    m[2].metric("Seconds", ar.get("seconds"))
    m[3].metric("Search attempts", ar.get("attempt"))
    _plot(st, pv.goal_ladder(gr), f"{kp}-ladder")
    st.caption("Search scores are optimistic once many configs were tried, so the finalists are re-scored on fresh folds and the held-out split is scored exactly once at the end.")
    _plot(st, pv.automl_trials(ar.get("trials", []), ar.get("search_metric", "")), f"{kp}-trials")
    st.markdown("#### How the search proceeded, round by round")
    for r in ar.get("rounds", []):
        with st.expander(f"Round {r['round']} [{r.get('source')}]: gain {r.get('gain')} → best {r.get('best')}"):
            st.write(r.get("diagnosis", ""))
            if r.get("actions"):
                st.dataframe(pd.DataFrame([{"action": a["type"], "detail": str(a.get("params", {}))[:80], "why": a.get("why", ""), "trials": a.get("trials"),
                                            "effect on best CV": (a.get("effect") or {}).get("mean_delta"), "gain": a.get("gain")} for a in r["actions"]]), use_container_width=True, hide_index=True)
            if r.get("rejected"):
                st.warning("Rejected by the guards: " + "; ".join(r["rejected"]))
    lc = ar.get("learning_curve")
    if lc and lc.get("points"):
        st.markdown(f"**Learning curve of the incumbent**: {lc.get('verdict')}")
        st.dataframe(pd.DataFrame(lc["points"]), use_container_width=True, hide_index=True)
    if (ar.get("ceiling") or {}).get("available"):
        st.info(f"Noise ceiling: {ar['ceiling']}")
    e = ar.get("ensemble") or {}
    if e.get("method") == "greedy":
        st.markdown("**Ensemble** (greedy selection on out-of-fold predictions)")
        st.dataframe(pd.DataFrame(e["members"]), use_container_width=True, hide_index=True)
    with st.expander("All trials"):
        st.dataframe(pd.DataFrame([{k: v for k, v in t.items() if k != "params"} | {"params": str(t.get("params"))[:100]} for t in ar.get("trials", [])]), use_container_width=True, hide_index=True)


def crew_panel(st, values: Dict[str, Any], kp: str) -> None:
    dec, board = values.get("decisions") or [], values.get("board") or []
    crew = values.get("crew") or {}
    if crew:
        st.caption("Specialists involved: " + ", ".join(k for k, v in crew.items() if v))
    _plot(st, pv.crew_gantt(values.get("timings") or {}), f"{kp}-gantt")
    st.markdown("#### Choices made")
    st.caption("Choices the assistant made while working, with the reason and the rule that bounded each one. `default` means a built-in default was used because the language model could not answer.")
    if dec:
        st.dataframe(pd.DataFrame([{"junction": d.get("junction"), "source": d.get("source"), "decision": str(d.get("decision"))[:90], "rationale": str(d.get("rationale") or d.get("diagnosis") or "")[:220],
                                    "guard": str(d.get("guard") or "")[:120]} for d in dec]), use_container_width=True, hide_index=True)
    st.markdown("#### Messages exchanged while the analysis ran")
    for mm in board:
        tag = {"high": "Important", "warning": "Note"}.get(mm.get("severity"), "")
        st.markdown(f"`{mm['t']}s` **{mm['sender'].capitalize()}**" + (f" · {tag}" if tag else "") + f": {mm['text']}")
