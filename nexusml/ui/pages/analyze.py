"""Analyses (landing): a large composer for a NEW analysis (or a follow-up that builds on an earlier one) with the history of analyses in a compact rail beside it.
A running analysis keeps running if you leave this page."""
from typing import Any, Dict, Optional

from ... import featurizers, imageprep, jobs, lineage, media, resume, workspace
from .. import components as c
from .. import context, history, nav, state
from ..options import METRICS, PROBLEMS, build_spec  # noqa: F401  (re-exported: other modules and tests import them from here)
from .._st import ss, st


FRIENDLY = {"Data Engineer": "Data", "Planner": "Plan", "Supervisor": "Plan", "AutoML Agent": "Model search", "Director": "Review", "Sentinel": "Checks", "Scout": "Features",
            "Statistician": "Statistics", "Analyst": "Findings", "Reporter": "Report", "Modeler": "Model", "Critic": "Review", "Explainer": "Explanation", "System": "System"}


def _progress(job_id: str) -> None:
    snap = jobs.get(job_id)
    if not snap:
        st.info("That analysis is no longer tracked in this session. Check Analyses for finished results.")
        return
    with st.container(border=True):
        top = st.columns([4, 1])
        top[0].markdown(f"**{snap['request'][:90] or 'Analysis'}**")
        top[0].caption(f"{snap['dataset_name']} · {snap['elapsed']:.0f} s")
        top[1].markdown(c.badge({"running": "Running", "queued": "Queued", "done": "Finished", "failed": "Failed", "cancelled": "Cancelled", "needs_input": "Needs input"}.get(snap["status"], snap["status"]), c.status_tone(snap["status"])), unsafe_allow_html=True)
        st.progress(snap["progress"] if snap["status"] != "done" else 1.0, text=snap["stage_label"] or "Starting…")
        if snap["log"]:
            with st.expander("What is happening", expanded=snap["status"] in ("running", "queued")):
                for e in snap["log"][-12:]:
                    st.markdown(f"<span class='nx-step'>{e['t']:.0f}s</span> **{FRIENDLY.get(e['agent'], str(e['agent']).replace(' Agent', ''))}** {e['message'][:220]}", unsafe_allow_html=True)
        if snap["status"] in ("running", "queued"):
            if st.button("Cancel", key=f"cancel-{job_id}"):
                jobs.cancel(job_id)
        elif snap["status"] == "done":
            st.success("The analysis is finished.")
            a, b = st.columns(2)
            if a.button("Open the results", type="primary", key=f"open-{job_id}", use_container_width=True):
                ss.open_run = ss.view_run = snap["experiment_id"]
                nav.go("history")
            if b.button("Ask the assistant about it", key=f"ask-{job_id}", use_container_width=True):
                ss.open_run = snap["experiment_id"]
                nav.go("assistant")
        elif snap["status"] == "needs_input":
            st.warning(snap["message"] or "I could not tell what to predict. Choose a target column and start again.")
            if st.button("Choose a column and retry", type="primary", key=f"fix-{job_id}"):
                history.retry_dialog({"kind": "job", "id": job_id})
        elif snap["status"] == "failed":
            st.error("The analysis could not be completed.")
            st.caption(snap["error"] or "")
            if st.button("Retry with changes", key=f"retry-{job_id}"):
                history.retry_dialog({"kind": "job", "id": job_id})
        elif snap["status"] == "cancelled":
            st.info("Cancelled.")


DEEP_CHOICES = {"Automatic (use when available)": "auto", "Off (hand-crafted features only)": "off", "ResNet-18": "resnet", "ResNet-50 (slower, stronger)": "resnet50", "Both": "both"}


def _step_widget(op: str) -> Dict[str, Any]:
    """Controls for one preprocessing step -> its validated settings."""
    step: Dict[str, Any] = {"op": op}
    for name, spec in imageprep.OPS[op].items():
        key = f"an-img-{op}-{name}"
        if isinstance(spec[1], list):
            step[name] = st.selectbox(name.capitalize(), spec[1], key=key)
        elif isinstance(spec[0], int):
            if op == "denoise":
                step[name] = st.selectbox("Window size", [3, 5, 7], key=key)
            else:
                step[name] = int(st.number_input(name.capitalize() + " (px)", int(spec[1]), int(spec[2]), int(spec[0]), key=key))
        else:
            step[name] = float(st.number_input(name.capitalize(), float(spec[1]), float(spec[2]), float(spec[0]), key=key))
    return step


def _image_options(df, dataset_path) -> Dict[str, Any]:
    """Preprocessing recipe + deep-network choice for image datasets (shown above the form so the preview updates as you choose)."""
    col = next((k for k in df.columns if featurizers.is_image_path(df[k])), None) if df is not None else None
    if col is None:
        return {}
    out: Dict[str, Any] = {}
    with st.expander("Image options", expanded=False):
        picked = st.multiselect("Preprocessing steps, applied in this order", list(imageprep.OPS), format_func=lambda k: imageprep.LABELS[k], key="an-img-steps",
                                help="Resize, blur, sharpen, edge detection and more. The same steps are applied to new photos at prediction time.")
        steps = []
        if picked:
            cols = st.columns(min(len(picked), 3))
            for i, op in enumerate(picked):
                with cols[i % len(cols)]:
                    st.caption(imageprep.LABELS[op])
                    steps.append(_step_widget(op))
            vals = [str(v) for v in df[col].dropna().head(40)]
            root = None if all(v.startswith(("/", "~")) or (len(v) > 1 and v[1] == ":") for v in vals) else media.locate_root(vals, str(dataset_path))
            paths = [featurizers._resolve(v, str(root) if root else None) for v in vals]
            pairs = imageprep.preview([p for p in paths if p is not None], steps, featurizers.load_rgb)
            if pairs:
                st.caption("Preview: original (top) and after the steps (bottom)")
                g = st.columns(len(pairs))
                for k, (a, b) in enumerate(pairs):
                    g[k].image(a, use_container_width=True)
                    g[k].image(b, use_container_width=True)
        choice = st.selectbox("Deep neural network features", list(DEEP_CHOICES), key="an-img-deep")
        import importlib.util
        if DEEP_CHOICES[choice] != "off":
            if importlib.util.find_spec("torch") is None:
                st.caption("PyTorch is not installed here, so only the hand-crafted models will run. Install torch and torchvision (see Dockerfile.multimodal) to add ResNet models.")
            else:
                st.caption("Pretrained ImageNet weights are downloaded on first use. Offline: point NEXUSML_DEEP_WEIGHTS at a saved weights file.")
        out = {"image_prep": steps, "deep_mode": DEEP_CHOICES[choice]}
    return out


def _extension_banner(parent_exp: str, path: Optional[str]) -> bool:
    """The 'building on' panel. Returns True if the person wants the exploration redone instead of reused."""
    parent = lineage.load_state(parent_exp)
    with st.container(border=True):
        c.eyebrow("Building on an earlier analysis")
        if not parent:
            st.warning("The analysis to build on could not be found. It may have been deleted.")
        else:
            st.markdown(f"**{(parent.get('user_request') or 'Analysis')[:140]}**")
            plan = parent.get("analysis_plan") or {}
            st.caption(" · ".join(b for b in [f"{plan.get('problem_type')}" if plan.get("problem_type") else "", f"target {plan['target']}" if plan.get("target") else "", parent.get("best_model_name") or "no model yet"] if b))
            try:
                sd = resume.seed(parent_exp, parent.get("dataset_path") or "")
                done = (["data profile"] if sd["resume"]["profile_reused"] else []) + [resume.LABEL[p] for p in sd["resume"]["carry"]]
                st.caption(("Reused, not recomputed: " + ", ".join(done) + ". Statistical tests are redone if the target or kind of analysis changes.") if done else "Nothing from it can be reused, so everything is recomputed.")
                for n in sd["resume"]["notes"]:
                    st.caption(n)
            except resume.ResumeError as e:
                st.warning(str(e))
        redo = st.checkbox("Redo the exploration instead of reusing it", value=False, key="an-redo", help="Recomputes the statistics, KPIs and trends from scratch. The data profile is still reused when the dataset is unchanged.")
        if st.button("Start a separate analysis instead", key="an-ext-cancel"):
            ss.extend_from = None
            st.rerun()
    return redo


def _composer() -> None:
    datasets = workspace.list_datasets()
    if not datasets:
        c.empty_state("Add a dataset first", "An analysis needs data. Add a table, or documents or images, on the Data page.", "Go to Data", key="an-nodata") and nav.go("data")
        return
    ext = ss.get("extend_from")
    parent = lineage.load_state(ext) if ext else None
    if ext and not parent:
        ss.extend_from = ext = None                                   # the analysis was deleted meanwhile: fall back to a plain new analysis
    pre = ss.get("prefill") or {}
    if parent:
        mine = next((d for d in datasets if str(workspace.dataset_path(d["id"])) == str(parent.get("dataset_path"))), None)
        if mine:
            ss.ds_id = mine["id"]
    meta, df = state.active_dataset()
    names = {d["id"]: d["name"] for d in datasets}
    default_id = pre.get("dataset_id") if pre.get("dataset_id") in names else (meta["id"] if meta else list(names)[0])
    st.markdown("## " + ("Follow-up" if ext else "New analysis"))
    st.markdown('<p class="nx-sub">' + ("Say what to do next. The earlier results are reused, and only the new work is done." if ext else
                                         "A complete, validated analysis: exploration, a tuned and tested model, and a written report. Start with a question in your own words.") + "</p>", unsafe_allow_html=True)
    redo = _extension_banner(ext, str(workspace.dataset_path(meta["id"])) if meta else None) if ext else False
    cols = list(df.columns) if df is not None else []
    img_kw = _image_options(df, workspace.dataset_path(meta["id"])) if meta else {}
    with st.form("new-analysis"):
        ds_id = st.selectbox("Dataset", list(names), index=list(names).index(default_id), format_func=lambda k: names[k], key="an-ds", disabled=bool(ext))
        request = st.text_area("What do you want to find out?" if not ext else "What should it do next?", value=pre.get("objective", ""), height=170, key="an-req",
                               placeholder="For example: why do customers cancel, and which ones are at risk next month?" if not ext else "For example: now build a predictive model for the customers who cancelled")
        a, b = st.columns(2)
        target_opts = ["Decide for me"] + cols
        t_idx = target_opts.index(pre["target"]) if pre.get("target") in target_opts else 0
        target = a.selectbox("Column to predict", target_opts, index=t_idx, key="an-target")
        kind = b.selectbox("Kind of analysis", list(PROBLEMS), key="an-kind")
        with st.expander("Set a target to reach (optional)"):
            g1, g2 = st.columns(2)
            m_opts = list(METRICS)
            pre_m = next((k for k, v in METRICS.items() if v == pre.get("goal_metric")), "No target")
            metric = g1.selectbox("Metric", m_opts, index=m_opts.index(pre_m), key="an-metric")
            value = g2.number_input("Value to reach", value=float(pre.get("goal_value") or 0.8), step=0.01, format="%.3f", key="an-value")
            g3, g4 = st.columns(2)
            second = g3.selectbox("…while keeping", m_opts, key="an-second")
            second_v = g4.number_input("at least", value=0.5, step=0.01, format="%.3f", key="an-second-v")
            st.caption("The analysis searches until the target is reached or the time runs out, then reports whether it was, checked on data it has not seen.")
        with st.expander("More options"):
            o1, o2 = st.columns(2)
            minutes = o1.slider("Time for the model search (minutes)", 1, 60, 5, key="an-min")
            workers = o2.slider("Parallel workers", 1, 8, 2, key="an-workers")
            exclude = st.checkbox("Keep sensitive columns (names, ids, protected attributes) out of the model", value=True, key="an-sens")
            tune = st.checkbox("Search for the best model settings", value=True, key="an-tune")
            share_words = st.checkbox("Share frequent words from text columns to name themes", value=False, key="an-words")
        go = st.form_submit_button("Start analysis" if not ext else "Start follow-up", type="primary", use_container_width=True)
    if go:
        if not request.strip():
            st.warning("Describe what you want to find out, even in a few words.")
            return
        ds_id = meta["id"] if ext and meta else ds_id
        m = workspace.get_dataset(ds_id)
        spec = build_spec(METRICS[metric], value, METRICS[second], second_v)
        kw: Dict[str, Any] = dict(enable_tuning=tune, exclude_sensitive=exclude, allow_llm_text=share_words, time_budget=minutes * 60.0, workers=workers, seed=42, **workspace.dataset_hints(m))
        if img_kw and meta and ds_id == meta["id"]:
            kw.update(img_kw)
        if target != "Decide for me":
            kw["target_column"] = target
        if PROBLEMS[kind]:
            kw["problem_type"] = PROBLEMS[kind]
        if spec:
            kw["target_spec"] = spec
        if ext:
            kw["resume_from"] = ext
            if redo:
                kw["resume_reuse"] = ["profile"]
        try:
            ss.job_id = jobs.submit(workspace.dataset_path(ds_id), request.strip(), ds_id, m["name"], options={"minutes": minutes}, **kw)
        except resume.ResumeError as e:
            st.error(str(e))
            return
        ss.prefill = None
        ss.extend_from = None
        st.rerun()


def render() -> None:
    state.init()
    context.set_view("analyses", extending=ss.get("extend_from"), job=ss.get("job_id"))
    main, rail = st.columns([7, 3], gap="large")
    with rail:
        history.render_rail()
    with main:
        if ss.get("job_id"):
            snap = jobs.get(ss.job_id)
            live = bool(snap and snap["status"] in ("running", "queued"))
            runner = st.fragment(run_every=1 if live else None)(_progress)
            runner(ss.job_id)
            if live:
                st.caption("This keeps running if you leave the page. You can start another analysis below, or find this one in the history.")
            elif st.button("Dismiss", key="an-another"):
                ss.job_id = None
                st.rerun()
        _composer()
