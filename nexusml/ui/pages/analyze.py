"""New analysis: set up a complete, validated analysis and watch it run. It keeps running if you leave this page."""
from typing import Any, Dict, Optional

from ... import featurizers, imageprep, jobs, media, workspace
from .. import components as c
from .. import nav, state
from .._st import ss, st

PROBLEMS = {"Decide for me": None, "Predict a category (classification)": "classification", "Predict a number (regression)": "regression", "Forecast over time": "time_series",
            "Find natural groups (segmentation)": "clustering", "Describe only, no model": "descriptive"}
METRICS = {"No target": None, "AUC": "roc_auc", "F1": "f1", "Accuracy": "accuracy", "Recall": "recall", "Precision": "precision", "R²": "r2", "RMSE (lower is better)": "rmse", "MAE (lower is better)": "mae"}


FRIENDLY = {"Data Engineer": "Data", "Planner": "Plan", "Supervisor": "Plan", "AutoML Agent": "Model search", "Director": "Review", "Sentinel": "Checks", "Scout": "Features",
            "Statistician": "Statistics", "Analyst": "Findings", "Reporter": "Report", "Modeler": "Model", "Critic": "Review", "Explainer": "Explanation", "System": "System"}


def build_spec(metric: Optional[str], value: float, second: Optional[str], second_value: float) -> Optional[Dict[str, Any]]:
    if not metric:
        return None
    spec: Dict[str, Any] = {"metric": metric, ("max" if metric in ("rmse", "mae") else "min"): float(value)}
    if second and second != metric:
        spec["subject_to"] = {"metric": second, ("max" if second in ("rmse", "mae") else "min"): float(second_value)}
    return spec


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
        elif snap["status"] == "failed":
            st.error("The analysis could not be completed.")
            st.caption(snap["error"] or "")
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


def render() -> None:
    state.init()
    c.page_header("New analysis", "A complete, validated analysis: exploration, a tuned and tested model, and a written report.")
    if ss.get("job_id"):
        snap = jobs.get(ss.job_id)
        live = bool(snap and snap["status"] in ("running", "queued"))
        runner = st.fragment(run_every=1 if live else None)(_progress)
        runner(ss.job_id)
        if live:
            st.caption("This keeps running if you leave the page. Open Analyses at any time to find it.")
        if st.button("Start another analysis", key="an-another"):
            ss.job_id = None
            st.rerun()
        if live:
            return
        if snap and snap["status"] == "done":
            return

    datasets = workspace.list_datasets()
    if not datasets:
        c.empty_state("Add a dataset first", "An analysis needs data. Add a table, or documents or images, on the Data page.", "Go to Data", key="an-nodata") and nav.go("data")
        return
    pre = ss.get("prefill") or {}
    meta, df = state.active_dataset()
    names = {d["id"]: d["name"] for d in datasets}
    default_id = pre.get("dataset_id") if pre.get("dataset_id") in names else (meta["id"] if meta else list(names)[0])
    cols = list(df.columns) if df is not None else []
    img_kw = _image_options(df, workspace.dataset_path(meta["id"])) if meta else {}
    with st.form("new-analysis"):
        ds_id = st.selectbox("Dataset", list(names), index=list(names).index(default_id), format_func=lambda k: names[k], key="an-ds")
        request = st.text_area("What do you want to find out?", value=pre.get("objective", ""), height=100, key="an-req",
                               placeholder="For example: why do customers cancel, and which ones are at risk next month?")
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
        go = st.form_submit_button("Start analysis", type="primary")
    if go:
        if not request.strip():
            st.warning("Describe what you want to find out, even in a few words.")
            return
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
        ss.job_id = jobs.submit(workspace.dataset_path(ds_id), request.strip(), ds_id, m["name"], options={"minutes": minutes}, **kw)
        ss.prefill = None
        st.rerun()
