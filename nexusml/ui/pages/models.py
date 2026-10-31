"""Models: the experiment registry, and scoring new data with a model you built."""
import pandas as pd

from ... import modeling, runstore, workspace
from ...storage import get_registry
from .. import components as c
from .. import legacy, state
from .._st import st


def _score() -> None:
    runs = [r for r in runstore.list_runs(200) if r.get("model_key")]
    if not runs:
        c.empty_state("No models yet", "Models appear here once an analysis has finished.")
        return
    label = {r["run_id"]: f"{r.get('target') or 'model'} · {r.get('best_model') or ''} · {c.time_ago(r['created'])}" for r in runs}
    rid = st.selectbox("Model", list(label), format_func=lambda k: label[k], key="mdl-pick")
    run = next(r for r in runs if r["run_id"] == rid)
    datasets = workspace.list_datasets()
    src = st.radio("Data to score", ["A dataset in the workspace", "Upload a file"], horizontal=True, key="mdl-src")
    df = None
    if src == "Upload a file":
        up = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls", "tsv"], key="mdl-up")
        if up is not None:
            try:
                import io
                from ... import ingest
                df = pd.read_csv(io.BytesIO(ingest.to_csv_bytes(up.name, up.getvalue())), low_memory=False)
            except Exception:
                st.error("That file could not be read as a table.")
    elif datasets:
        pick = st.selectbox("Dataset", [d["id"] for d in datasets], format_func=lambda k: next(d["name"] for d in datasets if d["id"] == k), key="mdl-ds")
        df = workspace.load_df(pick)
    else:
        st.info("Add a dataset on the Data page first.")
    if df is None:
        return
    if st.button("Score", type="primary", key="mdl-go"):
        try:
            bundle = get_registry().load(run["model_key"])
            out = modeling.predict_with_bundle(bundle, df)
        except Exception as e:
            st.error("This model could not score that data. Check that it has the same columns the model was trained on.")
            st.caption(f"{type(e).__name__}: {str(e)[:160]}")
            return
        res = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
        st.success(f"Scored {len(res):,} rows.")
        st.dataframe(res.head(500), use_container_width=True, hide_index=True, key="mdl-out")
        st.download_button("Download predictions", res.to_csv(index=False).encode(), file_name="predictions.csv", mime="text/csv", key="mdl-dl")


def render() -> None:
    state.init()
    c.page_header("Models", "Compare experiments, manage registered models, and score new data.")
    t_exp, t_score = st.tabs(["Experiments", "Score new data"])
    with t_exp:
        with legacy.guarded("Experiments"):
            legacy.experiments_view()
    with t_score:
        _score()
