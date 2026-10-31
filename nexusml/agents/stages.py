"""v6 graph stages: a CREW of agents, not a pipeline.

  intake -> frame -> [await_clarification] -> plan -> [human_review] -> prepare -> brief (Director, LLM)
      brief --+--> sentinel (leakage / drift / conflicts)  --+
              +--> explore  (statistics, KPIs, trends, text / vision analysis)
              +--> scout    (domain feature ideas)           +--> synthesize -> publish
              +--> automl <-> critic (LLM verdict: accept | refine) --> explain --+
  abort  (rejected or invalid plan)

  * The four branches run CONCURRENTLY and talk through the blackboard: the AutoML agent reads the sentinel's leakage proposal, the scout's feature ideas and the explorer's
    findings at every round, instead of waiting for them in a sequence.
  * Decisions that need judgement are made by the LLM (the Director) inside deterministic guards (director.py); validity, safety, approval and budgets stay deterministic.
  * `explain` is the loop's exit and the only join source on the modeling side (a looping node must never feed a join on real LangGraph).
  * Failure isolation inside parallel sub-tasks, per-stage timings, and every decision recorded in `decisions`.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .. import blackboard, config, leaderboard, llm, modeling, viz
from ..state import merge_update, merge_updates
from ..storage import get_registry
from ._util import jsonable, log, make_evidence
from . import analysis, business, context, crew, data_engineer, explainer, media, planner, reporter

SUPERVISED = modeling.SUPERVISED
PHASE_OF = {"statistical_analysis": "statistics", "kpi_analysis": "kpis", "trend_analysis": "trends", "media_analysis": "media",
            "visualize_eda": "charts", "context_research": "context"}
SKIPPED_DEFAULTS = {"statistical_analysis": {"stat_results": [], "stat_skipped": []}, "kpi_analysis": {"kpi_results": {}},
                    "trend_analysis": {"trend_results": {}, "outlier_results": {}}}


# telemetry + isolation
def stage(name: str, fn: Callable) -> Callable:
    """Wrap a stage: measure wall time and LLM calls into `timings` (accumulated across loop iterations)."""
    def node(state):
        t0 = time.perf_counter()
        wall0 = time.time()
        out = fn(state) or {}
        prev = (state.get("timings") or {}).get(name, {})
        recs = out.get("llm_calls") or []
        calls = len(recs)
        tok = 0
        for c in recs:                                                 # telemetry must never be able to stop a run: tolerate any record shape
            try:
                tok += int(c.get("tokens", 0) or 0) if isinstance(c, dict) else 0
            except (TypeError, ValueError):
                pass
        out = dict(out)
        out["timings"] = {name: {"seconds": round(prev.get("seconds", 0.0) + time.perf_counter() - t0, 2), "runs": prev.get("runs", 0) + 1,
                                 "llm_calls": prev.get("llm_calls", 0) + calls, "tokens": prev.get("tokens", 0) + tok,
                                 "first_start": prev.get("first_start", round(wall0, 3)), "last_end": round(time.time(), 3)}}
        return out
    node.__name__ = name
    return node


def _skip_update(sub: str, state) -> Dict[str, Any]:
    return {**SKIPPED_DEFAULTS.get(sub, {}), "agent_log": [log("Supervisor", f"Skipped '{sub}': not needed for this request.")]}


def _run_subtasks(stage_name: str, state, tasks: List[Tuple[str, Callable]], parallel: bool = True) -> Dict[str, Any]:
    """Run independent sub-tasks (concurrently when allowed), isolating failures. Updates are merged in LIST order, so results are deterministic."""
    skip = set(state.get("skip_phases") or [])

    def one(item):
        sub, fn = item
        if PHASE_OF.get(sub) in skip:
            return _skip_update(sub, state)
        t0 = time.perf_counter()
        try:
            with_lock = sub in ("media_analysis", "visualize_eda")
            if with_lock:
                with viz.PLOT_LOCK:
                    out = fn(dict(state)) or {}
            else:
                out = fn(dict(state)) or {}
            out = dict(out)
            out["timings"] = {f"{stage_name}.{sub}": {"seconds": round(time.perf_counter() - t0, 2)}}
            return out
        except llm.LLMRequired:
            raise                                                      # the LLM is mandatory: stop with its clear message
        except Exception as e:                                         # anything else: record, keep going
            msg = f"{sub} failed ({type(e).__name__}: {str(e)[:160]}); the run continued without it."
            return {"stage_errors": [{"stage": stage_name, "task": sub, "error": f"{type(e).__name__}: {str(e)[:300]}"}], "warnings": [msg],
                    "agent_log": [log("Supervisor", msg)], **SKIPPED_DEFAULTS.get(sub, {})}

    use_pool = parallel and config.PARALLEL and len(tasks) > 1
    if use_pool:
        with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as ex:
            results = list(ex.map(one, tasks))
    else:
        results = [one(t) for t in tasks]
    return merge_updates(*results)


def intake_node(state) -> Dict[str, Any]:
    return data_engineer.profile_dataset_node(state)


def frame_node(state) -> Dict[str, Any]:
    out = planner.understand_request_node(state)
    if not out.get("is_objective_clear"):                              # prepare the question NOW: the graph pauses before `await_clarification`
        out = merge_updates(out, planner.ask_clarification_node(merge_update(state, out)))
    return out


def await_clarification_node(state) -> Dict[str, Any]:
    """Resume after the human answered (target / problem type). The human's choices always win."""
    out = planner.await_clarification_node(state)
    if not out.get("is_objective_clear"):                              # still unclear: re-ask, listing the candidate columns again
        out = merge_updates(out, planner.ask_clarification_node(merge_update(state, out)))
    return out


def plan_node(state) -> Dict[str, Any]:
    return planner.build_plan_node(state)


def human_review_node(state) -> Dict[str, Any]:
    return planner.human_review_node(state)


def plan_errors(state) -> List[str]:
    """Pure validation of the (possibly human-edited) plan: used by the routers and by `abort`."""
    if state.get("approval_status") == "rejected":
        return ["Plan was rejected during human review."]
    plan, profile = state.get("analysis_plan") or {}, state.get("dataset_profile") or {}
    if not plan:
        return ["No analysis plan was produced."]
    errs: List[str] = []
    skip = {plan.get("target")} | set(plan.get("drop_columns", []))
    n_feat = len([c for c in profile.get("columns", {}) if c not in skip])
    pt = plan["problem_type"]
    if pt != "descriptive" and n_feat < 1:
        errs.append("No feature columns remain after drops.")
    if profile.get("n_rows", 0) < config.MIN_ROWS:
        errs.append(f"Need at least {config.MIN_ROWS} rows (have {profile.get('n_rows', 0)}).")
    if pt == "time_series" and not plan.get("date_column"):
        errs.append("Time-series analysis needs a date column.")
    return errs


def abort_node(state) -> Dict[str, Any]:
    errs = plan_errors(state) or ["The run was stopped before modeling."]
    return {"errors": errs, "current_status": "Aborted.", "active_agent": "Planner", "agent_log": [log("Planner", "Run ended without training: " + "; ".join(errs))]}


def pipeline_kind_of(mods: Dict[str, Any]) -> str:
    t, i = bool((mods or {}).get("text")), bool((mods or {}).get("image"))
    return "multimodal" if t and i else "text" if t else "image" if i else "tabular"


def prepare_node(state) -> Dict[str, Any]:
    out = data_engineer.prepare_data_node(state)
    kind = pipeline_kind_of(out.get("modality") or {})
    strat = {**(state.get("strategy") or {}), "pipeline_kind": kind}
    msg = {"text": "Text pipeline: representation x classifier comparison (word / char TF-IDF, LSA).",
           "image": "Vision pipeline: feature-backend x classifier comparison (HOG, LBP, colour, texture).",
           "multimodal": "Multimodal data: text and image columns are featurised inside one pipeline."}.get(kind)
    return {**out, "pipeline_kind": kind, "strategy": strat,
            "agent_log": list(out.get("agent_log", [])) + ([log("Data Engineer", msg)] if msg else [])}


# explore (parallel analysis)
def explore_node(state) -> Dict[str, Any]:
    b = blackboard.get_board(state["run_id"])
    try:
        st = state
        if not (state.get("crew") or {}).get("explore", True):
            st = {**state, "skip_phases": sorted(set(state.get("skip_phases") or []) | {"statistics", "kpis", "trends", "media"})}
        upd = _run_subtasks("explore", st, [("statistical_analysis", analysis.statistical_analysis_node), ("kpi_analysis", analysis.kpi_analysis_node),
                                            ("trend_analysis", analysis.trend_analysis_node), ("media_analysis", media.media_analysis_node)])
        try:                                                            # tell the AutoML agent what the analysis found (names and counts only)
            sr = upd.get("stat_results") or []
            names = [str(r.get("name") or r.get("test") or r.get("id") or "") for r in sr if isinstance(r, dict)][:4]
            if sr:
                b.post("explore", "findings", {"n_tests": len(sr)}, text=f"Statistics ran {len(sr)} tests; first: {', '.join(n for n in names if n)}.")
        except Exception:
            pass
        return upd
    finally:
        b.mark_done("explore")


# the crew (agents/crew.py)
brief_node, sentinel_node, scout_node, automl_node, critic_node = crew.brief_node, crew.sentinel_node, crew.scout_node, crew.automl_node, crew.critic_node


def explain_node(state) -> Dict[str, Any]:
    """Runs ONCE after the loop: explains the winner, builds the full leaderboard, the model card and the modality-specific post-model analyses."""
    pt = state["analysis_plan"]["problem_type"]
    if pt not in SUPERVISED or not state.get("models_key") or not state.get("candidate_metrics"):
        return {}
    with viz.PLOT_LOCK:
        upd = explainer.explain_results_node(state)
    st = merge_update(state, upd)
    reg = get_registry()
    extra: Dict[str, Any] = {}
    try:
        lb = leaderboard.build(st)
        if lb:
            ev = make_evidence("B", leaderboard.evidence_items(lb))
            html = leaderboard.to_html(lb)
            extra.update({"leaderboard": jsonable(lb), "evidence": ev,
                          "leaderboard_key": reg.save_text(html, f"{state['experiment_id']}/leaderboard.html")})
            reg.save_text(leaderboard.to_csv(lb), f"{state['experiment_id']}/leaderboard.csv")
            st = merge_update(st, extra)
            extra["model_card_key"] = reg.save_text(leaderboard.model_card(st), f"{state['experiment_id']}/model_card.md")
    except Exception as e:
        extra["warnings"] = [f"Leaderboard assembly failed ({type(e).__name__}: {str(e)[:120]})"]
    try:
        with viz.PLOT_LOCK:
            extra = merge_updates(extra, _post_model(st, reg))
    except Exception as e:
        extra = merge_updates(extra, {"warnings": [f"Post-model analysis partly skipped ({type(e).__name__}: {str(e)[:120]})"]})
    try:
        if st.get("automl_report"):
            extra = merge_updates(extra, {"charts": _automl_charts(st, reg)})
    except Exception as e:
        extra = merge_updates(extra, {"warnings": [f"AutoML charts skipped ({type(e).__name__}: {str(e)[:100]})"]})
    return merge_updates(upd, extra)


def _automl_charts(state, reg) -> List[Dict[str, Any]]:
    import matplotlib.pyplot as plt
    from .. import plotly_viz as pv
    ar, gr = state["automl_report"], (state["automl_report"].get("goal") or {})
    ts = [t for t in ar.get("trials", []) if t.get("mean") is not None and t.get("fidelity", 1) >= 0.999]
    g = viz.Gallery(reg, state["run_id"], "model")
    with viz.PLOT_LOCK:
        if ts:
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            for f in sorted({t["family"] for t in ts}):
                ax.scatter([t["id"] for t in ts if t["family"] == f], [t["mean"] for t in ts if t["family"] == f], s=22, label=f)
            ax.step([t["id"] for t in ts], np.maximum.accumulate([t["mean"] for t in ts]), where="post", color="black", lw=1.6, label="best so far")
            ax.set_xlabel("trial #")
            ax.set_ylabel(f"CV {ar.get('search_metric', '')}")
            ax.legend(fontsize=7, ncol=2, frameon=False)
            ax.set_title("AutoML search: every trial")
            g.add(fig, "automl_trials", "AutoML search: every trial", "Each dot is one configuration scored by cross-validation; the line is the best so far. A flat tail means more search would not have helped.", "modeling")
            g.attach("automl_trials", pv.automl_trials(ar["trials"], ar.get("search_metric", "")), "AutoML search: every trial", "Hover for the configuration.", "modeling")
        if gr.get("target") is not None:
            g.attach("goal_ladder", pv.goal_ladder(gr), "Goal: three looks at the same target", "Optimistic search estimate, fresh folds after the search, and the held-out score (once).", "modeling")
    return g.items


def _post_model(state, reg) -> Dict[str, Any]:
    """Text: mistakes, per-document word effects, stress test. Images: mistakes gallery, occlusion maps, robustness test. (Snippets stay out of the evidence.)"""
    from .. import cv, nlp
    from pathlib import Path
    mods = state.get("modality") or {}
    if not (mods.get("text") or mods.get("image")):
        return {}
    plan, pt = state["analysis_plan"], state["analysis_plan"]["problem_type"]
    splits = reg.load(state["splits_key"])
    est = reg.load(state["models_key"])["models"][state["best_model_name"]]
    X_te, y_te, classes = splits["X_test"], splits["y_test"], splits.get("classes")
    task = "classification" if pt == "classification" else "regression"
    g = viz.Gallery(reg, state["run_id"], "post")
    items: List[Tuple[str, str, Any]] = []
    out: Dict[str, Any] = {}
    if mods.get("text"):
        col = mods["text"][0]
        rep = dict(state.get("nlp_report") or {})
        if task == "classification":
            mis = nlp.misclassified(est, X_te, y_te, classes, col)
            for m in mis[:3]:
                try:
                    m["word_effects"] = nlp.explain_document(est, X_te.loc[[m["row"]]], col, classes)
                except Exception:
                    pass
            rep["misclassified"] = mis
            if mis:
                items.append(("nlp_errors", f"{len(mis)} held-out document(s) were misclassified with high confidence (top confidence {mis[0]['confidence']:.2f}); "
                                            "they are listed on the report page: they are often mislabelled, ambiguous or missing a signal.", {"n": len(mis)}))
        st = nlp.stress_test(est, X_te, y_te, col, task, seed=int(state.get("seed", 42)))
        rep["stress_test"] = st
        if st.get("worst"):
            items.append(("nlp_stress", f"Text stress test ({st['metric']}, clean {st['clean']}): the largest drop comes from '{st['worst']['perturbation']}' "
                                        f"(-{st['worst']['drop']:.3f}). " + ("The model is fragile to realistic text noise." if st["fragile"] else "No perturbation drops the score by more than 0.10."), st))
            g.add(cv.robustness_figure(st, "text"), "nlp_stress", "Text stress test", "Held-out score after degrading the text (typos, dropped words, shuffling ...).", "nlp")
        out["nlp_report"] = jsonable(rep)
    if mods.get("image"):
        col = mods["image"][0]
        root = Path(plan.get("image_root") or ".")
        rep = dict(state.get("vision_report") or {})
        if task == "classification":
            rows, fig = cv.misclassified_gallery(est, X_te, y_te, classes, col, root)
            rep["misclassified"] = rows
            if fig is not None:
                g.add(fig, "mistakes", "Most confident mistakes", "Held-out images predicted wrongly with the highest confidence (true → predicted).", "vision")
                items.append(("vision_errors", f"{len(rows)} held-out image(s) were misclassified with high confidence (top {rows[0]['confidence']:.2f}).", {"n": len(rows)}))
        occ = []
        pick = list(range(min(3, len(X_te))))
        for i in pick:
            try:
                r = cv.occlusion_map(est, X_te.iloc[[i]], col, root, classes if task == "classification" else None)
                g.add(r.pop("figure"), f"occlusion_{i}", f"Where the model looked (test image {i + 1})", "Occlusion sensitivity: red cells supported the prediction, blue cells argued against it.", "vision")
                occ.append({"row": int(X_te.index[i]), "label": r["label"], "score": r["score"]})
            except Exception:
                pass
        rep["occlusion_examples"] = occ
        rob = cv.robustness(est, X_te, y_te, col, root, task, seed=int(state.get("seed", 42)))
        rep["robustness"] = rob
        if rob.get("worst"):
            items.append(("vision_stress", f"Image robustness test ({rob['metric']}, clean {rob['clean']}): the largest drop comes from '{rob['worst']['perturbation']}' (-{rob['worst']['drop']:.3f}). "
                                           + ("The model is fragile to ordinary image changes: do not expect it to hold on photos taken differently from the training set." if rob["fragile"]
                                              else "No degradation drops the score by more than 0.10."), rob))
            g.add(cv.robustness_figure(rob), "robustness", "Image robustness test", "Held-out score after darkening, blurring, adding noise, rotating, compressing ...", "vision")
        out["vision_report"] = jsonable(rep)
    out["charts"] = g.items
    out["evidence"] = make_evidence("V", items)
    return out


def route_critic(state) -> str:
    return "retry" if state.get("retry_pending") else "done"


# synthesize + publish
def synthesize_node(state) -> Dict[str, Any]:
    """Charts and optional web context in parallel, then the business reasoning chain (root causes -> insights -> recommendations)."""
    upd = _run_subtasks("synthesize", state, [("visualize_eda", analysis.visualize_eda_node), ("context_research", context.context_research_node)])
    st = merge_update(state, upd)
    for nm, fn in (("root_cause", business.root_cause_node), ("insights", business.insights_node), ("recommendations", business.recommendations_node)):
        t0 = time.perf_counter()
        u = fn(st)
        u = merge_updates(u, {"timings": {f"synthesize.{nm}": {"seconds": round(time.perf_counter() - t0, 2)}}})
        upd = merge_updates(upd, u)
        st = merge_update(st, u)
    return upd


def publish_node(state) -> Dict[str, Any]:
    """Optional Power BI package, experiment tracking, report, final bookkeeping."""
    b = blackboard.get_board(state["run_id"])
    upd: Dict[str, Any] = {"board": b.export()}
    st = merge_update(state, upd)
    steps = ([("powerbi", reporter.powerbi_export_node)] if state.get("powerbi_enabled") else []) + [
        ("track", reporter.track_experiment_node), ("report", reporter.generate_report_node), ("finalize", reporter.finalize_node)]
    for nm, fn in steps:
        t0 = time.perf_counter()
        u = merge_updates(fn(st), {"timings": {f"publish.{nm}": {"seconds": round(time.perf_counter() - t0, 2)}}})
        upd = merge_updates(upd, u)
        st = merge_update(st, u)
    blackboard.release(state["run_id"])
    return upd
