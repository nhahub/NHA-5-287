"""What the assistant may know about a finished analysis, in bounded, aggregate-only pieces.

Each part answers one kind of question ("how good is the model?", "what did the statistics find?", "was anything suspicious?", "what does the report say about X?").
Nothing here returns raw rows. Text is cut to a fixed length, lists are capped, and every figure comes from the saved analysis, so the assistant can quote it
and the figure check in agent.py can verify it.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PARTS = ["summary", "overview", "metrics", "explanation", "trials", "statistics", "kpis", "trends", "findings", "evidence", "audit", "decisions", "charts", "report", "notes"]
MAX_TEXT = 3500


def _r(x: Any, nd: int = 4) -> Any:
    try:
        return round(float(x), nd) if x is not None and x == x else None
    except (TypeError, ValueError):
        return None


def _cut(x: Any, n: int = 260) -> str:
    return str(x or "").replace("\n", " ").strip()[:n]


def _primary(state: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    pm = state.get("primary_metric") or ""
    best = next((c for c in state.get("candidate_metrics") or [] if c.get("name") == state.get("best_model_name")), None)
    return pm, best


def overview(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    plan, ci, gr = state.get("analysis_plan") or {}, state.get("metric_ci") or {}, (state.get("automl_report") or {}).get("goal") or {}
    pm, best = _primary(state)
    ins = state.get("insights") or {}
    out = {"analysis_id": state.get("experiment_id"), "request": _cut(state.get("user_request"), 300), "business_question": _cut(state.get("business_question"), 300),
           "business_problem": state.get("analysis_type"), "analytical_problem": plan.get("problem_type"), "target": plan.get("target"), "date_column": plan.get("date_column"),
           "rows": (state.get("dataset_profile") or {}).get("n_rows"), "columns": (state.get("dataset_profile") or {}).get("n_cols"),
           "dropped_columns": (plan.get("drop_columns") or [])[:12], "phases_skipped": state.get("skip_phases") or [],
           "model": state.get("best_model_name"), "primary_metric": pm or None,
           "held_out_score": ci.get("value"), "held_out_interval": [ci.get("lo"), ci.get("hi")] if ci.get("lo") is not None else None,
           "passed_review": state.get("quality_passed"), "goal_status": gr.get("status"), "goal_text": _cut(gr.get("text"), 200) if gr else None,
           "headline": _cut(ins.get("headline"), 200), "summary": _cut(ins.get("executive_summary"), 600),
           "built_on": state.get("parent_experiment_id"), "n_charts": len(state.get("charts") or []), "n_evidence": len(state.get("evidence") or []),
           "warnings": [_cut(w, 160) for w in (state.get("warnings") or [])[:5]]}
    r = state.get("resume") or {}
    if r.get("parent"):
        out["reused_from_parent"] = (["data profile"] if r.get("profile_reused") else []) + [p for p in r.get("requested", []) if p != "profile"]
    return out, None


def summary(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """Everything needed to summarise an analysis in plain words, in one call: what was asked, what was found, how far to trust it, what to do next."""
    ov, _ = overview(state)
    ins = state.get("insights") or {}
    a = state.get("audit") or {}
    plan = state.get("analysis_plan") or {}
    out: Dict[str, Any] = {"asked": ov["request"], "business_question": ov["business_question"], "data": {"rows": ov["rows"], "columns": ov["columns"], "dropped_columns": ov["dropped_columns"]},
                           "approach": {"kind": ov["analytical_problem"], "target": ov["target"], "model": ov["model"], "phases_skipped": ov["phases_skipped"]},
                           "headline": ov["headline"], "summary": ov["summary"],
                           "key_findings": [{"title": _cut(i.get("title"), 100), "finding": _cut(i.get("insight"), 220)} for i in (ins.get("key_insights") or [])[:4]],
                           "recommendations": [{"action": _cut(r.get("action"), 180), "priority": r.get("priority")} for r in (state.get("recommendations") or [])[:4]],
                           "caveats": [_cut(c, 180) for c in (ins.get("caveats") or [])[:3]]}
    if ov["model"]:
        out["result"] = {"metric": ov["primary_metric"], "held_out_score": ov["held_out_score"], "interval": ov["held_out_interval"], "goal": ov["goal_status"], "goal_text": ov["goal_text"]}
        out["reliability"] = {"passed_review": ov["passed_review"], "review_issues": [_cut(i if isinstance(i, str) else (i.get("message") if isinstance(i, dict) else i), 160) for i in (a.get("issues") or [])[:4]],
                              "warnings": ov["warnings"][:3]}
    else:
        out["result"] = {"note": "No predictive model was built in this analysis (exploration and description only)."}
    if ov.get("built_on"):
        out["built_on_earlier_analysis"] = ov["built_on"]
    nn = notes(state)[0]
    if nn["count"]:
        out["user_notes"] = nn["notes"][-3:]
    return out, None


def metrics(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    pm, best = _primary(state)
    ci, lb = state.get("metric_ci") or {}, state.get("leaderboard") or {}
    if not state.get("candidate_metrics"):
        return {"modelled": False, "note": "This analysis did not train a model, so there are no model metrics."}, None
    rows = []
    for c in state["candidate_metrics"][:10]:
        row = {"model": c.get("name"), "selection_score": _r(c.get("select_score"))}
        for k in (pm, "accuracy", "f1_macro", "roc_auc", "r2", "rmse", "mae"):
            if k and c.get(k) is not None and k not in row:
                row[k] = _r(c.get(k))
        rows.append(row)
    sd = state.get("seed_stability") or {}
    summary = {"modelled": True, "primary_metric": pm, "winner": state.get("best_model_name"), "held_out": {"value": ci.get("value"), "low": ci.get("lo"), "high": ci.get("hi"), "test_rows": ci.get("n_test")},
               "baseline": lb.get("baseline"), "models_compared": len(state["candidate_metrics"]), "tied_with_winner": (lb.get("tied_with_winner") or [])[:5], "leaderboard_summary": _cut(lb.get("summary"), 500),
               "goal": {k: v for k, v in ((state.get("automl_report") or {}).get("goal") or {}).items() if k in ("status", "target", "text")},
               "seed_stability": {"mean": sd.get("mean"), "sd": sd.get("sd")} if sd else None, "data_split": {k: v for k, v in (state.get("data_summary") or {}).items() if k in ("split", "n_train", "n_test", "n_features", "minority_share")}}
    return summary, pd.DataFrame(rows)


def explanation(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    ex = state.get("explanations") or {}
    imps = ex.get("importances") or []
    if not imps:
        return {"available": False, "note": "No feature importances were computed for this analysis."}, None
    diag = ex.get("diagnostics") or {}
    summary = {"available": True, "model": state.get("best_model_name"), "top_features": [{"feature": i["feature"], "importance": _r(i["importance"]), "std": _r(i.get("std"))} for i in imps[:8]],
               "per_class": diag.get("per_class"), "confusion_matrix": diag.get("confusion_matrix"), "classes": diag.get("classes"),
               "reading": "Importance is how much the held-out score drops when a feature is shuffled. It shows what the model uses, not what causes the outcome."}
    return summary, pd.DataFrame([{"feature": i["feature"], "importance": _r(i["importance"]), "std": _r(i.get("std"))} for i in imps[:15]])


def trials(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    ar = state.get("automl_report") or {}
    if not ar:
        return {"available": False, "note": "No model search was run in this analysis."}, None
    ts = [t for t in ar.get("trials", []) if t.get("mean") is not None]
    by: Dict[str, List[float]] = {}
    for t in ts:
        by.setdefault(str(t.get("family")), []).append(float(t["mean"]))
    fam = [{"family": f, "trials": len(v), "best_cv_score": _r(max(v)), "mean_cv_score": _r(sum(v) / len(v))} for f, v in sorted(by.items(), key=lambda kv: -max(kv[1]))]
    summary = {"available": True, "trials": ar.get("n_trials"), "rounds": len(ar.get("rounds") or []), "seconds": _r(ar.get("seconds"), 1), "workers": ar.get("workers"), "search_metric": ar.get("search_metric"),
               "stopped_because": _cut(ar.get("stop_text"), 240), "noise_ceiling": ar.get("ceiling"), "dropped_columns": (ar.get("dropped_columns") or [])[:10], "ensemble": _cut(ar.get("ensemble"), 200) if ar.get("ensemble") else None,
               "protocol": _cut(ar.get("protocol"), 240)}
    return summary, pd.DataFrame(fam)


def statistics(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    sr = state.get("stat_results") or []
    if not sr:
        return {"available": False, "note": "No statistical tests were run in this analysis."}, None
    rows = [{"test": r.get("method"), "question": _cut(r.get("why"), 120), "adjusted_p": _r(r.get("p_adjusted"), 4), "effect": r.get("effect_label"), "significant": bool(r.get("significant")),
             "finding": _cut(r.get("interpretation"), 220)} for r in sr[:15]]
    return {"available": True, "tests_run": len(sr), "significant": sum(1 for r in sr if r.get("significant")), "tests_not_applicable": len(state.get("stat_skipped") or []),
            "findings": [r["finding"] for r in rows[:6]], "reading": "p-values are corrected for the number of tests; a small effect can be significant in large data."}, pd.DataFrame(rows)


def kpis(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    kr = state.get("kpi_results") or {}
    if not kr:
        return {"available": False, "note": "No KPIs were computed in this analysis."}, None
    rows, items = [], []
    for name, r in list(kr.items())[:8]:
        seg = next(iter((r.get("segments") or {}).items()), None)
        best = worst = None
        if seg and len(seg[1]) >= 2:
            best, worst = seg[1][0], seg[1][-1]
        rows.append({"kpi": name, "overall": r.get("overall_fmt"), "segment_column": seg[0] if seg else None, "highest_segment": best and best["segment"], "highest_value": best and best["value_fmt"],
                     "lowest_segment": worst and worst["segment"], "lowest_value": worst and worst["value_fmt"]})
        items.append({"kpi": name, "overall": r.get("overall_fmt")})
    return {"available": True, "kpis": items, "details": rows[:6]}, pd.DataFrame(rows)


def trends(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    tr, ol = state.get("trend_results") or {}, state.get("outlier_results") or {}
    if not tr and not ol:
        return {"available": False, "note": "No trend or outlier analysis was run in this analysis."}, None
    out: Dict[str, Any] = {"available": True}
    if tr.get("skipped"):
        out["trend"] = tr["skipped"]
    else:
        out["trend"] = {k: tr.get(k) for k in ("trend", "n_anomalies", "seasonality", "change_pct") if tr.get(k) is not None}
    cols = ol.get("columns") or []
    out["outlier_columns"] = [{"column": c["column"], "share_pct": c.get("pct")} for c in cols[:6]]
    return out, pd.DataFrame([{"column": c["column"], "outliers": c.get("n_outliers"), "share_pct": c.get("pct")} for c in cols[:10]]) if cols else None


def findings(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    ins = state.get("insights") or {}
    return {"headline": _cut(ins.get("headline"), 240), "summary": _cut(ins.get("executive_summary"), 700),
            "key_insights": [{"title": _cut(i.get("title"), 100), "insight": _cut(i.get("insight"), 240), "so_what": _cut(i.get("so_what"), 200), "evidence": i.get("evidence_ids")} for i in (ins.get("key_insights") or [])[:6]],
            "root_causes": [{"finding": _cut(r.get("finding"), 200), "hypothesis": _cut(r.get("hypothesis"), 200), "confidence": r.get("confidence")} for r in (state.get("root_causes") or [])[:5]],
            "recommendations": [{"action": _cut(r.get("action"), 200), "priority": r.get("priority"), "kpi_to_track": _cut(r.get("kpi_to_track"), 80)} for r in (state.get("recommendations") or [])[:6]],
            "caveats": [_cut(c, 200) for c in (ins.get("caveats") or [])[:5]]}, None


def evidence(state: Dict[str, Any], query: str = "") -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    ev = state.get("evidence") or []
    q = (query or "").strip().lower()
    if q:
        hit = [e for e in ev if q in e["text"].lower() or e["id"].lower() == q or q in str(e.get("source", "")).lower()]
    else:
        hit = ev
    rows = [{"id": e["id"], "source": e.get("source"), "fact": _cut(e["text"], 300)} for e in hit[:15]]
    return {"matching": len(hit), "of": len(ev), "facts": rows, "note": "Evidence ids are the numbered facts the analysis and its report are built on."}, pd.DataFrame(rows) if rows else None


def audit(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    a, sen, fair = state.get("audit") or {}, state.get("sentinel_report") or {}, state.get("fairness") or {}
    leaks = [{"column": s.get("column"), "severity": s.get("severity"), "score": _r(s.get("score")), "why": _cut(s.get("reason") or s.get("why"), 160)} for s in (sen.get("leakage_suspects") or [])[:6]]
    out = {"passed_review": state.get("quality_passed"), "review_issues": [_cut(i if isinstance(i, str) else (i.get("message") if isinstance(i, dict) else i), 220) for i in (a.get("issues") or [])[:8]],
           "gain_over_baseline_cv": a.get("gain_cv"), "gain_over_baseline_test": a.get("gain_test"), "leakage_suspects": leaks, "columns_quarantined": (state.get("quarantined") or [])[:10],
           "id_like_columns": (sen.get("id_like") or [])[:6], "drift": _cut(sen.get("drift"), 200) if sen.get("drift") else None, "label_conflicts": sen.get("conflicts"),
           "sensitive_columns": state.get("sensitive_columns") or [], "fairness_groups_checked": len(fair.get("groups") or []), "warnings": [_cut(w, 200) for w in (state.get("warnings") or [])[:8]],
           "stage_errors": [_cut(f"{e.get('stage')}: {e.get('error')}", 200) for e in (state.get("stage_errors") or [])[:5]]}
    return out, pd.DataFrame(leaks) if leaks else None


def decisions(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    ds = state.get("decisions") or []
    rows = [{"decision": d.get("junction"), "by": d.get("source"), "choice": _cut(d.get("choice"), 120), "why": _cut(d.get("rationale"), 200)} for d in ds[:15]]
    return {"decisions": len(ds), "board_messages": len(state.get("board") or []), "crew": state.get("crew"), "shown": rows[:8]}, pd.DataFrame(rows) if rows else None


def charts(state: Dict[str, Any], query: str = "") -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    cs = state.get("charts") or []
    q = (query or "").strip().lower()
    hit = [c for c in cs if not q or q in (c.get("title") or "").lower() or q in (c.get("id") or "").lower() or q in (c.get("section") or "").lower()]
    rows = [{"id": c.get("id"), "title": c.get("title"), "section": c.get("section"), "what_it_shows": _cut(c.get("caption"), 200)} for c in hit[:25]]
    return {"charts": len(cs), "matching": len(hit), "list": rows, "note": "Name a chart to have it shown to the person."}, pd.DataFrame(rows) if rows else None


def _sections(md: str) -> List[Tuple[str, str]]:
    parts = re.split(r"(?m)^(#{1,3})\s+(.+)$", md or "")
    out = [("Start", parts[0])] if parts and parts[0].strip() else []
    for i in range(1, len(parts) - 2, 3):
        out.append((parts[i + 1].strip(), parts[i + 2]))
    return out


def report(state: Dict[str, Any], query: str = "") -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    secs = _sections(state.get("report_markdown") or "")
    if not secs:
        return {"available": False, "note": "No written report is stored for this analysis."}, None
    q = (query or "").strip().lower()
    if not q:
        return {"available": True, "sections": [h for h, _ in secs][:40], "note": "Ask for a section by name to read it."}, None
    hit = [(h, b) for h, b in secs if q in h.lower()] or [(h, b) for h, b in secs if q in b.lower()]
    if not hit:
        return {"available": True, "matching_sections": 0, "sections": [h for h, _ in secs][:40]}, None
    text, used = "", []
    for h, b in hit[:3]:
        if len(text) > MAX_TEXT:
            break
        text += f"## {h}\n{b.strip()}\n\n"
        used.append(h)
    return {"available": True, "sections_shown": used, "text": text[:MAX_TEXT], "truncated": len(text) > MAX_TEXT}, None


def notes(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    from .. import runstore
    ns = runstore.list_notes(state.get("experiment_id") or "")
    return {"notes": [{"kind": n["kind"], "text": _cut(n["text"], 300)} for n in ns[-10:]], "count": len(ns)}, None


def part(state: Dict[str, Any], name: str, query: str = "") -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    fn = {"summary": summary, "overview": overview, "metrics": metrics, "explanation": explanation, "trials": trials, "statistics": statistics, "kpis": kpis, "trends": trends, "findings": findings,
          "audit": audit, "decisions": decisions, "notes": notes}.get(name)
    if fn:
        return fn(state)
    return {"evidence": evidence, "charts": charts, "report": report}[name](state, query)
