"""Business-analyst reasoning agents.

  Root Cause Analyst - WHY is it happening?   (driver analysis evidence + LLM hypotheses)
  Insight Analyst    - WHAT does it mean?     (executive summary, key insights)
  Recommender        - WHAT should we do?     (prioritised, measurable actions)

Grounding contract: every claim must cite evidence ids (Q/S/K/T/D/M/C...). Claims that cite no
valid id are discarded. Drivers are always labelled associations to validate, never proven causes.
"""
from typing import Any, Dict, List


from .. import claims, config, rca
from . import bus
from ._util import ANALYST_PERSONA, ask, evidence_brief, jsonable, log, valid_ids
from .analysis import _features, _load

CAUSAL_NOTE = ("Hypothesis from observational data: it shows where a difference comes from, not that it causes it. "
               "Validate with a controlled test or domain review before acting.")
CONF = {"high", "medium", "low"}


# root cause
def root_cause_node(state) -> Dict[str, Any]:
    A = "Root Cause Analyst"
    df = _load(state)
    plan, prof = state["analysis_plan"], state["dataset_profile"]
    target, pt = plan.get("target"), plan["problem_type"]
    first_spec = None
    for r in (state.get("kpi_results") or {}).values():
        if r["spec"]["agg"] != "count":
            first_spec = r["spec"]
            break
    positive = None
    if target and pt == "classification":
        positive = (state.get("data_summary") or {}).get("positive_class") or df[target].astype(str).value_counts().index[-1]
    m = rca.measure_vector(df, target, pt, positive, first_spec)
    new_ev: List[Dict[str, Any]] = []
    tables: Dict[str, Any] = {}
    warns_extra: List[str] = []
    groups_out: List[Dict[str, Any]] = []
    if m is not None:
        cols = [c for c in dict.fromkeys(_features(state) + state.get("segment_columns", []))
                if c != plan.get("date_column") and c in prof["columns"] and prof["columns"][c]["kind"] != "datetime"]
        lifts = rca.segment_lifts(df, m, cols)
        change = rca.contribution_change(df, plan["date_column"], m, cols) if plan.get("date_column") else None
        expl = []
        tr = state.get("trend_results") or {}
        if plan.get("date_column") and tr.get("anomalies"):
            for a in tr["anomalies"][:3]:
                try:
                    e = rca.explain_anomaly(df, plan["date_column"], m, cols, a["period"], tr.get("freq_code", "D"))
                    if e:
                        expl.append(e)
                except Exception:
                    pass
        evs = rca.evidence_from(lifts, change, m, expl)
        inter, groups = {"items": [], "screened": 0}, []
        try:
            inter = rca.interaction_segments(df, m, cols)
            evs = evs + rca.interaction_evidence(inter, m, len(evs))
        except Exception as ex:
            warns_extra.append(f"Combined-segment search skipped ({type(ex).__name__}).")
        try:
            top_drv = [i["feature"] for i in ((state.get("explanations") or {}).get("importances") or []) if i["importance"] > 0][:10]
            groups = rca.driver_groups(df, top_drv) if len(top_drv) >= 2 else []
            for g in groups:
                evs.append({"id": f"D{len(evs) + 1:02d}", "source": "driver_groups",
                            "text": (f"Overlapping drivers: {', '.join(g['members'])} are strongly associated with each other, so they describe largely the same "
                                     "records and should be treated as ONE dimension; actions aimed at each separately would target overlapping people."),
                            "data": g})
        except Exception as ex:
            warns_extra.append(f"Driver grouping skipped ({type(ex).__name__}).")
        new_ev = [{"id": e["id"], "source": e["source"], "text": e["text"], "data": jsonable(e["data"])} for e in evs]
        tables = {"measure": m["name"], "lifts": jsonable(lifts), "change": jsonable(change), "anomalies": jsonable(expl),
                  "interactions": jsonable(inter), "driver_groups": jsonable(groups)}
    # ---- agent-to-agent: ask the Statistician to test top driver columns that no selected test covered (bounded, logged, deterministic)
    msgs: List[Dict[str, Any]] = []
    used = int(state.get("agent_requests_used", 0))
    req_ev: List[Dict[str, Any]] = []
    try:
        if m is not None and target:
            covered = set()
            for r in state.get("stat_results", []) or []:
                pr = r.get("params", {})
                covered.add(frozenset(str(v) for v in pr.values() if isinstance(v, str)))
            seen_cols = []
            for l in tables.get("lifts", []) if isinstance(tables, dict) else []:
                if l["column"] not in seen_cols and l["column"] != target:
                    seen_cols.append(l["column"])
            for col in seen_cols:
                if frozenset({col, target}) in covered:
                    continue
                mm, res, used = bus.request({**state, "messages": list(state.get("messages", [])) + msgs}, A, "Statistician",
                                            "statistician/test_association", {"a": col, "b": target}, used)
                msgs += mm
                if res is not None and len(req_ev) < config.MAX_AGENT_REQUESTS:
                    req_ev.append({"id": f"R{len(req_ev) + 1:02d}", "source": "agent_request",
                                   "text": f"[requested by {A}] " + res["summary"], "data": jsonable({"test_id": "REQ", "columns": [col, target]})})
                if used >= config.MAX_AGENT_REQUESTS:
                    break
    except Exception as ex:
        msgs.append({"id": "MSG_ERR", "from": A, "to": "Statistician", "kind": "note", "topic": "request_failed",
                     "payload": {"error": f"{type(ex).__name__}: {ex}"}, "ts": ""})
    new_ev = new_ev + req_ev
    all_ev = list(state.get("evidence", [])) + new_ev

    payload = {"business_question": state.get("business_question"), "analysis_type": state.get("analysis_type"),
               "problem_type": pt, "hypotheses_to_test": state.get("hypotheses", []), "evidence": evidence_brief(all_ev),
               "instructions": ["Explain WHY the pattern may be happening: group related evidence into 2-5 root-cause hypotheses.",
                                "Every hypothesis must cite 1+ evidence ids. Do not state numbers that are not in the evidence.",
                                "Say which hypotheses from hypotheses_to_test the evidence supports or does not support.",
                                "Use cautious language; confidence is high only with several consistent evidence items.",
                                "Evidence ids starting with W are EXTERNAL web context: unverified background, never proof; say so when you use them."],
               "output_schema": {"root_causes": [{"finding": "what the data shows", "hypothesis": "plausible business reason",
                                                  "evidence_ids": ["D01"], "confidence": "high|medium|low",
                                                  "validation_step": "how to confirm (test, data to collect, owner to ask)"}]}}
    known = [e["id"] for e in evidence_brief(all_ev)]
    gate_rc, allow_roi = claims.Gate(), (state.get("cost_fn") is not None and state.get("cost_fp") is not None)

    def v_rc(d):
        out = []
        for rc in d.get("root_causes") or []:
            if not isinstance(rc, dict) or not rc.get("finding"):
                continue
            ids = valid_ids(rc.get("evidence_ids"), all_ev)
            if not ids:
                continue                                  # ungrounded claim: discarded
            conf = rc.get("confidence") if rc.get("confidence") in CONF else "low"
            if len(ids) < 2 and conf == "high":
                conf = "medium"
            out.append({"finding": str(rc["finding"])[:300], "hypothesis": str(rc.get("hypothesis", ""))[:300], "evidence_ids": ids,
                        "confidence": conf, "validation_step": str(rc.get("validation_step", ""))[:250], "note": CAUSAL_NOTE})
        if not out:
            raise ValueError(f"every root cause must cite evidence ids that exist, e.g. {known[:8]}")
        bad = gate_rc.check([(str(i), o["evidence_ids"], o["finding"] + " " + o["hypothesis"]) for i, o in enumerate(out)], all_ev, allow_roi)
        for i, o in enumerate(out):
            if str(i) in bad:
                o["unverified"] = bad[str(i)]
        return out

    causes, warns, calls = ask(state, "root_cause_analysis", ANALYST_PERSONA, payload, 1600, v_rc)
    by = "LLM"
    return {"root_causes": causes, "evidence": new_ev, "warnings": warns + warns_extra, "llm_calls": calls,
            "rca_tables": tables, "messages": msgs, "agent_requests_used": used, "driver_groups": tables.get("driver_groups", []),
            "interaction_segments": (tables.get("interactions") or {}).get("items", []),
            "current_status": "Root-cause analysis done.", "active_agent": A,
            "agent_log": [log(A, f"{len(causes)} grounded root-cause hypothesis(es) from {len(all_ev)} evidence items (by {by}); {len(new_ev)} new driver evidence item(s).")]}


# insights
def insights_node(state) -> Dict[str, Any]:
    A = "Insight Analyst"
    ev = state.get("evidence", [])
    payload = {"business_question": state.get("business_question"), "analysis_type": state.get("analysis_type"),
               "problem_type": state["analysis_plan"]["problem_type"], "evidence": evidence_brief(ev, 55),
               "root_causes": [{"finding": r["finding"], "hypothesis": r["hypothesis"], "ids": r["evidence_ids"], "confidence": r["confidence"]}
                               for r in state.get("root_causes", [])],
               "model_quality_passed": state.get("quality_passed"),
               "instructions": ["Write for an executive: lead with the answer to the business question.",
                                "Each insight = what we see, why it matters (so_what), and cite evidence ids.",
                                "Include data caveats an executive must know (data quality, correlation vs causation, weak models).",
                                "For model performance quote ONLY the held-out 'Headline' evidence (with its interval); cross-validated training scores are not the headline.",
                                "Use only figures that appear in the evidence. Do not mention ROI, savings or guarantees."],
               "output_schema": {"headline": "one sentence answer", "executive_summary": "3-5 sentences",
                                 "key_insights": [{"title": "", "insight": "", "so_what": "", "evidence_ids": ["K01"]}],
                                 "caveats": ["..."]}}
    known = [e["id"] for e in evidence_brief(ev, 55)]
    gate_in, allow_roi = claims.Gate(), (state.get("cost_fn") is not None and state.get("cost_fp") is not None)

    def v_ins(d):
        key = []
        for k in d.get("key_insights") or []:
            if isinstance(k, dict) and valid_ids(k.get("evidence_ids"), ev) and k.get("insight"):
                key.append({"title": str(k.get("title", ""))[:100], "insight": str(k["insight"])[:400], "so_what": str(k.get("so_what", ""))[:300],
                            "evidence_ids": valid_ids(k.get("evidence_ids"), ev)})
        if not key or not str(d.get("executive_summary", "")).strip():
            raise ValueError(f"need an executive_summary and key_insights that cite existing evidence ids, e.g. {known[:8]}")
        res = {"headline": str(d.get("headline", ""))[:240], "executive_summary": str(d["executive_summary"])[:1200], "key_insights": key[:6],
               "caveats": [str(c)[:240] for c in (d.get("caveats") or [])][:5], "source": "LLM"}
        all_ids = sorted({i for k in res["key_insights"] for i in k["evidence_ids"]})
        items = [("headline", all_ids, res["headline"]), ("summary", all_ids, res["executive_summary"])]
        items += [(f"insight{i}", k["evidence_ids"], k["insight"] + " " + k["so_what"]) for i, k in enumerate(res["key_insights"])]
        bad = gate_in.check(items, ev, allow_roi)
        for i, k in enumerate(res["key_insights"]):
            if f"insight{i}" in bad:
                k["unverified"] = bad[f"insight{i}"]
        res["unverified_summary"] = bad.get("headline", []) + bad.get("summary", [])
        return res

    ins, warns, calls = ask(state, "business_insights", ANALYST_PERSONA, payload, 1600, v_ins)
    q = next((e for e in ev if e["source"] == "data_quality"), None)
    if q and q["data"].get("score", 100) < 80:
        ins["caveats"].append(f"Data quality score is {q['data']['score']}/100: findings may be affected by missing/duplicate/outlier data.")
    if state.get("audit") and not state["audit"]["passed"]:
        codes = {i["code"] for i in state["audit"]["issues"] if i["severity"] == "blocker"}
        if codes - {"target_not_met"}:
            ins["caveats"].append("The predictive model did NOT clearly beat the baseline; do not rely on its predictions for decisions.")
        if "target_not_met" in codes:
            ins["caveats"].append("The requested performance target was not met; see the target report for the achieved held-out score and why the search stopped.")
    return {"insights": ins, "narrative": ins["executive_summary"], "warnings": warns, "llm_calls": calls,
            "current_status": "Insights written.", "active_agent": A,
            "agent_log": [log(A, f"{len(ins['key_insights'])} insight(s) written ({ins['source']}).")]}


# recommendations
def recommendations_node(state) -> Dict[str, Any]:
    A = "Recommender"
    ev = state.get("evidence", [])
    payload = {"business_question": state.get("business_question"), "analysis_type": state.get("analysis_type"),
               "insights": [{"title": k["title"], "insight": k["insight"], "ids": k["evidence_ids"]} for k in state["insights"]["key_insights"]],
               "root_causes": [{"finding": r["finding"], "hypothesis": r["hypothesis"], "ids": r["evidence_ids"], "confidence": r["confidence"]}
                               for r in state.get("root_causes", [])],
               "evidence": evidence_brief(ev, 35),
               "instructions": ["Give 3-6 SPECIFIC actions a business owner could start next week (who/what/which segment).",
                                "Each action must cite evidence ids and name the KPI that will show whether it worked.",
                                "Because evidence is observational, favour pilots / A-B tests for causal claims.",
                                "Order by priority (impact x confidence / effort). State the risks plainly.",
                                "Do not mention ROI, savings, payback or guarantees, and do not invent figures; describe impact qualitatively."],
               "output_schema": {"recommendations": [{"action": "", "rationale": "", "evidence_ids": ["D01"], "priority": "high|medium|low",
                                                      "effort": "low|medium|high", "expected_impact": "qualitative; no invented numbers",
                                                      "kpi_to_track": "", "risk": "", "owner": "suggested function"}]}}
    known = [e["id"] for e in evidence_brief(ev, 35)]
    gate_rec, allow_roi = claims.Gate(), (state.get("cost_fn") is not None and state.get("cost_fp") is not None)

    def v_rec(d):
        out = []
        for r in d.get("recommendations") or []:
            if not isinstance(r, dict) or not r.get("action"):
                continue
            ids = valid_ids(r.get("evidence_ids"), ev)
            if not ids:
                continue
            out.append({"action": str(r["action"])[:300], "rationale": str(r.get("rationale", ""))[:400], "evidence_ids": ids,
                        "priority": r.get("priority") if r.get("priority") in CONF else "medium",
                        "effort": r.get("effort") if r.get("effort") in CONF else "medium",
                        "expected_impact": str(r.get("expected_impact", ""))[:240], "kpi_to_track": str(r.get("kpi_to_track", ""))[:120],
                        "risk": str(r.get("risk", ""))[:240], "owner": str(r.get("owner", ""))[:60]})
        if not out:
            raise ValueError(f"give 3-6 actions, each citing existing evidence ids, e.g. {known[:8]}")
        bad = gate_rec.check([(str(i), o["evidence_ids"], o["action"] + " " + o["rationale"] + " " + o["expected_impact"]) for i, o in enumerate(out)], ev, allow_roi)
        for i, o in enumerate(out):
            if str(i) in bad:
                o["unverified"] = bad[str(i)]
        return out

    recs, warns, calls = ask(state, "recommendations", ANALYST_PERSONA, payload, 1800, v_rec)
    by = "LLM"
    order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: order[r["priority"]])
    return {"recommendations": recs[:7], "warnings": warns, "llm_calls": calls, "current_status": "Recommendations ready.", "active_agent": A,
            "agent_log": [log(A, f"{len(recs[:7])} grounded recommendation(s) ({by}).")]}
