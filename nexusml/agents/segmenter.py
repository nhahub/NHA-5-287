"""Segmenter (clustering problem_type): k tuned by silhouette, LLM names the segments."""
from typing import Any, Dict

from .. import config, interactive, modeling, viz
from ..storage import get_registry
from ._util import ANALYST_PERSONA, ask, jsonable, log, make_evidence

AGENT = "Segmenter"


def cluster_segments_node(state) -> Dict[str, Any]:
    reg = get_registry()
    splits = reg.load(state["splits_key"])
    cr = modeling.cluster_segments(splits["X_train"])
    labels, pipe = cr.pop("labels"), cr.pop("pipeline")
    # LLM turns statistical profiles into business personas (names/descriptions only)
    payload = {"business_question": state.get("business_question"), "analysis_type": state.get("analysis_type"),
               "segments": [{"cluster": p["cluster"], "share": p["share"], "defining_features": p["defining_features"],
                             "top_categories": p["top_categories"]} for p in cr["profiles"]],
               "output_schema": {"segments": [{"cluster": 0, "name": "2-4 word business name", "description": "one sentence: who they are and how they differ"}]}}
    want = {p["cluster"] for p in cr["profiles"]}

    def v_names(d):
        got = {x.get("cluster"): x for x in (d.get("segments") or []) if isinstance(x, dict) and x.get("name")}
        if not want <= set(got):
            raise ValueError(f"name every cluster id in {sorted(want)}")
        return got

    names, warns, calls = ask(state, "name_segments", ANALYST_PERSONA + " Name and describe customer/record segments from their statistical profile.",
                              payload, 900, v_names)
    items = []
    for p in cr["profiles"]:
        nm = names[p["cluster"]]
        p["name"] = str(nm.get("name") or f"Segment {p['cluster']}")[:40]
        p["description"] = str(nm.get("description") or "")[:240]
        feats = ", ".join(f"{f['feature']} ({f['z_vs_overall']:+.1f} sd)" for f in p["defining_features"][:3])
        items.append(("segmentation", f"Segment '{p['name']}' ({p['share']:.0%} of records) is distinguished by {feats}."
                                      + (f" {p['description']}" if p["description"] else ""), p))
    quality = "clear" if cr["silhouette"] >= config.MIN_SILHOUETTE else "weak"
    warn = [] if quality == "clear" else [f"Segment structure is weak (silhouette {cr['silhouette']} < {config.MIN_SILHOUETTE}); treat segments as descriptive, not distinct groups."]
    items.insert(0, ("segmentation", f"{cr['k']} segments chosen by silhouette search (silhouette {cr['silhouette']}, {quality} separation).", {}))
    key = reg.save({"models": {"KMeans": pipe}, "labels": labels}, "models")
    g = viz.Gallery(reg, state["run_id"], "cluster")
    try:
        viz.cluster_charts(g, cr)
        interactive.cluster(g, cr)
    except Exception as e:
        warn.append(f"Cluster charts failed: {e}")
    cr["quality"] = quality
    return {"cluster_results": jsonable(cr), "models_key": key, "best_model_name": "KMeans", "charts": g.items,
            "evidence": make_evidence("C", items), "warnings": warns + warn, "llm_calls": calls, "quality_passed": quality == "clear",
            "current_status": "Segments built.", "active_agent": AGENT,
            "agent_log": [log(AGENT, f"k={cr['k']} (silhouette {cr['silhouette']}, {quality}); segments named by the LLM.")]}
