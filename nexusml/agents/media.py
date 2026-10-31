"""Media analysis agent: profiles image / document collections (and free-text or image-path columns in any table)."""
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .. import cv, mediamining, nlp, viz
from ..storage import get_registry
from ._util import ANALYST_PERSONA, ask, jsonable, log, make_evidence

AGENT = "Media Analyst"


def _redact_terms(items):
    """Privacy: the evidence register is what the narrative LLM reads. Unless the user opted in (allow_llm_text), items that carry document TERMS
    (keywords, theme words, class-distinguishing words) are reduced to counts, in the text AND in the data field. The full terms stay in the report and the app."""
    out = []
    for s, t, d in items:
        if s == "media_keywords":
            n = len((d or {}).get("terms", []))
            out.append((s, f"{n} characteristic terms were extracted (shown in the report and the app; not sent to the AI unless you opt in).", {"n_terms": n}))
        elif s == "media_themes":
            th = (d or {}).get("themes", [])
            m = re.search(r"(The themes [^()]*\(ARI [-0-9.]+\)[^.]*\.)", t)
            sil = (d or {}).get("silhouette")
            out.append((s, f"{len(th)} natural themes" + (f" (silhouette {sil:.2f})" if sil is not None else "") + f" with {', '.join(str(x['size']) for x in th)} documents each." + (" " + m.group(1) if m else ""),
                        {"sizes": [x["size"] for x in th], "silhouette": sil}))
        elif s == "media_class_terms":
            n = t.count(";") + 1
            out.append((s, f"Class-distinguishing terms were computed for {n} classes (shown in the report and the app; not sent to the AI unless you opt in).", {"n_classes": n}))
        else:
            out.append((s, t, d))
    return out


def _topic_namer(state, calls: List[Dict[str, Any]]):
    """Optional (opt-in): the LLM only receives the top TERMS of each theme (never raw document text) and returns short names."""
    if not state.get("allow_llm_text"):
        return None

    def namer(tops):
        payload = {"themes": [{"theme": i, "top_terms": t} for i, t in enumerate(tops)],
                   "output_schema": {"names": [{"theme": 0, "name": "2-4 word topic name"}]}}

        def v(d):
            got = {int(x["theme"]): str(x["name"])[:40] for x in (d.get("names") or []) if isinstance(x, dict) and "theme" in x and x.get("name")}
            if len(got) < len(tops):
                raise ValueError("name every theme")
            return got
        try:
            out, _, rec = ask(state, "name_doc_topics", ANALYST_PERSONA + " You name document themes from their top terms.", payload, 500, v)
            calls.extend(rec)
            return out
        except Exception:
            return None
    return namer


def media_analysis_node(state) -> Dict[str, Any]:
    mods = state.get("modality") or {}
    if not (mods.get("text") or mods.get("image")):
        return {}                                                             # nothing to do for plain tabular data
    reg = get_registry()
    df = pd.read_csv(state["dataset_path"])
    plan = state["analysis_plan"]
    root = Path(plan.get("image_root") or Path(state["dataset_path"]).parent)
    target = plan.get("target")
    label = target if (plan.get("problem_type") == "classification" and target in df.columns) else None
    g = viz.Gallery(reg, state["run_id"], "media")
    items: List[Any] = []
    extra: Dict[str, Any] = {}
    calls: List[Dict[str, Any]] = []
    warns: List[str] = []
    vis_rep: Optional[Dict[str, Any]] = None
    nlp_rep: Optional[Dict[str, Any]] = None
    for col in mods.get("image", [])[:1]:
        it, figs = mediamining.image_insights(df, col, root, label)
        items += it
        for name, title, cap, fig in figs:
            g.add(fig, name, title, cap, "media")
        try:                                                                   # v5: the vision-model view (features, label issues, outliers, feature map)
            it2, figs2, vrep = cv.analyze(df, col, root, label, seed=int(state.get("seed", 42)))
            items += it2
            for name, title, cap, fig in figs2:
                g.add(fig, name, title, cap, "vision")
            vis_rep = jsonable(vrep)
        except Exception as e:
            warns.append(f"Vision analysis partly skipped ({type(e).__name__}: {str(e)[:100]})")
    for col in mods.get("text", [])[:1]:
        name_col = next((c for c in ("document", "file", "filename", "name") if c in df.columns), None)
        it, figs, extra = mediamining.document_insights(df, col, label, name_col, _topic_namer(state, calls))
        items += it
        for name, title, cap, fig in figs:
            g.add(fig, name, title, cap, "media")
        try:                                                                   # v5: the full text view (topics, class terms, label issues, PII, sentiment ...)
            it2, figs2, nrep = nlp.analyze(df, col, label, allow_terms=bool(state.get("allow_llm_text")), seed=int(state.get("seed", 42)))
            items += it2
            for name, title, cap, fig in figs2:
                g.add(fig, name, title, cap, "nlp")
            nlp_rep = jsonable(nrep)
        except Exception as e:
            warns.append(f"NLP analysis partly skipped ({type(e).__name__}: {str(e)[:100]})")
    try:                                                                       # what ingestion could not use
        import json
        mf = json.loads((root / "media_manifest.json").read_text())
        if mf.get("unreadable"):
            items.insert(1, ("media_unreadable", f"{len(mf['unreadable'])} file(s) could not be read and were excluded: {', '.join(mf['unreadable'][:4])}"
                                                  + (f" (+{len(mf['unreadable']) - 4} more)." if len(mf["unreadable"]) > 4 else "."), {"files": mf["unreadable"][:50]}))
        if mf.get("text_problems"):
            items.append(("media_text_problems", f"{len(mf['text_problems'])} document(s) had extraction problems, e.g. {mf['text_problems'][0]}", {"problems": mf["text_problems"][:20]}))
        if mf.get("set_aside"):
            items.append(("media_set_aside", f"The upload also contained {mf['set_aside']} that were set aside; this run analyses the larger group ({mf['kind']}).", mf["set_aside"]))
    except Exception:
        pass
    if not state.get("allow_llm_text"):
        items = _redact_terms(items)
    ev = make_evidence("X", [(k, t, d) for k, t, d in items])
    extra = dict(extra or {})
    upd: Dict[str, Any] = {**({"nlp_report": nlp_rep} if nlp_rep else {}), **({"vision_report": vis_rep} if vis_rep else {})}
    return {**upd, "evidence": ev, "charts": g.items, "media_report": jsonable({"images": mods.get("image"), "text": mods.get("text"), **extra}),
            "llm_calls": calls, "warnings": warns, "current_status": "Media analysis complete.", "active_agent": AGENT,
            "agent_log": [log(AGENT, f"Analysed {len(df):,} items: {len(ev)} findings, {len(g.items)} chart(s).")]}
