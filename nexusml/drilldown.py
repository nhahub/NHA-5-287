"""Targeted follow-up analyses ("re-run just this"). A whitelist of deterministic actions, executed on the dataset of a finished
run; each produces a new evidence item (prefix F) that later answers can cite. The LLM only CHOOSES among actions; code validates
the columns and runs them. At most MAX_ACTIONS per question."""
import re
from typing import Any, Dict, List

import pandas as pd

from . import rca
from .agents import bus

MAX_ACTIONS = 3
ACTIONS = {
    "segment_profile": {"column": "a column to break the outcome down by"},
    "test": {"a": "column", "b": "column"},
    "combinations": {"columns": ["2-6 columns to search for combined segments"]},
}
HINT = re.compile(r"\b(break ?down|by |split|compare|versus|vs\.?|test|combination|combine|drill|re-?run|between|segment|interact\w*|differ\w*)\b", re.I)


def wants_drilldown(question: str) -> bool:
    return bool(HINT.search(question or ""))


def _measure(state, df):
    plan = state.get("analysis_plan") or {}
    target, pt = plan.get("target"), plan.get("problem_type")
    first = next((r["spec"] for r in (state.get("kpi_results") or {}).values() if r["spec"]["agg"] != "count"), None)
    positive = (state.get("data_summary") or {}).get("positive_class")
    if target and pt == "classification" and positive is None and target in df.columns:
        positive = df[target].astype(str).value_counts().index[-1]
    return rca.measure_vector(df, target, pt, positive, first), target


def validate_actions(raw: Any, columns: List[str]) -> List[Dict[str, Any]]:
    ok = []
    for a in (raw or [])[:MAX_ACTIONS]:
        if not isinstance(a, dict) or a.get("action") not in ACTIONS:
            continue
        act = a["action"]
        if act == "segment_profile" and a.get("column") in columns:
            ok.append({"action": act, "column": a["column"]})
        elif act == "test" and a.get("a") in columns and a.get("b") in columns and a["a"] != a["b"]:
            ok.append({"action": act, "a": a["a"], "b": a["b"]})
        elif act == "combinations":
            cols = [c for c in (a.get("columns") or []) if c in columns][:6]
            if len(cols) >= 2:
                ok.append({"action": act, "columns": cols})
    return ok


def run(state: Dict[str, Any], actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Execute validated actions; returns new evidence dicts (ids F01, F02, ... continuing any earlier follow-up evidence)."""
    df = pd.read_csv(state["dataset_path"])
    start = sum(1 for e in state.get("evidence", []) if e["id"].startswith("F"))
    m, target = _measure(state, df)
    out: List[Dict[str, Any]] = []

    def add(text, data):
        out.append({"id": f"F{start + len(out) + 1:02d}", "source": "followup_analysis", "text": text, "data": data})
    for a in actions:
        try:
            if a["action"] == "segment_profile" and m is not None:
                seg = rca._segment_frame(df, [a["column"]]).get(a["column"])
                if seg is None:
                    add(f"Could not segment by '{a['column']}' (needs 2-20 distinct values or a numeric column).", {})
                    continue
                y = m["y"]
                rows = [(str(lv), int(((seg == lv) & y.notna()).sum()), float(y[(seg == lv)].mean())) for lv in seg.dropna().unique()]
                rows = [r for r in rows if r[1] >= 10]
                rows.sort(key=lambda r: -r[2])
                txt = "; ".join(f"{lv}: {rca._f(v, m['fmt'])} (n={n:,})" for lv, n, v in rows[:8])
                add(f"{m['name']} by {a['column']}: {txt}.", {"rows": rows[:12]})
            elif a["action"] == "test":
                r = bus.run_single_test(df, a["a"], a["b"])
                add(r["interpretation"] if r and "skipped" not in r else f"No applicable test for {a['a']} vs {a['b']}: {(r or {}).get('skipped', '')}", {"columns": [a["a"], a["b"]]})
            elif a["action"] == "combinations" and m is not None:
                res = rca.interaction_segments(df, m, a["columns"])
                evs = rca.interaction_evidence(res, m, 0)
                if not evs:
                    add(f"No combination of {a['columns']} is clearly different from both of its single factors ({res['screened']} screened).", {})
                for e in evs[:3]:
                    add(e["text"], e["data"])
        except Exception as ex:
            add(f"Follow-up analysis '{a['action']}' failed: {type(ex).__name__}: {ex}", {})
    return out
