"""A compact picture of the whole workspace, sent with every assistant turn: where the person is, what is open, what exists and what is running.

It is what lets "this", "here" and "the open analysis" mean something on any page, and lets the assistant offer the right next step without the person having to
explain the state of the application first. Names, counts and statuses only: no rows, no keys, nothing from the environment.
"""
import importlib.metadata as md
from typing import Any, Dict, List, Optional

KEY_PACKAGES = ["pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "plotly", "streamlit", "langgraph", "mlflow", "torch", "torchvision", "openpyxl", "pypdf", "pdfplumber", "pymupdf",
                "python-docx", "beautifulsoup4", "odfpy", "statsmodels", "xgboost", "lightgbm", "catboost", "imbalanced-learn", "shap", "pillow"]


def package_versions(query: str = "", limit: int = 60) -> Dict[str, Any]:
    """Installed Python packages. With no query: the ones this application cares about. With a query: every installed package whose name contains it."""
    q = (query or "").strip().lower()
    seen: Dict[str, str] = {}
    for d in md.distributions():
        try:
            name = (d.metadata["Name"] or "").strip()
        except Exception:
            continue
        if name:
            seen.setdefault(name.lower().replace("_", "-"), d.version or "?")
    if q:
        hit = {n: v for n, v in sorted(seen.items()) if q in n}
        return {"query": q, "found": dict(list(hit.items())[:limit]), "matches": len(hit), "installed_total": len(seen)}
    have = {p: seen[p] for p in KEY_PACKAGES if p in seen}
    return {"installed": have, "not_installed": [p for p in KEY_PACKAGES if p not in seen], "installed_total": len(seen)}


def digest(view: Optional[Dict[str, Any]] = None, open_run: Optional[str] = None, max_runs: int = 6) -> Dict[str, Any]:
    from .. import config, jobs, lineage, runstore, workspace
    out: Dict[str, Any] = {"page": {k: v for k, v in (view or {}).items() if v not in (None, "", [], {})}}
    try:
        ds = workspace.list_datasets()
        out["datasets"] = [{"name": d["name"], "rows": d["rows"], "columns": d["cols"]} for d in ds[:8]] + ([{"more": len(ds) - 8}] if len(ds) > 8 else [])
    except Exception:
        out["datasets"] = []
    try:
        runs = lineage.list_state_summaries(60)
        out["analyses"] = {"total": len(runs), "open": open_run,
                           "recent": [{"id": r["id"], "request": (r["request"] or "")[:70], "target": r["target"], "model": r["best_model"], "metric": r["metric"],
                                       "held_out": round(r["heldout"], 3) if isinstance(r.get("heldout"), (int, float)) else None, "built_on": r["parent"]} for r in runs[:max_runs]]}
    except Exception:
        out["analyses"] = {"total": 0}
    try:
        live = [j for j in jobs.active() if j["status"] in ("running", "queued", "needs_input")]
        out["running"] = [{"id": j["id"], "status": j["status"], "stage": j["stage_label"], "request": (j["request"] or "")[:60]} for j in live[:4]]
        failed = [a for a in runstore.list_attempts(20) if a["status"] in ("failed", "cancelled")]
        out["unfinished"] = len(failed)
    except Exception:
        pass
    try:
        out["language_model"] = {"provider": config.LLM_PROVIDER_NAME, "model": config.LLM_MODEL}
    except Exception:
        pass
    return out
