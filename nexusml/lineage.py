"""Managing finished analyses: load one back, delete one cleanly, and rebuild the arguments needed to run one again.

An analysis leaves files in several places (state, reports, charts, the model bundle, tracking records, registry versions, an index entry). `delete_run` removes all of
them, but never a file another analysis still points at (analyses that were extended from this one keep working). Nothing here imports Streamlit or the language model.
"""
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from . import modelio, runstore
from .storage import get_registry

# arguments that describe the run itself and are passed on their own, never as options
_NOT_OPTIONS = {"dataset_path", "request", "run_args", "resume_from", "resume_reuse", "clarify", "on_event", "on_clarify", "on_approval", "auto_approve", "graph", "thread_id", "options"}


def state_path(exp: str) -> Path:
    return get_registry().path_of(f"runs/{exp}/state.json")


def load_state(exp: Optional[str]) -> Optional[Dict[str, Any]]:
    """A finished analysis exactly as it was saved, or None."""
    if not exp:
        return None
    row = runstore.get_run(exp)
    key = (row or {}).get("state_key")
    for p in ([get_registry().path_of(key)] if key else []) + [state_path(exp)]:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def _keys_of(state: Dict[str, Any], exp: str) -> Set[str]:
    """Every registry key (file or folder) that belongs to this analysis."""
    keys: Set[str] = set()
    for k in ("splits_key", "models_key", "model_key", "report_key", "report_html_key", "exec_summary_key", "leaderboard_key", "model_card_key", "repro_key"):
        if state.get(k):
            keys.add(str(state[k]))
    for ch in state.get("charts") or []:
        for k in ("key", "spec_key"):
            if isinstance(ch, dict) and ch.get(k):
                keys.add(str(ch[k]))
    for rid in {state.get("run_id")} - {None}:
        keys |= {f"charts/{rid}", f"runs/{rid}"}                      # charts, and the run log used for replay
    if state.get("thread_id"):
        keys.add(f"threads/{state['thread_id']}.json")                # graph checkpoint
    keys |= {f"{exp}", f"reports/{exp}", f"runs/{exp}", f"powerbi/{exp}", f"models/{exp}.joblib"}   # {exp}/ holds the reproducibility kit
    return keys


def _referenced_elsewhere(candidates: Set[str], skip_exp: str) -> Set[str]:
    """Candidate keys that the saved state of ANOTHER analysis still mentions."""
    if not candidates:
        return set()
    found: Set[str] = set()
    for r in runstore.list_runs(10_000):
        if r["run_id"] == skip_exp:
            continue
        try:
            text = state_path(r["run_id"]).read_text(encoding="utf-8")
        except OSError:
            continue
        for k in candidates:
            if k in text:
                found.add(k)
        if found == candidates:
            break
    return found


def _remove(key: str) -> int:
    reg = get_registry()
    try:
        p = reg.path_of(key)
    except ValueError:
        return 0
    n = 0
    if p.is_dir():
        n = sum(1 for _ in p.rglob("*") if _.is_file())
        shutil.rmtree(p, ignore_errors=True)
    elif p.exists():
        for f in (p, modelio.sidecar(p)):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return n


def delete_run(exp: str) -> Dict[str, Any]:
    """Remove a finished analysis and what it left behind. Returns {removed_files, kept_shared, children}."""
    state = load_state(exp) or {}
    reg = get_registry()
    children = runstore.children_of(exp)
    keys = _keys_of(state, exp)
    keep = _referenced_elsewhere({k for k in keys if k.startswith("charts/") or k.startswith(("splits_", "models_"))}, exp)
    removed = sum(_remove(k) for k in sorted(keys - keep, key=len, reverse=True))
    try:                                                              # experiment index
        rows = [r for r in reg.list_runs() if r.get("experiment_id") != exp]
        reg._write_json(reg.index, rows)
    except (OSError, ValueError):
        pass
    try:                                                              # tracking records + registered model versions that point at this analysis
        from . import tracking
        mine = [r for r in tracking.list_runs() if (r.get("tags") or {}).get("experiment_id") == exp]
        ids = {r["run_id"] for r in mine}
        ids |= {r["run_id"] for r in tracking.list_runs() if r.get("parent") in ids}
        for rid in ids:
            for sub in ("runs", "diag"):
                try:
                    (tracking._root() / sub / f"{rid}.json").unlink()
                except OSError:
                    pass
        regd = tracking.registry()
        changed = False
        for name in list(regd["models"]):
            vs = regd["models"][name]["versions"]
            left = [v for v in vs if v.get("run_id") not in ids]
            if len(left) != len(vs):
                changed = True
                if left:
                    regd["models"][name]["versions"] = left
                else:
                    del regd["models"][name]
        if changed:
            tracking._write(tracking._reg_path(), regd)
    except Exception:
        pass                                                          # the analysis itself is gone; leftover tracking records are harmless
    runstore.delete_run_rows(exp)
    runstore.audit("run_deleted", exp, {"files": removed, "children": children})
    return {"removed_files": removed, "kept_shared": sorted(keep), "children": children}


def delete_attempt(att_id: str) -> None:
    runstore.delete_attempt(att_id)
    try:
        from . import jobs
        jobs.forget(att_id)
    except Exception:
        pass


def options_of(state: Dict[str, Any]) -> Dict[str, Any]:
    """The options a saved analysis was started with (everything except the dataset and the request), ready to pass to jobs.submit."""
    args = dict(state.get("run_args") or {})
    return {k: v for k, v in args.items() if k not in _NOT_OPTIONS and v is not None}


def retry_spec(exp: Optional[str] = None, attempt_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """What the Retry dialog needs: {request, dataset_id, dataset_path, kwargs, parent, source}. For a finished analysis or one that did not finish."""
    if attempt_id:
        a = runstore.get_attempt(attempt_id)
        if not a:
            return None
        kw = {k: v for k, v in (a.get("kwargs") or {}).items() if k not in _NOT_OPTIONS}
        return {"request": a.get("request") or "", "dataset_id": a.get("dataset_id"), "dataset_path": (a.get("kwargs") or {}).get("dataset_path"), "kwargs": kw, "options": (a.get("kwargs") or {}).get("options") or {},
                "parent": a.get("parent"), "source": "attempt", "id": attempt_id}
    state = load_state(exp)
    if not state:
        return None
    return {"request": state.get("user_request") or "", "dataset_id": None, "dataset_path": state.get("dataset_path"), "kwargs": options_of(state), "options": {"minutes": max(1, int((state.get("time_budget") or 300) // 60))},
            "parent": state.get("parent_experiment_id"), "source": "run", "id": exp}


def family(exp: str) -> Dict[str, Any]:
    """{parent, children} for display. Ids of analyses that no longer exist are reported as None / dropped."""
    par = runstore.parent_of(exp)
    return {"parent": par if par and runstore.get_run(par) else None, "parent_deleted": bool(par and not runstore.get_run(par)),
            "children": [c for c in runstore.children_of(exp) if runstore.get_run(c)]}


def list_state_summaries(limit: int = 50) -> List[Dict[str, Any]]:
    """Light rows for the assistant and the history rail: id, request, when, dataset, target, model, metric, review result, parent."""
    out = []
    for r in runstore.list_runs(limit):
        out.append({"id": r["run_id"], "request": r.get("request"), "created": r.get("created"), "dataset": Path(r.get("dataset") or "").name or None, "target": r.get("target"),
                    "problem_type": r.get("problem_type"), "best_model": r.get("best_model"), "metric": r.get("primary_metric"), "heldout": r.get("heldout"),
                    "passed_review": bool(r.get("passed")) if r.get("passed") is not None else None, "parent": runstore.parent_of(r["run_id"])})
    return out
