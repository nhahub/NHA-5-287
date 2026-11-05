"""Persistent run index + audit log in SQLite (stdlib, no server). Lives next to the artifacts (`<artifacts>/runs.sqlite`), so mounting
the artifact directory as a volume makes runs, approvals and reports survive container restarts."""
import contextvars
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, created REAL, dataset TEXT, request TEXT, analysis_type TEXT, problem_type TEXT,
  target TEXT, best_model TEXT, primary_metric TEXT, heldout REAL, ci_lo REAL, ci_hi REAL, passed INTEGER, seed INTEGER,
  llm_calls INTEGER, tokens INTEGER, llm_seconds REAL, report_html TEXT, report_md TEXT, exec_summary TEXT, model_key TEXT, state_key TEXT, user TEXT);
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, user TEXT, action TEXT, run_id TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS lineage(run_id TEXT PRIMARY KEY, parent TEXT);
CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY, created REAL, user TEXT, dataset_id TEXT, dataset_name TEXT, request TEXT, status TEXT, error TEXT, message TEXT, kwargs TEXT, parent TEXT);
CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, user TEXT, run_id TEXT, kind TEXT, text TEXT);
"""


def _path() -> Path:
    p = Path(config.artifacts_dir()) / "runs.sqlite"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_path(), timeout=15)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


_USER: contextvars.ContextVar = contextvars.ContextVar("nexusml_user", default=None)


def set_user(name: Optional[str]) -> None:
    """Per-session identity. A web server runs many people's sessions in one process, so the name lives in a context variable (never in os.environ, which is shared)."""
    _USER.set(name or None)


def current_user() -> str:
    return _USER.get() or os.environ.get("NEXUSML_USER", "local")


def audit(action: str, run_id: Optional[str] = None, detail: Any = None, user: Optional[str] = None) -> None:
    """Who did what: approvals, rejections, clarifications, promotions, model downloads."""
    try:
        with _conn() as c:
            c.execute("INSERT INTO audit(ts,user,action,run_id,detail) VALUES(?,?,?,?,?)",
                      (time.time(), user or current_user(), action, run_id, json.dumps(detail, default=str)[:2000]))
    except sqlite3.Error:
        pass                                              # an audit failure must never break an analysis


def audit_log(run_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    with _conn() as c:
        q, a = ("SELECT * FROM audit WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit)) if run_id else ("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.execute(q, a)]


def record_run(state: Dict[str, Any]) -> None:
    from . import runlog
    plan = state.get("analysis_plan") or {}
    ci = state.get("metric_ci") or {}
    try:
        summ = runlog.summary(state.get("run_id", "")) if state.get("run_id") else {}
    except Exception:
        summ = {}
    row = (state.get("experiment_id") or state.get("run_id"), time.time(), state.get("dataset_path"), (state.get("user_request") or "")[:500],
           state.get("analysis_type"), plan.get("problem_type"), plan.get("target"), state.get("best_model_name"), state.get("primary_metric"),
           ci.get("value"), ci.get("lo"), ci.get("hi"), int(bool(state.get("quality_passed"))), state.get("seed"),
           summ.get("llm_calls"), summ.get("tokens"), summ.get("llm_seconds"), state.get("report_html_key"), state.get("report_key"),
           state.get("exec_summary_key"), state.get("model_key"), state.get("state_key"), current_user())
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO runs VALUES(" + ",".join("?" * 23) + ")", row)
        if state.get("parent_experiment_id"):
            c.execute("INSERT OR REPLACE INTO lineage VALUES(?,?)", (row[0], state["parent_experiment_id"]))


def list_runs(limit: int = 100) -> List[Dict[str, Any]]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY created DESC LIMIT ?", (limit,))]


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(r) if r else None


def delete_run_rows(run_id: str) -> None:
    """Remove a run from the index (its files are removed by lineage.delete_run). Its notes go too; the audit log keeps a trace on purpose."""
    with _conn() as c:
        c.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        c.execute("DELETE FROM lineage WHERE run_id=?", (run_id,))
        c.execute("DELETE FROM notes WHERE run_id=?", (run_id,))


def parent_of(run_id: str) -> Optional[str]:
    with _conn() as c:
        r = c.execute("SELECT parent FROM lineage WHERE run_id=?", (run_id,)).fetchone()
    return r["parent"] if r else None


def children_of(run_id: str) -> List[str]:
    with _conn() as c:
        return [r["run_id"] for r in c.execute("SELECT run_id FROM lineage WHERE parent=? ORDER BY run_id", (run_id,))]


# analyses that did not finish (failed, cancelled, waiting for an answer). Finished ones live in `runs`; these make the failures visible, retryable and deletable
# after a restart, which the in-memory job table cannot.
def record_attempt(att: Dict[str, Any]) -> None:
    try:
        with _conn() as c:
            c.execute("INSERT OR REPLACE INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      (att["id"], att.get("created") or time.time(), att.get("user") or current_user(), att.get("dataset_id"), att.get("dataset_name"), (att.get("request") or "")[:1000],
                       att.get("status"), (att.get("error") or "")[:600], (att.get("message") or "")[:600], json.dumps(att.get("kwargs") or {}, default=str), att.get("parent")))
    except sqlite3.Error:
        pass


def list_attempts(limit: int = 100) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM attempts WHERE user=? ORDER BY created DESC LIMIT ?", (current_user(), limit))]
    for r in rows:
        try:
            r["kwargs"] = json.loads(r.get("kwargs") or "{}")
        except ValueError:
            r["kwargs"] = {}
    return rows


def get_attempt(att_id: str) -> Optional[Dict[str, Any]]:
    return next((a for a in list_attempts(500) if a["id"] == att_id), None)


def delete_attempt(att_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM attempts WHERE id=? AND user=?", (att_id, current_user()))


# notes: what a person said about an analysis (a concern, a question to revisit). Shown next to the analysis and visible to the assistant.
def add_note(run_id: str, text: str, kind: str = "concern") -> None:
    with _conn() as c:
        c.execute("INSERT INTO notes(ts,user,run_id,kind,text) VALUES(?,?,?,?,?)", (time.time(), current_user(), run_id, kind, text.strip()[:1000]))
    audit("note_" + kind, run_id, text[:300])


def list_notes(run_id: str) -> List[Dict[str, Any]]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT ts,user,kind,text FROM notes WHERE run_id=? ORDER BY id", (run_id,))]
