"""Background analyses. A full analysis takes minutes; it must not freeze the page or die when the person navigates away.

submit() starts the run on a worker thread and returns immediately. The page polls `get()` for live progress (stage, log lines) and, when the job is done, opens the
saved result. A job can be cancelled: it stops at the next stage boundary (a stage that is already running finishes first, because a thread cannot be interrupted safely).
"""
import contextvars
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional

from . import llm, runstore, runtime

STAGE_LABELS = {"intake": "Reading the dataset", "frame": "Understanding the question", "await_clarification": "Waiting for clarification", "plan": "Planning the analysis",
                "human_review": "Plan review", "prepare": "Preparing the data", "brief": "Briefing the team", "sentinel": "Checking for leakage and drift", "explore": "Exploring the data",
                "scout": "Looking for useful features", "automl": "Searching for the best model", "critic": "Reviewing the result", "explain": "Explaining the model", "synthesize": "Writing up findings",
                "publish": "Publishing the report", "abort": "Stopped"}
ORDER = ["intake", "frame", "plan", "prepare", "brief", "sentinel", "explore", "scout", "automl", "critic", "explain", "synthesize", "publish"]


class JobCancelled(Exception):
    pass


class Job:
    def __init__(self, dataset_id: Optional[str], dataset_name: str, request: str, options: Dict[str, Any], path: str = "", kwargs: Optional[Dict[str, Any]] = None):
        self.id = uuid.uuid4().hex[:10]
        self.path, self.kwargs = path, dict(kwargs or {})              # how it was started: what Retry needs to run it again
        self.dataset_id, self.dataset_name, self.request, self.options = dataset_id, dataset_name, request, options
        self.status = "queued"                      # queued | running | done | failed | cancelled | needs_input
        self.stage, self.stages_done = "", []
        self.log: List[Dict[str, Any]] = []
        self.started, self.finished = time.time(), None
        self.experiment_id: Optional[str] = None
        self.error: Optional[str] = None
        self.message: Optional[str] = None
        self.cancel_requested = False
        self._lock = threading.Lock()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            done = [s for s in ORDER if s in self.stages_done]
            return {"id": self.id, "status": self.status, "stage": self.stage, "stage_label": STAGE_LABELS.get(self.stage, self.stage), "progress": min(1.0, len(done) / len(ORDER)),
                    "log": list(self.log), "elapsed": round((self.finished or time.time()) - self.started, 1), "experiment_id": self.experiment_id, "error": self.error, "message": self.message,
                    "dataset_name": self.dataset_name, "request": self.request, "options": self.options, "dataset_id": self.dataset_id,
                    "parent": self.kwargs.get("resume_from"), "started": self.started}


_JOBS: Dict[str, Job] = {}
_LOCK = threading.Lock()


def submit(path: str, request: str, dataset_id: Optional[str] = None, dataset_name: str = "", options: Optional[Dict[str, Any]] = None, clarify: Optional[Dict[str, Any]] = None,
           runner: Optional[Callable[..., Dict[str, Any]]] = None, **state_kwargs) -> str:
    """Start an analysis in the background. `clarify` = {"target_column", "problem_type"} answers the one question the run may otherwise pause on."""
    if state_kwargs.get("resume_from"):                               # a follow-up must be possible NOW, so the person is told here rather than by a failed background run
        from . import resume
        resume.seed(state_kwargs["resume_from"], path, state_kwargs.get("resume_reuse"))
    job = Job(dataset_id, dataset_name or path, request, options or {}, path, state_kwargs)
    with _LOCK:
        _JOBS[job.id] = job
    runner = runner or runtime.run_to_completion

    def on_event(node: str, upd: Dict[str, Any]) -> None:
        if job.cancel_requested:
            raise JobCancelled()
        with job._lock:
            job.stage = node
            if node not in job.stages_done:
                job.stages_done.append(node)
            for e in (upd or {}).get("agent_log", []):
                job.log.append({"t": round(time.time() - job.started, 1), "agent": e.get("agent"), "message": e.get("message"), "stage": node})

    def work() -> None:
        job.status = "running"
        try:
            values = runner(path, request, on_event=on_event, on_clarify=(lambda v: dict(clarify)) if clarify else None, auto_approve=True, **state_kwargs)
            job.experiment_id = values.get("experiment_id")
            if values.get("errors"):
                job.status, job.error = "failed", "; ".join(values["errors"])[:400]
            elif values.get("clarification_question") and not values.get("analysis_plan"):
                job.status, job.message = "needs_input", str(values["clarification_question"])[:400]
            else:
                job.status = "done"
        except JobCancelled:
            job.status = "cancelled"
        except llm.LLMRequired as e:
            job.status, job.error = "failed", str(e)[:400]
        except Exception as e:                                           # a worker must never die silently
            job.status, job.error = "failed", f"{type(e).__name__}: {str(e)[:300]}"
            job.log.append({"t": round(time.time() - job.started, 1), "agent": "System", "message": traceback.format_exc()[-600:], "stage": job.stage})
        finally:
            job.finished = time.time()
            if job.status in ("failed", "cancelled", "needs_input"):      # keep what did not finish: visible after a restart, retryable, deletable
                runstore.record_attempt({"id": job.id, "created": job.started, "dataset_id": job.dataset_id, "dataset_name": job.dataset_name, "request": job.request, "status": job.status,
                                         "error": job.error, "message": job.message, "parent": job.kwargs.get("resume_from"),
                                         "kwargs": {**_plain(job.kwargs), "dataset_path": job.path, "options": job.options}})
    ctx = contextvars.copy_context()                                    # the worker runs as the person who started it
    threading.Thread(target=lambda: ctx.run(work), name=f"analysis-{job.id}", daemon=True).start()
    return job.id


def _plain(d: Dict[str, Any]) -> Dict[str, Any]:
    """Only values that survive being stored as JSON (callbacks and objects are left out)."""
    import json
    out = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            pass
    return out


def get(job_id: str) -> Optional[Dict[str, Any]]:
    j = _JOBS.get(job_id)
    return j.snapshot() if j else None


def cancel(job_id: str) -> None:
    j = _JOBS.get(job_id)
    if j and j.status in ("queued", "running"):
        j.cancel_requested = True


def active() -> List[Dict[str, Any]]:
    return [j.snapshot() for j in sorted(_JOBS.values(), key=lambda j: -j.started)]


def forget(job_id: str) -> None:
    with _LOCK:
        _JOBS.pop(job_id, None)
