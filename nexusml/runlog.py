"""Run log: reproducibility and audit trail.

One JSON-lines file per run (`<artifacts>/runs/<run_id>/runlog.jsonl`) records the seed, every LLM call
(prompt hash, response, latency, tokens) and every routing decision, so a run can be inspected and replayed.
Set NEXUSML_REPLAY=<runlog.jsonl> to answer LLM calls from a previous log instead of the network
(strict: a prompt that was not recorded fails loudly).
"""
import hashlib
import json
import os
import random
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import config

_lock = threading.Lock()
_cur: Dict[str, Any] = {"run_id": None, "seed": None, "replay": None}


def run_dir(run_id: str) -> Path:
    p = Path(config.artifacts_dir()) / "runs" / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def path(run_id: str) -> Path:
    return run_dir(run_id) / "runlog.jsonl"


def phash(system: str, prompt: str) -> str:
    return hashlib.sha256((system + "\n" + prompt).encode("utf-8", "replace")).hexdigest()[:16]


def seed_everything(seed: int) -> None:
    config.RANDOM_STATE = int(seed)
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32 - 1))


def start(run_id: str, seed: int, meta: Optional[Dict[str, Any]] = None) -> None:
    with _lock:
        _cur.update(run_id=run_id, seed=seed, replay=None)
    seed_everything(seed)
    rp = os.environ.get("NEXUSML_REPLAY")
    if rp and Path(rp).is_file():
        q: Dict[Any, deque] = defaultdict(deque)
        for e in _read_file(Path(rp)):
            if e.get("kind") == "llm" and e.get("ok") and e.get("response") is not None:
                q[(e["purpose"], e["prompt_hash"])].append(e["response"])
        with _lock:
            _cur["replay"] = q
    event("run_start", seed=seed, **(meta or {}))


def end_run() -> None:
    """Leave replay mode when a run ends, so later calls in the same process (follow-up questions, searches) use the live LLM again."""
    with _lock:
        _cur["replay"] = None


def current_run() -> Optional[str]:
    return _cur.get("run_id")


def event(kind: str, **data: Any) -> None:
    rid = _cur.get("run_id")
    if not rid:
        return
    line = json.dumps({"ts": round(time.time(), 3), "kind": kind, **data}, default=str)
    try:
        with _lock, open(path(rid), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass                                       # logging must never break a run


def llm_event(purpose: str, h: str, response: Any, error: Optional[str], rec: Dict[str, Any]) -> None:
    event("llm", purpose=purpose, prompt_hash=h, ok=error is None, error=error, response=response if error is None else None,
          model=rec.get("model"), seconds=rec.get("seconds", 0), tokens=rec.get("tokens", 0))


def decision(router: str, result: Any, **ctx: Any) -> None:
    event("route", router=router, result=result, **ctx)


def replay_active() -> bool:
    return _cur.get("replay") is not None


def replay_lookup(purpose: str, h: str) -> Optional[Dict[str, Any]]:
    q = _cur.get("replay")
    if q is None:
        return None
    dq = q.get((purpose, h))
    if not dq:
        raise RuntimeError(f"Replay log has no recorded answer for '{purpose}' (prompt {h}); the run is not identical to the recorded one.")
    return dq.popleft()


def _read_file(p: Path) -> List[Dict[str, Any]]:
    out = []
    if p.is_file():
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def read(run_id: str) -> List[Dict[str, Any]]:
    return _read_file(path(run_id))


def summary(run_id: str) -> Dict[str, Any]:
    ev = read(run_id)
    calls = [e for e in ev if e.get("kind") == "llm"]
    toks = sum(int(e.get("tokens") or 0) for e in calls)
    by: Dict[str, Dict[str, float]] = {}
    for e in calls:
        b = by.setdefault(e["purpose"], {"calls": 0, "seconds": 0.0, "tokens": 0, "failed": 0})
        b["calls"] += 1
        b["seconds"] += float(e.get("seconds") or 0)
        b["tokens"] += int(e.get("tokens") or 0)
        b["failed"] += 0 if e.get("ok") else 1
    return {"llm_calls": len(calls), "failed_calls": sum(1 for e in calls if not e.get("ok")), "tokens": toks,
            "llm_seconds": round(sum(float(e.get("seconds") or 0) for e in calls), 2),
            "est_cost_usd": round(toks / 1e6 * config.LLM_PRICE_PER_MTOK, 4) if config.LLM_PRICE_PER_MTOK else None,
            "routes": [(e["router"], e["result"]) for e in ev if e.get("kind") == "route"],
            "by_purpose": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in by.items()},
            "seed": next((e.get("seed") for e in ev if e.get("kind") == "run_start"), None)}
