"""A thread-safe blackboard: how the PARALLEL agents (sentinel, scout, explore, automl) talk to each other while the run is still going.

LangGraph state only merges when a node finishes, so concurrent branches cannot see each other through it. The blackboard is the live channel:
the sentinel posts leakage suspects, the scout posts feature ideas, explore posts what it found, and the AutoML agent reads its inbox at every round.
Every message is also returned in the final state (`board`), so the conversation is part of the audit trail and shown in the app.
"""
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class Blackboard:
    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._msgs: List[Dict[str, Any]] = []
        self._done: set = set()
        self.t0 = time.time()

    def post(self, sender: str, kind: str, payload: Optional[Dict[str, Any]] = None, text: str = "", severity: str = "info", to: Optional[str] = None) -> Dict[str, Any]:
        with self._cv:
            m = {"id": len(self._msgs), "t": round(time.time() - self.t0, 2), "sender": sender, "to": to, "kind": kind, "severity": severity, "text": text, "payload": payload or {}}
            self._msgs.append(m)
            self._cv.notify_all()
            return m

    def read(self, reader: str, since: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """Messages after cursor `since` that the reader did not write itself and that are addressed to everyone or to the reader."""
        with self._cv:
            new = [m for m in self._msgs[since:] if m["sender"] != reader and m["to"] in (None, reader)]
            return list(new), len(self._msgs)

    def mark_done(self, agent: str) -> None:
        with self._cv:
            self._done.add(agent)
            self._cv.notify_all()

    def wait_for(self, agents: List[str], timeout: float) -> bool:
        """Block (up to `timeout` s) until the named agents finished, so a decision is not taken before a fast peer reported."""
        end = time.time() + timeout
        with self._cv:
            while not set(agents) <= self._done:
                left = end - time.time()
                if left <= 0:
                    return False
                self._cv.wait(left)
            return True

    def export(self) -> List[Dict[str, Any]]:
        with self._cv:
            return [dict(m) for m in self._msgs]

    def count(self) -> int:
        with self._cv:
            return len(self._msgs)


_BOARDS: Dict[str, Blackboard] = {}
_LOCK = threading.Lock()


def get_board(run_id: str) -> Blackboard:
    with _LOCK:
        return _BOARDS.setdefault(str(run_id), Blackboard())


def release(run_id: str) -> None:
    with _LOCK:
        _BOARDS.pop(str(run_id), None)
