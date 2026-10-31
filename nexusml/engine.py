"""Minimal, dependency-free stand-in for the subset of LangGraph this project uses.

v4: supports FAN-OUT (several edges from one node), JOINS (`add_edge([a, b], c)` waits for all sources) and runs the
nodes of one superstep concurrently (threads) when NEXUSML_PARALLEL=1. Updates are merged in a fixed order, so results
are identical to a sequential run.


It mirrors LangGraph's public API (StateGraph / compile(interrupt_before=...) /
stream / get_state / update_state, Annotated reducers) so the exact same graph
definition runs on either backend. `graph.py` prefers real LangGraph when it is
installed; this engine is the offline fallback and what the test-suite exercises.
"""
import json
import operator
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Callable, Dict, Iterator, List, Optional, get_args, get_origin, get_type_hints

START = "__start__"
END = "__end__"
MAX_STEPS = 200


class MemorySaver:
    """Persists thread state as JSON under `path` (or $NEXUSML_ARTIFACTS/threads) so a paused run
    survives a process restart. State is JSON-friendly by design (heavy objects live in storage)."""

    def __init__(self, path: str = ""):
        self.path = path

    def _dir(self) -> Path:
        base = self.path or os.path.join(os.environ.get("NEXUSML_ARTIFACTS", ".agent_artifacts"), "threads")
        d = Path(base)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, tid: str, thread: Dict[str, Any]) -> None:
        tmp = self._dir() / f"{tid}.tmp"
        tmp.write_text(json.dumps(thread, default=str), encoding="utf-8")
        os.replace(tmp, self._dir() / f"{tid}.json")

    def load(self, tid: str):
        f = self._dir() / f"{tid}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


class StateGraph:
    def __init__(self, schema):
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, List[str]] = {}
        self.joins: List[tuple] = []          # (tuple(sources), dst): dst fires once ALL sources have completed
        self.cond: Dict[str, tuple] = {}
        self.reducers: Dict[str, Callable] = {}
        for key, hint in get_type_hints(schema, include_extras=True).items():
            if get_origin(hint) is Annotated:
                for meta in get_args(hint)[1:]:
                    if callable(meta):
                        self.reducers[key] = meta

    def add_node(self, name: str, fn: Callable):
        self.nodes[name] = fn

    def add_edge(self, src, dst: str):
        if isinstance(src, (list, tuple)):
            self.joins.append((tuple(src), dst))
        else:
            self.edges.setdefault(src, []).append(dst)

    def add_conditional_edges(self, src: str, router: Callable, path_map=None):
        if isinstance(path_map, (list, tuple)):
            path_map = {p: p for p in path_map}
        self.cond[src] = (router, path_map)

    def compile(self, checkpointer=None, interrupt_before: Optional[List[str]] = None):
        return CompiledGraph(self, set(interrupt_before or []), checkpointer)


class CompiledGraph:
    def __init__(self, g: StateGraph, interrupt_before: set, saver=None):
        self.g, self.interrupt_before = g, interrupt_before
        self.saver = saver if isinstance(saver, MemorySaver) else None
        self._threads: Dict[str, Dict[str, Any]] = {}

    def _thread(self, tid: str):
        t = self._threads.get(tid)
        if t is None and self.saver is not None:
            t = self.saver.load(tid)
            if t is not None:
                self._threads[tid] = t
        return t

    def _persist(self, tid: str) -> None:
        if self.saver is not None and tid in self._threads:
            self.saver.save(tid, self._threads[tid])

    @staticmethod
    def _tid(config) -> str:
        return (config or {}).get("configurable", {}).get("thread_id", "default")

    def _apply(self, values: Dict[str, Any], update: Dict[str, Any]) -> None:
        for k, v in (update or {}).items():
            fn = self.g.reducers.get(k)
            if fn is None:
                values[k] = v
            elif fn is operator.add:
                values[k] = fn(list(values.get(k) or []), list(v or []))
            else:
                values[k] = fn(values.get(k), v)

    def _targets(self, node: str, values: Dict[str, Any]) -> List[str]:
        if node in self.g.cond:
            router, pmap = self.g.cond[node]
            key = router(values)
            return [pmap[key] if pmap else key]
        return list(self.g.edges.get(node, []))

    def _advance(self, t: Dict[str, Any], done: List[str]) -> List[str]:
        """Next frontier after `done` nodes completed: direct edges plus any join whose sources have all completed."""
        out: List[str] = []
        seen: Dict[str, List[str]] = t.setdefault("join_seen", {})
        for node in done:
            for dst in self._targets(node, t["values"]):
                if dst != END and dst not in out:
                    out.append(dst)
            for srcs, dst in self.g.joins:
                if node in srcs:
                    key = "+".join(sorted(srcs)) + ">" + dst
                    got = seen.setdefault(key, [])
                    if node not in got:
                        got.append(node)
                    if set(got) >= set(srcs):
                        seen[key] = []
                        if dst not in out:
                            out.append(dst)
        order = {n: i for i, n in enumerate(self.g.nodes)}
        return sorted(out, key=lambda n: order.get(n, 10 ** 6))          # deterministic order, independent of finish time

    def _run_nodes(self, nodes: List[str], values: Dict[str, Any]) -> List[Dict[str, Any]]:
        parallel = os.environ.get("NEXUSML_PARALLEL", "1") == "1" and len(nodes) > 1
        if not parallel:
            return [self.g.nodes[n](dict(values)) or {} for n in nodes]
        with ThreadPoolExecutor(max_workers=min(len(nodes), 4)) as ex:
            futs = [ex.submit(self.g.nodes[n], dict(values)) for n in nodes]
            return [f.result() or {} for f in futs]          # exceptions propagate; order = frontier order

    def stream(self, input: Optional[Dict[str, Any]], config=None, stream_mode: str = "updates") -> Iterator[Dict]:
        tid = self._tid(config)
        if input is not None:
            t = {"values": {}, "frontier": [], "paused_at": None, "join_seen": {}}
            self._apply(t["values"], input)
            t["frontier"] = self._advance(t, [START])
            self._threads[tid] = t
        else:
            t = self._thread(tid)
            if t is None:
                raise ValueError(f"No state for thread {tid!r}")
            if "frontier" not in t:                                   # thread persisted by an older version
                nxt = t.get("next")
                t["frontier"] = [] if nxt in (None, END) else [nxt]
                t.setdefault("join_seen", {})
        steps = 0
        while t["frontier"]:
            front = list(t["frontier"])
            gate = [n for n in front if n in self.interrupt_before]
            if gate and t["paused_at"] != gate[0]:
                t["paused_at"] = gate[0]
                t["frontier"] = gate[:1] + [n for n in front if n not in gate]
                self._persist(tid)
                return
            t["paused_at"] = None
            updates = self._run_nodes(front, t["values"])
            for node, upd in zip(front, updates):                     # merge in frontier order: deterministic
                self._apply(t["values"], upd)
            for node, upd in zip(front, updates):
                yield {node: upd}
            t["frontier"] = self._advance(t, front)
            self._persist(tid)
            steps += 1
            if steps > MAX_STEPS:
                raise RuntimeError("Graph exceeded max steps (possible routing loop)")

    def get_state(self, config=None):
        t = self._thread(self._tid(config))
        if t is None:
            return SimpleNamespace(values={}, next=())
        if "frontier" in t:
            nxt = tuple(t["frontier"])
        else:
            nxt = () if t.get("next") in (None, END) else (t["next"],)
        return SimpleNamespace(values=dict(t["values"]), next=nxt)

    def update_state(self, config, values: Dict[str, Any], as_node: Optional[str] = None):
        tid = self._tid(config)
        t = self._thread(tid)
        if t is None:
            raise ValueError("No state to update")
        self._apply(t["values"], values)
        self._persist(tid)
        return config
