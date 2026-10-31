"""Typed contracts for what agents exchange. validate_update() checks a node's returned update against the graph state schema
and the shapes of evidence / messages; the graph runs it on every node when NEXUSML_VALIDATE=1 (the test-suite does)."""
from typing import Any, Dict, TypedDict, get_type_hints

from .state import DataScientistState


class Evidence(TypedDict):
    id: str
    source: str
    text: str
    data: Any


class Message(TypedDict):
    id: str
    to: str
    kind: str
    topic: str
    payload: Dict[str, Any]
    ts: str


class SchemaError(ValueError):
    pass


STATE_KEYS = set(get_type_hints(DataScientistState, include_extras=True))
KINDS = {"request", "response", "note"}


def validate_evidence(e: Any) -> None:
    if not isinstance(e, dict) or not {"id", "source", "text", "data"} <= set(e):
        raise SchemaError(f"evidence item must have id/source/text/data: {str(e)[:80]}")
    if not isinstance(e["id"], str) or not e["id"][:1].isalpha() or not e["id"][1:].isdigit():
        raise SchemaError(f"bad evidence id {e['id']!r}")
    if not isinstance(e["text"], str) or not e["text"].strip():
        raise SchemaError(f"evidence {e['id']} has empty text")


def validate_update(node: str, update: Dict[str, Any]) -> None:
    unknown = set(update) - STATE_KEYS
    if unknown:
        raise SchemaError(f"node '{node}' returned keys that are not in the state schema: {sorted(unknown)}")
    for e in update.get("evidence") or []:
        validate_evidence(e)
    for m in update.get("messages") or []:
        if not isinstance(m, dict) or m.get("kind") not in KINDS or "from" not in m or "to" not in m or "topic" not in m:
            raise SchemaError(f"node '{node}' returned a malformed message: {str(m)[:80]}")
    if "evidence" in update:
        ids = [e["id"] for e in update["evidence"]]
        if len(ids) != len(set(ids)):
            raise SchemaError(f"node '{node}' produced duplicate evidence ids")
