"""Per-session state and the lookups pages share."""
import json
import uuid
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .. import runstore, workspace
from ..storage import get_registry
from ._st import ss


def init() -> None:
    """Make sure every key the pages read exists. Safe to call on every run."""
    ss.setdefault("uid", uuid.uuid4().hex[:8])
    ss.setdefault("user", runstore.current_user())
    ss.setdefault("ds_id", None)
    ss.setdefault("conv_id", workspace.new_conversation_id())
    ss.setdefault("messages", [])
    ss.setdefault("prefill", None)
    ss.setdefault("pending_prompt", None)
    ss.setdefault("job_id", None)
    ss.setdefault("open_run", None)
    ss.setdefault("_doc_idx", 0)
    ss.setdefault("share_labels", workspace.get_pref("share_labels", True))
    runstore.set_user(ss.user)


def active_dataset() -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    meta = workspace.get_dataset(ss.get("ds_id"))
    if meta is None:
        ds = workspace.list_datasets()
        if ds and ss.get("ds_id") is None:
            ss.ds_id = ds[0]["id"]
            meta = ds[0]
    return (meta, workspace.load_df(meta["id"])) if meta else (None, None)


def new_chat() -> None:
    ss.conv_id = workspace.new_conversation_id()
    ss.messages = []
    ss.pending_prompt = None


def open_chat(cid: str) -> bool:
    c = workspace.load_conversation(cid)
    if not c:
        return False
    ss.conv_id, ss.messages = c["id"], c["messages"]
    if c.get("dataset_id") and workspace.get_dataset(c["dataset_id"]):
        ss.ds_id = c["dataset_id"]
    return True


def save_chat() -> None:
    workspace.save_conversation(ss.conv_id, ss.messages, ss.get("ds_id"))


def load_run(exp: str) -> Optional[Dict[str, Any]]:
    """A finished analysis, exactly as it was saved."""
    for r in runstore.list_runs(500):
        if r["run_id"] == exp and r.get("state_key"):
            try:
                return json.loads(get_registry().path_of(r["state_key"]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
    return None
