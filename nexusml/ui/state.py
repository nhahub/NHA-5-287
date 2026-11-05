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
    if not ss.get("_llm_applied"):                       # re-apply the provider the person chose in Settings (a preference, never a key)
        ss["_llm_applied"] = True
        try:
            from .. import config
            saved = workspace.get_pref("llm_provider", None)
            if saved in config.PROVIDERS and config.provider_configured(saved):
                config.set_provider(saved, workspace.get_pref("llm_model", "") or "")
        except Exception:
            pass


def active_dataset() -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    meta = workspace.get_dataset(ss.get("ds_id"))
    if meta is None:
        ds = workspace.list_datasets()
        if ds and ss.get("ds_id") is None:
            ss.ds_id = ds[0]["id"]
            meta = ds[0]
    return (meta, workspace.load_df(meta["id"])) if meta else (None, None)


def sync_pick(key: str, valid: Any, empty: Any = None) -> None:
    """Make a dataset picker follow the workspace's selected dataset.

    A picker with a `key` keeps what it last showed, so after an upload (or a pasted link) it would still show the old dataset and the page would then switch back to it.
    Whenever the selected dataset changed since this picker last looked, point the picker at it. `valid` are the picker's options."""
    cur = ss.get("ds_id")
    if cur != ss.get(f"_seen:{key}"):
        if cur in valid:
            ss[key] = cur
        elif cur is None and empty is not None:
            ss[key] = empty
    ss[f"_seen:{key}"] = cur


def new_chat() -> None:
    ss.conv_id = workspace.new_conversation_id()
    ss.messages = []
    ss.attachments = []                                              # pictures and documents belong to the conversation they were dropped into
    ss.pending_prompt = None


def open_chat(cid: str) -> bool:
    c = workspace.load_conversation(cid)
    if not c:
        return False
    ss.conv_id, ss.messages = c["id"], c["messages"]
    ss.attachments = []
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
