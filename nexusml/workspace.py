"""The workspace: a library of datasets and the saved conversations that belong to the current user.

Kept in its own small SQLite file next to the other artifacts (no server needed). Datasets are stored once, as CSV, whatever the upload format was; conversations store
messages with their tables and figures so that reopening one shows exactly what was shown before.
"""
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config, ingest, runstore

MAX_UPLOAD_MB = float(os.environ.get("NEXUSML_MAX_UPLOAD_MB", "200"))
_CACHE: Dict[str, Any] = {}


def _root() -> Path:
    p = Path(config.artifacts_dir()) / "workspace"
    (p / "datasets").mkdir(parents=True, exist_ok=True)
    return p


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_root() / "workspace.sqlite"), timeout=15)
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, user TEXT, name TEXT, rows INTEGER, cols INTEGER, bytes INTEGER, added REAL, file TEXT, sha TEXT, note TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS prefs(user TEXT, key TEXT, value TEXT, PRIMARY KEY(user, key))")
    c.execute("CREATE TABLE IF NOT EXISTS conversations(id TEXT, user TEXT, title TEXT, dataset_id TEXT, created REAL, updated REAL, pinned INTEGER DEFAULT 0, messages TEXT, PRIMARY KEY(id, user))")
    return c


# datasets
def safe_name(name: str) -> str:
    base = re.sub(r"[^\w.\- ]+", "_", Path(str(name)).name).strip() or "dataset"
    return base[:80]


def add_dataset(name: str, raw: bytes, note: str = "") -> Dict[str, Any]:
    """Store an uploaded table. Accepts CSV, TSV, Excel, JSON. Raises ValueError with a message fit to show the person."""
    if not raw:
        raise ValueError("The file is empty.")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"The file is larger than the {MAX_UPLOAD_MB:.0f} MB limit.")
    try:
        data = ingest.to_csv_bytes(name, raw)
        import io
        df = pd.read_csv(io.BytesIO(data), low_memory=False)
    except UnicodeDecodeError as e:
        raise ValueError("This does not look like a text or spreadsheet file, so it could not be read as a table.") from e
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"This file could not be read as a table ({type(e).__name__}). Check that it is a CSV, TSV, Excel or JSON file with a header row.") from e
    if df.shape[1] < 1 or len(df) < 1:
        raise ValueError("The table has no rows or no columns.")
    sha = hashlib.sha1(data).hexdigest()
    user = runstore.current_user()
    with _conn() as c:
        row = c.execute("SELECT * FROM datasets WHERE sha=? AND user=?", (sha, user)).fetchone()
        if row:
            return dict(row)
        did = uuid.uuid4().hex[:10]
        file = f"{did}.csv"
        (_root() / "datasets" / file).write_bytes(data)
        meta = (did, user, safe_name(name), int(len(df)), int(df.shape[1]), len(data), time.time(), file, sha, note)
        c.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?,?,?)", meta)
    return get_dataset(did)


def list_datasets() -> List[Dict[str, Any]]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM datasets WHERE user=? ORDER BY added DESC", (runstore.current_user(),))]


def get_dataset(did: Optional[str]) -> Optional[Dict[str, Any]]:
    if not did:
        return None
    with _conn() as c:
        r = c.execute("SELECT * FROM datasets WHERE id=? AND user=?", (did, runstore.current_user())).fetchone()
    return dict(r) if r else None


def dataset_path(did: str) -> Optional[str]:
    m = get_dataset(did)
    return str(_root() / "datasets" / m["file"]) if m else None


def load_df(did: str) -> Optional[pd.DataFrame]:
    p = dataset_path(did)
    if not p or not Path(p).exists():
        return None
    key = (did, os.path.getmtime(p))
    if key not in _CACHE:
        if len(_CACHE) > 6:
            _CACHE.clear()
        _CACHE[key] = pd.read_csv(p, low_memory=False)
    return _CACHE[key]


def delete_dataset(did: str) -> None:
    m = get_dataset(did)
    if not m:
        return
    with _conn() as c:
        c.execute("DELETE FROM datasets WHERE id=? AND user=?", (did, runstore.current_user()))
    try:
        (_root() / "datasets" / m["file"]).unlink()
    except OSError:
        pass
    _CACHE.clear()


def rename_dataset(did: str, name: str) -> None:
    with _conn() as c:
        c.execute("UPDATE datasets SET name=? WHERE id=? AND user=?", (safe_name(name), did, runstore.current_user()))


# conversations
def new_conversation_id() -> str:
    return uuid.uuid4().hex[:12]


def title_from(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user" and (m.get("text") or "").strip():
            t = m["text"].strip().replace("\n", " ")
            return t[:56] + ("…" if len(t) > 56 else "")
    return "New conversation"


def save_conversation(cid: str, messages: List[Dict[str, Any]], dataset_id: Optional[str] = None, title: Optional[str] = None) -> None:
    if not any(m.get("role") == "user" for m in messages):
        return                                                         # nothing worth keeping yet
    user, now = runstore.current_user(), time.time()
    with _conn() as c:
        row = c.execute("SELECT created, pinned, title FROM conversations WHERE id=? AND user=?", (cid, user)).fetchone()
        c.execute("INSERT OR REPLACE INTO conversations VALUES(?,?,?,?,?,?,?,?)",
                  (cid, user, (title or (row["title"] if row else None) or title_from(messages))[:120], dataset_id, row["created"] if row else now, now, row["pinned"] if row else 0,
                   json.dumps(messages, default=str)))


def load_conversation(cid: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        r = c.execute("SELECT * FROM conversations WHERE id=? AND user=?", (cid, runstore.current_user())).fetchone()
    if not r:
        return None
    try:
        msgs = json.loads(r["messages"])
    except ValueError:
        return None
    return {"id": r["id"], "title": r["title"], "dataset_id": r["dataset_id"], "updated": r["updated"], "pinned": bool(r["pinned"]), "messages": msgs}


def list_conversations(query: str = "", limit: int = 60) -> List[Dict[str, Any]]:
    q, a = "SELECT id,title,dataset_id,updated,pinned FROM conversations WHERE user=?", [runstore.current_user()]
    if query.strip():
        q += " AND title LIKE ?"
        a.append(f"%{query.strip()}%")
    with _conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY pinned DESC, updated DESC LIMIT ?", a + [limit])]


def delete_conversation(cid: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM conversations WHERE id=? AND user=?", (cid, runstore.current_user()))


def rename_conversation(cid: str, title: str) -> None:
    with _conn() as c:
        c.execute("UPDATE conversations SET title=? WHERE id=? AND user=?", (title.strip()[:120] or "Untitled", cid, runstore.current_user()))


def pin_conversation(cid: str, flag: bool) -> None:
    with _conn() as c:
        c.execute("UPDATE conversations SET pinned=? WHERE id=? AND user=?", (int(flag), cid, runstore.current_user()))


def clear_conversations() -> int:
    with _conn() as c:
        return c.execute("DELETE FROM conversations WHERE user=?", (runstore.current_user(),)).rowcount


# preferences + documents/images
def get_pref(key: str, default: Any = None) -> Any:
    with _conn() as c:
        r = c.execute("SELECT value FROM prefs WHERE user=? AND key=?", (runstore.current_user(), key)).fetchone()
    if not r:
        return default
    try:
        return json.loads(r["value"])
    except ValueError:
        return default


def set_pref(key: str, value: Any) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO prefs VALUES(?,?,?)", (runstore.current_user(), key, json.dumps(value)))


TABLE_EXT = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".ods"}
MEDIA_EXT = {".pdf", ".docx", ".odt", ".html", ".htm", ".md", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".zip"}


def is_table(name: str) -> bool:
    return Path(name).suffix.lower() in TABLE_EXT


def add_upload(name: str, raw: bytes) -> Dict[str, Any]:
    """One entry point for everything a person can drop in: a table is stored as is; documents, images and zip archives are turned into a table of their contents
    (one row per document / image) that every analysis can use. Raises ValueError with a message fit to show."""
    ext = Path(name).suffix.lower()
    if ext in TABLE_EXT or ext == "":
        return add_dataset(name, raw)
    if ext not in MEDIA_EXT:
        raise ValueError(f"Files of type '{ext}' are not supported. Add a CSV, Excel or JSON table, or documents, images or a zip of them.")
    import tempfile
    from . import media
    with tempfile.TemporaryDirectory(prefix="ws_media_") as d:
        src = Path(d) / safe_name(name)
        src.write_bytes(raw)
        info = media.prepare_paths([src], Path(config.artifacts_dir()) / "workspace" / "media")
    if not info.get("csv"):
        raise ValueError("Nothing readable was found in that file. " + "; ".join(info.get("notes", [])[:2]))
    raw_csv = Path(info["csv"]).read_bytes()
    if info.get("kind") == "images":
        # the images stay in the media folder, so the table must point at them by absolute path
        import io
        import pandas as pd
        tbl = pd.read_csv(io.BytesIO(raw_csv))
        tbl["image"] = [str((Path(info["root"]) / str(r)).resolve()) for r in tbl["image"]]
        raw_csv = tbl.to_csv(index=False).encode()
    meta = add_dataset(Path(name).stem + ".csv", raw_csv, note=json.dumps({"media": True, "suggest": {k: v for k, v in (info.get("suggest") or {}).items() if v}}))
    return meta


def dataset_hints(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Run options that came with a media dataset (which column holds the text / images, the label, the image folder ...)."""
    try:
        return (json.loads(meta["note"]) or {}).get("suggest", {}) if meta and meta.get("note") else {}
    except ValueError:
        return {}
