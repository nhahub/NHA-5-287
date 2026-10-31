"""Artifact registry. State stores *keys*; objects live on disk.

Hardening vs. the original: keys are relative and resolved under the base dir
(joblib/pickle loading arbitrary paths from state would be a code-execution
hole), and the experiment index is written atomically.
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


from . import config, modelio


class ArtifactRegistry:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir).resolve()
        self.base.mkdir(parents=True, exist_ok=True)
        self.index = self.base / "experiments_index.json"
        if not self.index.exists():
            self._write_json(self.index, [])

    # ---- paths
    def path_of(self, key: str) -> Path:
        p = (self.base / key).resolve()
        if self.base != p and self.base not in p.parents:
            raise ValueError(f"Artifact key escapes registry: {key!r}")
        return p

    @staticmethod
    def _stamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)

    # ---- objects
    def save(self, obj: Any, prefix: str) -> str:
        key = f"{prefix}_{self._stamp()}.joblib"
        modelio.dump(obj, self.path_of(key))                    # joblib + SHA-256 sidecar (tamper check on load)
        return key

    def load(self, key: str) -> Any:
        p = self.path_of(key)
        if not p.exists():
            raise FileNotFoundError(f"Artifact not found: {key}")
        return modelio.load(p, trusted=True)                    # files in the registry are ours; a MODIFIED file (hash mismatch) is still refused

    def save_text(self, text: str, name: str) -> str:
        key = f"reports/{name}"
        p = self.path_of(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return key

    def save_bytes(self, data: bytes, key: str) -> str:
        p = self.path_of(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return key

    def save_upload(self, data: bytes, name: str) -> str:
        key = f"uploads/{self._stamp()}_{Path(name).name}"
        p = self.path_of(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return str(p)

    # ---- experiment index
    def log_run(self, meta: Dict[str, Any]) -> str:
        runs = self.list_runs()
        runs.append(meta)
        self._write_json(self.index, runs)
        return meta.get("experiment_id", "")

    def list_runs(self) -> List[Dict[str, Any]]:
        with open(self.index) as f:
            return json.load(f)


_CACHE: Dict[str, ArtifactRegistry] = {}


def get_registry() -> ArtifactRegistry:
    base = os.path.abspath(config.artifacts_dir())
    if base not in _CACHE:
        _CACHE[base] = ArtifactRegistry(base)
    return _CACHE[base]
