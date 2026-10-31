"""Safer model files. joblib/pickle can execute code when loaded, so every saved model gets a SHA-256 sidecar (`<file>.sha256`) that is
verified before loading; a file that has no sidecar, or whose hash differs, is refused unless you explicitly trust it.
Optional: NEXUSML_MODEL_FORMAT=skops stores bundles with skops (data-only format; untrusted types are listed, never executed) when installed.
Loading a model file from someone you do not trust is still unsafe: only load files you created or can verify."""
import hashlib
import os
from pathlib import Path
from typing import Any

import joblib


class UntrustedModelError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar(p: Path) -> Path:
    return p.with_name(p.name + ".sha256")


def dump(obj: Any, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("NEXUSML_MODEL_FORMAT") == "skops":
        try:
            import skops.io as sio
            sio.dump(obj, p)
            sidecar(p).write_text(sha256_file(p) + "  skops\n")
            return p
        except ImportError:
            pass                                          # skops not installed: fall back to joblib
    joblib.dump(obj, p)
    sidecar(p).write_text(sha256_file(p) + "  joblib\n")
    return p


def load(path, trusted: bool = False) -> Any:
    """Verify the sidecar hash, then load. `trusted=True` (or NEXUSML_TRUST_MODELS=1) skips the requirement for a sidecar, never a mismatch."""
    p = Path(path)
    sc = sidecar(p)
    if sc.is_file():
        want, _, fmt = sc.read_text().strip().partition("  ")
        if sha256_file(p) != want:
            raise UntrustedModelError(f"{p.name}: hash does not match its .sha256 file; the file was modified. Refusing to load.")
        if fmt.strip() == "skops":
            import skops.io as sio
            return sio.load(p, trusted=sio.get_untrusted_types(file=p) if trusted else [])
    elif not (trusted or os.environ.get("NEXUSML_TRUST_MODELS") == "1"):
        raise UntrustedModelError(f"{p.name} has no .sha256 sidecar. Loading unverified pickles can run arbitrary code; "
                                  "pass trusted=True (or set NEXUSML_TRUST_MODELS=1) only for files you created.")
    return joblib.load(p)
