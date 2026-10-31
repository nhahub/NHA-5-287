"""Start / reuse / stop the MLflow web interface from inside the app (no terminal).

  start()  -> launches `mlflow server` (one worker; falls back to `mlflow ui`) as a background process on 127.0.0.1, waits until it answers /health,
              and returns its URL. If an MLflow server is already running (started by the app earlier or by you) it is REUSED, never duplicated.
  stop()   -> stops only a server this app started (the whole process tree on Windows).
Arguments are passed as a list (no shell), so folders with spaces ("bad nerd") need no quoting. Output goes to <artifacts>/mlflow_ui.log.
Environment: NEXUSML_MLFLOW_UI_PORT (default 5000), NEXUSML_MLFLOW_UI_HOST (default 127.0.0.1)."""
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import config, tracking

_OURS: Dict[str, Any] = {"proc": None}


def host() -> str:
    return os.environ.get("NEXUSML_MLFLOW_UI_HOST", "127.0.0.1")


def preferred_port() -> int:
    try:
        return int(os.environ.get("NEXUSML_MLFLOW_UI_PORT", "5000"))
    except ValueError:
        return 5000


def _dir() -> Path:
    p = Path(config.artifacts_dir())
    p.mkdir(parents=True, exist_ok=True)
    return p


def state_file() -> Path:
    return _dir() / "mlflow_ui.json"


def log_file() -> Path:
    return _dir() / "mlflow_ui.log"


def is_mlflow(port: int, h: Optional[str] = None, timeout: float = 1.0) -> bool:
    """True only for a real MLflow server: its /health endpoint answers 'OK' (any other web app on the port does not)."""
    try:
        with urllib.request.urlopen(f"http://{h or host()}:{port}/health", timeout=timeout) as r:
            return r.status == 200 and r.read(8).strip().startswith(b"OK")
    except Exception:
        return False


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host(), port)) == 0


def free_port(start: int) -> int:
    for p in range(start, start + 40):
        if not _port_in_use(p):
            return p
    raise RuntimeError(f"no free port found between {start} and {start + 39}")


def _read_state() -> Dict[str, Any]:
    try:
        return json.loads(state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def status() -> Dict[str, Any]:
    s = _read_state()
    port = s.get("port")
    running = bool(port) and is_mlflow(int(port))
    return {"running": running, "port": port, "url": s.get("url") if running else None, "pid": s.get("pid"), "ours": bool(s.get("ours")) and running}


def default_candidates(uri: str, port: int) -> List[List[str]]:
    """`mlflow server` with ONE worker first (Windows + Python 3.14 fails to open the port in some of the default four workers),
    then plain `mlflow ui` in case this MLflow version lacks --workers."""
    h = host()
    return [[sys.executable, "-m", "mlflow", "server", "--backend-store-uri", uri, "--host", h, "--port", str(port), "--workers", "1"],
            [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", uri, "--host", h, "--port", str(port)]]


def _popen_kwargs(logf) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"stdout": logf, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "cwd": str(_dir())}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    return kw


def _kill_tree(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=15)
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def _log_tail(n: int = 12) -> str:
    try:
        return "\n".join(log_file().read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def start(candidates: Optional[List[List[str]]] = None, wait: float = 45.0, popen: Callable = subprocess.Popen) -> Dict[str, Any]:
    """Make sure an MLflow UI is running and return {"ok", "url", "reused", "ours", "message"}."""
    st = tracking.mlflow_status()
    if not st["active"]:
        return {"ok": False, "message": st["reason"]}
    cur = status()
    if cur["running"]:
        return {"ok": True, "url": cur["url"], "reused": True, "ours": cur["ours"], "message": f"MLflow is already running at {cur['url']}."}
    pref = preferred_port()
    if is_mlflow(pref):                                           # an MLflow you started yourself (e.g. in a terminal): reuse it
        url = f"http://{host()}:{pref}"
        state_file().write_text(json.dumps({"port": pref, "pid": None, "ours": False, "url": url}), encoding="utf-8")
        return {"ok": True, "url": url, "reused": True, "ours": False,
                "message": f"An MLflow server was already running on port {pref}; I opened it. If it points at a different database, stop it and press the button again."}
    port = free_port(pref)
    uri = tracking.tracking_uri()
    cands = [[a.replace("{port}", str(port)) for a in c] for c in (candidates if candidates is not None else default_candidates(uri, port))]
    log_file().write_text("", encoding="utf-8")
    for argv in cands:
        with open(log_file(), "ab") as logf:
            logf.write(("$ " + " ".join(argv) + "\n").encode())
            try:
                proc = popen(argv, **_popen_kwargs(logf))
            except Exception as e:
                logf.write(f"could not start: {type(e).__name__}: {e}\n".encode())
                continue
        deadline = time.time() + wait
        while time.time() < deadline:
            if proc.poll() is not None:                           # exited: wrong flags / missing package -> try the next command
                break
            if is_mlflow(port):
                url = f"http://{host()}:{port}"
                _OURS["proc"] = proc
                state_file().write_text(json.dumps({"port": port, "pid": proc.pid, "ours": True, "url": url, "started": time.time()}), encoding="utf-8")
                return {"ok": True, "url": url, "reused": False, "ours": True, "message": f"MLflow started at {url}."}
            time.sleep(0.4)
        else:
            _kill_tree(proc.pid)                                  # still starting after `wait` seconds: give up on this command
    return {"ok": False, "message": "MLflow could not be started. Last lines of its log (" + str(log_file()) + "):\n" + (_log_tail() or "(empty)")}


def stop() -> bool:
    """Stop the server THIS APP started. A server you started yourself is left alone. Returns True if something was stopped."""
    s = _read_state()
    proc = _OURS.get("proc")
    pid = (proc.pid if proc is not None else None) or (s.get("pid") if s.get("ours") else None)
    if not pid:
        return False
    _kill_tree(int(pid))
    if proc is not None:
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
    _OURS["proc"] = None
    try:
        state_file().unlink()
    except OSError:
        pass
    return True


def _cleanup() -> None:
    if _OURS.get("proc") is not None:
        stop()


atexit.register(_cleanup)
