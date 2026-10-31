"""A stand-in for the `streamlit` module so pages can run in a sandbox without Streamlit.

It records every call, returns sensible widget defaults, supports `with` blocks, session state, queued chat input, one-shot button clicks, and the control-flow
exceptions (rerun / stop / switch_page). What it CANNOT do is render: it proves a page runs and calls Streamlit coherently, not that it looks right.

  FakeSt(answers={"method": value})   backward-compatible: a fixed return value (or callable) for a method name
  FakeSt(values={"label or key": v})  value returned by an input widget with that label / key
  FakeSt(clicks=["Run"])              buttons with that label / key return True once
  FakeSt(chat=["find outliers"])      messages returned by st.chat_input, one per call
"""
from typing import Any, Callable, Dict, Iterable, List, Optional


class FakeRerun(Exception):
    pass


class FakeStop(Exception):
    pass


class _Session(dict):
    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def __delattr__(self, k: str) -> None:
        self.pop(k, None)


_MISSING = object()


class FakeSt:
    def __init__(self, answers: Optional[Dict[str, Any]] = None, calls: Optional[List[Any]] = None, values: Optional[Dict[str, Any]] = None,
                 clicks: Iterable[str] = (), chat: Iterable[Any] = (), session: Optional[Dict[str, Any]] = None, _core: Optional[Dict[str, Any]] = None):
        self._core = _core if _core is not None else {"calls": calls if calls is not None else [], "values": dict(values or {}), "clicks": set(clicks), "chat": list(chat),
                                                      "session": _Session(session or {}), "switched": [], "query": {}}
        self.answers = answers or {}

    @property
    def calls(self) -> List[Any]:
        return self._core["calls"]

    @property
    def session_state(self) -> _Session:
        return self._core["session"]

    @property
    def query_params(self) -> Dict[str, Any]:
        return self._core["query"]

    @property
    def switched(self) -> List[Any]:
        return self._core["switched"]

    @property
    def sidebar(self) -> "FakeSt":
        return self._child()

    def _child(self) -> "FakeSt":
        return FakeSt(self.answers, _core=self._core)

    def add_click(self, label: str) -> None:
        self._core["clicks"].add(label)

    def __enter__(self) -> "FakeSt":
        return self

    def __exit__(self, *a: Any) -> bool:
        return False

    def _lookup(self, a: tuple, k: dict) -> Any:
        for cand in (k.get("key"), a[0] if a and isinstance(a[0], str) else None):
            if cand is not None and cand in self._core["values"]:
                return self._core["values"][cand]
        return _MISSING

    def _clicked(self, a: tuple, k: dict) -> bool:
        for cand in (k.get("key"), a[0] if a and isinstance(a[0], str) else None):
            if cand is not None and cand in self._core["clicks"]:
                self._core["clicks"].discard(cand)
                return True
        return False

    def __getattr__(self, name: str) -> Callable:
        if name.startswith("_"):
            raise AttributeError(name)

        def f(*a: Any, **k: Any) -> Any:
            self._core["calls"].append((name, a, k))
            if name in self.answers:
                v = self.answers[name]
                return v(*a, **k) if callable(v) else v
            if name == "columns":
                n = a[0] if isinstance(a[0], int) else len(a[0])
                return [self._child() for _ in range(n)]
            if name == "tabs":
                return [self._child() for _ in a[0]]
            if name in ("expander", "container", "spinner", "form", "popover", "chat_message", "empty", "status", "progress", "navigation"):
                return self._child()
            if name in ("update", "run", "toast"):
                return None
            if name == "dialog":
                return lambda fn: fn
            if name in ("fragment", "cache_data", "cache_resource"):
                return a[0] if a and callable(a[0]) else (lambda fn: fn)
            if name == "Page":
                return {"page": a[0] if a else None, **k}
            if name == "switch_page":
                self._core["switched"].append(a[0] if a else None)
                raise FakeStop()
            if name == "rerun":
                raise FakeRerun()
            if name == "stop":
                raise FakeStop()
            if name in ("button", "form_submit_button", "download_button", "link_button"):
                return self._clicked(a, k)
            if name == "chat_input":
                q = self._core["chat"]
                return q.pop(0) if q else None
            v = self._lookup(a, k)
            if name in ("text_input", "text_area", "password_input"):
                return k.get("value", "") if v is _MISSING else v
            if name in ("number_input", "slider", "select_slider"):
                return k.get("value", k.get("min_value", 0)) if v is _MISSING else v
            if name in ("checkbox", "toggle"):
                return k.get("value", False) if v is _MISSING else v
            if name in ("selectbox", "radio", "pills", "segmented_control"):
                if v is not _MISSING:
                    return v
                opts = list(k.get("options", a[1] if len(a) > 1 else []))
                idx = k.get("index", None if name in ("pills", "segmented_control") else 0)
                return opts[idx] if opts and idx is not None and idx < len(opts) else None
            if name == "multiselect":
                return k.get("default", []) if v is _MISSING else v
            if name == "file_uploader":
                return None if v is _MISSING else v
            if name == "data_editor":
                return a[0] if a else None
            return None
        return f

    def count(self, name: str) -> int:
        return sum(1 for c in self._core["calls"] if c[0] == name)

    def texts(self) -> List[str]:
        """All plain text a page wrote, for assertions."""
        return [a[0] for n, a, k in self._core["calls"] if n in ("markdown", "write", "caption", "info", "success", "warning", "error", "subheader", "header", "title", "text") and a and isinstance(a[0], str)]


def run_page(render: Callable[[], Any], max_runs: int = 4) -> int:
    """Run a page function the way Streamlit would: re-run after st.rerun(), end on st.stop() / switch_page. Returns the number of runs."""
    runs = 0
    while runs < max_runs:
        runs += 1
        try:
            render()
            return runs
        except FakeRerun:
            continue
        except FakeStop:
            return runs
    return runs
