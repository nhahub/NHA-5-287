"""Late-bound access to Streamlit. Pages import `st` and `ss` from here, so they can be exercised in tests with a stand-in (see tests/fake_st.py) and the module
never needs Streamlit at import time."""
from typing import Any

_override: Any = None


def use(obj: Any) -> None:
    """Route all calls to `obj` (tests). Pass None to go back to the real Streamlit."""
    global _override
    _override = obj


def real() -> Any:
    if _override is not None:
        return _override
    import streamlit
    return streamlit


class _Proxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(real(), name)


class _Session:
    """st.session_state with the same attribute and item access, resolved late."""

    def _s(self) -> Any:
        return real().session_state

    def __getattr__(self, n: str) -> Any:
        return getattr(self._s(), n)

    def __setattr__(self, n: str, v: Any) -> None:
        setattr(self._s(), n, v)

    def __getitem__(self, k: str) -> Any:
        return self._s()[k]

    def __setitem__(self, k: str, v: Any) -> None:
        self._s()[k] = v

    def __delitem__(self, k: str) -> None:
        del self._s()[k]

    def __contains__(self, k: str) -> bool:
        return k in self._s()

    def get(self, k: str, default: Any = None) -> Any:
        return self._s().get(k, default)

    def setdefault(self, k: str, v: Any) -> Any:
        s = self._s()
        if k not in s:
            s[k] = v
        return s[k]

    def pop(self, k: str, *d: Any) -> Any:
        return self._s().pop(k, *d)


st = _Proxy()
ss = _Session()
