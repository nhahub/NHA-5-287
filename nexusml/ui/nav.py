"""Page registry: lets one page send the person to another (`go("analyze")`) without importing it."""
from typing import Any, Dict

from ._st import st

PAGES: Dict[str, Any] = {}


def register(key: str, page: Any) -> None:
    PAGES[key] = page


def go(key: str) -> None:
    page = PAGES.get(key)
    if page is not None:
        st.switch_page(page)
