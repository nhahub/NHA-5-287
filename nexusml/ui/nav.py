"""Page registry: lets one page send the person to another (`go("analyze")`) without importing it."""
from typing import Any, Dict

from ._st import st

PAGES: Dict[str, Any] = {}


def register(key: str, page: Any) -> None:
    PAGES[key] = page


def go(key: str) -> None:
    """Send the person to a page. "analyze" means the composer, so an analysis that was open in full is closed first."""
    page = PAGES.get(key)
    if page is None:
        return
    if key == "analyze":
        from ._st import ss
        ss.pop("view_run", None)
        ss.open_run = None
        try:
            st.query_params.clear()
        except Exception:
            pass
    st.switch_page(page)
