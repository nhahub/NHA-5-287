"""Where the person is, so the assistant can be reached from every page and still know what "this" means.

Each page calls `set_view(...)` as it starts; the assistant dock and the assistant page read it with `get_view()` and send it with every question.
"""
from typing import Any, Dict, List

from ._st import ss

LABEL = {"home": "Home", "assistant": "Assistant", "data": "Data", "analyses": "Analyses", "analysis": "an analysis", "models": "Models", "settings": "Settings"}


def set_view(page: str, **info: Any) -> None:
    ss["view_ctx"] = {"page": page, **{k: v for k, v in info.items() if v not in (None, "", [], {})}}


def get_view() -> Dict[str, Any]:
    return dict(ss.get("view_ctx") or {})


def describe(view: Dict[str, Any]) -> str:
    """One short line for the person: what the assistant can see right now."""
    page = view.get("page") or "home"
    if page == "analysis" and view.get("request"):
        return f"Looking at the analysis: {str(view['request'])[:70]}"
    if page == "data" and view.get("dataset"):
        return f"Looking at the dataset {view['dataset']}"
    if page == "analyses":
        return "On the Analyses page" + (f", setting up a follow-up to {view['extending'][-8:]}" if view.get("extending") else "")
    return f"On the {LABEL.get(page, page)} page"


def suggestions(view: Dict[str, Any]) -> List[str]:
    page = view.get("page")
    if page == "analysis":
        return ["Summarise what this analysis found", "Is there anything in this analysis I should be concerned about?", "What would you check or do next?"]
    if page == "analyses":
        return ["What have I analysed so far?", "What is running or did not finish?", "Which of my analyses has the best model?"]
    if page == "data":
        return ["What stands out in this dataset?", "Which columns have quality problems?", "What could I predict with this data?"]
    if page == "models":
        return ["Which registered model is best, and why?", "Compare my latest analyses", "What did my best model rely on?"]
    if page == "settings":
        return ["Which packages are installed?", "Which language model is connected?"]
    return ["What have I analysed so far?", "What is in my workspace?", "What should I look at first?"]
