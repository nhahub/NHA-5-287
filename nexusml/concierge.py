"""Front-door supervisor: decides what to do with a chat message BEFORE any pipeline runs.

  general   -> answer directly from data-science / ML knowledge (no dataset needed)
  followup  -> the user is asking about the analysis that is already open
  needs_data-> the user asks about THEIR data but none is loaded: answer helpfully and explain what to upload

Routing is deterministic (cheap, predictable); only the answer itself uses the LLM. The pipeline's own supervisor (which phases to run)
is separate (agents/supervisor.py)."""
import re
from typing import Any, Dict, Sequence, Tuple

from .agents._util import ANALYST_PERSONA, ask

CAPABILITIES = """What this tool can do when you give it data: analyse tables (CSV / Excel / JSON ...: statistics, KPIs, trends, root causes, recommendations),
train and evaluate prediction models (classification, regression, forecasting, clustering) with confidence intervals and leakage checks,
combine several tables, analyse images (describe a photo with a vision model, quality / duplicates / classes) and documents (PDF / Word / text:
summaries, themes, search, classification), explain models, export reports / Power BI / deployable model packages, and monitor drift."""

YOUR_DATA = re.compile(r"\b(my|our|the attached|this)\s+(data|dataset|file|csv|excel|sheet|table|spreadsheet|sales|customers?|churn|numbers|results?)\b|"
                       r"\b(analy[sz]e|predict|forecast|cluster|segment)\s+(my|our|the|this)\b|\bin (my|our) (data|file)\b|\bwhy (is|are|do|does) (my|our)\b", re.I)
REFERS_TO_RUN = re.compile(r"\b(this|that|the|my)\s+(analysis|model|result|results|report|data|dataset|chart|insight|recommendation|segment|cluster|kpi|feature|column)\b|"
                           r"\b(evidence|root cause|top driver|why did|how reliable|how accurate|what does .* mean here)\b", re.I)


def route(text: str, has_results: bool, columns: Sequence[str] = ()) -> str:
    t = text or ""
    if has_results:
        if REFERS_TO_RUN.search(t) or any(c and len(c) > 2 and re.search(rf"\b{re.escape(str(c))}\b", t, re.I) for c in columns):
            return "followup"
        return "general"
    return "needs_data" if YOUR_DATA.search(t) else "general"


def answer(state: Dict[str, Any], question: str, history: Sequence[Tuple[str, str]], intent: str = "general") -> Dict[str, Any]:
    """Answer from general knowledge. `history` = [(question, answer), ...]. Needs the LLM (no dataset)."""
    nudge = ("The user is asking about THEIR OWN data but nothing is loaded. Give useful general guidance and say what file to upload and what analysis you would run."
             if intent == "needs_data" else "")
    payload = {"question": question, "recent_conversation": [{"q": q, "a": a[:600]} for q, a in list(history)[-5:]], "about_this_tool": CAPABILITIES,
               "rules": ["Answer like a senior data scientist: correct, practical, concise, with a short example or code (python / pandas / scikit-learn) only when it helps.",
                         "Do not invent facts about the user's data. For facts that change quickly (library versions, new models, prices) say your knowledge may be out of date.",
                         "If the question is ambiguous, answer the most likely reading and state the assumption.",
                         "Reply in the user's language.", nudge],
               "output_schema": {"answer": "markdown text", "needs_data": "true if a dataset would materially improve the answer", "next_steps": ["0-3 short suggestions"]}}

    def v(d):
        a = str(d.get("answer", "")).strip()
        if not a:
            raise ValueError("write the answer")
        return {"answer": a[:6000], "needs_data": bool(d.get("needs_data")) or intent == "needs_data",
                "next_steps": [str(x)[:140] for x in (d.get("next_steps") or [])][:3]}
    out, _, _ = ask(state, "general_chat", ANALYST_PERSONA + " You are also a patient teacher of data science and machine learning.", payload, 1500, v)
    return out
