"""The conversational analyst: decide what the question needs, do exactly that, answer.

  "find the outliers"          -> one tool call, one answer      (seconds, no pipeline)
  "why is revenue falling?"    -> a few tools in sequence, then an answer that cites them
  "hello" / "what is a p-value?" -> a direct answer, no data touched
  "build me a model that ..."  -> a proposal the person confirms (the full pipeline is never started on its own)

Safety and honesty:
  * the model never sees rows: only column names, summary statistics and category counts (see `digest` and tools.py)
  * every tool call is validated against the real columns; a rejected call goes back to the model as an observation so it can correct itself
  * the final answer's figures are checked against the tool results; unverified figures are flagged instead of silently shown
  * a hard step limit and per-tool caps bound the cost of any question
"""
import json
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .. import claims, llm, profiling
from ..agents._util import ask
from ..util import jsonable
from . import tools as T

MAX_STEPS = 6
MAX_OBS_CHARS = 6000

SYSTEM = (
    "You are the analyst built into a data workspace. People ask you things about their data, their analyses, or data work in general. Work like a careful colleague: "
    "do the smallest amount of work that answers the question correctly, say what you did, and be straight about what the data does and does not show.\n"
    "At each step output JSON with EITHER one tool call OR the final reply, never both. A tool call is {\"action\": {\"tool\": name, \"args\": {...}}}; the final answer is {\"reply\": markdown}.\n"
    "Rules: (1) Use a tool when the answer depends on the data; answer directly (no tool) for greetings, definitions and general advice. (2) Call tools one at a time and read the "
    "observation before deciding the next step; most questions need one or two. (3) Only use column names from the dataset description. (4) Every number in your reply must come from "
    "an observation or the dataset description; never invent or estimate figures. (5) Lead with the answer, then the supporting figures, then what to check or do next; keep it short. "
    "(6) Flagged rows or correlations are leads, not proof: say so when it matters. (7) Use propose_analysis ONLY when the person clearly wants a complete modelling analysis or a model that "
    "meets an accuracy goal; for a quick 'what drives X' use drivers. (8) If a tool returns an error, fix the call or explain the limitation; do not repeat the same failing call. "
    "(9) Tables and charts from tools are shown to the person automatically: refer to them, do not re-type tables. (10) If no dataset is loaded and the question needs one, say what to add."
)


# what the model may know about the data
def digest(df: pd.DataFrame, name: str, share_labels: bool = True) -> Dict[str, Any]:
    """Column names, types and aggregates. No rows, no free text. Category labels with counts only for low-cardinality columns (and only if the person allows it)."""
    cols = []
    for c in list(df.columns)[:80]:
        s = df[c]
        k = profiling.column_kind(s)
        d: Dict[str, Any] = {"name": str(c), "type": k, "distinct": int(s.nunique()), "missing_pct": round(float(100 * s.isna().mean()), 1)}
        if k == "numeric" and s.notna().any():
            d.update(min=_f(s.min()), median=_f(s.median()), max=_f(s.max()))
        elif k == "datetime":
            dt = pd.to_datetime(s, errors="coerce")
            d.update(first=str(dt.min().date()) if dt.notna().any() else None, last=str(dt.max().date()) if dt.notna().any() else None)
        elif share_labels and 1 < s.nunique() <= 8:
            d["top_values"] = {str(i): int(v) for i, v in s.astype(str).where(s.notna()).value_counts().head(6).items()}
        cols.append(d)
    return {"name": name, "rows": int(len(df)), "columns": int(df.shape[1]), "column_list": cols}


def _f(x: Any) -> Any:
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


# step validation
def validate_step(d: Any) -> Dict[str, Any]:
    if not isinstance(d, dict):
        raise ValueError("answer with a JSON object")
    act, rep = d.get("action"), d.get("reply")
    if act and rep:
        raise ValueError("give either an action or a reply, not both")
    if not act and not (isinstance(rep, str) and rep.strip()):
        raise ValueError("give an action {tool, args} or a non-empty reply")
    if act and not (isinstance(act, dict) and isinstance(act.get("tool"), str)):
        raise ValueError("action must be {\"tool\": name, \"args\": {...}}")
    fu = d.get("followups")
    return {"reasoning": str(d.get("reasoning", ""))[:300], "action": {"tool": act["tool"], "args": act.get("args") if isinstance(act.get("args"), dict) else {}} if act else None,
            "reply": rep.strip() if isinstance(rep, str) else None, "followups": [str(x)[:90] for x in fu][:3] if isinstance(fu, list) else []}


def _pool(obs: List[Dict[str, Any]], dg: Optional[Dict[str, Any]], message: str) -> List[float]:
    acc: List[float] = []
    claims._flatten([o.get("summary") for o in obs], acc)
    if dg:
        claims._flatten(dg, acc)
    acc += [n["val"] for n in claims._numbers(message)]
    return acc


def unverified_figures(reply: str, obs: List[Dict[str, Any]], dg: Optional[Dict[str, Any]], message: str) -> List[str]:
    pool = _pool(obs, dg, message)
    out = []
    for c in claims._numbers(reply):
        if not claims._match(c, pool):
            out.append(c["raw"])
    return out[:6]


def _compact_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in (history or [])[-8:]:
        out.append({"role": m.get("role"), "text": str(m.get("text", ""))[:700], **({"tools_used": [s["tool"] for s in m.get("steps", []) if s.get("ok")]} if m.get("steps") else {})})
    return out


def _obs_payload(obs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for o in obs:
        s = json.dumps(o.get("summary"), default=str) if "summary" in o else ""
        out.append({**o, "summary": json.loads(s) if len(s) <= MAX_OBS_CHARS else {"truncated": True, "text": s[:MAX_OBS_CHARS]}} if "summary" in o else o)
    return out


# the loop
def respond(message: str, history: Optional[List[Dict[str, Any]]] = None, ctx: Optional[T.ToolContext] = None, share_labels: bool = True, max_steps: int = MAX_STEPS,
            on_event: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """-> {text, steps, artifacts, proposal, followups, warnings, calls, seconds}. Never raises for model/network problems: they come back as a plain explanation."""
    t0 = time.time()
    emit = on_event or (lambda e: None)
    has_data = ctx is not None and ctx.df is not None and len(ctx.df) > 0
    dg = digest(ctx.df, ctx.name, share_labels) if has_data else None
    has_run = bool(ctx is not None and ctx.run_state is not None and ctx.answer_run is not None)
    obs: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    warnings: List[str] = []
    proposal: Optional[Dict[str, Any]] = None
    calls = 0
    violation: Optional[str] = None
    retried_figures = False
    reply: Optional[str] = None
    followups: List[str] = []
    state: Dict[str, Any] = {}

    for step_no in range(1, max_steps + 1):
        payload = {"message": message, "conversation": _compact_history(history or []), "dataset": dg, "tools": T.catalog(has_run), "observations": _obs_payload(obs),
                   "steps_left": max_steps - step_no + 1, "a_finished_analysis_is_open": has_run,
                   "output_schema": {"reasoning": "one short sentence on why", "action": {"tool": "name", "args": {}}, "reply": "markdown, only when you are done", "followups": ["up to 3 short next questions"]}}
        if violation:
            payload["problem_with_your_last_reply"] = violation
        emit({"type": "thinking", "step": step_no})
        try:
            d, warns, c = ask(state, "assistant_step", SYSTEM, payload, 1800, validate_step)
        except llm.LLMRequired as e:
            msg = str(e)
            warnings.append("The language model could not be reached.")
            reply = ("I can't reach the language model right now, so I can't interpret questions. " + ("Check the connection settings. " if "key" in msg.lower() else "Please try again in a moment. ") +
                     "The Data page still lets you browse, filter and profile the dataset by hand.")
            break
        calls += len(c)
        warnings += [w for w in warns if w]
        violation = None
        if d["action"] is None:
            figs = unverified_figures(d["reply"], obs, dg, message)
            if figs and not retried_figures and step_no < max_steps:
                retried_figures = True
                violation = f"These figures are not in any observation or the dataset description: {figs}. Rewrite the reply using only figures that appear there (or call a tool to get them)."
                steps.append({"tool": "check", "ok": False, "note": f"Reply held back: {len(figs)} figure(s) could not be verified", "args": {}})
                continue
            reply, followups = d["reply"], d["followups"]
            if figs:
                warnings.append(f"Some figures in this answer could not be matched to the data: {', '.join(figs)}.")
            break
        tool, args = d["action"]["tool"], d["action"]["args"]
        emit({"type": "tool_start", "tool": tool, "args": args, "reasoning": d["reasoning"]})
        try:
            res = T.run_tool(ctx or T.ToolContext(pd.DataFrame()), tool, args)
        except T.ToolError as e:
            obs.append({"tool": tool, "args": args, "error": str(e)})
            steps.append({"tool": tool, "args": args, "ok": False, "note": str(e)[:200], "reasoning": d["reasoning"]})
            emit({"type": "tool_error", "tool": tool, "error": str(e)})
            continue
        obs.append({"tool": tool, "args": args, "title": res.title, "summary": res.summary})
        steps.append({"tool": tool, "args": args, "ok": True, "note": res.text, "reasoning": d["reasoning"], "seconds": res.summary.get("_seconds")})
        art = {"title": res.title, "tool": tool, "table": table_payload(res.table), "figure": res.figure, "notes": res.notes}
        if art["table"] or art["figure"]:
            artifacts.append(art)
        if res.proposal:
            proposal = res.proposal
        emit({"type": "tool_done", "tool": tool, "artifact": art, "text": res.text})
    if reply is None:                                                    # ran out of steps without an answer: say so, show what was found
        warnings.append("I stopped after reaching the step limit.")
        found = [s["note"] for s in steps if s.get("ok")]
        reply = ("I ran out of steps before finishing, but here is what I did find:\n\n" + "\n".join(f"- {f}" for f in found)) if found else "I wasn't able to answer that with the tools available. Could you rephrase or narrow the question?"
    return {"text": reply, "steps": steps, "artifacts": artifacts, "proposal": proposal, "followups": followups, "warnings": list(dict.fromkeys(warnings)), "calls": calls, "seconds": round(time.time() - t0, 2)}


def table_payload(df: Optional[pd.DataFrame], max_rows: int = 200) -> Optional[Dict[str, Any]]:
    """A table in a JSON-safe form (so a conversation can be stored and replayed exactly)."""
    if df is None or len(df.columns) == 0:
        return None
    d = df.head(max_rows)
    return {"columns": [str(c) for c in d.columns], "rows": jsonable(d.astype(object).where(d.notna(), None).values.tolist()), "total_rows": int(len(df))}


def suggestions(df: Optional[pd.DataFrame]) -> List[str]:
    """Starter questions that fit THIS dataset (deterministic: derived from column types, no model call)."""
    if df is None or df.empty:
        return ["What can you help me with?", "How do I know if a model is good enough to trust?", "Explain the difference between correlation and causation"]
    kinds = {c: profiling.column_kind(df[c]) for c in df.columns}
    num = [c for c, k in kinds.items() if k == "numeric" and df[c].nunique() > 8]
    cat = [c for c, k in kinds.items() if k in ("categorical", "boolean") and 1 < df[c].nunique() <= 12]
    dt = [c for c, k in kinds.items() if k == "datetime"]
    out = ["Give me an overview of this data", "Find outliers"]
    if df.isna().any().any():
        out.append("Where is data missing?")
    if num and dt:
        out.append(f"How does {num[0]} change over time?")
    if num and cat:
        out.append(f"Does {num[0]} differ by {cat[0]}?")
    if num:
        out.append(f"What drives {num[-1]}?")
    if len(num) >= 3:
        out.append("Which columns are strongly correlated?")
    return out[:6]
