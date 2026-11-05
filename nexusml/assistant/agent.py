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
MAX_BATCH = 4                  # independent tool calls the model may request in ONE step (one model round-trip)
OLD_OBS_CHARS = 1500           # observations from earlier steps are shortened when resent; the newest ones stay in full

SYSTEM = (
    "You are a helpful assistant built into a data workspace. You can talk about anything the person asks, like any good assistant: general questions, explanations, writing, advice, everyday conversation. You also have extra abilities for their data, their analyses and the files they drop in. Work like a careful colleague: "
    "do the smallest amount of work that answers the question correctly, say what you did, and be straight about what the data does and does not show.\n"
    "At each step output JSON with EITHER tool calls OR the final reply, never both. One tool call is {\"action\": {\"tool\": name, \"args\": {...}}}. When several checks do NOT depend on each other, request them together as {\"actions\": [{\"tool\": name, \"args\": {...}}, ...]} (at most " + str(MAX_BATCH) + "): every extra step is a slow, costly round-trip. The final answer is {\"reply\": markdown}.\n"
    "Rules: (1) Use a tool when the answer depends on the data; answer directly (no tool) for greetings, definitions and general advice. (2) Wait for an observation only when the next call depends on it; "
    "otherwise batch. Never repeat a call that already ran (its result is in observations) and never repeat a call that failed. Most questions need one to three tool calls in total. "
    "(3) Only use column names from the dataset description. (4) Every number in your reply must come from "
    "an observation or the dataset description; never invent or estimate figures. (5) Lead with the answer, then the supporting figures, then what to check or do next; keep it short. "
    "(6) Flagged rows or correlations are leads, not proof: say so when it matters. (7) Use propose_analysis ONLY when the person clearly wants a complete modelling analysis or a model that "
    "meets an accuracy goal; for a quick 'what drives X' use drivers. (8) If a tool returns an error, fix the call or explain the limitation; do not repeat the same failing call. "
    "(9) Tables and charts from tools are shown to the person automatically: refer to them, do not re-type tables. (10) Never refuse or hold back a general question because no dataset is loaded. Only when the question is about THEIR data and none is loaded, say what to add. "
    "(11) The 'workspace' block says which page the person is on, what is open, which datasets and analyses exist and what is running: use it to resolve 'this', 'here' and 'the open analysis'. "
    "To answer about a finished analysis (findings, metrics, trial history, statistics, KPIs, charts, the report, warnings, what the model uses) call analyses / analysis / analysis_chart; "
    "for what is running use jobs_status; for datasets, registered models and settings use workspace_info; for installed software use packages. These need no dataset. "
    "(12) When the person voices a concern or doubts a result, take it seriously: look at the relevant evidence first (audit for leakage and review issues, metrics with their uncertainty, explanation), "
    "say plainly whether the concern is supported by what the analysis shows, and say what would settle it. Offer note_concern if they want it kept with the analysis, and propose_extension if a concrete follow-up would test it. "
    "Do not defend the analysis and do not give in: report what the evidence supports. "
    "(13) To CONTINUE a finished analysis ('now build a model', 'go deeper on X', 'also check Y') use propose_extension: it reuses the earlier results and only does the new work. Use propose_analysis only for a new, separate analysis. "
    "You never start an analysis yourself; the person confirms a proposal. "
    "(14) Set \"about\" in every reply: \"general\" when the reply is general knowledge or conversation that states no fact about their data, analyses or files; otherwise \"workspace\". "
    "(15) The \"attachments\" block lists files the person dropped into this conversation (a picture you were shown, a document you summarised). Answer questions about them from that block and say plainly when it does not contain the detail asked for. "
    "(16) To summarise a finished analysis call analysis with part=summary, then write a short plain-language summary: what was asked, what was found, how reliable it is, what to do next."
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
    """Accepts one tool call ("action"), several independent ones ("actions", at most MAX_BATCH), or the final "reply". Returns all three forms
    ("action" is the first call, kept for older callers)."""
    if not isinstance(d, dict):
        raise ValueError("answer with a JSON object")
    act, acts, rep = d.get("action"), d.get("actions"), d.get("reply")
    raw: List[Any] = []
    if acts is not None:
        if not isinstance(acts, list):
            raise ValueError("actions must be a list of {\"tool\": name, \"args\": {...}}")
        raw += acts
    if act:
        raw.insert(0, act)
    raw = [a for a in raw if a]
    if raw and rep:
        raise ValueError("give tool calls or a reply, not both")
    if not raw and not (isinstance(rep, str) and rep.strip()):
        raise ValueError("give an action {tool, args}, a list of actions, or a non-empty reply")
    clean: List[Dict[str, Any]] = []
    for a in raw:
        if not (isinstance(a, dict) and isinstance(a.get("tool"), str)):
            raise ValueError("every action must be {\"tool\": name, \"args\": {...}}")
        clean.append({"tool": a["tool"], "args": a.get("args") if isinstance(a.get("args"), dict) else {}})
    dropped = max(0, len(clean) - MAX_BATCH)                         # over the batch size: run the first ones, tell the model (no extra round-trip to reject)
    clean = clean[:MAX_BATCH]
    fu = d.get("followups")
    return {"reasoning": str(d.get("reasoning", ""))[:300], "action": clean[0] if clean else None, "actions": clean, "dropped": dropped,
            "reply": rep.strip() if isinstance(rep, str) else None, "followups": [str(x)[:90] for x in fu][:3] if isinstance(fu, list) else [],
            "about": "general" if str(d.get("about", "")).strip().lower() == "general" else "workspace"}


def _pool(obs: List[Dict[str, Any]], dg: Optional[Dict[str, Any]], message: str, ws: Optional[Dict[str, Any]] = None, att: Optional[List[Dict[str, Any]]] = None) -> List[float]:
    acc: List[float] = []
    if att:                                                                # numbers in a document or a picture are inside its TEXT, so read them out of it
        claims._flatten(att, acc)
        acc += [n["val"] for n in claims._numbers(json.dumps(att, default=str))]
    claims._flatten([o.get("summary") for o in obs], acc)
    if dg:
        claims._flatten(dg, acc)
    if ws:
        claims._flatten(ws, acc)
    acc += [n["val"] for n in claims._numbers(message)]
    return acc


def unverified_figures(reply: str, obs: List[Dict[str, Any]], dg: Optional[Dict[str, Any]], message: str, ws: Optional[Dict[str, Any]] = None, att: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    pool = _pool(obs, dg, message, ws, att)
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


def _obs_payload(obs: List[Dict[str, Any]], newest_from: int = 0) -> List[Dict[str, Any]]:
    """What the model is shown. Timing noise is removed; observations from earlier steps are shortened (the model already read them), the newest stay whole."""
    out = []
    for o in obs:
        if "summary" not in o:
            out.append({k: v for k, v in o.items() if k != "step"})
            continue
        sm = {k: v for k, v in o["summary"].items() if not str(k).startswith("_")} if isinstance(o["summary"], dict) else o["summary"]
        s = json.dumps(sm, default=str)
        cap = MAX_OBS_CHARS if o.get("step", 0) >= newest_from else OLD_OBS_CHARS
        out.append({"tool": o["tool"], "args": o["args"], "title": o.get("title"), "summary": json.loads(s) if len(s) <= cap else {"truncated": True, "text": s[:cap]}})
    return out


# the loop
def respond(message: str, history: Optional[List[Dict[str, Any]]] = None, ctx: Optional[T.ToolContext] = None, share_labels: bool = True, max_steps: int = MAX_STEPS,
            on_event: Optional[Callable[[Dict[str, Any]], None]] = None, workspace: Optional[Dict[str, Any]] = None,
            attachments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """-> {text, steps, artifacts, proposal, followups, warnings, calls, seconds, stats}. Never raises for model/network problems: they come back as a plain explanation.

    Where the model calls were being wasted, and what is done about it:
      * one tool per step            -> independent tools are requested together ("actions") and run in one step
      * a repeated successful call   -> not run again; the model is told the result is already in its observations
      * a repeated FAILING call      -> not run again; after two, the model must answer with what it has (the old loop retried it until the step limit)
      * the last step used for a tool -> the last step is reply-only, so the tool call is never run and then thrown away without an answer
      * a near-miss column name      -> corrected by the tool layer (tools.resolve_column) instead of costing a round-trip to retype it
    """
    t0 = time.time()
    emit = on_event or (lambda e: None)
    has_data = ctx is not None and ctx.df is not None and len(ctx.df) > 0
    ws = workspace or None                                                # the workspace digest (workspace_ctx.digest): where the person is, what exists, what runs
    dg = digest(ctx.df, ctx.name, share_labels) if has_data else None
    has_run = bool(ctx is not None and ctx.run_state is not None and ctx.answer_run is not None)
    att = (attachments or [])[-4:] or None                               # files dropped into the conversation (attachments.context): a picture description, a document summary
    catalog = T.catalog_compact(has_run, has_data)
    obs: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    warnings: List[str] = []
    proposal: Optional[Dict[str, Any]] = None
    calls = 0
    violation: Optional[str] = None
    notes_for_model: List[str] = []
    retried_figures = False
    reply: Optional[str] = None
    followups: List[str] = []
    state: Dict[str, Any] = {}
    done: Dict[str, bool] = {}                 # call key -> already ran successfully
    failed: Dict[str, str] = {}                # call key -> why it failed
    stats = {"tool_runs": 0, "tool_batches": 0, "skipped_repeats": 0, "skipped_failures": 0, "name_corrections": 0, "payload_chars": 0}
    repeat_failures = 0
    force_reply = False
    newest_from = 1

    for step_no in range(1, max_steps + 1):
        reply_only = force_reply or step_no == max_steps
        payload = {"message": message, "conversation": _compact_history(history or []), "dataset": dg, "tools": [] if reply_only else catalog, "observations": _obs_payload(obs, newest_from), **({"workspace": ws} if ws else {}), **({"attachments": att} if att else {}),
                   "steps_left": 1 if reply_only else max_steps - step_no + 1, "a_finished_analysis_is_open": has_run,
                   "output_schema": {"reasoning": "one short sentence", "actions": [{"tool": "name", "args": {}}], "reply": "markdown, only when you are done", "about": "general | workspace", "followups": ["up to 3 short next questions"]}}
        if reply_only:
            payload["must_reply_now"] = "No more tool calls are possible. Answer with what the observations show, and say plainly what you could not check."
        if violation:
            payload["problem_with_your_last_reply"] = violation
        if notes_for_model:
            payload["notes"] = notes_for_model
        stats["payload_chars"] += len(json.dumps(jsonable(payload), default=str))
        notes_for_model, newest_from = [], step_no
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
        if not d["actions"]:
            # a plain general answer ("what is a p-value?") states nothing about the person's data, so there is nothing to verify it against; anything else is checked
            figs = [] if (d["about"] == "general" and not obs) else unverified_figures(d["reply"], obs, dg, message, ws, att)
            if figs and not retried_figures and step_no < max_steps:
                retried_figures = True
                violation = f"These figures are not in any observation or the dataset description: {figs}. Rewrite the reply using only figures that appear there (or call a tool to get them)."
                steps.append({"tool": "check", "ok": False, "note": f"Reply held back: {len(figs)} figure(s) could not be verified", "args": {}})
                continue
            reply, followups = d["reply"], d["followups"]
            if figs:
                warnings.append(f"Some figures in this answer could not be matched to the data: {', '.join(figs)}.")
            break
        if reply_only:                                                  # asked for tools when none may run: do not run them; fall through to the summary of what was found
            break
        if d["dropped"]:
            notes_for_model.append(f"Only the first {MAX_BATCH} tool calls were run; ask for the rest in the next step if you still need them.")
        stats["tool_batches"] += 1
        new_info = False
        for act in d["actions"]:
            tool, args = act["tool"], act["args"]
            key = T.call_key(tool, args)
            if key in done:                                             # already answered: do not run it again
                stats["skipped_repeats"] += 1
                notes_for_model.append(f"{tool} with these arguments already ran; use its earlier observation.")
                continue
            if key in failed:
                stats["skipped_failures"] += 1
                repeat_failures += 1
                notes_for_model.append(f"{tool} with these arguments already failed ({failed[key][:160]}); change the call or answer without it.")
                steps.append({"tool": tool, "args": args, "ok": False, "note": "Skipped: the same call already failed. " + failed[key][:160], "reasoning": d["reasoning"]})
                continue
            emit({"type": "tool_start", "tool": tool, "args": args, "reasoning": d["reasoning"]})
            try:
                res = T.run_tool(ctx or T.ToolContext(pd.DataFrame()), tool, args)
            except T.ToolError as e:
                failed[key] = str(e)
                obs.append({"tool": tool, "args": args, "error": str(e), "step": step_no})
                steps.append({"tool": tool, "args": args, "ok": False, "note": str(e)[:200], "reasoning": d["reasoning"]})
                emit({"type": "tool_error", "tool": tool, "error": str(e)})
                new_info = True
                continue
            done[key] = True
            new_info = True
            stats["tool_runs"] += 1
            stats["name_corrections"] += len(res.summary.get("column_names_corrected", []))
            obs.append({"tool": tool, "args": args, "title": res.title, "summary": res.summary, "step": step_no})
            steps.append({"tool": tool, "args": args, "ok": True, "note": res.text, "reasoning": d["reasoning"], "seconds": res.summary.get("_seconds")})
            art = {"title": res.title, "tool": tool, "table": table_payload(res.table), "figure": res.figure, "notes": res.notes, **({"image_key": res.image} if res.image else {})}
            if art["table"] or art["figure"] or art.get("image_key"):
                artifacts.append(art)
            if res.proposal:
                proposal = res.proposal
            emit({"type": "tool_done", "tool": tool, "artifact": art, "text": res.text})
        if repeat_failures >= 2:                                        # the model keeps repeating a failing call: stop spending rounds, make it answer
            force_reply = True
        elif not new_info:                                              # a whole step of repeats teaches the model nothing: one nudge, then a forced answer
            if step_no >= 2 and all("already" in n for n in notes_for_model):
                force_reply = True
    if reply is None:                                                    # ran out of steps without an answer: say so, show what was found
        if force_reply:
            warnings.append("I stopped early because the same call kept failing or repeating, so another attempt would not have helped.")
        else:
            warnings.append("I stopped after reaching the step limit.")
        found = [s["note"] for s in steps if s.get("ok")]
        reply = ("I ran out of steps before finishing, but here is what I did find:\n\n" + "\n".join(f"- {f}" for f in found)) if found else "I wasn't able to answer that with the tools available. Could you rephrase or narrow the question?"
    return {"text": reply, "steps": steps, "artifacts": artifacts, "proposal": proposal, "followups": followups, "warnings": list(dict.fromkeys(warnings)), "calls": calls, "stats": stats, "seconds": round(time.time() - t0, 2)}


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
