"""Model-suggested analysis options. After a table is uploaded the assistant scans it (cheap, local, no LLM call) and PROPOSES settings in
the chat: which column looks like the target, whether sensitive attributes should be excluded, whether business costs matter, what it noticed
about data quality. The user can change anything and press Start. Nothing here is sent to an LLM."""
import re
from typing import Any, Dict, List

import pandas as pd

from . import fairness, featurizers, profiling

TARGET_NAMES = re.compile(r"churn|attrition|target|label|outcome|default|fraud|convert|survived|cancel|response|status|class|risk|won|lost", re.I)
FORECAST = re.compile(r"\b(forecast|next (week|month|quarter|year)|future|projection)\b", re.I)


def suggest_options(csv_path: str, request: str = "") -> Dict[str, Any]:
    df = pd.read_csv(csv_path, nrows=50000)
    prof = profiling.profile_dataset(df)
    cols = prof["columns"]
    mentioned = [c for c in df.columns if len(str(c)) > 2 and re.search(rf"\b{re.escape(str(c))}\b", request or "", re.I)]
    named = [c for c in df.columns if TARGET_NAMES.search(str(c)) and cols[c]["kind"] in ("categorical", "boolean", "numeric") and cols[c]["unique_ratio"] < 0.5]
    cand = [c for c in mentioned if c in named] or named or [c for c in mentioned if cols[c]["kind"] != "text"]
    target = cand[0] if cand else None
    why = ""
    ptype = None
    date_cols = [c for c, i in cols.items() if i["kind"] == "datetime"]
    if target:
        info = cols[target]
        ptype = "regression" if (info["kind"] == "numeric" and info["unique_count"] > 15) else "classification"
        why = (f"'{target}' " + ("was named in your question" if target in mentioned else "looks like the outcome (its name and values suggest it)")
               + f" and has {info['unique_count']} distinct values, so this is a {ptype} problem.")
        if date_cols and FORECAST.search(request or "") and info["kind"] == "numeric":
            ptype = "time_series"
            why = f"You asked for a forecast and '{date_cols[0]}' is a date column, so I would treat '{target}' as a time series."
    sens = fairness.detect_sensitive([c for c in df.columns if c != target])
    mods = featurizers.modalities(df)
    notes: List[str] = []
    if sens:
        notes.append(f"I found sensitive attributes ({', '.join(sens)}). Using them to score individuals carries fairness and legal risk, so I suggest excluding them. "
                     "A fairness check still runs on them either way.")
    miss = [c for c, i in cols.items() if i["missing_pct"] >= 20]
    if miss:
        notes.append(f"{len(miss)} column(s) have 20% or more missing values ({', '.join(miss[:4])}{'...' if len(miss) > 4 else ''}); I will handle them and say what it costs.")
    const = [c for c, i in cols.items() if i["unique_count"] <= 1]
    ids = [c for c, i in cols.items() if i["unique_ratio"] >= 0.95 and i["kind"] != "numeric"]
    if const or ids:
        notes.append(f"Columns that carry no signal will be dropped (constant: {const or 'none'}; looks like an ID: {ids[:4] or 'none'}); you approve any drop.")
    if prof["duplicates"]:
        notes.append(f"{prof['duplicates']} duplicate row(s) found.")
    if mods["text"] or mods["image"]:
        notes.append(f"Free-text column(s) {mods['text'] or 'none'} and image-path column(s) {mods['image'] or 'none'} will be turned into features.")
    costs = bool(ptype == "classification" and target and cols[target]["unique_count"] == 2)
    if costs:
        notes.append("This predicts a yes/no outcome. If you tell me what a MISSED case costs compared with a FALSE ALARM, I can choose the cutoff that fits your business; "
                     "otherwise I use a clearly labelled default.")
    if prof["n_rows"] < 300:
        notes.append(f"Only {prof['n_rows']} rows: model scores will be noisy and I will show wide confidence intervals.")
    return {"rows": prof["n_rows"], "cols": prof["n_cols"], "columns": list(df.columns), "target": target, "target_reason": why, "problem_type": ptype,
            "sensitive": sens, "exclude_sensitive": bool(sens and ptype in ("classification", "regression")), "costs_relevant": costs,
            "date_columns": date_cols, "has_text": bool(mods["text"]), "has_images": bool(mods["image"]), "notes": notes}
