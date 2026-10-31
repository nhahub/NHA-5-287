"""Single-document understanding: analyse, summarise, and answer questions about ONE document (PDF, Word, text, HTML ...).

Local (always, nothing leaves the machine): reading time, structure, keywords, key numbers / dates / links, an EXTRACTIVE summary
(the most representative sentences, chosen by TF-IDF), language/script, extraction problems (e.g. scanned PDF without a text layer).
AI (needs the user's consent, the TEXT is sent to the LLM provider): an abstractive summary with key points and action items;
long documents are summarised chunk by chunk (map-reduce) and the answer says how much of the document it covered.
Questions are answered only from retrieved passages of the document.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import media, mediamining
from .agents._util import ANALYST_PERSONA, ask

CHUNK_CHARS = 6000
MAX_CHUNKS = 12
SINGLE_CALL_CHARS = 9000
_DATE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4})\b")
_MONEY = re.compile(r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|EGP|SAR|AED)\b)")
_SENT_SPLIT = re.compile(r"(?<=[.!?؟])\s+|\n{2,}")


def sentences(text: str) -> List[str]:
    out = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip().split()) >= 5]
    return out[:5000]


def extractive_summary(text: str, n: int = 5) -> List[str]:
    """The n sentences closest to the document's overall TF-IDF centroid, in their original order."""
    sents = sentences(text)
    if len(sents) <= n:
        return sents
    try:
        arabic = mediamining.script_share(pd.Series([text])).get("arabic", 0) > 0.3
        X = TfidfVectorizer(stop_words=None if arabic else "english", sublinear_tf=True).fit_transform(sents)
        centroid = np.asarray(X.mean(axis=0)).ravel()
        score = X @ centroid
        score = np.asarray(score).ravel()
        pick = sorted(np.argsort(-score)[:n])
        return [sents[i] for i in pick]
    except ValueError:
        return sents[:n]


def keywords(text: str, k: int = 12) -> List[str]:
    arabic = mediamining.script_share(pd.Series([text])).get("arabic", 0) > 0.3
    try:
        cv = CountVectorizer(ngram_range=(1, 2), stop_words=None if arabic else "english", max_features=300, min_df=1)
        X = cv.fit_transform([text])
    except ValueError:
        return []
    counts = np.asarray(X.sum(axis=0)).ravel()
    terms = cv.get_feature_names_out()
    order = counts.argsort()[::-1]
    out: List[str] = []
    for i in order:
        t = terms[i]
        if counts[i] < 2 or any(t in o or o in t for o in out):        # skip words that only repeat an already chosen phrase
            continue
        out.append(t)
        if len(out) >= k:
            break
    return out


def analyze(path: Path) -> Dict[str, Any]:
    text, pages, note = media.extract_text(path)
    clean = re.sub(r"[ \t]+", " ", text).strip()
    if len(clean.split()) < 5:
        raise media.MediaError(note or "I could not extract any text from that file.")
    st = media.text_stats(clean)
    heads = [l.strip() for l in text.splitlines() if 3 <= len(l.strip()) <= 70 and not l.strip().endswith((".", ",", ";")) and (l.isupper() or re.match(r"^(\d+[.)]|[A-Z][\w ]{2,})", l.strip()))]
    return {"name": path.name, "text": clean, "pages": pages, "note": note, "stats": st, "reading_minutes": round(st["n_words"] / 200, 1),
            "keywords": keywords(clean), "extractive_summary": extractive_summary(clean, 5), "headings": heads[:12],
            "dates": sorted(set(_DATE.findall(clean)))[:8], "amounts": sorted(set(_MONEY.findall(clean)))[:8],
            "script": mediamining.script_share(pd.Series([clean])), "truncated": len(text) > media.LIMITS["max_doc_chars"]}


def format_local(a: Dict[str, Any]) -> str:
    s = a["stats"]
    L = [f"**{a['name']}**: {s['n_words']:,} words" + (f", {a['pages']} page(s)" if a["pages"] else "") + f", about {a['reading_minutes']} min to read."]
    if a.get("note"):
        L.append(f"*Note: {a['note']}.*")
    if a["script"].get("arabic", 0) > 0.3:
        L.append("Arabic script detected: English readability and stop-word handling are not applied.")
    elif s.get("readability") is not None:
        L.append(f"Readability (Flesch) {s['readability']:.0f}.")
    if a["keywords"]:
        L.append("**Key terms:** " + ", ".join(a["keywords"]) + ".")
    if a["headings"]:
        L.append("**Possible headings:** " + "; ".join(a["headings"][:8]) + ".")
    if a["dates"] or a["amounts"]:
        L.append("**Dates / amounts mentioned:** " + ", ".join(a["dates"][:5] + a["amounts"][:5]) + ".")
    L.append("**Most representative sentences (extractive summary, no AI):**\n" + "\n".join(f"- {x}" for x in a["extractive_summary"]))
    return "\n\n".join(L)


def _chunks(text: str) -> List[str]:
    paras, cur, out = re.split(r"(?<=[.!?؟])\s+", text), "", []
    for p in paras:
        if len(cur) + len(p) > CHUNK_CHARS and cur:
            out.append(cur)
            cur = ""
        cur += p + " "
    if cur.strip():
        out.append(cur)
    return out


def summarize_with_ai(state: Dict[str, Any], a: Dict[str, Any], focus: Optional[str] = None) -> Dict[str, Any]:
    """Abstractive summary. THE TEXT IS SENT TO THE LLM PROVIDER (caller must have consent)."""
    text = a["text"]
    chunks = [text] if len(text) <= SINGLE_CALL_CHARS else _chunks(text)
    covered = f"all {len(chunks)} part(s)"
    if len(chunks) > MAX_CHUNKS:                                                  # sample evenly, and say so
        idx = np.linspace(0, len(chunks) - 1, MAX_CHUNKS).astype(int)
        chunks = [chunks[i] for i in idx]
        covered = f"{MAX_CHUNKS} evenly spaced parts out of {len(_chunks(text))} (the document is long)"
    lang = "Arabic" if a["script"].get("arabic", 0) > 0.3 else "the same language as the document"
    schema = {"summary": "5-8 sentence summary", "key_points": ["5-8 short bullet points"], "action_items": ["tasks / decisions mentioned, may be empty"],
              "topics": ["3-6 topics"], "questions_to_ask": ["2-3 useful follow-up questions"]}

    def v(d):
        s = str(d.get("summary", "")).strip()
        if not s:
            raise ValueError("write a summary")
        return {"summary": s[:2500], "key_points": [str(x)[:300] for x in (d.get("key_points") or [])][:10],
                "action_items": [str(x)[:300] for x in (d.get("action_items") or [])][:8], "topics": [str(x)[:60] for x in (d.get("topics") or [])][:8],
                "questions_to_ask": [str(x)[:200] for x in (d.get("questions_to_ask") or [])][:4]}
    if len(chunks) == 1:
        out, _, _ = ask(state, "summarize_document", ANALYST_PERSONA + f" Summarise strictly from the document; write in {lang}.",
                        {"document_name": a["name"], "focus": focus, "text": chunks[0], "output_schema": schema}, 1300, v)
        return {**out, "covered": covered}
    partial = []
    for i, c in enumerate(chunks):
        r, _, _ = ask(state, "summarize_chunk", ANALYST_PERSONA + f" Summarise this part of a longer document in {lang}.",
                      {"part": i + 1, "of": len(chunks), "text": c, "output_schema": {"summary": "3-5 sentences", "key_points": ["3-5 bullets"]}}, 700,
                      lambda d: {"summary": str(d["summary"])[:1200], "key_points": [str(x)[:250] for x in (d.get("key_points") or [])][:6]})
        partial.append(r)
    out, _, _ = ask(state, "summarize_document", ANALYST_PERSONA + f" Combine the part summaries into one coherent summary in {lang}. Use only the supplied summaries.",
                    {"document_name": a["name"], "focus": focus, "part_summaries": partial, "output_schema": schema}, 1300, v)
    return {**out, "covered": covered}


def format_ai(r: Dict[str, Any]) -> str:
    L = ["**Summary:** " + r["summary"]]
    if r["key_points"]:
        L.append("**Key points:**\n" + "\n".join(f"- {x}" for x in r["key_points"]))
    if r["action_items"]:
        L.append("**Actions / decisions mentioned:**\n" + "\n".join(f"- {x}" for x in r["action_items"]))
    if r["topics"]:
        L.append("**Topics:** " + ", ".join(r["topics"]) + ".")
    L.append(f"*Based on {r['covered']}. AI summaries can miss or misstate details; check important facts against the document.*")
    if r["questions_to_ask"]:
        L.append("You could ask: " + " · ".join(r["questions_to_ask"]))
    return "\n\n".join(L)


def as_frame(a: Dict[str, Any]) -> pd.DataFrame:
    """One-row table in the same shape media.build_dataset writes, so media_search can index it."""
    return pd.DataFrame([{"document": a["name"], "text": a["text"]}])
