"""Tasks on a collection of files: keyword search over document passages, "find similar" for documents and images,
and (opt-in) grounded question answering that sends only the few retrieved passages to the LLM.
Retrieval is deterministic TF-IDF / feature similarity; nothing is sent anywhere unless the user explicitly asks a question with consent."""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import featurizers

PASSAGE_WORDS = 110


def passages(df: pd.DataFrame, text_col: str, name_col: str, label_col: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        words = str(r[text_col]).split()
        for i in range(0, max(len(words), 1), PASSAGE_WORDS):
            chunk = " ".join(words[i:i + PASSAGE_WORDS])
            if chunk.strip():
                rows.append({"document": r[name_col], "label": r[label_col] if label_col and label_col in df else None, "start_word": i, "text": chunk})
    return pd.DataFrame(rows)


class DocIndex:
    def __init__(self, df: pd.DataFrame):
        text_col = "text" if "text" in df else None
        name_col = next((c for c in ("document", "file", "filename", "name") if c in df.columns), None)
        if text_col is None or name_col is None:
            raise ValueError("this collection has no text column to search")
        self.p = passages(df, text_col, name_col, "label" if "label" in df else None)
        if self.p.empty:
            raise ValueError("no document contains extractable text")
        self.vec = TfidfVectorizer(sublinear_tf=True, max_df=0.95)
        self.X = self.vec.fit_transform(self.p["text"])
        self.df, self.name_col = df.reset_index(drop=True), name_col

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = self.vec.transform([query])
        if q.nnz == 0:
            return []
        sims = cosine_similarity(q, self.X).ravel()
        order = np.argsort(-sims)
        out, seen = [], {}
        for i in order:
            if sims[i] <= 0 or len(out) >= k:
                break
            doc = self.p.iloc[i]["document"]
            if seen.get(doc, 0) >= 2:                          # at most 2 passages per document
                continue
            seen[doc] = seen.get(doc, 0) + 1
            txt = self.p.iloc[i]["text"]
            terms = [t for t in re.findall(r"\w+", query.lower()) if t in txt.lower()]
            pos = min([txt.lower().find(t) for t in terms] or [0])
            out.append({"document": doc, "label": self.p.iloc[i]["label"], "score": round(float(sims[i]), 3),
                        "snippet": ("..." if pos > 60 else "") + txt[max(0, pos - 60):pos + 240] + "..."})
        return out

    def similar_to_text(self, text: str, k: int = 5, exclude: Optional[str] = None) -> List[Dict[str, Any]]:
        d = self.vec.transform([text])
        if d.nnz == 0:
            return []
        doc_sim = pd.DataFrame({"document": self.p["document"], "s": cosine_similarity(d, self.X).ravel()}).groupby("document")["s"].max().sort_values(ascending=False)
        return [{"document": n, "similarity": round(float(s), 3)} for n, s in doc_sim.items() if n != exclude][:k]


def image_similarity(df: pd.DataFrame, image_col: str, root: Path, query_path: Path, k: int = 5) -> List[Dict[str, Any]]:
    """Nearest images by simple visual features (colour, brightness, edges, thumbnail), standardised then cosine."""
    q = featurizers.handcrafted_image_features(query_path)
    if q is None:
        raise ValueError("that image could not be read")
    feats, names = [], []
    for p in df[image_col]:
        f = featurizers.handcrafted_image_features(Path(root) / p)
        if f is not None:
            feats.append(f)
            names.append(p)
    M = np.vstack(feats)
    mu, sd = M.mean(axis=0), M.std(axis=0) + 1e-6
    sims = cosine_similarity(((q - mu) / sd).reshape(1, -1), (M - mu) / sd).ravel()
    return [{"image": names[i], "similarity": round(float(sims[i]), 3)} for i in np.argsort(-sims)[:k]]


def answer(state, question: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Opt-in grounded answer. ONLY the retrieved passages leave the machine; the answer must cite documents from `hits`."""
    from .agents._util import ANALYST_PERSONA, ask
    allowed = [h["document"] for h in hits]
    payload = {"question": question, "passages": [{"document": h["document"], "text": h["snippet"][:500]} for h in hits],
               "rules": ["Answer ONLY from the passages. If they do not contain the answer, say so.", "Cite the documents you used."],
               "output_schema": {"answer": "text", "sources": ["document names"]}}

    def v(d):
        a = str(d.get("answer", "")).strip()
        src = [s for s in (d.get("sources") or []) if s in allowed]
        if not a:
            raise ValueError("give an answer")
        if not src and "do not" not in a.lower() and "not contain" not in a.lower():
            raise ValueError(f"cite at least one of {allowed}")
        return {"answer": a[:1200], "sources": src}
    out, _, _ = ask(state, "document_question", ANALYST_PERSONA + " You answer questions strictly from supplied passages.", payload, 600, v)
    return out
