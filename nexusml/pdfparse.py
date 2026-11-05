"""PDF ingestion: tables, multi-page tables and unstructured text blocks -> pandas.

Backends are tried best-first and each one is optional:
  1. PyMuPDF (`pip install pymupdf`)  fastest; native table finder + layout-aware text blocks
  2. pdfplumber                       robust table extraction (ruled and whitespace-aligned tables)
  3. pypdf                            text only (always available, it is in requirements.txt)

What you get from `parse(raw)`:
  tables       list of DataFrames. A table that continues over several pages (same columns, header repeated or not) is stitched
               into ONE frame. Header row detected, duplicate / empty headers repaired, numbers coerced ("$1,200", "(45)", "12%").
  blocks       DataFrame [page, block, kind, text, n_words]: paragraphs of running text with page headers/footers removed
               and words broken across lines ("custo-\nmer") re-joined.
  backend, pages, notes.

`to_dataframe(raw)` is what the data pipeline uses: the largest table when the PDF has tables, otherwise the text blocks.
Scanned (image-only) PDFs have no text layer: that is reported in `notes`, never silently returned as an empty table.
"""
import io
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MAX_PAGES = 400
MIN_TABLE_ROWS = 2          # data rows (excluding the header) for a grid to count as a table
MIN_TABLE_COLS = 2


# ---------------------------------------------------------------- backends
def _have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def available_backends() -> List[str]:
    return [n for n, m in (("pymupdf", "fitz"), ("pdfplumber", "pdfplumber"), ("pypdf", "pypdf")) if _have(m)]


def _raw_pymupdf(raw: bytes, max_pages: int) -> Tuple[List[List[List[Any]]], List[str], int, List[str]]:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=raw, filetype="pdf")
    if doc.needs_pass:
        raise ValueError("The PDF is password-protected.")
    tables: List[List[List[Any]]] = []
    page_tables: List[int] = []
    texts: List[str] = []
    for i in range(min(len(doc), max_pages)):
        page = doc[i]
        found = []
        try:
            found = list(page.find_tables().tables)
        except Exception:
            found = []
        boxes = []
        for t in found:
            rows = t.extract()
            if rows:
                tables.append(rows)
                page_tables.append(i + 1)
                boxes.append(fitz.Rect(t.bbox))
        # running text WITHOUT the cells already captured as tables
        blocks = page.get_text("blocks", sort=True)
        keep = []
        for b in blocks:
            if b[6] != 0:                                 # 0 = text block (1 = image)
                continue
            r = fitz.Rect(b[:4])
            if any((r & bx).get_area() > 0.5 * max(r.get_area(), 1e-6) for bx in boxes):
                continue
            keep.append(b[4].strip())
        texts.append("\n\n".join(k for k in keep if k))
    return list(zip(page_tables, tables)), texts, len(doc), []


def _raw_pdfplumber(raw: bytes, max_pages: int):
    import pdfplumber
    tables, texts = [], []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        n = len(pdf.pages)
        for i, page in enumerate(pdf.pages[:max_pages]):
            found = []
            for settings in ({}, {"vertical_strategy": "text", "horizontal_strategy": "text", "snap_tolerance": 3, "intersection_tolerance": 5}):
                try:
                    found = [t for t in page.extract_tables(settings) if t and len(t) >= 2 and max(len(r) for r in t) >= MIN_TABLE_COLS]
                except Exception:
                    found = []
                if settings == {} and found:
                    break                                    # ruled tables found: do not also run the (noisier) whitespace strategy
                if settings != {} and found:                 # whitespace strategy: keep only grids that look like real tables
                    found = [t for t in found if _grid_quality(t) >= 0.6]
            for t in found:
                tables.append((i + 1, t))
            try:                                             # text outside table areas
                tbs = [tb.bbox for tb in page.find_tables()] if found else []
                p2 = page
                for bb in tbs:
                    p2 = p2.outside_bbox(bb)
                texts.append(_plumber_text(p2))
            except Exception:
                texts.append(page.extract_text() or "")
    return tables, texts, n, []


_HEAD = "\u0001H\u0001"          # sentinel marking a heading paragraph between a backend and _blocks_frame


def _plumber_text(page) -> str:
    """Page text as paragraphs: a blank line wherever the vertical gap is clearly larger than normal line spacing, and headings
    (larger or bold, short lines) marked. pdfplumber's plain extract_text() has no paragraph breaks at all."""
    try:
        lines = page.extract_text_lines(return_chars=True)
    except Exception:
        return page.extract_text() or ""
    if not lines:
        return ""
    sizes = [np.median([c.get("size", 0) for c in l["chars"]]) for l in lines if l.get("chars")]
    body = float(np.median(sizes)) if sizes else 0.0
    line_h = float(np.median([l["bottom"] - l["top"] for l in lines])) or 10.0
    gaps = [lines[i + 1]["top"] - lines[i]["bottom"] for i in range(len(lines) - 1)]
    tight = float(np.percentile([g for g in gaps if g >= 0] or [0.0], 25))          # ordinary line spacing (robust to a few huge gaps)
    para_gap = max(0.55 * line_h, 2.0 * tight)
    out, prev_head = [], False
    for i, l in enumerate(lines):
        chars = l.get("chars") or []
        size = float(np.median([c.get("size", 0) for c in chars])) if chars else body
        bold = bool(chars) and sum("bold" in str(c.get("fontname", "")).lower() for c in chars) > 0.6 * len(chars)
        head = len(l["text"]) < 90 and ((body and size >= 1.15 * body) or (bold and size >= body * 0.99 and not l["text"].rstrip().endswith((".", ",", ";"))))
        gap = (l["top"] - lines[i - 1]["bottom"]) if i else 0.0
        if i and (head or prev_head or gap > para_gap):
            out.append("")
        out.append((_HEAD if head else "") + l["text"])
        prev_head = head
    return "\n".join(out)


def _raw_pypdf(raw: bytes, max_pages: int):
    from pypdf import PdfReader
    r = PdfReader(io.BytesIO(raw))
    if r.is_encrypted:
        try:
            r.decrypt("")
        except Exception as e:
            raise ValueError("The PDF is password-protected.") from e
    n = len(r.pages)
    return [], [(r.pages[i].extract_text() or "") for i in range(min(n, max_pages))], n, []


def _grid_quality(t: List[List[Any]]) -> float:
    """Share of non-empty cells in a candidate grid: ordinary paragraphs split by a whitespace strategy are mostly empty."""
    cells = [c for r in t for c in r]
    return sum(1 for c in cells if c not in (None, "")) / max(1, len(cells))


# ---------------------------------------------------------------- table cleaning
_NUM = re.compile(r"^\(?[-+]?[$€£¥]?\s*\d[\d,\s]*(\.\d+)?\s*%?\)?$")


def _to_num(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s or not _NUM.match(s):
        return v
    neg = (s.startswith("(") and s.endswith(")")) or s.startswith("-")             # accounting negatives: (45) -> -45
    s = re.sub(r"[()$€£¥%,\s+\-]", "", s)                                           # currency, thousands separators, percent sign
    try:
        x = float(s)
    except ValueError:
        return v
    return -x if neg else x                                                       # a percentage keeps its printed number (12% -> 12.0)


def _clean_cell(c: Any) -> Any:
    if c is None:
        return None
    s = re.sub(r"\s+", " ", str(c).replace("\n", " ")).strip()
    return s or None


def _looks_like_header(row: List[Any]) -> bool:
    """A header row is mostly filled-in, mostly non-numeric labels."""
    cells = [c for c in row if c]
    if len(cells) < max(2, len(row) // 2):
        return False
    return sum(1 for c in cells if not isinstance(_to_num(str(c)), float)) / len(cells) >= 0.6


def _unique_names(names: List[Any]) -> List[str]:
    out, seen = [], {}
    for i, n in enumerate(names):
        base = re.sub(r"\s+", " ", str(n)).strip() if n not in (None, "") else f"col_{i + 1}"
        k = seen.get(base, 0)
        seen[base] = k + 1
        out.append(base if k == 0 else f"{base}_{k + 1}")
    return out


def _grid_to_frame(grid: List[List[Any]], page: int) -> Optional[Dict[str, Any]]:
    rows = [[_clean_cell(c) for c in r] for r in grid if r]
    width = max((len(r) for r in rows), default=0)
    rows = [r + [None] * (width - len(r)) for r in rows]
    rows = [r for r in rows if any(c is not None for c in r)]
    if len(rows) < 1 + MIN_TABLE_ROWS or width < MIN_TABLE_COLS:
        return None
    keep = [j for j in range(width) if any(r[j] is not None for r in rows)]       # drop fully empty columns
    rows = [[r[j] for j in keep] for r in rows]
    head, body = rows[0], rows[1:]
    has_header = _looks_like_header(head)
    cols = _unique_names(head) if has_header else [f"col_{i + 1}" for i in range(len(keep))]
    data = body if has_header else rows
    df = pd.DataFrame(data, columns=cols)
    return {"df": df, "page": page, "has_header": has_header, "cols": cols, "header": head if has_header else None}


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        conv = df[c].map(_to_num)
        num = pd.to_numeric(conv, errors="coerce")
        nonnull = df[c].notna().sum()
        if nonnull and num.notna().sum() >= 0.8 * nonnull:                          # column is numeric (allowing a few footnote cells)
            df[c] = num.astype("Int64") if (num.dropna() % 1 == 0).all() and num.notna().all() else num      # whole numbers stay integers
        else:
            df[c] = df[c].where(df[c].notna(), None)
    return df.dropna(how="all").reset_index(drop=True)


def _stitch(frames: List[Dict[str, Any]]) -> List[pd.DataFrame]:
    """Join tables that continue across pages: same width and (same header OR a headerless continuation on the next page)."""
    groups: List[Dict[str, Any]] = []
    for f in frames:
        w = f["df"].shape[1]
        prev = groups[-1] if groups else None
        cont = prev is not None and prev["width"] == w and f["page"] in (prev["last_page"], prev["last_page"] + 1) and (
            (f["has_header"] and prev["cols"] == f["cols"]) or not f["has_header"])
        if cont:
            d = f["df"].copy()
            if not f["has_header"]:
                d.columns = prev["cols"]
            prev["dfs"].append(d)
            prev["last_page"] = f["page"]
        else:
            groups.append({"width": w, "cols": f["cols"], "dfs": [f["df"]], "last_page": f["page"], "first_page": f["page"]})
    out = []
    for g in groups:
        df = pd.concat(g["dfs"], ignore_index=True)
        if g["dfs"] and len(g["dfs"]) > 1:                                          # a repeated header row inside the data (some PDFs repeat it unlabelled)
            hdr = [str(c) for c in g["cols"]]
            df = df[~df.apply(lambda r: [str(x) for x in r.tolist()] == hdr, axis=1)]
        df = _finish(df)
        df.attrs.update({"first_page": g["first_page"], "last_page": g["last_page"], "n_pages": len(g["dfs"])})
        out.append(df)
    return out


# ---------------------------------------------------------------- text cleaning
_PAGE_NO = re.compile(r"^(page|p\.|pg\.?)?\s*\d+\s*((of|/)\s*\d+)?$|^-\s*\d+\s*-$", re.I)


def _strip_running_heads(texts: List[str]) -> List[str]:
    """Remove page furniture: lines that repeat VERBATIM at the top/bottom of >= 50% of pages (running headers/footers) and bare
    page numbers ("3", "Page 3 of 9"). Body lines and section headings are never touched: only the first/last 3 lines are candidates
    and a line that merely contains a digit is not a repeat."""
    if len(texts) < 3:
        return [_drop_page_numbers(t) for t in texts]
    count: Dict[str, int] = {}
    for t in texts:
        ls = [x.replace(_HEAD, "").strip() for x in t.splitlines() if x.strip()]
        for l in set(ls[:3] + ls[-3:]):
            count[l] = count.get(l, 0) + 1
    common = {l for l, n in count.items() if n >= max(3, 0.5 * len(texts)) and not _PAGE_NO.match(l)}
    out = []
    for t in texts:
        ls = t.splitlines()
        idx = [i for i, x in enumerate(ls) if x.strip()]
        edge = set(idx[:3] + idx[-3:])
        out.append(_drop_page_numbers("\n".join(l for i, l in enumerate(ls) if not (i in edge and l.replace(_HEAD, "").strip() in common))))
    return out


def _drop_page_numbers(t: str) -> str:
    ls = t.splitlines()
    idx = [i for i, x in enumerate(ls) if x.strip()]
    edge = set(idx[:2] + idx[-2:])
    return "\n".join(l for i, l in enumerate(ls) if not (i in edge and _PAGE_NO.match(l.replace(_HEAD, "").strip())))


def _paragraphs(text: str) -> List[str]:
    text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)                                     # re-join words hyphenated at a line break
    paras = []
    for chunk in re.split(r"\n\s*\n", text):
        p = re.sub(r"\s*\n\s*", " ", chunk).strip()
        p = re.sub(r"\s{2,}", " ", p)
        if len(p) >= 3:
            paras.append(p)
    return paras


def _kind(p: str) -> str:
    if p.startswith(_HEAD):
        return "heading"
    if len(p) < 80 and not p.endswith((".", "?", "!", ",", ";")) and (p.isupper() or p.istitle() or re.match(r"^\d+(\.\d+)*\s+\S", p)):
        return "heading"
    if re.match(r"^([-•*·▪]|\d+[.)])\s+", p):
        return "list"
    return "paragraph"


def _blocks_frame(texts: List[str]) -> pd.DataFrame:
    rows, b = [], 0
    for pg, t in enumerate(_strip_running_heads(texts), 1):
        for p in _paragraphs(t):
            b += 1
            k = _kind(p)
            p = p.replace(_HEAD, "").strip()
            rows.append({"page": pg, "block": b, "kind": k, "text": p, "n_words": len(p.split())})
    return pd.DataFrame(rows, columns=["page", "block", "kind", "text", "n_words"])


# ---------------------------------------------------------------- public API
def parse(raw: bytes, max_pages: int = MAX_PAGES, backend: Optional[str] = None) -> Dict[str, Any]:
    if not raw or not raw.lstrip()[:5] == b"%PDF-":
        raise ValueError("This is not a PDF file.")
    import os
    avail = available_backends()
    pref = backend or os.environ.get("NEXUSML_PDF_BACKEND", "")
    order = [pref] if pref in avail else sorted(avail, key=lambda b: {"pdfplumber": 0, "pymupdf": 1, "pypdf": 2}[b])
    if not order:
        raise ValueError("Reading PDFs needs one of: pymupdf, pdfplumber or pypdf (pip install pdfplumber pymupdf).")
    fn = {"pymupdf": _raw_pymupdf, "pdfplumber": _raw_pdfplumber, "pypdf": _raw_pypdf}
    last_err: Optional[Exception] = None
    best: Optional[Dict[str, Any]] = None
    for name in order:
        try:
            grids, texts, n_pages, _ = fn[name](raw, max_pages)
        except ValueError:
            raise
        except Exception as e:                                                        # a backend that chokes on this file: try the next one
            last_err = e
            continue
        frames = [f for f in (_grid_to_frame(g, pg) for pg, g in grids) if f]
        tables = _stitch(frames)
        blocks = _blocks_frame(texts)
        notes: List[str] = []
        n_chars = int(blocks["text"].str.len().sum()) if len(blocks) else 0
        if n_pages and not tables and n_chars < 20 * min(n_pages, 5):
            notes.append("The PDF has almost no text layer (scanned images?). Run OCR first, or upload the source spreadsheet.")
        if n_pages > max_pages:
            notes.append(f"Only the first {max_pages} of {n_pages} pages were read.")
        res = {"backend": name, "pages": n_pages, "tables": tables, "blocks": blocks, "notes": notes}
        if tables or n_chars >= 20:
            return res                                                                # first backend that produced something usable wins
        best = best or res
    if best is not None:
        return best
    raise ValueError(f"The PDF could not be read ({type(last_err).__name__ if last_err else 'no backend'}).")


def to_dataframe(raw: bytes, prefer: str = "auto") -> pd.DataFrame:
    """The table a data pipeline should use. 'auto': the largest table if there is one, else the text blocks."""
    res = parse(raw)
    tabs = res["tables"]
    if prefer != "text" and tabs:
        return max(tabs, key=lambda d: d.shape[0] * d.shape[1])
    if len(res["blocks"]):
        return res["blocks"][["page", "block", "kind", "text", "n_words"]]
    raise ValueError(" ".join(res["notes"]) or "No table or text could be extracted from this PDF.")


def summarize(raw: bytes) -> Dict[str, Any]:
    r = parse(raw)
    return {"backend": r["backend"], "pages": r["pages"], "n_tables": len(r["tables"]),
            "tables": [{"shape": list(t.shape), "pages": f"{t.attrs.get('first_page')}-{t.attrs.get('last_page')}", "columns": [str(c) for c in t.columns][:12]} for t in r["tables"]],
            "n_text_blocks": int(len(r["blocks"])), "notes": r["notes"]}
