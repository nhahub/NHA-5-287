"""Media ingestion: images, PDFs, Word/text/HTML documents, or a ZIP of folders -> ONE analysis table the existing pipeline understands.

  * folders become class labels when the upload has >= 2 folders ("cats/", "dogs/"): -> classification
  * no folders -> unlabeled: the system finds groups / themes (clustering) and profiles the collection
  * every image / document also gets interpretable metadata columns (brightness, sharpness, word count, readability ...)
    so the normal statistics, root-cause and reporting steps work on them
Safety: ZIPs are extracted with path-traversal, size, entry-count and compression-ratio limits; only whitelisted extensions are written.
Scanned (image-only) PDFs have no text layer: they are read with OCR when the `tesseract` (+ `pdftoppm`) command-line tools are installed
(NEXUSML_OCR=0 disables it; NEXUSML_OCR_LANG sets the language, default 'eng'); otherwise they are flagged.
"""
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
DOC_EXT = {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".odt"}
TABULAR_EXT = {".csv", ".xlsx", ".xls", ".tsv", ".json", ".jsonl", ".xml", ".parquet"}
UPLOAD_TYPES = sorted(e.lstrip(".") for e in IMAGE_EXT | DOC_EXT | TABULAR_EXT) + ["zip"]
LIMITS = {"max_files": 5000, "max_total_mb": 500, "max_ratio": 200, "max_doc_chars": 30000, "max_pdf_pages": 200, "max_rows": 5000}


class MediaError(ValueError):
    pass


def kind_of(name: str) -> Optional[str]:
    e = Path(name).suffix.lower()
    return ("image" if e in IMAGE_EXT else "document" if e in DOC_EXT else "tabular" if e in TABULAR_EXT else "zip" if e == ".zip" else None)


def classify(names: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"image": [], "document": [], "tabular": [], "zip": [], "unsupported": []}
    for n in names:
        out[kind_of(n) or "unsupported"].append(n)
    return out


def _clean_rel(name: str) -> Optional[PurePosixPath]:
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or any(part.startswith(".") or part == "__MACOSX" for part in p.parts):
        return None
    return p if p.parts else None


def extract_zip(data: bytes, dest: Path) -> Dict[str, Any]:
    """Safe extraction: no path traversal, whitelisted extensions only, entry / size / compression-ratio limits."""
    dest.mkdir(parents=True, exist_ok=True)
    wrote, skipped, total = 0, 0, 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if len(infos) > LIMITS["max_files"] * 2:
            raise MediaError(f"the ZIP has {len(infos):,} entries (limit {LIMITS['max_files'] * 2:,})")
        for i in infos:
            rel = _clean_rel(i.filename)
            ext = Path(i.filename).suffix.lower()
            if rel is None or ext not in IMAGE_EXT | DOC_EXT | TABULAR_EXT:
                skipped += 1
                continue
            if i.compress_size and i.file_size / max(i.compress_size, 1) > LIMITS["max_ratio"] and i.file_size > 5_000_000:
                raise MediaError(f"'{i.filename}' has a suspicious compression ratio (possible ZIP bomb)")
            total += i.file_size
            if total > LIMITS["max_total_mb"] * 1_000_000:
                raise MediaError(f"the ZIP expands to more than {LIMITS['max_total_mb']} MB")
            target = dest.joinpath(*rel.parts)
            if dest.resolve() not in target.resolve().parents:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            wrote += 1
    return {"files": wrote, "skipped": skipped}


def save_uploads(files: List[Tuple[str, bytes]], root: Path) -> Dict[str, Any]:
    """Write uploads under `root` (ZIPs are expanded). Returns {'extracted': n, 'skipped': n}."""
    root.mkdir(parents=True, exist_ok=True)
    n = sk = 0
    for name, data in files:
        k = kind_of(name)
        if k == "zip":
            r = extract_zip(data, root)
            n, sk = n + r["files"], sk + r["skipped"]
        elif k in ("image", "document"):
            rel = _clean_rel(Path(name).name)
            if rel is None:
                sk += 1
                continue
            (root / rel.name).write_bytes(data)
            n += 1
        else:
            sk += 1
    return {"extracted": n, "skipped": sk}


# images
def image_meta(path: Path) -> Optional[Dict[str, Any]]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            rgb = im.convert("RGB")
        small = rgb.copy()
        small.thumbnail((256, 256))
        a = np.asarray(small, dtype=float)
        g = a.mean(axis=2)
        lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:] if min(g.shape) > 3 else np.zeros((1, 1))
        hsv = np.asarray(small.convert("HSV"), dtype=float)
        return {"width": int(w), "height": int(h), "aspect_ratio": round(w / max(h, 1), 3), "file_kb": round(path.stat().st_size / 1024, 1),
                "brightness": round(float(g.mean()), 1), "contrast": round(float(g.std()), 1), "sharpness": round(float(lap.var()), 1),
                "mean_red": round(float(a[..., 0].mean()), 1), "mean_green": round(float(a[..., 1].mean()), 1), "mean_blue": round(float(a[..., 2].mean()), 1),
                "saturation": round(float(hsv[..., 1].mean()), 1),
                "is_grayscale": int(np.abs(a[..., 0] - a[..., 1]).mean() < 3 and np.abs(a[..., 1] - a[..., 2]).mean() < 3)}
    except Exception:
        return None


# documents
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text_builtin(path: Path) -> str:
    """DOCX without python-docx: the file is a ZIP of XML. Reads body, text boxes, headers, footers, footnotes (one line per paragraph)."""
    import xml.etree.ElementTree as ET
    out: List[str] = []
    with zipfile.ZipFile(path) as z:
        names = ["word/document.xml"] + sorted(n for n in z.namelist() if re.match(r"word/(header|footer|footnotes|endnotes)\d*\.xml$", n))
        for n in names:
            if n not in z.namelist():
                continue
            root = ET.fromstring(z.read(n))
            for p in root.iter(_W + "p"):
                parts = []
                for el in p.iter():
                    if el.tag == _W + "t" and el.text:
                        parts.append(el.text)
                    elif el.tag in (_W + "tab", _W + "br"):
                        parts.append(" ")
                line = "".join(parts).strip()
                if line:
                    out.append(line)
    return "\n".join(out)


def _odt_text_builtin(path: Path) -> str:
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))
    ns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
    return "\n".join("".join(el.itertext()).strip() for el in root.iter() if el.tag in (ns + "p", ns + "h") and "".join(el.itertext()).strip())


def _html_text(raw: bytes) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        return soup.get_text(" ")
    except ImportError:                                           # standard-library fallback
        from html.parser import HTMLParser

        class P(HTMLParser):
            def __init__(self):
                super().__init__()
                self.skip, self.out = 0, []

            def handle_starttag(self, tag, attrs):
                self.skip += tag in ("script", "style", "noscript")

            def handle_endtag(self, tag):
                self.skip -= tag in ("script", "style", "noscript") and self.skip > 0

            def handle_data(self, d):
                if not self.skip:
                    self.out.append(d)
        p = P()
        p.feed(raw.decode("utf-8", "replace"))
        return " ".join(p.out)


def ocr_available() -> bool:
    return os.environ.get("NEXUSML_OCR", "1") != "0" and bool(shutil.which("tesseract"))


def ocr_image(path: Path, timeout: int = 60) -> str:
    """Text from one image via the tesseract CLI ('' when unavailable or nothing was recognised)."""
    if not ocr_available():
        return ""
    try:
        r = subprocess.run(["tesseract", str(path), "stdout", "-l", os.environ.get("NEXUSML_OCR_LANG", "eng")], capture_output=True, timeout=timeout)
        return r.stdout.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def ocr_pdf(path: Path, max_pages: int = 10, dpi: int = 150) -> Tuple[str, int]:
    """OCR the first pages of a scanned PDF (pdftoppm renders pages, tesseract reads them). -> (text, pages_read)."""
    if not (ocr_available() and shutil.which("pdftoppm")):
        return "", 0
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ds_ocr_") as d:
        try:
            subprocess.run(["pdftoppm", "-r", str(dpi), "-l", str(max_pages), "-png", str(path), str(Path(d) / "p")], capture_output=True, timeout=120)
        except Exception:
            return "", 0
        pages = sorted(Path(d).glob("p*.png"))
        return "\n".join(ocr_image(p) for p in pages), len(pages)


def _pdf_pdftotext(path: Path) -> str:
    if not shutil.which("pdftotext"):
        return ""
    try:
        return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, timeout=90).stdout.decode("utf-8", "replace")
    except Exception:
        return ""


def dependency_status() -> Dict[str, Dict[str, Any]]:
    """Which file types can be read on THIS machine, and how (shown by `nexusml.cli doctor` and in the app sidebar)."""
    def has(mod):
        try:
            __import__(mod)
            return True
        except Exception:
            return False
    pdf_lib, pdf_bin = has("pypdf"), bool(shutil.which("pdftotext"))
    return {
        "PDF": {"ok": pdf_lib or pdf_bin, "how": "pypdf" if pdf_lib else "pdftotext (system tool)" if pdf_bin else "MISSING",
                "fix": "" if (pdf_lib or pdf_bin) else "pip install pypdf"},
        "Word .docx": {"ok": True, "how": "python-docx" if has("docx") else "built-in reader", "fix": ""},
        "Text / Markdown": {"ok": True, "how": "built-in", "fix": ""},
        "HTML": {"ok": True, "how": "beautifulsoup4" if has("bs4") else "built-in reader", "fix": ""},
        "OpenDocument .odt": {"ok": True, "how": "odfpy" if has("odf") else "built-in reader", "fix": ""},
        "Images": {"ok": has("PIL"), "how": "pillow" if has("PIL") else "MISSING", "fix": "" if has("PIL") else "pip install pillow"},
        "OCR (scanned PDFs)": {"ok": True, "how": "tesseract" + (" + pdftoppm" if shutil.which("pdftoppm") else " (pdftoppm missing: PDFs cannot be rendered)") if shutil.which("tesseract")
                               else "optional, not installed: scanned PDFs are flagged instead of read", "fix": ""},
        "Excel": {"ok": has("openpyxl"), "how": "openpyxl" if has("openpyxl") else "MISSING", "fix": "" if has("openpyxl") else "pip install openpyxl"},
    }


def extract_text(path: Path) -> Tuple[str, int, str]:
    """(text, n_pages, note). `note` explains a problem (empty text, encrypted, missing library) instead of failing silently."""
    ext = path.suffix.lower()
    try:
        if ext in (".txt", ".md"):
            raw = path.read_bytes()
            for enc in ("utf-8-sig", "utf-16", "cp1256", "latin-1"):
                try:
                    return raw.decode(enc), 1, ""
                except UnicodeDecodeError:
                    continue
        if ext in (".html", ".htm"):
            return _html_text(path.read_bytes()), 1, ""
        if ext == ".docx":
            text, pages = _docx_text_builtin(path), 1                 # built-in XML reader: body, tables, text boxes, headers, footers
            try:
                from docx import Document
                d = Document(str(path))
                pages = max(1, len(d.sections))
                if len(text.strip()) < 20:                            # second opinion if the XML route found almost nothing
                    parts = [p.text for p in d.paragraphs]
                    for t in d.tables:
                        for row in t.rows:
                            parts.append(" | ".join(c.text for c in row.cells))
                    text = "\n".join(parts) or text
            except ImportError:
                pass
            return text, pages, "" if text.strip() else "the Word file contains no text (it may only hold images)"
        if ext == ".odt":
            try:
                from odf import teletype, text as odtext
                from odf.opendocument import load
                doc = load(str(path))
                t = "\n".join(teletype.extractText(p) for p in doc.getElementsByType(odtext.P))
            except ImportError:
                t = _odt_text_builtin(path)
            return t, 1, "" if t.strip() else "the document contains no text"
        if ext == ".pdf":
            txt, n, note = "", 0, ""
            try:
                from pypdf import PdfReader
                r = PdfReader(str(path))
                if r.is_encrypted:
                    try:
                        r.decrypt("")
                    except Exception:
                        return "", 0, "the PDF is password-protected"
                n = len(r.pages)
                txt = "\n".join((r.pages[i].extract_text() or "") for i in range(min(n, LIMITS["max_pdf_pages"])))
            except ImportError:
                note = "reading PDFs needs the 'pypdf' package: run  pip install pypdf  (or install the 'poppler-utils' system tool)"
            if len(txt.strip()) < 20:
                alt = _pdf_pdftotext(path)
                if len(alt.strip()) > len(txt.strip()):
                    txt, note = alt, ""
                    n = n or max(1, alt.count("\f"))
            if len(txt.strip()) < 20 and not note:
                o_txt, o_pages = ocr_pdf(path, LIMITS["max_pdf_pages"] if LIMITS["max_pdf_pages"] < 25 else 25)
                if len(o_txt.strip()) >= 20:
                    return o_txt, n or o_pages, f"text recovered by OCR from {o_pages} page(s) (a scanned PDF); check accuracy"
                note = ("no text layer found (a scanned/image-only PDF) and OCR found no text" if ocr_available() else
                        "no text layer found (a scanned/image-only PDF); install the 'tesseract' and 'poppler-utils' tools to read it with OCR")
            return txt, n, note
    except Exception as e:
        if isinstance(e, zipfile.BadZipFile) or type(e).__name__ == "PackageNotFoundError":
            return "", 0, "the file is damaged or is not really a Word / OpenDocument file (an old .doc must be re-saved as .docx)"
        return "", 0, f"could not read ({type(e).__name__}: {str(e)[:80]})"
    return "", 0, "unsupported file type"


_SENT = re.compile(r"[.!?؟۔]+")
_WORD = re.compile(r"\w+", re.U)


def _syll(w: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", w.lower())))


def text_stats(text: str) -> Dict[str, Any]:
    words = _WORD.findall(text)
    n = len(words)
    sents = [s for s in _SENT.split(text) if s.strip()]
    ascii_share = (sum(c.isascii() and c.isalpha() for c in text) / max(1, sum(c.isalpha() for c in text)))
    flesch = None
    if n >= 30 and ascii_share > 0.8 and sents:             # Flesch Reading Ease is only meaningful for English prose
        flesch = round(206.835 - 1.015 * (n / len(sents)) - 84.6 * (sum(_syll(w) for w in words) / n), 1)
    return {"n_words": n, "n_chars": len(text), "avg_word_len": round(float(np.mean([len(w) for w in words])), 2) if n else 0.0,
            "avg_sentence_len": round(n / max(1, len(sents)), 1), "lexical_diversity": round(len(set(w.lower() for w in words)) / n, 3) if n else 0.0,
            "readability": flesch, "n_numbers": len(re.findall(r"\d+(?:[.,]\d+)?", text)), "n_urls": len(re.findall(r"https?://\S+", text)),
            "n_emails": len(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text))}


# dataset building
def _collect(root: Path, exts) -> List[Path]:
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts
                  and not any(part.startswith(".") or part == "__MACOSX" for part in p.relative_to(root).parts))


def folder_labels(rels: List[Path]) -> Optional[Dict[str, str]]:
    """Class label from the first folder level that splits the files into >= 2 groups of >= 3 files (single wrapper folders are skipped)."""
    parts = [r.parts for r in rels]
    depth = 0
    while parts and all(len(p) - depth > 1 for p in parts) and len({p[depth] for p in parts}) == 1:
        depth += 1
    groups: Dict[str, int] = {}
    for p in parts:
        if len(p) - depth > 1:
            groups[p[depth]] = groups.get(p[depth], 0) + 1
    in_folders = sum(groups.values())
    if len(groups) >= 2 and all(v >= 3 for v in groups.values()) and in_folders >= 0.7 * len(parts):
        return {str(r): (r.parts[depth] if len(r.parts) - depth > 1 else "(unfiled)") for r in rels}
    return None


def build_dataset(root: Path) -> Dict[str, Any]:
    """Scan `root`, build `media_dataset.csv` (+ manifest). The collection kind is the larger of images / documents; the other kind is reported as set aside."""
    root = Path(root)
    imgs, docs = _collect(root, IMAGE_EXT), _collect(root, DOC_EXT)
    if not imgs and not docs:
        raise MediaError("I could not find any supported images or documents in the upload.")
    kind = "images" if len(imgs) >= len(docs) else "documents"
    rels = (imgs if kind == "images" else docs)[:LIMITS["max_rows"]]
    labels = folder_labels(rels)
    rows, unreadable, notes = [], [], []
    for rel in rels:
        full = root / rel
        base: Dict[str, Any] = {("image" if kind == "images" else "document"): str(rel).replace("\\", "/")}
        if kind == "images":
            m = image_meta(full)
            if m is None:
                unreadable.append(str(rel))
                continue
            base.update(m)
        else:
            text, pages, note = extract_text(full)
            if note:
                notes.append(f"{rel}: {note}")
            base.update({"text": re.sub(r"\s+", " ", text).strip()[:LIMITS["max_doc_chars"]], "n_pages": pages, **text_stats(text)})
        if labels:
            base["label"] = labels[str(rel)]
        rows.append(base)
    if not rows:
        raise MediaError(f"None of the {len(rels)} {kind} could be read" + (f" (e.g. {unreadable[0]})." if unreadable else ".") + (" " + notes[0] if notes else ""))
    df = pd.DataFrame(rows)
    if kind == "documents":
        df["has_text"] = (df["n_words"] >= 20).astype(int)
    out_csv = root / "media_dataset.csv"
    df.to_csv(out_csv, index=False)
    set_aside = {"documents": len(docs)} if kind == "images" and docs else {"images": len(imgs)} if kind == "documents" and imgs else {}
    class_counts = df["label"].value_counts().to_dict() if "label" in df else None
    suggest: Dict[str, Any] = {"analysis_type": "general_exploration"}
    warn: List[str] = []
    if class_counts and min(class_counts.values()) >= 5:
        suggest.update(target_column="label", problem_type="classification")
    else:
        if class_counts:
            warn.append(f"Some folders have fewer than 5 files ({class_counts}); I will group the {kind} by similarity instead of predicting folder names.")
        suggest.update(target_column=None, problem_type="clustering")
        df = df.drop(columns=[c for c in ("label",) if c in df])
        df.to_csv(out_csv, index=False)
    if len(df) < config.MIN_ROWS:
        warn.append(f"Only {len(df)} {kind} are readable; at least {config.MIN_ROWS} are needed for a reliable analysis.")
    manifest = {"kind": kind, "n": int(len(df)), "classes": class_counts, "unreadable": unreadable, "text_problems": notes[:50], "set_aside": set_aside,
                "suggest": suggest, "warnings": warn, "csv": str(out_csv), "root": str(root)}
    (root / "media_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def prepare(files: List[Tuple[str, bytes]], work_dir: Path) -> Dict[str, Any]:
    """Uploads -> dataset. Raises MediaError with a plain-language reason."""
    stamp = hashlib.sha1(b"".join(n.encode() + hashlib.md5(d).digest() for n, d in files)).hexdigest()[:10]
    root = Path(work_dir) / f"media_{stamp}"
    if root.exists():
        shutil.rmtree(root)
    save_uploads(files, root)
    return build_dataset(root)


def locate_root(rels: List[str], dataset_csv: Optional[str] = None) -> Optional[Path]:
    """Folder that relative image paths resolve against: the dataset's own folder, else the stored upload that contains most of them.
    Repairs datasets saved before image paths were stored as absolute paths."""
    rels = [str(r).strip() for r in rels if str(r).strip()][:25]
    if not rels:
        return None
    cands: List[Path] = []
    if dataset_csv:
        cands.append(Path(dataset_csv).resolve().parent)
    media_dirs = [Path(config.artifacts_dir()) / "workspace" / "media"]
    if dataset_csv:                                    # <workspace>/datasets/<id>.csv  ->  <workspace>/media
        media_dirs.insert(0, Path(dataset_csv).resolve().parent.parent / "media")
    for media_dir in media_dirs:
        if not media_dir.is_dir():
            continue
        cands += sorted((d for d in media_dir.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    best, best_hits = None, 0
    for c in cands:
        hits = sum((c / r).is_file() for r in rels)
        if hits > best_hits:
            best, best_hits = c, hits
    return best if best_hits >= max(1, len(rels) // 2) else None


def prepare_paths(paths: List[str], work_dir: Path) -> Dict[str, Any]:
    """CLI helper: files, a folder, or a ZIP on disk."""
    files: List[Tuple[str, bytes]] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for f in sorted(pp.rglob("*")):
                if f.is_file() and kind_of(f.name) in ("image", "document"):
                    rel = f.relative_to(pp)
                    files.append((str(PurePosixPath(pp.name, *rel.parts)), f.read_bytes()))
        else:
            files.append((pp.name, pp.read_bytes()))
    zips = [(n, d) for n, d in files if kind_of(n) == "zip"]
    flat = [(n, d) for n, d in files if kind_of(n) != "zip"]
    # folder uploads keep their folder structure by zipping it in memory (so labels from sub-folders survive)
    if flat and any("/" in n for n, _ in flat):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for n, d in flat:
                z.writestr(n, d)
        return prepare(zips + [("folder.zip", buf.getvalue())], work_dir)
    return prepare(files, work_dir)


def row_for_file(path: Path, columns: List[str], root: Path) -> Dict[str, Any]:
    """One table row for a NEW file (Predict tab): same columns the model was trained on."""
    k = kind_of(path.name)
    row: Dict[str, Any] = {}
    if k == "image":
        m = image_meta(path)
        if m is None:
            raise MediaError("that image could not be read")
        row = {"image": str(path), **m}
    elif k == "document":
        text, pages, note = extract_text(path)
        if not text.strip():
            raise MediaError(note or "no text could be extracted")
        row = {"document": path.name, "text": re.sub(r"\s+", " ", text).strip()[:LIMITS["max_doc_chars"]], "n_pages": pages, **text_stats(text),
               "has_text": int(len(_WORD.findall(text)) >= 20)}
    else:
        raise MediaError("unsupported file type")
    return {c: row.get(c) for c in columns}
