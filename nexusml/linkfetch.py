"""Add data by LINK instead of downloading and re-uploading it: Kaggle datasets / competitions, GitHub files, Dropbox / Google Drive share links, or any direct file URL.

    from nexusml import linkfetch
    r = linkfetch.add_from_link("https://www.kaggle.com/datasets/shenba/time-series-datasets")      # -> datasets are in the workspace, ready to analyse

Kaggle requires a login even for public data. Put your token in ENVIRONMENT VARIABLES (never in code or in the app), either
    KAGGLE_API_TOKEN=<access token from kaggle.com/settings>          (the newer single token)
    KAGGLE_USERNAME=<name> and KAGGLE_KEY=<key>                       (the classic API key)   or the usual ~/.kaggle/kaggle.json file.

Safety, because a link is untrusted input
  * https only (http needs NEXUSML_ALLOW_HTTP=1); every redirect hop is re-checked; hosts that resolve to private / loopback / link-local addresses are refused
    (NEXUSML_ALLOW_PRIVATE_URLS=1 lifts that, for tests and intranets)
  * downloads are streamed and stopped at NEXUSML_MAX_DOWNLOAD_MB (default 100); the Kaggle size is checked BEFORE downloading
  * zip archives are unpacked in memory with limits on member count, total size and names (no path tricks), and only supported file types are kept
"""
import io
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TABLE_EXT = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl", ".parquet", ".ods"}
DOC_EXT = {".pdf", ".docx", ".odt", ".html", ".htm", ".md"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
MAX_ZIP_MEMBERS = 5000
MAX_TABLES = 8


class LinkError(ValueError):
    """A problem with the link, worded for the person who pasted it."""


def max_bytes() -> int:
    try:
        return int(float(os.environ.get("NEXUSML_MAX_DOWNLOAD_MB", "100")) * 1024 * 1024)
    except ValueError:
        return 100 * 1024 * 1024


def kaggle_api() -> str:
    return os.environ.get("NEXUSML_KAGGLE_API", "https://www.kaggle.com/api/v1").rstrip("/")


# ------------------------------------------------------------------ parsing
def find_urls(text: str) -> List[str]:
    return [u.rstrip(").,;>]'\"") for u in re.findall(r"https?://[^\s<>\"']+", text or "")]


def parse(url: str) -> Dict[str, Any]:
    """-> {'kind': kaggle_dataset | kaggle_competition | direct, ...}. Rewrites GitHub / Dropbox / Drive share links to their download form."""
    url = (url or "").strip()
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.netloc:
        raise LinkError("That does not look like a web link. Paste the full address, starting with https://")
    host = u.netloc.lower().split(":")[0]
    parts = [p for p in u.path.split("/") if p]
    q = urllib.parse.parse_qs(u.query)
    if host in ("www.kaggle.com", "kaggle.com"):
        if len(parts) >= 3 and parts[0] == "datasets":
            return {"kind": "kaggle_dataset", "owner": parts[1], "slug": parts[2], "select": (q.get("select") or [None])[0], "url": url}
        if len(parts) >= 2 and parts[0] == "competitions":
            return {"kind": "kaggle_competition", "slug": parts[1], "url": url}
        if parts and parts[0] in ("code", "notebooks", "models", "discussions"):
            raise LinkError("That is a Kaggle notebook / model / discussion link. Paste the link to the DATASET itself (it looks like kaggle.com/datasets/owner/name).")
        raise LinkError("I could not find a dataset in that Kaggle link. It should look like https://www.kaggle.com/datasets/owner/name")
    if host == "github.com" and len(parts) >= 5 and parts[2] == "blob":
        return {"kind": "direct", "url": f"https://raw.githubusercontent.com/{parts[0]}/{parts[1]}/{'/'.join(parts[3:])}"}
    if host.endswith("dropbox.com"):
        q2 = {k: v for k, v in q.items() if k != "dl"}
        q2["dl"] = ["1"]
        return {"kind": "direct", "url": urllib.parse.urlunparse(u._replace(query=urllib.parse.urlencode(q2, doseq=True)))}
    if host == "drive.google.com":
        m = re.search(r"/file/d/([\w-]+)", u.path) or re.search(r"[?&]id=([\w-]+)", url)
        if m:
            return {"kind": "direct", "url": f"https://drive.google.com/uc?export=download&id={m.group(1)}"}
    return {"kind": "direct", "url": url}


# ------------------------------------------------------------------ credentials
def _clean(v: str) -> str:
    """Windows' variable editor and .env files often leave quotes or stray spaces around a pasted secret."""
    return (v or "").strip().strip('"').strip("'").strip()


def _kaggle_dir() -> Path:
    return Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))


def kaggle_auth() -> Tuple[Dict[str, str], str]:
    """-> (headers, where the credential came from). Never returns the secret itself in the description."""
    tok = _clean(os.environ.get("KAGGLE_API_TOKEN", ""))
    if tok:
        return {"Authorization": f"Bearer {tok}"}, "the KAGGLE_API_TOKEN environment variable"
    for name in ("access_token", "access_token.txt"):                                   # the file the newer Kaggle tools use
        try:
            tok = _clean((_kaggle_dir() / name).read_text())
        except OSError:
            tok = ""
        if tok:
            return {"Authorization": f"Bearer {tok}"}, f"the file ~/.kaggle/{name}"
    user, key = _clean(os.environ.get("KAGGLE_USERNAME", "")), _clean(os.environ.get("KAGGLE_KEY", ""))
    src = "the KAGGLE_USERNAME / KAGGLE_KEY environment variables"
    if not (user and key):
        try:
            j = json.loads((_kaggle_dir() / "kaggle.json").read_text())
            user, key, src = _clean(str(j.get("username", ""))), _clean(str(j.get("key", ""))), "the file ~/.kaggle/kaggle.json"
        except (OSError, ValueError):
            pass
    if user and key:
        import base64
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()}, src
    raise LinkError("Kaggle needs your API token, even for public datasets. Create one at kaggle.com/settings, then set it as an environment variable "
                    "(KAGGLE_API_TOKEN, or KAGGLE_USERNAME and KAGGLE_KEY) and restart this app from a NEW terminal window so it sees the variable. "
                    "It is read from your environment only and never stored by this app.")


def kaggle_headers() -> Dict[str, str]:
    return kaggle_auth()[0]


def kaggle_configured() -> bool:
    try:
        kaggle_headers()
        return True
    except LinkError:
        return False


def kaggle_check() -> str:
    """Make one cheap authenticated call so a bad token is found now, with a plain explanation. -> a success message; raises LinkError otherwise."""
    hdr, src = kaggle_auth()
    tok = hdr["Authorization"].split(" ", 1)[1] if hdr["Authorization"].startswith("Bearer ") else ""
    if tok and not tok.startswith("KGAT_"):
        raise LinkError("The value found in KAGGLE_API_TOKEN does not start with KGAT_, so it is not a Kaggle access token. Copy the token again from kaggle.com/settings (API section).")
    try:
        download(f"{kaggle_api()}/datasets/list?pageSize=1", hdr, limit=2_000_000, timeout=30)
    except LinkError as e:
        if "401" in str(e):
            raise LinkError(f"Kaggle rejected the credential read from {src}. The token may be expired, revoked, or copied incompletely. Create a new one at kaggle.com/settings.") from None
        raise
    return f"Kaggle accepted the credential from {src}."


# ------------------------------------------------------------------ safe download
def _check_host(host: str, scheme: str) -> None:
    if scheme == "http" and os.environ.get("NEXUSML_ALLOW_HTTP") != "1" and os.environ.get("NEXUSML_ALLOW_PRIVATE_URLS") != "1":
        raise LinkError("Only secure (https) links are allowed.")
    if os.environ.get("NEXUSML_ALLOW_PRIVATE_URLS") == "1":
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        raise LinkError(f"Could not reach {host}. Check the link and your connection.")
    for fam, _, _, _, sa in infos:
        ip = ipaddress.ip_address(sa[0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise LinkError("That link points to a private or internal address, which is not allowed.")


class _Guard(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        p = urllib.parse.urlparse(newurl)
        _check_host(p.hostname or "", p.scheme)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(req.full_url).netloc != p.netloc:
            new.headers = {k: v for k, v in new.headers.items() if k.lower() != "authorization"}          # never forward the Kaggle token to another host
            new.unredirected_hdrs = {}
        return new


def _filename(resp, url: str) -> str:
    cd = resp.headers.get("Content-Disposition", "") or ""
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd)
    name = urllib.parse.unquote(m.group(1)) if m else Path(urllib.parse.urlparse(url).path).name
    return re.sub(r"[^\w.\- ]", "_", name or "download")[:120] or "download"


def download(url: str, headers: Optional[Dict[str, str]] = None, limit: Optional[int] = None, timeout: int = 60) -> Tuple[bytes, str, str]:
    """-> (bytes, filename, content_type). Stops, with a clear message, when the file is larger than the limit."""
    limit = limit or max_bytes()
    p = urllib.parse.urlparse(url)
    _check_host(p.hostname or "", p.scheme)
    req = urllib.request.Request(url, headers={"User-Agent": "NexusML-linkfetch/1", **(headers or {})})
    opener = urllib.request.build_opener(_Guard)
    try:
        resp = opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise LinkError({401: "The site refused the login (HTTP 401). Check your token.", 403: "Access denied (HTTP 403). The data may be private, or its rules need to be accepted on the website first.",
                         404: "Nothing found at that address (HTTP 404). Check the link."}.get(e.code, f"The site answered with an error (HTTP {e.code}).")) from None
    except urllib.error.URLError as e:
        raise LinkError(f"Could not download from that link ({getattr(e, 'reason', e)}).") from None
    size = resp.headers.get("Content-Length")
    if size and size.isdigit() and int(size) > limit:
        raise LinkError(f"That file is {int(size) / 1048576:.0f} MB, over the {limit / 1048576:.0f} MB limit. Pick a smaller file (on Kaggle, add ?select=<file name> to the link) or raise NEXUSML_MAX_DOWNLOAD_MB.")
    buf, total = io.BytesIO(), 0
    while True:
        chunk = resp.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise LinkError(f"The download is larger than the {limit / 1048576:.0f} MB limit and was stopped. Pick a smaller file or raise NEXUSML_MAX_DOWNLOAD_MB.")
        buf.write(chunk)
    return buf.getvalue(), _filename(resp, url), (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()


# ------------------------------------------------------------------ zip
def unzip(raw: bytes, limit: Optional[int] = None) -> List[Tuple[str, bytes]]:
    limit = (limit or max_bytes()) * 3
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise LinkError("The downloaded archive is damaged or is not a zip file.") from None
    infos = [i for i in z.infolist() if not i.is_dir()]
    if len(infos) > MAX_ZIP_MEMBERS:
        raise LinkError(f"The archive has {len(infos)} files; the limit is {MAX_ZIP_MEMBERS}.")
    if sum(i.file_size for i in infos) > limit:
        raise LinkError("The archive expands to more than the size limit.")
    out = []
    for i in infos:
        name = i.filename.replace("\\", "/")
        if name.startswith("/") or ".." in name.split("/") or "__MACOSX" in name or Path(name).name.startswith("."):
            continue
        if Path(name).suffix.lower() in TABLE_EXT | DOC_EXT | IMG_EXT:
            out.append((name, z.read(i)))
    return out


# ------------------------------------------------------------------ fetch + add
def _kaggle_fetch(info: Dict[str, Any]) -> Tuple[List[Tuple[str, bytes]], str, List[str]]:
    hdr = kaggle_headers()
    api, notes = kaggle_api(), []
    if info["kind"] == "kaggle_dataset":
        owner, slug = info["owner"], info["slug"]
        label = f"{owner}/{slug}"
        try:                                                                    # size check BEFORE downloading
            meta = json.loads(download(f"{api}/datasets/view/{owner}/{slug}", hdr, limit=5_000_000)[0])
            tb = meta.get("totalBytes") or meta.get("total_bytes")
            if isinstance(tb, (int, float)) and tb > max_bytes() and not info.get("select"):
                raise LinkError(f"This Kaggle dataset is {tb / 1048576:.0f} MB, over the {max_bytes() / 1048576:.0f} MB limit. Link to one file instead by adding ?select=<file name> "
                                "(for example ...?select=train.csv), or raise NEXUSML_MAX_DOWNLOAD_MB.")
        except LinkError as e:
            if "MB" in str(e) and "limit" in str(e):
                raise
        except (ValueError, TypeError):
            pass
        sel = info.get("select")
        if sel:
            try:
                raw, fname, _ = download(f"{api}/datasets/download/{owner}/{slug}/{urllib.parse.quote(sel)}", hdr)
                if not fname.lower().endswith(".zip") and Path(sel).suffix.lower() in TABLE_EXT | DOC_EXT | IMG_EXT:
                    return [(Path(sel).name, raw)], label, notes
            except LinkError:
                pass
        raw, fname, _ = download(f"{api}/datasets/download/{owner}/{slug}", hdr)
        files = unzip(raw) if raw[:2] == b"PK" else [(fname, raw)]
        if sel:
            hit = [f for f in files if f[0] == sel or Path(f[0]).name == Path(sel).name]
            if hit:
                files = hit
            else:
                notes.append(f"The file '{sel}' was not in the dataset; added everything instead.")
        return files, label, notes
    slug = info["slug"]
    raw, fname, _ = download(f"{api}/competitions/data/download-all/{slug}", hdr)
    return (unzip(raw) if raw[:2] == b"PK" else [(fname, raw)]), slug, ["Competition data: you must have accepted the competition rules on kaggle.com first."]


def fetch(url: str) -> Dict[str, Any]:
    """Download whatever the link points to. -> {'files': [(name, bytes)], 'source': str, 'notes': [str], 'was_zip': bool, 'zip_bytes': bytes|None}"""
    info = parse(url)
    if info["kind"] in ("kaggle_dataset", "kaggle_competition"):
        files, source, notes = _kaggle_fetch(info)
        return {"files": files, "source": source, "notes": notes, "info": info}
    raw, fname, ctype = download(info["url"])
    if raw[:2] == b"PK" and Path(fname).suffix.lower() not in (".xlsx", ".docx", ".odt", ".ods"):
        return {"files": unzip(raw), "source": fname, "notes": [], "info": info, "zip_bytes": raw, "zip_name": fname}
    ext = Path(fname).suffix.lower()
    if ext not in TABLE_EXT | DOC_EXT | IMG_EXT:
        if "csv" in ctype or raw[:200].count(b",") > 3 and b"\n" in raw[:2000]:
            fname = (Path(fname).stem or "data") + ".csv"
        elif "json" in ctype:
            fname = (Path(fname).stem or "data") + ".json"
        elif raw[:5] == b"%PDF-":
            fname = (Path(fname).stem or "data") + ".pdf"
        elif "html" in ctype:
            raise LinkError("That link opens a web page, not a data file. Use the link to the file itself (a download link), or a Kaggle dataset link.")
        else:
            raise LinkError(f"I could not tell what kind of file that is ('{ext or 'no extension'}'). Supported: tables (csv, xlsx, json, parquet), documents (pdf, docx), images and zips of them.")
    return {"files": [(fname, raw)], "source": fname, "notes": [], "info": info}


def add_from_link(url: str) -> Dict[str, Any]:
    """Fetch the link and add it to the workspace. -> {'datasets': [meta, ...], 'primary': meta, 'source': str, 'notes': [str]}. Raises LinkError."""
    from . import workspace
    got = fetch(url)
    files, notes, metas = got["files"], list(got["notes"]), []
    if not files:
        raise LinkError("The link worked, but it contained no supported files (tables, documents or images).")
    imgs = [f for f in files if Path(f[0]).suffix.lower() in IMG_EXT]
    tables = sorted([f for f in files if Path(f[0]).suffix.lower() in TABLE_EXT], key=lambda f: -len(f[1]))
    docs = [f for f in files if Path(f[0]).suffix.lower() in DOC_EXT]
    zip_raw = got.get("zip_bytes")
    try:
        if len(imgs) >= 3 and len(imgs) >= len(tables):                         # an image collection: keep folder structure (class folders) by adding the archive
            if zip_raw is None:
                bio = io.BytesIO()
                with zipfile.ZipFile(bio, "w", zipfile.ZIP_STORED) as z:
                    for n, b in files:
                        z.writestr(n, b)
                zip_raw = bio.getvalue()
            metas.append(workspace.add_upload((got.get("zip_name") or re.sub(r"[^\w.-]", "_", got["source"]) + ".zip"), zip_raw))
            tables, docs = [], []
        for n, b in tables[:MAX_TABLES]:
            try:
                metas.append(workspace.add_upload(Path(n).name, b))
            except ValueError as e:
                notes.append(f"{Path(n).name}: {e}")
        if len(tables) > MAX_TABLES:
            notes.append(f"{len(tables) - MAX_TABLES} more tables were skipped (limit {MAX_TABLES}).")
        if not tables and not metas:
            for n, b in docs[:MAX_TABLES]:
                try:
                    metas.append(workspace.add_upload(Path(n).name, b))
                except ValueError as e:
                    notes.append(f"{Path(n).name}: {e}")
    except ValueError as e:
        raise LinkError(str(e)) from None
    if not metas:
        raise LinkError("Nothing in that link could be read as data. " + " ".join(notes[:2]))
    return {"datasets": metas, "primary": metas[0], "source": got["source"], "notes": notes}


def materialize(url: str, dest_dir: str) -> str:
    """For the command line: fetch the link and write its main table (or the whole image archive) to disk. -> a path to pass to the normal pipeline."""
    got = fetch(url)
    d = Path(dest_dir)
    d.mkdir(parents=True, exist_ok=True)
    files = got["files"]
    imgs = [f for f in files if Path(f[0]).suffix.lower() in IMG_EXT]
    tables = sorted([f for f in files if Path(f[0]).suffix.lower() in TABLE_EXT], key=lambda f: -len(f[1]))
    if len(imgs) >= 3 and len(imgs) >= len(tables):
        for n, b in files:
            p = d / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
        return str(d)
    pick = tables[0] if tables else (files[0] if files else None)
    if pick is None:
        raise LinkError("The link contained no supported files.")
    p = d / re.sub(r"[^\w.\- ]", "_", Path(pick[0]).name)
    p.write_bytes(pick[1])
    return str(p)
