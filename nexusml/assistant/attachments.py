"""Files dropped into the assistant conversation: a picture is described, a document is summarised.

This is for the assistant only. Nothing here adds anything to the workspace's datasets and nothing here is used by an analysis. Dropping a file into the conversation is the
person's consent to send it to the language model provider (the reply says so). Each read returns a small record that is shown in the chat and kept for follow-up questions:
what a picture shows, what a document says. The file itself is not kept.
"""
import io
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config, docchat, llm, media

IMAGE_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
DOC_EXT = {".pdf", ".docx", ".odt", ".html", ".htm", ".md"}
MAX_IMAGE_BYTES = 3_500_000                                      # providers cap the size of an inline picture


def kind_of(name: str) -> Optional[str]:
    ext = Path(name).suffix.lower()
    return "image" if ext in IMAGE_EXT else "document" if ext in DOC_EXT else None


def _shrink(raw: bytes) -> Optional[bytes]:
    """A picture that is too large is re-saved smaller (same picture, fewer pixels). None if that is not possible."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        for side in (1800, 1300, 900):
            small = im.copy()
            small.thumbnail((side, side))
            buf = io.BytesIO()
            small.save(buf, "JPEG", quality=82)
            if buf.tell() <= MAX_IMAGE_BYTES:
                return buf.getvalue()
    except Exception:
        return None
    return None


IMAGE_SYSTEM = "You describe pictures for a person who dropped one into a chat. Be accurate and specific; say what you can see, not what you guess. If text or numbers are visible, read them out."
IMAGE_SCHEMA = {"description": "3-5 sentences: what the picture shows overall", "type": "photo | chart | table | screenshot | document | diagram | drawing | other",
                "objects": ["the main things or people visible, at most 12"], "visible_text": "text or numbers visible in the picture, verbatim, or empty", "details": ["notable details worth mentioning, at most 5"]}


def read_image(name: str, raw: bytes) -> Dict[str, Any]:
    ext = Path(name).suffix.lower()
    out: Dict[str, Any] = {"kind": "image", "name": name, "ok": False, "mime": IMAGE_EXT.get(ext, "image/jpeg")}
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        out["size"] = f"{im.width} x {im.height}"
    except Exception:
        out["size"] = None
    if len(raw) > MAX_IMAGE_BYTES:
        small = _shrink(raw)
        if small is None:
            out["error"] = "The picture is too large to look at (over 3.5 MB) and could not be made smaller."
            return out
        raw, out["mime"] = small, "image/jpeg"
    if not llm.available():
        out["error"] = "The language model is not connected, so I can't look at pictures yet. Add your API key in Settings."
        return out
    data, err, _ = llm.ask_vision(IMAGE_SYSTEM, "Describe this picture. Answer as JSON with these keys: " + str(IMAGE_SCHEMA), raw, out["mime"], 700)
    if not data or not str(data.get("description", "")).strip():
        out["error"] = f"I could not look at the picture ({err or 'no description came back'})."
        return out
    out.update(ok=True, description=str(data["description"]).strip()[:1500], type=str(data.get("type") or "other")[:30],
               objects=[str(x)[:60] for x in (data.get("objects") or [])][:12], visible_text=str(data.get("visible_text") or "").strip()[:800],
               details=[str(x)[:200] for x in (data.get("details") or [])][:5])
    return out


def read_document(name: str, raw: bytes) -> Dict[str, Any]:
    out: Dict[str, Any] = {"kind": "document", "name": name, "ok": False}
    with tempfile.TemporaryDirectory(prefix="asst_doc_") as d:
        p = Path(d) / Path(name).name
        p.write_bytes(raw)
        try:
            a = docchat.analyze(p)
        except media.MediaError as e:
            out["error"] = str(e)
            return out
        except Exception as e:                                                # a damaged file must give a message, not a crash
            out["error"] = f"I could not read that file ({type(e).__name__})."
            return out
    out.update(ok=True, pages=a["pages"], words=a["stats"]["n_words"], keywords=a["keywords"], dates=a["dates"], amounts=a["amounts"], headings=a["headings"],
               excerpt=a["text"][:2500], local_summary=a["extractive_summary"], note=a.get("note"), mode="local", covered=None, truncated=a["truncated"])
    if llm.available():
        try:
            r = docchat.summarize_with_ai({}, a)
            out.update(mode="model", summary=r["summary"], key_points=r["key_points"], topics=r["topics"], action_items=r["action_items"], covered=r["covered"], questions=r["questions_to_ask"])
        except Exception:                                                     # the model was unreachable or gave nothing usable: the local summary still helps
            out["model_failed"] = True
    return out


def read(name: str, raw: bytes) -> Dict[str, Any]:
    k = kind_of(name)
    return read_image(name, raw) if k == "image" else read_document(name, raw)


def context(att: Dict[str, Any]) -> Dict[str, Any]:
    """What the assistant is told about the file in later questions (short; the file itself is not kept)."""
    if att["kind"] == "image":
        return {"name": att["name"], "kind": "image", "what_it_shows": att["description"][:900], "type": att.get("type"), "objects": att.get("objects", [])[:12],
                "visible_text": att.get("visible_text", "")[:600], "details": att.get("details", [])}
    return {"name": att["name"], "kind": "document", "pages": att.get("pages"), "words": att.get("words"), "summary": (att.get("summary") or " ".join(att.get("local_summary", [])))[:1400],
            "key_points": att.get("key_points", [])[:6], "topics": att.get("topics", [])[:6], "dates": att.get("dates", [])[:6], "amounts": att.get("amounts", [])[:6], "excerpt": att.get("excerpt", "")[:2000]}


def message_text(att: Dict[str, Any]) -> str:
    """The reply shown in the chat when the file arrives."""
    sent = f"*The {'picture' if att['kind'] == 'image' else 'text of the file'} was sent to {config.LLM_PROVIDER_NAME} to write this.*"
    if not att["ok"]:
        return f"**{att['name']}**: {att['error']}"
    if att["kind"] == "image":
        L = [f"**{att['name']}**" + (f" ({att['size']})" if att.get("size") else ""), att["description"]]
        if att.get("objects"):
            L.append("**Main things I can see:** " + ", ".join(att["objects"]) + ".")
        if att.get("visible_text"):
            L.append("**Text in the picture:** " + att["visible_text"])
        if att.get("details"):
            L.append("**Also worth noting:**\n" + "\n".join(f"- {x}" for x in att["details"]))
        L.append(sent)
        return "\n\n".join(L)
    head = f"**{att['name']}**: {att['words']:,} words" + (f", {att['pages']} page(s)" if att.get("pages") else "") + "."
    if att["mode"] == "model":
        L = [head, "**Summary:** " + att["summary"]]
        if att["key_points"]:
            L.append("**Key points:**\n" + "\n".join(f"- {x}" for x in att["key_points"]))
        if att["action_items"]:
            L.append("**Actions or decisions mentioned:**\n" + "\n".join(f"- {x}" for x in att["action_items"]))
        L.append(sent + f" Based on {att['covered']}. A summary can miss or misstate details, so check important facts against the document.")
        return "\n\n".join(L)
    L = [head]
    if att.get("note"):
        L.append(f"*Note: {att['note']}.*")
    L.append("I could not write a full summary" + (" (the language model did not answer)" if att.get("model_failed") else " (the language model is not connected)") + ", so here are the most representative sentences:\n"
             + "\n".join(f"- {x}" for x in att["local_summary"]))
    if att.get("keywords"):
        L.append("**Key terms:** " + ", ".join(att["keywords"]) + ".")
    return "\n\n".join(L)


def followups(att: Dict[str, Any]) -> List[str]:
    if not att["ok"]:
        return []
    if att["kind"] == "image":
        return ["What else can you tell me about this picture?", "Is there any text or number I should check?"]
    return (att.get("questions") or [])[:2] or ["What are the key dates or amounts?", "What should I do about this?"]
