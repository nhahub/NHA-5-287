"""Single-image understanding ("what is this a picture of?").

Three tiers:
  1. LOCAL measurements (always): size, brightness, sharpness, dominant colours, quality flags. These never leave the machine.
  2. VISION AI (needs the user's consent, because the image is sent to the provider): Groq's vision model (default Llama 4 Scout) names the
     main subject, objects, colours, visible text, and can answer a question about the picture.
  3. LOCAL classifier (optional): if torch + torchvision are installed, a pretrained ResNet-18 gives ImageNet labels (1000 classes, incl. dog breeds);
     the weights are downloaded on first use. Untested in the reference environment.
If neither tier 2 nor 3 is available the answer says so plainly: handcrafted features cannot recognise objects.
"""
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import llm, media

SYSTEM = ("You are a careful image analyst. Describe only what is visible. If you are not sure, say so and lower the confidence. "
          "Never identify real people by name. Answer in the language of the user's question if one is given, otherwise English.")


def prepare_image(path: Path, max_side: int = 1280, max_bytes: int = 3_000_000) -> bytes:
    """Resize / re-encode so the base64 request stays well under provider limits."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        q = 88
        while True:
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=q)
            if buf.tell() <= max_bytes or q <= 40:
                return buf.getvalue()
            q -= 12
            if q < 60:
                im.thumbnail((im.width * 3 // 4, im.height * 3 // 4))


def local_facts(path: Path) -> Dict[str, Any]:
    m = media.image_meta(path)
    if m is None:
        raise media.MediaError("that image could not be read (is it a valid png / jpg / gif / webp / bmp / tiff?)")
    flags = []
    if m["brightness"] < 40:
        flags.append("very dark")
    if m["brightness"] > 215:
        flags.append("very bright")
    if m["contrast"] < 15:
        flags.append("low contrast")
    if m["sharpness"] < 25:
        flags.append("blurry")
    if m["is_grayscale"]:
        flags.append("grayscale")
    from PIL import Image
    with Image.open(path) as im:
        small = im.convert("RGB").resize((48, 48))
        pal = small.quantize(5).convert("RGB").getcolors(2500) or []
    cols = [f"#{r:02x}{g:02x}{b:02x}" for _, (r, g, b) in sorted(pal, reverse=True)[:4]]
    return {**m, "quality_flags": flags, "dominant_colours": cols}


def vision_available() -> bool:
    return llm.available()


def describe(path: Path, question: Optional[str] = None, send_to_ai: bool = False) -> Dict[str, Any]:
    """Returns {facts, source, subject, confidence, summary, objects, details, text_in_image, answer, note}."""
    out: Dict[str, Any] = {"facts": local_facts(path), "source": "none", "note": ""}
    if send_to_ai:
        if not vision_available():
            out["note"] = f"{llm.config.LLM_KEY_ENV} is not set, so I cannot use the vision AI."
        else:
            prompt = ("Analyse this image. Return JSON: {\"main_subject\": \"what the image mainly shows, in 1-4 words\", "
                      "\"confidence\": \"high|medium|low\", \"summary\": \"one or two sentences\", "
                      "\"objects\": [{\"label\": \"...\", \"confidence\": \"high|medium|low\"}], "
                      "\"details\": [\"colours, setting, breed/type if clear, condition\"], \"text_in_image\": \"any readable text or empty\", "
                      "\"answer\": \"direct answer to the question or empty\"}.")
            if question:
                prompt += f" The user asks: {question!r}."
            data, err, _ = llm.ask_vision(SYSTEM, prompt, prepare_image(path))
            if err:
                out["note"] = err
            else:
                objs = [o for o in (data.get("objects") or []) if isinstance(o, dict) and o.get("label")][:8]
                out.update(source="vision AI", subject=str(data.get("main_subject", ""))[:80], confidence=str(data.get("confidence", "")),
                           summary=str(data.get("summary", ""))[:600], objects=objs, details=[str(d)[:140] for d in (data.get("details") or [])][:6],
                           text_in_image=str(data.get("text_in_image", ""))[:600], answer=str(data.get("answer", ""))[:800])
                if not out["subject"] and not out["summary"]:
                    out.update(source="none", note="The vision model replied without a description.")
    if out["source"] == "none":
        loc = local_classify(path)
        if loc:
            out.update(source="local ResNet-18 (ImageNet labels)", subject=loc[0]["label"], confidence="model score %.2f" % loc[0]["score"], objects=loc[:5],
                       summary=f"Most likely: {', '.join(x['label'] for x in loc[:3])}.")
        elif not out["note"]:
            out["note"] = ("I can measure the image, but telling you WHAT it shows needs a vision model. Allow the vision AI (the image is sent to Groq), "
                           "or install torch + torchvision for an on-machine ImageNet classifier.")
    return out


_RESNET: Dict[str, Any] = {}


def local_classify(path: Path, k: int = 5) -> Optional[List[Dict[str, Any]]]:
    try:
        import torch
        import torchvision
        from PIL import Image
        if "m" not in _RESNET:
            w = torchvision.models.ResNet18_Weights.DEFAULT
            _RESNET.update(m=torchvision.models.resnet18(weights=w).eval(), t=w.transforms(), c=w.meta["categories"])
        with Image.open(path) as im, torch.no_grad():
            p = torch.softmax(_RESNET["m"](_RESNET["t"](im.convert("RGB")).unsqueeze(0))[0], dim=0)
        top = p.topk(k)
        return [{"label": _RESNET["c"][i], "score": float(s)} for s, i in zip(top.values, top.indices)]
    except Exception:
        return None


def format_answer(d: Dict[str, Any]) -> str:
    f = d["facts"]
    lines = []
    if d["source"] != "none":
        lines.append(f"**It looks like: {d['subject']}**" + (f" ({d['confidence']} confidence)" if d.get("confidence") else "") + f"  \n*Source: {d['source']}.*")
        if d.get("summary"):
            lines.append(d["summary"])
        if d.get("answer"):
            lines.append(f"**Answer to your question:** {d['answer']}")
        if d.get("objects"):
            lines.append("Also visible: " + ", ".join(o["label"] for o in d["objects"][:6]) + ".")
        if d.get("details"):
            lines.append("Details: " + "; ".join(d["details"]) + ".")
        if d.get("text_in_image"):
            lines.append(f"Text in the image: \"{d['text_in_image']}\"")
        lines.append("*A vision model can be wrong, especially for similar-looking breeds or species; treat low confidence as a guess.*")
    else:
        lines.append("**I cannot tell what this image shows yet.** " + d["note"])
    lines.append(f"Measured on this machine: {f['width']}x{f['height']} px, brightness {f['brightness']}/255, sharpness {f['sharpness']}, "
                 f"dominant colours {', '.join(f['dominant_colours'])}" + (f"; flags: {', '.join(f['quality_flags'])}" if f["quality_flags"] else "") + ".")
    return "\n\n".join(lines)
