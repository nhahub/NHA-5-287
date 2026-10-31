"""Image preprocessing steps shared by every image feature backend (hand-crafted and deep).

A recipe is a list of steps, applied in order: {"op": "blur", "radius": 1.5}. Short forms are accepted too ("blur:1.5", "grayscale").
Pillow + numpy only. The same recipe is stored with the model, so predictions on new photos are preprocessed identically.

  resize        size (px, 16-1024), mode: squash | fit (keep aspect, pad with white) | fill (keep aspect, centre-crop)
  grayscale     drop colour
  autocontrast  cutoff (% of extreme pixels ignored)
  equalize      histogram equalisation
  blur          radius (Gaussian)
  sharpen       amount (%)  - unsharp mask
  denoise       size (3, 5 or 7) - median filter
  edges         edge map (Sobel gradient magnitude) as a grayscale image
"""
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# op -> {param: (default, low, high)}; string params list their choices
OPS: Dict[str, Dict[str, Any]] = {
    "resize": {"size": (128, 16, 1024), "mode": ("squash", ["squash", "fit", "fill"])},
    "grayscale": {},
    "autocontrast": {"cutoff": (1.0, 0.0, 20.0)},
    "equalize": {},
    "blur": {"radius": (1.5, 0.1, 25.0)},
    "sharpen": {"amount": (150.0, 10.0, 500.0)},
    "denoise": {"size": (3, 3, 7)},
    "edges": {},
}
LABELS = {"resize": "Resize", "grayscale": "Grayscale", "autocontrast": "Stretch contrast", "equalize": "Equalise histogram",
          "blur": "Blur", "sharpen": "Sharpen", "denoise": "Remove noise (median)", "edges": "Edge detection"}

Step = Dict[str, Any]


def _clean(op: str, params: Dict[str, Any]) -> Step:
    if op not in OPS:
        raise ValueError(f"Unknown image step '{op}'. Choose from: {', '.join(OPS)}.")
    out: Step = {"op": op}
    for k, spec in OPS[op].items():
        v = params.get(k, spec[0])
        if isinstance(spec[1], list):                                   # choice
            if v not in spec[1]:
                raise ValueError(f"'{k}' for {op} must be one of {spec[1]}.")
            out[k] = v
        else:
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"'{k}' for {op} must be a number.")
            lo, hi = spec[1], spec[2]
            if not lo <= v <= hi:
                raise ValueError(f"'{k}' for {op} must be between {lo:g} and {hi:g}.")
            out[k] = int(v) if isinstance(spec[0], int) else v
    if op == "denoise" and out["size"] not in (3, 5, 7):
        raise ValueError("'size' for denoise must be 3, 5 or 7.")
    unknown = set(params) - set(OPS[op]) - {"op"}
    if unknown:
        raise ValueError(f"Unknown setting(s) for {op}: {', '.join(sorted(unknown))}.")
    return out


def parse(spec: Union[None, str, Sequence[Union[str, Dict[str, Any]]]]) -> List[Step]:
    """None / '' / JSON text / list of dicts or short strings -> a validated list of steps."""
    if spec is None or spec == "" or spec == []:
        return []
    if isinstance(spec, str):
        spec = json.loads(spec) if spec.strip().startswith(("[", "{")) else [s for s in spec.replace(";", ",").split(",") if s.strip()]
    if isinstance(spec, dict):
        spec = [spec]
    steps: List[Step] = []
    for item in spec:
        if isinstance(item, str):
            name, _, val = item.strip().partition(":")
            params: Dict[str, Any] = {}
            if val:
                first = next(iter(OPS.get(name.strip(), {})), None)
                if first is None:
                    raise ValueError(f"'{name}' takes no value.")
                params[first] = val.strip()
            steps.append(_clean(name.strip(), params))
        elif isinstance(item, dict):
            steps.append(_clean(str(item.get("op", "")), {k: v for k, v in item.items() if k != "op"}))
        else:
            raise ValueError("Image steps must be names or {'op': ...} objects.")
    return steps


def signature(steps: Optional[List[Step]]) -> str:
    """Stable text id of a recipe (part of every feature-cache key)."""
    return json.dumps(steps, sort_keys=True) if steps else ""


def describe(steps: Optional[List[Step]]) -> str:
    if not steps:
        return "none"
    out = []
    for s in steps:
        extra = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items() if k != "op")
        out.append(LABELS[s["op"]] + (f" ({extra})" if extra else ""))
    return " -> ".join(out)


def _sobel(gray: np.ndarray) -> np.ndarray:
    p = np.pad(gray, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.hypot(gx, gy)


def apply_step(im, step: Step):
    from PIL import Image, ImageFilter, ImageOps
    op = step["op"]
    if op == "resize":
        n, mode = int(step["size"]), step["mode"]
        if mode == "squash":
            return im.resize((n, n), Image.Resampling.LANCZOS)
        if mode == "fit":
            fitted = ImageOps.contain(im, (n, n), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (n, n), (255, 255, 255))
            canvas.paste(fitted, ((n - fitted.width) // 2, (n - fitted.height) // 2))
            return canvas
        return ImageOps.fit(im, (n, n), Image.Resampling.LANCZOS)
    if op == "grayscale":
        return ImageOps.grayscale(im).convert("RGB")
    if op == "autocontrast":
        return ImageOps.autocontrast(im, cutoff=float(step["cutoff"]))
    if op == "equalize":
        return ImageOps.equalize(im)
    if op == "blur":
        return im.filter(ImageFilter.GaussianBlur(radius=float(step["radius"])))
    if op == "sharpen":
        return im.filter(ImageFilter.UnsharpMask(radius=2, percent=int(step["amount"]), threshold=2))
    if op == "denoise":
        return im.filter(ImageFilter.MedianFilter(size=int(step["size"])))
    if op == "edges":
        g = np.asarray(ImageOps.grayscale(im), dtype=float)
        e = _sobel(g)
        e = e / (np.percentile(e, 99) + 1e-9) * 255.0
        return Image.fromarray(np.clip(e, 0, 255).astype("uint8")).convert("RGB")
    raise ValueError(f"Unknown image step '{op}'.")


def apply(im, steps: Optional[List[Step]]):
    for s in steps or []:
        im = apply_step(im, s)
    return im


def preview(paths: Sequence[Any], steps: Optional[List[Step]], load, limit: int = 3, size: int = 160) -> List[Tuple[Any, Any]]:
    """(original, processed) thumbnails for the first readable paths. `load(path)` returns an RGB image or raises."""
    out = []
    for p in paths:
        if len(out) >= limit:
            break
        try:
            im = load(p)
            out.append((_thumb(im, size), _thumb(apply(im.copy(), steps), size)))
        except Exception:
            continue
    return out


def _thumb(im, size: int):
    im = im.copy()
    im.thumbnail((size, size))
    return im
