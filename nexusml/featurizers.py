"""Text and image featurizers that plug into the SAME sklearn pipeline as tabular columns (so the critic, tuning, explanation and
reporting stay unchanged). Everything is fit inside CV folds like any other preprocessing step.

  text   -> TF-IDF (1-2 grams, capped vocabulary), dense
  image  -> a column of image FILE PATHS -> handcrafted colour/brightness/edge/thumbnail features (Pillow + numpy, no GPU)
            optional: NEXUSML_IMAGE_BACKEND=resnet uses torchvision's ResNet-18 features when torch/torchvision are installed
            (untested in the reference environment; falls back to handcrafted if unavailable)
Images are never sent to the LLM; the LLM only sees aggregated column statistics.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config

IMAGE_ROOT: Optional[str] = None          # folder that relative image paths are resolved against (set when the data is prepared)
_IMG_RE = re.compile(r"\.(png|jpe?g|bmp|gif|webp|tiff?)$", re.I)


PREP: List[Dict[str, Any]] = []              # image preprocessing recipe (see imageprep.py), set when the data is prepared
PREP_SIG = ""
DEEP_MODE = "auto"                           # auto | off | resnet | resnet50 | both


def set_image_prep(steps) -> None:
    from . import imageprep
    global PREP, PREP_SIG
    PREP = imageprep.parse(steps)
    PREP_SIG = imageprep.signature(PREP)


def set_deep_mode(mode: Optional[str]) -> None:
    global DEEP_MODE
    DEEP_MODE = mode if mode in ("auto", "off", "resnet", "resnet50", "both") else "auto"


def set_image_root(path: Optional[str]) -> None:
    global IMAGE_ROOT
    IMAGE_ROOT = str(path) if path else None


def is_image_path(s: pd.Series) -> bool:
    nn = s.dropna().astype(str).head(300)
    return len(nn) >= 5 and float(nn.map(lambda v: bool(_IMG_RE.search(v.strip()))).mean()) >= 0.8


def is_text(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    nn = s.dropna().astype(str)
    if len(nn) < 20 or is_image_path(s):
        return False
    words = nn.head(500).str.split().map(len).mean()
    return bool(words >= config.TEXT_MIN_AVG_WORDS and nn.nunique() / len(nn) > 0.5)


def modalities(X: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"text": [], "image": []}
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype).startswith("str"):
            if is_image_path(X[c]):
                out["image"].append(c)
            elif is_text(X[c]):
                out["text"].append(c)
    return out


class TextFeaturizer(BaseEstimator, TransformerMixin):
    """TF-IDF on ONE text column (pass the column name, not a list)."""

    def __init__(self, max_features: int = None):
        self.max_features = max_features

    def fit(self, X, y=None):
        docs = pd.Series(X).fillna("").astype(str)
        mf = self.max_features or config.TEXT_MAX_FEATURES
        self.vec_ = TfidfVectorizer(max_features=mf, ngram_range=(1, 2), min_df=3 if len(docs) > 150 else 1, stop_words="english",
                                    sublinear_tf=True, lowercase=True)
        try:
            self.vec_.fit(docs)
        except ValueError:                                  # empty vocabulary (e.g. only stop-words)
            self.vec_ = TfidfVectorizer(max_features=mf, min_df=1, lowercase=True).fit(docs.where(docs.str.len() > 0, "empty"))
        return self

    def transform(self, X):
        docs = pd.Series(X).fillna("").astype(str)
        return self.vec_.transform(docs).toarray()

    def get_feature_names_out(self, names=None):
        return np.array(["text_" + t for t in self.vec_.get_feature_names_out()])


def _resolve(p: Any, root: Optional[str]) -> Optional[Path]:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    pp = Path(str(p).strip())
    if pp.is_absolute() and pp.is_file():
        return pp
    for base in (os.environ.get("NEXUSML_IMAGE_ROOT"), root, IMAGE_ROOT, "."):
        if base and (Path(base) / pp).is_file():
            return Path(base) / pp
    return None


def load_rgb(path: Path, size: Optional[int] = None):
    """Open an image the same way for every feature backend: honour the EXIF rotation, put transparency on white,
    scale 16-bit / palette / greyscale files to plain RGB, and optionally resize to size x size."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I", "F"):
            a = np.asarray(im, dtype=float)
            lo, hi = float(a.min()), float(a.max())
            im = Image.fromarray(((a - lo) / max(hi - lo, 1e-9) * 255).astype("uint8"))
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        if PREP:
            from . import imageprep
            im = imageprep.apply(im, PREP)
        if size:
            im = im.resize((size, size), Image.Resampling.BILINEAR if max(im.size) <= size else Image.Resampling.LANCZOS)
        else:
            im.load()
        return im


def check_images(values, root: Optional[str], min_found: float = 0.5) -> None:
    """Stop with a clear message when most image files cannot be found, instead of training on empty features."""
    vals = [v for v in pd.Series(values).dropna().astype(str).head(200)]
    if not vals:
        return
    found = sum(_resolve(v, root) is not None for v in vals)
    if found / len(vals) < min_found:
        raise ValueError(f"Only {found} of {len(vals)} sampled image files could be found (looked in: {root or 'the working folder'}). "
                         "Add the images again as a zip or folder so they are stored with the dataset.")


def handcrafted_image_features(path: Path) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        with Image.open(path) as im0:
            w, h = im0.size
        rgb = load_rgb(path, config.IMAGE_SIZE)
    except Exception:
        return None
    a = np.asarray(rgb, dtype=float) / 255.0
    gray = a.mean(axis=2)
    gx, gy = np.abs(np.diff(gray, axis=1)).mean(), np.abs(np.diff(gray, axis=0)).mean()
    hist = [np.histogram(a[:, :, c], bins=4, range=(0, 1))[0] / a[:, :, c].size for c in range(3)]
    thumb = np.asarray(rgb.convert("L").resize((8, 8)), dtype=float).ravel() / 255.0
    n = config.IMAGE_SIZE // 4 * 4                       # 4 x 4 colour layout: where the colours sit, not only how much
    layout = a[:n, :n].reshape(4, n // 4, 4, n // 4, 3).mean(axis=(1, 3)).ravel()
    return np.concatenate([a.mean(axis=(0, 1)), a.std(axis=(0, 1)), np.concatenate(hist), [gray.mean(), gray.std(), gx, gy, w / max(h, 1)], thumb, layout])


DEEP_ARCHS = {"resnet": ("resnet18", "ResNet18_Weights", "ResNet-18"), "resnet50": ("resnet50", "ResNet50_Weights", "ResNet-50")}
_DEEP: Dict[str, Any] = {}                   # arch -> {"embed": fn} or {"error": text}
EMBEDDER_OVERRIDE: Dict[str, Any] = {}       # arch -> fn(list of PIL images) -> (n, d) array; lets tests and other backends plug in


def _build_embedder(arch: str):
    """ImageNet-pretrained network with its classifier removed -> fn(list of PIL images) -> (n, d) features.
    NEXUSML_DEEP_WEIGHTS: 'default' (download the torchvision weights once), 'none' (random weights, plumbing tests only) or a path to a .pth state dict (offline)."""
    import torch
    import torchvision
    name, enum_name, _ = DEEP_ARCHS[arch]
    src = os.environ.get("NEXUSML_DEEP_WEIGHTS", "default")
    ctor = getattr(torchvision.models, name)
    if src == "default":
        w = getattr(torchvision.models, enum_name).DEFAULT
        model, tf = ctor(weights=w), w.transforms()
    else:
        model = ctor(weights=None)
        if src != "none":
            model.load_state_dict(torch.load(src, map_location="cpu"))
        T = torchvision.transforms
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model.fc = torch.nn.Identity()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    def embed(images):
        with torch.no_grad():
            x = torch.stack([tf(im) for im in images]).to(device)
            return model(x).cpu().numpy()
    return embed


def deep_embedder(arch: str):
    if arch in EMBEDDER_OVERRIDE:
        return EMBEDDER_OVERRIDE[arch]
    if arch not in _DEEP:
        try:
            _DEEP[arch] = {"embed": _build_embedder(arch)}
        except Exception as e:                                       # not installed, no weights offline, bad file ...
            _DEEP[arch] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
    if "error" in _DEEP[arch]:
        raise RuntimeError(_DEEP[arch]["error"])
    return _DEEP[arch]["embed"]


def deep_status(arch: str = "resnet") -> Dict[str, Any]:
    """{'ready': bool, 'message': text}. Loads the network once (may download weights on first use)."""
    try:
        deep_embedder(arch)
        return {"ready": True, "message": f"{DEEP_ARCHS[arch][2]} ready"}
    except Exception as e:
        return {"ready": False, "message": str(e)}


def deep_archs_selected() -> List[str]:
    """Which deep networks the model search should try, from the run option (and NEXUSML_IMAGE_BACKEND for compatibility)."""
    mode = DEEP_MODE
    if mode == "auto" and os.environ.get("NEXUSML_IMAGE_BACKEND", "") in DEEP_ARCHS:
        mode = os.environ["NEXUSML_IMAGE_BACKEND"]
    if mode == "off":
        return []
    want = ["resnet", "resnet50"] if mode == "both" else ["resnet"] if mode in ("auto", "resnet") else [mode]
    return [a for a in want if deep_status(a)["ready"]]


def deep_features(path: Path, arch: str = "resnet") -> Optional[np.ndarray]:
    try:
        return np.asarray(deep_embedder(arch)([load_rgb(path)])[0], dtype=float)
    except Exception:
        return None


def resnet_features(path: Path) -> Optional[np.ndarray]:           # kept for older callers
    return deep_features(path, "resnet")


class _using_prep:
    """Run a block with a given preprocessing recipe (the one the model was trained with), then restore the previous one."""

    def __init__(self, steps):
        self.steps = steps

    def __enter__(self):
        self.old = list(PREP)
        if self.steps is not None:
            set_image_prep(self.steps)

    def __exit__(self, *exc):
        set_image_prep(self.old)


class ImageFeaturizer(BaseEstimator, TransformerMixin):
    """A DataFrame with ONE column of image file paths -> numeric feature matrix (NaN rows for unreadable files)."""

    def __init__(self, backend: str = None):
        self.backend = backend

    def fit(self, X, y=None):
        self.root_ = IMAGE_ROOT
        self.prep_ = list(PREP)
        self.backend_ = self.backend or os.environ.get("NEXUSML_IMAGE_BACKEND", "handcrafted")
        self.n_features_ = None
        probe = None
        for v in pd.DataFrame(X).iloc[:, 0].head(50):
            p = _resolve(v, self.root_)
            if p is not None:
                probe = self._one(p)
                if probe is not None:
                    break
        self.n_features_ = len(probe) if probe is not None else 1
        return self

    def _one(self, p: Path):
        if self.backend_ in DEEP_ARCHS:
            from . import cv
            f = cv.extract(p, self.backend_)
            if f is not None:
                return f
        elif self.backend_ in ("hog", "lbp", "hsv", "texture", "color", "shape", "shapecolor"):
            from . import cv
            f = cv.extract(p, self.backend_)                 # cached per file + backend
            if f is not None:
                return f
        return handcrafted_image_features(p)

    def transform(self, X):
        col = pd.DataFrame(X).iloc[:, 0]
        out = np.full((len(col), self.n_features_), np.nan)
        paths = [_resolve(v, self.root_) for v in col]
        with _using_prep(getattr(self, "prep_", None)):
            if self.backend_ in DEEP_ARCHS:
                from . import cv
                cv.prefetch([p for p in paths if p is not None], self.backend_)      # batched network passes
            for i, p in enumerate(paths):
                f = self._one(p) if p is not None else None
                if f is not None and len(f) == self.n_features_:
                    out[i] = f
        return out


# leakage: duplicates across train/test
def _norm_text(s: str) -> str:
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def _ahash(path: Path) -> Optional[int]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            px = np.asarray(im.convert("L").resize((8, 8)), dtype=float)
        bits = (px > px.mean()).ravel()
        return int("".join("1" if b else "0" for b in bits), 2)
    except Exception:
        return None


def duplicate_report(X_train: pd.DataFrame, X_test: pd.DataFrame, mods: Dict[str, List[str]]) -> Dict[str, Any]:
    """Share of test rows whose text / image is an exact or near duplicate of a training row (inflates held-out scores)."""
    rep: Dict[str, Any] = {}
    for c in mods.get("text", []):
        tr = set(X_train[c].dropna().map(_norm_text))
        te = X_test[c].dropna().map(_norm_text)
        exact = float(te.isin(tr).mean()) if len(te) else 0.0
        near = exact
        try:
            vec = TfidfVectorizer(stop_words="english").fit(list(tr)[:5000] + list(te)[:5000])
            A, B = vec.transform(list(tr)[:5000]), vec.transform(list(te)[:2000])
            sims = (B @ A.T).max(axis=1).toarray().ravel()
            near = float((sims >= 0.95).mean())
        except Exception:
            pass
        rep[c] = {"kind": "text", "exact_dup_share": round(exact, 4), "near_dup_share": round(max(near, exact), 4)}
    for c in mods.get("image", []):
        def hashes(s):
            out = []
            for v in s.dropna().head(3000):
                p = _resolve(v, IMAGE_ROOT)
                out.append(_ahash(p) if p else None)
            return [h for h in out if h is not None]
        htr, hte = hashes(X_train[c]), hashes(X_test[c])
        if hte:
            trs = np.array(htr, dtype=object)
            near = np.mean([any(bin(h ^ t).count("1") <= 2 for t in trs[:3000]) for h in hte[:500]]) if len(trs) else 0.0
            rep[c] = {"kind": "image", "exact_dup_share": round(float(np.mean([h in set(htr) for h in hte])), 4), "near_dup_share": round(float(near), 4)}
    return rep
