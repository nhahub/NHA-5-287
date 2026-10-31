"""Computer-vision pipeline. Runs on CPU with Pillow + numpy + scikit-learn; a pretrained ResNet-18 backend is an OPTIONAL extra (torch + torchvision).

  feature backends  color (colour, edge and thumbnail statistics), hog (edges), lbp (texture), hsv (colour histogram), texture (all of them), shape (rotation-invariant outline of the main object), shapecolor, resnet (optional deep embedding)
  candidates        representation x classifier grid (LogReg / SVC-RBF / KNN / RF / HGB). They are ordinary sklearn Pipelines (the image column is featurised
                    INSIDE the pipeline), so tuning, the critic, the leaderboard, MLflow, the saved bundle and the Predict tab work unchanged.
  collection        class montage, mean image per class, feature-space map, label-issue suspects, visual outliers          -> analyze
  after the model   most-confident mistakes gallery, occlusion-sensitivity heatmaps (model-agnostic "where did it look?"),
                    robustness stress test (brightness, blur, noise, rotation, JPEG, downscale, flip, crop)               -> misclassified_gallery / occlusion_map / robustness
  utilities         ImageIndex (find similar images)

Limits: hand-crafted features recognise colour, edge, texture and shape statistics; they cannot recognise objects the way a deep network does.
The report says so, and shows the ResNet candidate only when torch is installed.
"""
import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


FEATURE_SIZE = 64
_CACHE: Dict[Tuple[str, float, str, str], np.ndarray] = {}
_CACHE_MAX = 20000
BACKEND_LABEL = {"color": "basic image stats", "hog": "HOG", "lbp": "LBP texture", "hsv": "HSV histogram", "texture": "HOG+LBP+colour", "shape": "outline shape",
                 "shapecolor": "shape+colour", "resnet": "ResNet-18", "resnet50": "ResNet-50"}


# feature backends
def _load(path: Path, size: int, mode: str):
    from . import featurizers
    return featurizers.load_rgb(path, size).convert(mode)


def hog_features(g: np.ndarray, cell: int = 8, bins: int = 9) -> np.ndarray:
    gx, gy = np.zeros_like(g), np.zeros_like(g)
    gx[:, 1:-1], gy[1:-1, :] = g[:, 2:] - g[:, :-2], g[2:, :] - g[:-2, :]
    mag, ang = np.hypot(gx, gy), (np.arctan2(gy, gx) * 180 / np.pi) % 180
    n = g.shape[0] // cell
    b = (ang / (180 / bins)).astype(int).clip(0, bins - 1)
    H = np.stack([(mag * (b == k))[: n * cell, : n * cell].reshape(n, cell, n, cell).sum(axis=(1, 3)) for k in range(bins)], axis=-1)
    H = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-6)
    return H.ravel()


def lbp_features(g: np.ndarray, grid: int = 2) -> np.ndarray:
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.int32)
    for k, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
        code += ((g[1 + dy: g.shape[0] - 1 + dy, 1 + dx: g.shape[1] - 1 + dx] >= c).astype(np.int32) << k)
    code = code >> 3                                                  # 256 -> 32 bins
    h, w = code.shape
    out = []
    for i in range(grid):
        for j in range(grid):
            blk = code[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
            hist = np.bincount(blk.ravel(), minlength=32).astype(float)
            out.append(hist / max(hist.sum(), 1))
    return np.concatenate(out)


def shape_features(a, nang=32, nharm=8):
    """a: HxWx3 float 0..1. Rotation/scale/position-invariant outline descriptors of the main object."""
    H,W,_=a.shape
    border=np.concatenate([a[0],a[-1],a[:,0],a[:,-1]]); bg=np.median(border,axis=0)
    d=np.linalg.norm(a-bg,axis=2)
    # Otsu on distance-to-background
    hist,edges=np.histogram(d,bins=64); p=hist/hist.sum(); w=np.cumsum(p); mu=np.cumsum(p*(edges[:-1]+edges[1:])/2); mt=mu[-1]
    sb=(mt*w-mu)**2/(w*(1-w)+1e-12); t=edges[1:][np.argmax(sb)]
    m=d>t
    out=np.zeros(5+7+nharm+3)
    n=int(m.sum())
    if n<8 or n>0.97*H*W: return out
    ys,xs=np.nonzero(m); cy,cx=ys.mean(),xs.mean()
    er=m.copy(); er[1:-1,1:-1]=m[1:-1,1:-1]&m[:-2,1:-1]&m[2:,1:-1]&m[1:-1,:-2]&m[1:-1,2:]
    bd=m&~er; by,bx=np.nonzero(bd)
    per=max(len(by),1); comp=4*np.pi*n/(per**2)
    cov=np.cov(np.vstack([xs-cx,ys-cy])); ev=np.sort(np.linalg.eigvalsh(cov))[::-1]; elong=np.sqrt(ev[1]/(ev[0]+1e-9))
    sol=0.0
    try:
        from scipy.spatial import ConvexHull
        pts=np.column_stack([bx,by]); sol=n/(ConvexHull(pts).volume+1e-9)
    except Exception: pass
    # normalised central moments -> first 4 Hu invariants
    x=xs-cx; y=ys-cy
    def eta(p,q): return ((x**p)*(y**q)).sum()/n**(1+(p+q)/2)
    n20,n02,n11,n30,n03,n21,n12=eta(2,0),eta(0,2),eta(1,1),eta(3,0),eta(0,3),eta(2,1),eta(1,2)
    hu=[n20+n02,(n20-n02)**2+4*n11**2,(n30-3*n12)**2+(3*n21-n03)**2,(n30+n12)**2+(n21+n03)**2,
        (n30-3*n12)*(n30+n12)*((n30+n12)**2-3*(n21+n03)**2)+(3*n21-n03)*(n21+n03)*(3*(n30+n12)**2-(n21+n03)**2),
        (n20-n02)*((n30+n12)**2-(n21+n03)**2)+4*n11*(n30+n12)*(n21+n03),
        (3*n21-n03)*(n30+n12)*((n30+n12)**2-3*(n21+n03)**2)-(n30-3*n12)*(n21+n03)*(3*(n30+n12)**2-(n21+n03)**2)]
    hu=[np.sign(h)*np.log10(abs(h)+1e-12) for h in hu]
    r=np.hypot(bx-cx,by-cy); ang=np.arctan2(by-cy,bx-cx); r=r/(r.mean()+1e-9)
    bins=((ang+np.pi)/(2*np.pi)*nang).astype(int)%nang
    prof=np.array([r[bins==k].mean() if (bins==k).any() else 1.0 for k in range(nang)])
    f=np.abs(np.fft.rfft(prof-prof.mean()))[1:nharm+1]/nang
    out[:]=np.concatenate([[n/(H*W),comp,elong,sol,per/np.sqrt(n)],hu,f,[r.std(),r.max()/r.min() if r.min()>0 else 0,np.mean(((r-r.mean())/(r.std()+1e-9))**3)]])
    return np.nan_to_num(out)


def hsv_features(path: Path) -> np.ndarray:
    a = np.asarray(_load(path, 32, "HSV"), dtype=float) / 255.0
    return np.concatenate([np.histogram(a[..., k], bins=8, range=(0, 1))[0] / a[..., k].size for k in range(3)])


def _key(path: Path, backend: str):
    from . import featurizers
    try:
        return (str(path), os.path.getmtime(path), backend, featurizers.PREP_SIG)
    except OSError:
        return None


def prefetch(paths: List[Path], backend: str, batch: int = 32) -> None:
    """Fill the feature cache for a deep backend with batched network passes (much faster than one image at a time)."""
    from . import featurizers
    if backend not in featurizers.DEEP_ARCHS:
        return
    todo = [p for p in dict.fromkeys(paths) if (k := _key(p, backend)) is not None and k not in _CACHE]
    if not todo:
        return
    try:
        embed = featurizers.deep_embedder(backend)
    except Exception:
        return
    for i in range(0, len(todo), batch):
        ims, ok = [], []
        for p in todo[i:i + batch]:
            try:
                ims.append(featurizers.load_rgb(p))
                ok.append(p)
            except Exception:
                continue
        if not ims:
            continue
        try:
            feats = np.asarray(embed(ims), dtype=float)
        except Exception:
            continue
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.clear()
        for p, v in zip(ok, feats):
            _CACHE[_key(p, backend)] = v


def extract(path: Path, backend: str) -> Optional[np.ndarray]:
    """Feature vector for one image (None if unreadable). Cached by file + modification time + backend, so cross-validation does not re-read the disk."""
    from . import featurizers
    key = _key(path, backend)
    if key is None:
        return None
    if key in _CACHE:
        return _CACHE[key]
    try:
        if backend in ("color", "handcrafted"):
            v = featurizers.handcrafted_image_features(path)
        elif backend in featurizers.DEEP_ARCHS:
            v = featurizers.deep_features(path, backend)
        else:
            g = np.asarray(_load(path, FEATURE_SIZE, "L"), dtype=float) / 255.0
            if backend == "hog":
                v = hog_features(g)
            elif backend == "lbp":
                v = lbp_features(g)
            elif backend == "hsv":
                v = hsv_features(path)
            elif backend in ("shape", "shapecolor"):
                v = shape_features(np.asarray(_load(path, 64, "RGB"), dtype=float) / 255.0)
                if backend == "shapecolor":
                    col = featurizers.handcrafted_image_features(path)
                    v = np.concatenate([v] + ([col] if col is not None else []))
            elif backend == "texture":
                col = featurizers.handcrafted_image_features(path)
                v = np.concatenate([hog_features(g), lbp_features(g), hsv_features(path)] + ([col] if col is not None else []))
            else:
                v = None
    except Exception:
        v = None
    if v is not None:
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = np.asarray(v, dtype=float)
    return v


def resnet_available() -> bool:
    """A deep network can be used right now (torch installed and its weights loadable)."""
    from . import featurizers
    return bool(featurizers.deep_archs_selected())


# candidates
def candidate_specs(task: str, strategy: Dict[str, Any], rs: int) -> Dict[str, Any]:
    """Representation x model candidates for an image task (pixels only: tabular metadata such as brightness would be a shortcut, not vision)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from . import featurizers
    from .modeling import Spec
    cw, rg = strategy.get("class_weight"), bool(strategy.get("regularize"))
    L = BACKEND_LABEL
    m: Dict[str, Any] = {}
    if task == "classification":
        c = 0.1 if rg else 1.0
        m[f"LogisticRegression · {L['color']}"] = Spec(LogisticRegression(max_iter=3000, class_weight=cw, C=c), image_rep="color", meta=False)
        m[f"LogisticRegression · {L['hog']}"] = Spec(LogisticRegression(max_iter=3000, class_weight=cw, C=c), image_rep="hog", meta=False)
        m[f"LogisticRegression · {L['shape']}"] = Spec(LogisticRegression(max_iter=3000, class_weight=cw, C=c), image_rep="shape", meta=False)
        m[f"SVC · {L['shape']}"] = Spec(SVC(C=0.5 if rg else 2.0, kernel="rbf", probability=True, class_weight=cw, random_state=rs), image_rep="shape", meta=False)
        m[f"RandomForest · {L['shapecolor']}"] = Spec(RandomForestClassifier(n_estimators=200, class_weight=cw, n_jobs=-1, random_state=rs, **({"min_samples_leaf": 3} if rg else {})),
                                                      image_rep="shapecolor", meta=False)
        m[f"LogisticRegression · {L['texture']}"] = Spec(LogisticRegression(max_iter=3000, class_weight=cw, C=c), image_rep="texture", meta=False)
        m[f"SVC · {L['texture']}"] = Spec(SVC(C=0.5 if rg else 2.0, kernel="rbf", probability=True, class_weight=cw, random_state=rs), image_rep="texture", meta=False)
        m[f"RandomForest · {L['texture']}"] = Spec(RandomForestClassifier(n_estimators=200, class_weight=cw, n_jobs=-1, random_state=rs, **({"min_samples_leaf": 3} if rg else {})),
                                                   image_rep="texture", meta=False)
        m[f"KNN · {L['texture']}"] = Spec(KNeighborsClassifier(n_neighbors=25 if rg else 7), image_rep="texture", meta=False)
        m[f"HistGradientBoosting · {L['color']}"] = Spec(HistGradientBoostingClassifier(class_weight=cw, random_state=rs, **({"l2_regularization": 1.0, "max_depth": 4} if rg else {})),
                                                         image_rep="color", meta=False)
        for arch in featurizers.deep_archs_selected():
            m[f"LogisticRegression · {L[arch]}"] = Spec(LogisticRegression(max_iter=3000, class_weight=cw, C=0.1 if rg else 0.5), image_rep=arch, meta=False)
            m[f"SVC · {L[arch]}"] = Spec(SVC(C=0.5 if rg else 2.0, kernel="rbf", probability=True, class_weight=cw, random_state=rs), image_rep=arch, meta=False)
    else:
        for arch in featurizers.deep_archs_selected():
            m[f"Ridge · {L[arch]}"] = Spec(Ridge(alpha=10.0 if rg else 1.0), image_rep=arch, meta=False)
        m[f"Ridge · {L['texture']}"] = Spec(Ridge(alpha=10.0 if rg else 1.0), image_rep="texture", meta=False)
        m[f"RandomForest · {L['texture']}"] = Spec(RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=rs, **({"min_samples_leaf": 3} if rg else {})), image_rep="texture", meta=False)
        m[f"HistGradientBoosting · {L['color']}"] = Spec(HistGradientBoostingRegressor(random_state=rs), image_rep="color", meta=False)
    return m


# helpers
def _paths(df: pd.DataFrame, col: str, root: Path) -> List[Path]:
    return [Path(root) / str(p) if not Path(str(p)).is_absolute() else Path(str(p)) for p in df[col]]


def feature_matrix(paths: List[Path], backend: str = "texture") -> Tuple[np.ndarray, List[int]]:
    rows, ok = [], []
    prefetch(paths, backend)
    for i, p in enumerate(paths):
        v = extract(p, backend)
        if v is not None:
            rows.append(v)
            ok.append(i)
    n = max((len(r) for r in rows), default=0)
    return (np.vstack([r for r in rows if len(r) == n]) if rows else np.zeros((0, 0))), ok


def _thumb(path: Path, size: int = 96):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        return np.asarray(im)


def _montage(paths: List[Path], titles: List[str], ncols: int = 6, size: int = 96, title: str = ""):
    import matplotlib.pyplot as plt
    n = len(paths)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.75 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p, t in zip(axes.ravel(), paths, titles):
        try:
            ax.imshow(_thumb(p, size))
        except Exception:
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center", fontsize=7)
        ax.set_title(t, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# collection analysis
def label_issues_images(X: np.ndarray, labels: List[str], seed: int = 42, top: int = 12) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(labels)
    vc = pd.Series(y).value_counts()
    if len(y) < 40 or vc.min() < 5 or len(vc) < 2 or X.shape[0] != len(y):
        return {}
    classes = sorted(vc.index)
    yi = np.searchsorted(classes, y)
    try:
        P = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)), X, yi,
                              cv=StratifiedKFold(min(5, int(vc.min())), shuffle=True, random_state=seed), method="predict_proba")
    except Exception:
        return {}
    sc = P[np.arange(len(yi)), yi]
    thr = np.array([sc[yi == c].mean() for c in range(len(classes))])
    pred = P.argmax(axis=1)
    sus = [i for i in range(len(yi)) if pred[i] != yi[i] and sc[i] < thr[yi[i]] and P[i, pred[i]] >= 0.6]
    sus.sort(key=lambda i: -(P[i, pred[i]] - sc[i]))
    return {"n_suspects": len(sus), "share": round(len(sus) / len(yi), 4),
            "suspects": [{"index": int(i), "given": str(classes[yi[i]]), "suggested": str(classes[pred[i]]), "p_given": round(float(sc[i]), 3), "p_suggested": round(float(P[i, pred[i]]), 3)} for i in sus[:top]]}


def analyze(df: pd.DataFrame, col: str, root: Path, label: Optional[str] = None, seed: int = 42, max_images: int = 1500):
    """-> (evidence_items, figures, report). Complements mediamining.image_insights (quality, duplicates, shortcut risk) with the vision-model view."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    d = df.reset_index(drop=True)
    if len(d) > max_images:
        d = d.sample(max_images, random_state=seed).reset_index(drop=True)
    paths = _paths(d, col, root)
    items: List[Tuple[str, str, Any]] = []
    figs: List[Tuple[str, str, str, Any]] = []
    rep: Dict[str, Any] = {"image_column": col, "n_images": len(d), "resnet_available": resnet_available()}
    X, ok = feature_matrix(paths, "texture")
    labels = d[label].astype(str).tolist() if label and label in d else None
    lab_ok = [labels[i] for i in ok] if labels else None
    rep["n_features"] = int(X.shape[1])
    items.append(("vision_features", f"Vision pipeline: {len(ok):,} of {len(d):,} images featurised (HOG shape + LBP texture + HSV colour + colour statistics = {X.shape[1]} features each); "
                                     f"{'a pretrained ResNet-18 candidate is also available' if rep['resnet_available'] else 'no deep model is installed, so candidates use hand-crafted features (they capture colour, texture and shape, not objects)'}.", {"n_features": int(X.shape[1])}))
    if len(ok) < 10:
        return items, figs, rep
    # ---- montage and mean images per class
    if lab_ok:
        classes = sorted(set(lab_ok))[:8]
        sel, tt = [], []
        for c in classes:
            idx = [ok[j] for j, l in enumerate(lab_ok) if l == c][:5]
            sel += [paths[i] for i in idx]
            tt += [str(c)[:14]] * len(idx)
        figs.append(("montage", "Sample images per class", "Look for classes that differ only in background, lighting or image source rather than in the object itself.", _montage(sel, tt, ncols=5, title="Samples per class")))
        fig, axes = plt.subplots(1, len(classes), figsize=(1.9 * len(classes), 2.2), squeeze=False)
        for ax, c in zip(axes[0], classes):
            ims = [np.asarray(_load(paths[ok[j]], 64, "RGB"), dtype=float) for j, l in enumerate(lab_ok) if l == c][:60]
            ax.imshow(np.mean(ims, axis=0).astype(np.uint8))
            ax.set_title(str(c)[:14], fontsize=8)
            ax.axis("off")
        fig.suptitle("Mean image per class", fontsize=10)
        figs.append(("mean_images", "Mean image per class", "A mean image with obvious structure (a bright centre, a coloured border) shows what the classes share.", fig))
    else:
        figs.append(("montage", "Sample images", "A random sample of the collection.", _montage([paths[i] for i in ok[:18]], [""] * min(18, len(ok)), ncols=6, title="Sample images")))
    # ---- feature-space map
    Z = StandardScaler().fit_transform(X)
    pts = PCA(n_components=2, random_state=seed).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    if lab_ok:
        for c in sorted(set(lab_ok))[:10]:
            m = np.asarray(lab_ok) == c
            ax.scatter(pts[m, 0], pts[m, 1], s=14, alpha=0.7, label=str(c)[:16])
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.7, color="#2f6fed")
    ax.set_title("Image feature space (PCA of HOG+LBP+colour)", fontsize=10)
    figs.append(("feature_map", "Feature-space map", "Classes that overlap heavily here are hard for hand-crafted features; separated clouds are easy.", fig))
    # ---- label issues
    if lab_ok:
        li = label_issues_images(X, lab_ok, seed)
        if li:
            for s in li["suspects"]:
                s["file"] = str(d[col].iloc[ok[s["index"]]])
                s["index"] = int(ok[s["index"]])
            rep["label_issues"] = li
            items.append(("vision_label_issues", f"{li['n_suspects']} image(s) ({li['share']:.1%}) look mislabelled: an out-of-fold model confidently prefers another class. Review them before trusting the held-out score.",
                          {"n": li["n_suspects"], "share": li["share"]}))
            sus = li["suspects"][:12]
            if sus:
                figs.append(("label_issues", "Possible label errors", "Title: given label -> suggested label.",
                             _montage([paths[s["index"]] for s in sus], [f"{s['given'][:8]}→{s['suggested'][:8]}" for s in sus], ncols=6, title="Possible label errors")))
    # ---- visual outliers
    if len(ok) >= 30:
        iso = IsolationForest(n_estimators=150, contamination="auto", random_state=seed).fit(Z)
        sc = -iso.score_samples(Z)
        top = np.argsort(-sc)[:6]
        rep["outliers"] = [{"index": int(ok[i]), "file": str(d[col].iloc[ok[i]]), "score": round(float(sc[i]), 3)} for i in top]
        figs.append(("outliers", "Most unusual images", "Images far from the rest in feature space: wrong subject, corrupted, or a different source.",
                     _montage([paths[ok[i]] for i in top], [f"{sc[i]:.2f}" for i in top], ncols=6, title="Most unusual images")))
        items.append(("vision_outliers", f"Isolation-forest check on image features flags {int((sc > np.percentile(sc, 97)).sum())} visually unusual image(s) (top 3%).", {}))
    return items, figs, rep


# perturbations (stress test)
def _p_brightness(f):
    def fn(im, rng):
        from PIL import ImageEnhance
        return ImageEnhance.Brightness(im).enhance(f)
    return fn


def _p_blur(im, rng):
    from PIL import ImageFilter
    return im.filter(ImageFilter.GaussianBlur(2.0))


def _p_noise(im, rng):
    from PIL import Image
    a = np.asarray(im, dtype=float) + rng.normal(0, 0.08 * 255, np.asarray(im).shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _p_rotate(im, rng):
    return im.rotate(15, expand=False)


def _p_jpeg(im, rng):
    from PIL import Image
    b = io.BytesIO()
    im.save(b, "JPEG", quality=15)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def _p_down(im, rng):
    w, h = im.size
    return im.resize((max(8, w // 4), max(8, h // 4))).resize((w, h))


def _p_flip(im, rng):
    from PIL import ImageOps
    return ImageOps.mirror(im)


def _p_crop(im, rng):
    w, h = im.size
    return im.crop((int(w * .1), int(h * .1), int(w * .9), int(h * .9))).resize((w, h))


PERTURBATIONS: Dict[str, Callable] = {"darker (x0.6)": _p_brightness(0.6), "brighter (x1.5)": _p_brightness(1.5), "blur": _p_blur, "gaussian noise": _p_noise,
                                      "rotate 15°": _p_rotate, "JPEG quality 15": _p_jpeg, "low resolution (÷4)": _p_down, "mirror flip": _p_flip, "centre crop 80%": _p_crop}


def _perturbed_frame(X: pd.DataFrame, col: str, root: Path, fn: Callable, tmp: Path, seed: int) -> pd.DataFrame:
    from PIL import Image
    rng = np.random.RandomState(seed)
    Xp = X.copy()
    out = []
    for i, p in enumerate(_paths(X, col, root)):
        try:
            with Image.open(p) as im:
                q = fn(im.convert("RGB"), rng)
            f = tmp / f"p{i}.png"
            q.save(f)
            out.append(str(f))
        except Exception:
            out.append(str(p))
    Xp[col] = out
    return Xp


def robustness(est, X_te: pd.DataFrame, y_te: np.ndarray, col: str, root: Path, task: str = "classification", seed: int = 42, max_rows: int = 120) -> Dict[str, Any]:
    """Held-out score of the winning model after degrading the IMAGES. Large drops mean the model relies on fragile cues (exact brightness, sharpness, framing)."""
    from sklearn.metrics import accuracy_score, r2_score
    rng = np.random.RandomState(seed)
    n = min(len(X_te), max_rows)
    idx = rng.choice(len(X_te), n, replace=False) if len(X_te) > n else np.arange(len(X_te))
    X, y = X_te.iloc[idx].copy(), np.asarray(y_te)[idx]
    score = accuracy_score if task == "classification" else r2_score
    base = float(score(y, est.predict(X)))
    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="ds_robust_"))
    try:
        for name, fn in PERTURBATIONS.items():
            sub = tmp / name.split()[0].strip("()")
            sub.mkdir(parents=True, exist_ok=True)
            try:
                s = float(score(y, est.predict(_perturbed_frame(X, col, root, fn, sub, seed))))
                rows.append({"perturbation": name, "score": round(s, 4), "drop": round(base - s, 4)})
            except Exception as e:
                rows.append({"perturbation": name, "error": type(e).__name__})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = [r for r in rows if "drop" in r]
    worst = max(ok, key=lambda r: r["drop"]) if ok else None
    return {"metric": "accuracy" if task == "classification" else "r2", "clean": round(base, 4), "n_rows": int(n), "results": rows, "worst": worst, "fragile": bool(worst and worst["drop"] > 0.10)}


def robustness_figure(rob: Dict[str, Any], what: str = "images"):
    import matplotlib.pyplot as plt
    rows = [r for r in rob.get("results", []) if "drop" in r]
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * max(3, len(rows)) + 0.8))
    ax.barh([r["perturbation"] for r in rows][::-1], [r["score"] for r in rows][::-1], color="#2f6fed")
    ax.axvline(rob["clean"], color="#e45b5b", ls="--", lw=1, label=f"clean {rob['clean']}")
    ax.set_xlabel(rob["metric"])
    ax.set_title(f"Held-out score under {what} degradations", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


# explainability
def occlusion_map(est, row: pd.DataFrame, col: str, root: Path, classes: Optional[List[str]] = None, grid: int = 6) -> Dict[str, Any]:
    """Model-agnostic 'where did it look?': grey out one cell of a grid at a time and measure how much the predicted class probability falls.
    Needs no gradients, so it works for any pipeline (hand-crafted features, SVM, forest, ResNet probe). Returns {'heat': grid x grid array, 'label', 'score', 'figure'}."""
    import matplotlib.pyplot as plt
    from PIL import Image
    path = _paths(row.iloc[[0]], col, root)[0]
    with Image.open(path) as im0:
        im = im0.convert("RGB").resize((192, 192))
    a = np.asarray(im)
    mean = tuple(int(v) for v in a.reshape(-1, 3).mean(axis=0))
    step = 192 // grid
    tmp = Path(tempfile.mkdtemp(prefix="ds_occ_"))
    try:
        files = []
        base = tmp / "base.png"
        im.save(base)
        files.append(str(base))
        for i in range(grid):
            for j in range(grid):
                v = im.copy()
                v.paste(mean, (j * step, i * step, (j + 1) * step, (i + 1) * step))
                f = tmp / f"o{i}_{j}.png"
                v.save(f)
                files.append(str(f))
        X = pd.concat([row.iloc[[0]]] * len(files), ignore_index=True)
        X[col] = files
        if classes is not None and hasattr(est, "predict_proba"):
            P = np.asarray(est.predict_proba(X))
            c = int(P[0].argmax())
            s, label = P[:, c], classes[c]
        else:
            s, label = np.asarray(est.predict(X), dtype=float), None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    heat = (s[0] - s[1:]).reshape(grid, grid)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.9))
    a1.imshow(a)
    a1.set_title(f"{label or 'prediction'}  ({s[0]:.2f})", fontsize=9)
    a1.axis("off")
    a2.imshow(a)
    hm = a2.imshow(np.kron(heat, np.ones((step, step))), cmap="coolwarm", alpha=0.55, vmin=-max(abs(heat).max(), 1e-6), vmax=max(abs(heat).max(), 1e-6))
    a2.set_title("evidence (red = supports)", fontsize=9)
    a2.axis("off")
    fig.colorbar(hm, ax=a2, fraction=0.046)
    fig.tight_layout()
    return {"heat": np.round(heat, 4).tolist(), "label": label, "score": round(float(s[0]), 4), "figure": fig}


def misclassified_gallery(est, X_te: pd.DataFrame, y_te: np.ndarray, classes: List[str], col: str, root: Path, k: int = 12):
    """The held-out images the model got wrong with the highest confidence. -> (rows, figure or None)."""
    if not hasattr(est, "predict_proba"):
        return [], None
    P = np.asarray(est.predict_proba(X_te))
    pred = P.argmax(axis=1)
    wrong = np.where(pred != np.asarray(y_te))[0]
    wrong = wrong[np.argsort(-P[wrong, pred[wrong]])][:k]
    if not len(wrong):
        return [], None
    paths = _paths(X_te.iloc[wrong], col, root)
    rows = [{"file": str(X_te[col].iloc[i]), "true": classes[int(y_te[i])], "predicted": classes[int(pred[i])], "confidence": round(float(P[i, pred[i]]), 3)} for i in wrong]
    fig = _montage(paths, [f"{r['true'][:7]}→{r['predicted'][:7]} {r['confidence']:.2f}" for r in rows], ncols=6, title="Most confident mistakes (true → predicted)")
    return rows, fig


class ImageIndex:
    """Find visually similar images: standardised texture features, cosine similarity."""

    def __init__(self, paths: List[Path], backend: str = "texture"):
        X, ok = feature_matrix(paths, backend)
        self.paths, self.ok, self.backend = paths, ok, backend
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    def search(self, path: Path, k: int = 5) -> List[Tuple[int, float]]:
        v = extract(path, self.backend)
        if v is None:
            return []
        z = (v - self.mu) / self.sd
        s = self.Z @ (z / (np.linalg.norm(z) + 1e-9))
        return [(self.ok[int(i)], round(float(s[i]), 3)) for i in np.argsort(-s)[:k]]
