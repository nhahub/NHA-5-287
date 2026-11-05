"""Deterministic analyses for image and document collections. Everything is computed by code; results become evidence items (prefix X).

images   : composition, resolution, quality flags (dark / bright / flat / blurry), exact + near duplicates (and duplicates with CONFLICTING
           labels), outliers, and whether classes differ so much in simple colour/brightness that a model could "cheat" (shortcut risk).
documents: composition, keywords, themes (clustering of TF-IDF) and whether they match the folder labels, distinguishing terms per class,
           near-duplicates, readability, extractable-text coverage, script detection (Arabic etc.).
"""
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from . import config, featurizers

Item = Tuple[str, str, Dict[str, Any]]
Fig = Tuple[str, str, str, Any]          # (name, title, caption, matplotlib figure)


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _names(v, k=4) -> str:
    v = list(v)
    return ", ".join(str(x) for x in v[:k]) + (f" (+{len(v) - k} more)" if len(v) > k else "")


# images
def image_insights(df: pd.DataFrame, col: str, root: Path, label: Optional[str]) -> Tuple[List[Item], List[Fig]]:
    import matplotlib.pyplot as plt
    from PIL import Image
    items: List[Item] = []
    figs: List[Fig] = []
    n = len(df)
    comp = f"{n:,} images"
    if label and label in df:
        vc = df[label].value_counts()
        comp += f" in {len(vc)} classes ({_names(f'{k}: {v}' for k, v in vc.items())})"
        if vc.max() / max(vc.min(), 1) >= 3:
            comp += f". Classes are imbalanced ({vc.max() / vc.min():.1f}x between largest and smallest)."
    items.append(("media_composition", comp, {"n": n}))
    if {"width", "height"} <= set(df.columns):
        size = (df.width.astype(str) + "x" + df.height.astype(str))
        mode_share = size.value_counts(normalize=True).iloc[0]
        items.append(("media_resolution", f"Resolution: median {int(df.width.median())}x{int(df.height.median())} px (min {int(df.width.min())}x{int(df.height.min())}, "
                                          f"max {int(df.width.max())}x{int(df.height.max())}); {_pct(mode_share)} of images share the most common size"
                                          + ("; sizes are mixed, so images are resized before feature extraction." if mode_share < 0.8 else "."),
                      {"mode_share": float(mode_share)}))
    flags = {}
    if "brightness" in df:
        flags["too dark (brightness < 40/255)"] = df.index[df.brightness < 40]
        flags["too bright (brightness > 215/255)"] = df.index[df.brightness > 215]
    if "contrast" in df:
        flags["flat / low contrast (contrast < 15)"] = df.index[df.contrast < 15]
    if "sharpness" in df:
        flags["blurry (sharpness < 25)"] = df.index[df.sharpness < 25]
    bad = {k: v for k, v in flags.items() if len(v)}
    if bad:
        txt = "; ".join(f"{len(v)} {k}" for k, v in bad.items())
        items.append(("media_quality", f"Image quality flags: {txt}. Examples: {_names(df.loc[next(iter(bad.values())), col])}.", {k: len(v) for k, v in bad.items()}))
    else:
        items.append(("media_quality", "No image is flagged as too dark, too bright, flat or blurry.", {}))
    # duplicates
    paths = [root / p for p in df[col]]
    md5: Dict[str, List[int]] = {}
    ah: List[Optional[int]] = []
    for i, p in enumerate(paths):
        try:
            md5.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append(i)
        except OSError:
            pass
        ah.append(featurizers._ahash(p))
    exact = [v for v in md5.values() if len(v) > 1]
    near, seen = [], set()
    first_of_bytes = {v[0] for v in md5.values()}                   # one representative per byte-identical group, so exact copies are not counted twice
    valid = [(i, h) for i, h in enumerate(ah) if h is not None and i in first_of_bytes]
    for a in range(len(valid)):
        i, hi = valid[a]
        if i in seen:
            continue
        grp = [i] + [j for j, hj in valid[a + 1:] if bin(hi ^ hj).count("1") <= 3 and j not in seen and featurizers.confirm_near_duplicate(paths[i], paths[j])]
        if len(grp) > 1:
            near.append(grp)
            seen.update(grp)
    if exact or near:
        conflict = 0
        if label and label in df:
            conflict = sum(1 for g in exact + near if df.iloc[g][label].nunique() > 1)
        ex = df.iloc[(exact or near)[0]][col].tolist()
        items.append(("media_duplicates", f"Duplicates: {sum(len(g) - 1 for g in exact)} byte-identical copy(ies) and {len(near)} group(s) of visually near-identical but different files "
                                          f"(e.g. {_names(ex, 3)})." + (f" {conflict} group(s) carry CONFLICTING labels, which is a labelling error." if conflict else "")
                                          + " Duplicates across train and test inflate held-out scores.", {"exact": len(exact), "near": len(near), "conflicting": conflict}))
    # outliers
    feat = [c for c in ("brightness", "contrast", "sharpness", "mean_red", "mean_green", "mean_blue", "saturation", "aspect_ratio", "file_kb", "width", "height") if c in df]
    if feat and n >= 20:
        z = (df[feat] - df[feat].median()) / (stats.median_abs_deviation(df[feat], scale="normal", axis=0) + 1e-9)
        score = z.abs().max(axis=1)
        top = score[score >= 6].sort_values(ascending=False).head(5)
        if len(top):
            why = z.loc[top.index[0]].abs().idxmax()
            items.append(("media_outliers", f"{len(top)} unusual image(s) stand out (most extreme on {why}): {_names(df.loc[top.index, col])}.", {"index": [int(i) for i in top.index]}))
    # class shortcut risk
    if label and label in df and df[label].nunique() >= 2 and feat:
        best = None
        for c in feat:
            groups = [g[c].dropna().values for _, g in df.groupby(label) if len(g) >= 3]
            if len(groups) >= 2 and all(np.std(g) > 0 for g in groups):
                f, p = stats.f_oneway(*groups)
                allv = np.concatenate(groups)
                ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
                eta2 = ss_b / max(((allv - allv.mean()) ** 2).sum(), 1e-9)
                if best is None or eta2 > best[1]:
                    best = (c, eta2, p)
        if best:
            c, eta2, p = best
            risk = eta2 >= 0.5
            items.append(("media_shortcut", f"Classes differ most in '{c}' (eta-squared {eta2:.2f}, p={p:.2g}). "
                          + ("This is a strong, simple difference: a model may be learning '" + c + "' instead of the real content (shortcut risk); check how the images were collected."
                             if risk else "No single simple property separates the classes, so a model must use richer image content."), {"feature": c, "eta2": float(eta2)}))
    # figures
    with featurizers_lock():
        k = min(n, 12)
        if label and label in df and df[label].nunique() <= 6:
            sel = pd.concat([g.head(max(1, 12 // df[label].nunique())) for _, g in df.groupby(label)])
        else:
            sel = df.sample(k, random_state=config.RANDOM_STATE)
        sel = sel.head(12)
        ncols = 4
        nrows = int(np.ceil(len(sel) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, 2.2 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, (_, r) in zip(axes, sel.iterrows()):
            try:
                with Image.open(root / r[col]) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(str(r[label]) if label and label in df else Path(str(r[col])).name[:18], fontsize=8)
            except Exception:
                pass
        fig.suptitle("Sample images" + (" by class" if label and label in df else ""), fontsize=10)
        figs.append(("contact_sheet", "Sample images", "A random sample (one row per class when labelled) so you can sanity-check what the model sees.", fig))
        if feat:
            fig2, axs = plt.subplots(1, 3, figsize=(9, 2.8))
            for ax, c in zip(axs, [x for x in ("brightness", "sharpness", "contrast") if x in df][:3]):
                if label and label in df and df[label].nunique() <= 6:
                    for lv, g in df.groupby(label):
                        ax.hist(g[c], bins=15, alpha=0.55, label=str(lv))
                    ax.legend(fontsize=6)
                else:
                    ax.hist(df[c], bins=20, color="#4C78A8")
                ax.set_title(c)
            fig2.tight_layout()
            figs.append(("quality_hist", "Image quality distributions", "Brightness, sharpness and contrast across the collection (by class when labelled).", fig2))
    return items, figs


def featurizers_lock():
    from . import viz
    return viz.MPL_LOCK


# documents
_ARABIC = re.compile(r"[\u0600-\u06FF]")


def script_share(texts: pd.Series) -> Dict[str, float]:
    s = " ".join(texts.fillna("").astype(str).head(200))
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return {}
    ar = sum(bool(_ARABIC.match(c)) for c in letters) / len(letters)
    return {"arabic": round(ar, 3), "latin": round(sum(c.isascii() for c in letters) / len(letters), 3)}


def _themes(X, terms, k_max=6) -> Tuple[Optional[np.ndarray], int, List[List[str]], float]:
    n = X.shape[0]
    if n < 12:
        return None, 0, [], 0.0
    Z = normalize(TruncatedSVD(n_components=min(30, X.shape[1] - 1, n - 1), random_state=config.RANDOM_STATE).fit_transform(X))
    best = None
    for k in range(2, min(k_max, n // 4) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(Z)
        sil = silhouette_score(Z, km.labels_)
        if best is None or sil > best[0]:
            best = (sil, k, km.labels_)
    if best is None:
        return None, 0, [], 0.0
    sil, k, lab = best
    tops = []
    for c in range(k):
        m = np.asarray(X[lab == c].mean(axis=0)).ravel()
        tops.append([terms[i] for i in m.argsort()[::-1][:6]])
    return lab, k, tops, float(sil)


def document_insights(df: pd.DataFrame, col: str, label: Optional[str], name_col: Optional[str] = None,
                      topic_namer=None) -> Tuple[List[Item], List[Fig], Dict[str, Any]]:
    import matplotlib.pyplot as plt
    items: List[Item] = []
    figs: List[Fig] = []
    extra: Dict[str, Any] = {}
    txt = df[col].fillna("").astype(str)
    n = len(df)
    words = txt.str.split().map(len)
    comp = f"{n:,} documents, {int(words.sum()):,} words in total (median {int(words.median())} words per document, longest {int(words.max())})"
    if label and label in df:
        vc = df[label].value_counts()
        comp += f"; classes: {_names(f'{k}: {v}' for k, v in vc.items())}"
    items.append(("media_composition", comp + ".", {"n": n, "words": int(words.sum())}))
    empty = df.index[words < 20]
    if len(empty):
        nm = df.loc[empty, name_col] if name_col and name_col in df else empty
        items.append(("media_text_coverage", f"{len(empty)} document(s) have almost no extractable text (< 20 words), e.g. {_names(nm, 3)}. "
                                             "Scanned / image-only PDFs have no text layer and OCR is not included, so they carry no signal here.", {"empty": int(len(empty))}))
    sc = script_share(txt)
    if sc.get("arabic", 0) > 0.3:
        items.append(("media_language", f"Arabic script detected ({_pct(sc['arabic'])} of letters). Word features use the generic tokenizer without an Arabic stop-word list, "
                                        "and English readability scores are not computed.", sc))
    norm = txt.map(lambda s: re.sub(r"\W+", " ", s.lower()).strip())
    grp = norm[words >= 5].groupby(norm[words >= 5]).groups
    dup_groups = [list(ix) for ix in grp.values() if len(ix) > 1]
    if dup_groups:
        nmcol = df[name_col] if name_col and name_col in df else pd.Series(df.index, index=df.index)
        conflict = sum(1 for g in dup_groups if label and label in df and df.loc[g, label].nunique() > 1)
        items.append(("media_exact_duplicates", f"{sum(len(g) - 1 for g in dup_groups)} document(s) have identical text to another document (e.g. {_names(nmcol.loc[dup_groups[0]], 3)})."
                      + (f" {conflict} group(s) carry different labels, which is a labelling error." if conflict else "")
                      + " Duplicates across train and test inflate held-out scores.", {"groups": len(dup_groups), "conflicting": int(conflict)}))
    use = df[words >= 5]
    if len(use) >= 12:
        english = sc.get("latin", 1) > 0.8
        vec = TfidfVectorizer(max_features=2000, min_df=2 if len(use) > 30 else 1, max_df=0.9, stop_words="english" if english else None, sublinear_tf=True)
        try:
            X = vec.fit_transform(use[col].astype(str))
        except ValueError:
            X = None
        if X is not None and X.shape[1] >= 3:
            terms = vec.get_feature_names_out()
            kw = np.asarray(X.mean(axis=0)).ravel().argsort()[::-1][:12]
            items.append(("media_keywords", "Most characteristic terms across all documents: " + ", ".join(terms[kw]) + ".", {"terms": [str(t) for t in terms[kw]]}))
            lab, k, tops, sil = _themes(X, terms)
            if lab is not None:
                names = topic_namer(tops) if topic_namer else None
                themes = []
                for c in range(k):
                    nm = (names or {}).get(c) or " / ".join(tops[c][:3])
                    themes.append({"theme": int(c), "name": nm, "size": int((lab == c).sum()), "terms": tops[c]})
                extra["themes"] = themes
                agree = ""
                if label and label in use:
                    ari = adjusted_rand_score(use[label].astype(str), lab)
                    agree = (f" The themes match the folder labels closely (ARI {ari:.2f})." if ari >= 0.5 else
                             f" The themes only partly follow the folder labels (ARI {ari:.2f}), so some folders mix topics or some topics span folders.")
                    extra["theme_label_ari"] = round(float(ari), 3)
                items.append(("media_themes", f"{k} natural themes (silhouette {sil:.2f}): " + "; ".join(f"[{t['name']}] {t['size']} docs ({', '.join(t['terms'][:4])})" for t in themes) + "." + agree,
                              {"themes": themes, "silhouette": sil}))
                with featurizers_lock():
                    fig, axs = plt.subplots(1, k, figsize=(2.6 * k + 1, 2.6), squeeze=False)
                    for c, ax in enumerate(axs[0]):
                        m = np.asarray(X[lab == c].mean(axis=0)).ravel()
                        idx = m.argsort()[::-1][:6][::-1]
                        ax.barh([terms[i] for i in idx], m[idx], color="#4C78A8")
                        ax.set_title(f"{themes[c]['name'][:22]} (n={themes[c]['size']})", fontsize=8)
                        ax.tick_params(labelsize=7)
                    fig.tight_layout()
                    figs.append(("themes", "Document themes", "Top terms of each automatically found theme.", fig))
            if label and label in use and use[label].nunique() >= 2:
                from sklearn.feature_selection import chi2
                ch, _ = chi2(X, use[label].astype(str))
                per = []
                for lv in sorted(use[label].astype(str).unique()):
                    mask = (use[label].astype(str) == lv).values
                    diff = np.asarray(X[mask].mean(axis=0)).ravel() - np.asarray(X[~mask].mean(axis=0)).ravel()
                    per.append(f"{lv}: " + ", ".join(terms[diff.argsort()[::-1][:4]]))
                items.append(("media_class_terms", "Terms that distinguish each class: " + "; ".join(per) + ".", {}))
            # near duplicates (identical texts were reported above)
            sims = (X @ X.T).toarray() if X.shape[0] <= 1500 else None
            if sims is not None:
                np.fill_diagonal(sims, 0)
                ii, jj = np.where(np.triu(sims) >= 0.95)
                nn_ = norm.loc[use.index].values
                keep = [(i, j) for i, j in zip(ii, jj) if nn_[i] != nn_[j]]
                if keep:
                    nm = use[name_col].values if name_col and name_col in use else use.index.values
                    pairs = [f"{nm[i]} ~ {nm[j]}" for i, j in keep[:3]]
                    conflict = int(sum(use[label].values[i] != use[label].values[j] for i, j in keep)) if label and label in use else 0
                    items.append(("media_duplicates", f"{len(keep)} near-duplicate document pair(s) (cosine >= 0.95, not identical), e.g. {'; '.join(pairs)}."
                                  + (f" {conflict} pair(s) carry different labels, which is a labelling error." if conflict else "")
                                  + " Duplicates across train and test inflate held-out scores.", {"pairs": len(keep), "conflicting": conflict}))
    if "readability" in df and df.readability.notna().sum() >= 5:
        r = df.readability.dropna()
        lvl = "very easy" if r.median() > 80 else "plain English" if r.median() > 60 else "fairly difficult" if r.median() > 30 else "very difficult"
        items.append(("media_readability", f"Readability (Flesch): median {r.median():.0f} ({lvl}); range {r.min():.0f} to {r.max():.0f}.", {"median": float(r.median())}))
    ents = {c: int(df[c].sum()) for c in ("n_urls", "n_emails", "n_numbers") if c in df}
    if ents:
        items.append(("media_entities", f"Mentions found by pattern matching: {ents.get('n_numbers', 0):,} numbers, {ents.get('n_urls', 0):,} links, {ents.get('n_emails', 0):,} e-mail addresses.", ents))
    with featurizers_lock():
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.hist(words.clip(upper=words.quantile(0.98) + 1), bins=25, color="#4C78A8")
        ax.set_xlabel("words per document")
        ax.set_ylabel("documents")
        figs.append(("length_hist", "Document length", "How long the documents are (top 2% clipped).", fig))
    return items, figs, extra
