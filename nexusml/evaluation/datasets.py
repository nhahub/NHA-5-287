"""Datasets with KNOWN ground truth. Each generator writes CSV file(s) into `out_dir` and returns a case dict:
{name, path | paths, frame (how the stub LLM frames it), truth (what the system should find), kind}."""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _kpi_rate(col, pos):
    return [{"name": f"{col} rate", "agg": "rate", "column": col, "positive": pos, "format": "percent", "higher_is_better": False, "group_by": ["region"]},
            {"name": "Records", "agg": "count"}]


def planted_classification(out_dir: Path, seed: int = 0, n: int = 1800) -> Dict[str, Any]:
    r = np.random.RandomState(seed)
    tenure, income, age = r.normal(0, 1, n), r.normal(0, 1, n), r.normal(0, 1, n)
    overtime = r.choice(["Yes", "No"], n, p=[0.3, 0.7])
    logit = -1.8 + 1.5 * (overtime == "Yes") - 0.9 * tenure + 0.6 * income + 1.2 * ((overtime == "Yes") & (tenure < -0.3))
    y = (r.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"employee_id": np.arange(n), "tenure": tenure, "income": income, "age": age, "overtime": overtime,
                       "region": r.choice(list("NSEW"), n), "noise_a": r.normal(size=n), "noise_b": r.normal(size=n), "noise_c": r.normal(size=n),
                       "attrition": np.where(y == 1, "Yes", "No")})
    p = out_dir / "planted_classification.csv"
    df.to_csv(p, index=False)
    return {"name": "planted_classification", "kind": "classification", "path": str(p),
            "frame": {"analysis_type": "churn", "problem_type": "classification", "target": "attrition", "date_column": None,
                      "segment_columns": ["overtime", "region"], "kpis": _kpi_rate("attrition", "Yes")},
            "truth": {"target": "attrition", "problem_type": "classification", "drivers": ["overtime", "tenure", "income"],
                      "interaction": ["overtime", "tenure"], "noise": ["age", "region", "noise_a", "noise_b", "noise_c"]}}


def planted_regression(out_dir: Path, seed: int = 1, n: int = 1500) -> Dict[str, Any]:
    r = np.random.RandomState(seed)
    area, rooms, dist = r.uniform(40, 200, n), r.randint(1, 7, n), r.uniform(0, 30, n)
    price = 1500 * area + 8000 * rooms - 3000 * dist + r.normal(0, 15000, n)
    df = pd.DataFrame({"area": area, "rooms": rooms, "distance": dist, "noise_a": r.normal(size=n), "noise_b": r.uniform(size=n), "price": price})
    p = out_dir / "planted_regression.csv"
    df.to_csv(p, index=False)
    return {"name": "planted_regression", "kind": "regression", "path": str(p),
            "frame": {"analysis_type": "profitability", "problem_type": "regression", "target": "price", "date_column": None, "segment_columns": [],
                      "kpis": [{"name": "Avg price", "agg": "mean", "column": "price", "format": "currency"}]},
            "truth": {"target": "price", "problem_type": "regression", "drivers": ["area", "rooms", "distance"], "noise": ["noise_a", "noise_b"]}}


def planted_timeseries(out_dir: Path, seed: int = 2, n: int = 420) -> Dict[str, Any]:
    r = np.random.RandomState(seed)
    d = pd.date_range("2023-01-01", periods=n, freq="D")
    base = 200 + 0.3 * np.arange(n) + 25 * np.sin(2 * np.pi * np.arange(n) / 7) + r.normal(0, 6, n)
    base[300:] += 60                                            # level shift at day 300
    df = pd.DataFrame({"date": d, "promo": r.choice(["yes", "no"], n, p=[.2, .8]), "sales": base})
    df.loc[df.promo == "yes", "sales"] += 20
    p = out_dir / "planted_timeseries.csv"
    df.to_csv(p, index=False)
    return {"name": "planted_timeseries", "kind": "time_series", "path": str(p),
            "frame": {"analysis_type": "sales_performance", "problem_type": "time_series", "target": "sales", "date_column": "date", "segment_columns": ["promo"],
                      "kpis": [{"name": "Total sales", "agg": "sum", "column": "sales", "format": "currency", "group_by": ["promo"]}]},
            "truth": {"target": "sales", "problem_type": "time_series", "level_shift_index": 300}}


def planted_clusters(out_dir: Path, seed: int = 3, n: int = 600) -> Dict[str, Any]:
    r = np.random.RandomState(seed)
    lab = r.randint(0, 3, n)
    centers = np.array([[0, 0, 0], [6, 6, 0], [0, 6, 8]])
    X = centers[lab] + r.normal(0, 1.0, (n, 3))
    df = pd.DataFrame({"spend": X[:, 0], "visits": X[:, 1], "tenure": X[:, 2], "region": r.choice(list("NSEW"), n)})
    p = out_dir / "planted_clusters.csv"
    df.to_csv(p, index=False)
    return {"name": "planted_clusters", "kind": "clustering", "path": str(p),
            "frame": {"analysis_type": "customer_analysis", "problem_type": "clustering", "target": None, "date_column": None, "segment_columns": ["region"],
                      "kpis": [{"name": "Avg spend", "agg": "mean", "column": "spend"}]},
            "truth": {"problem_type": "clustering", "labels": lab.tolist(), "k": 3}}


def leakage_trap(out_dir: Path, seed: int = 4, n: int = 800) -> Dict[str, Any]:
    r = np.random.RandomState(seed)
    x = r.normal(size=n)
    y = (x + r.normal(0, 1.2, n) > 0).astype(int)
    df = pd.DataFrame({"x": x, "z": r.normal(size=n), "outcome_code": y * 1.0 + r.normal(0, 0.001, n), "label": np.where(y == 1, "Yes", "No")})
    p = out_dir / "leakage_trap.csv"
    df.to_csv(p, index=False)
    return {"name": "leakage_trap", "kind": "edge", "path": str(p),
            "frame": {"analysis_type": "churn", "problem_type": "classification", "target": "label", "date_column": None, "segment_columns": [],
                      "kpis": [{"name": "Records", "agg": "count"}]},
            "truth": {"target": "label", "problem_type": "classification", "must_drop": ["outcome_code"]}}


def constant_and_missing(out_dir: Path, seed: int = 5, n: int = 500) -> Dict[str, Any]:
    r = np.random.RandomState(seed)
    x = r.normal(size=n)
    y = (x + r.normal(0, 1, n) > 0.3).astype(int)
    df = pd.DataFrame({"x": x, "const": 1, "mostly_missing": np.where(r.uniform(size=n) < 0.8, np.nan, r.normal(size=n)),
                       "cat": r.choice(list("abc"), n), "label": np.where(y == 1, "Yes", "No")})
    p = out_dir / "constant_and_missing.csv"
    df.to_csv(p, index=False)
    return {"name": "constant_and_missing", "kind": "edge", "path": str(p),
            "frame": {"analysis_type": "churn", "problem_type": "classification", "target": "label", "date_column": None, "segment_columns": ["cat"],
                      "kpis": [{"name": "Records", "agg": "count"}]},
            "truth": {"target": "label", "problem_type": "classification", "must_drop": ["const", "mostly_missing"]}}


def multitable_churn(out_dir: Path, seed: int = 6, n: int = 700) -> Dict[str, Any]:
    r = np.random.RandomState(seed)
    n_orders = r.poisson(6, n) + 1
    churn = (r.uniform(size=n) < 1 / (1 + np.exp(-(1.2 - 0.45 * n_orders)))).astype(int)
    cust = pd.DataFrame({"customer_id": np.arange(n), "region": r.choice(list("NSEW"), n), "churn": np.where(churn == 1, "Yes", "No")})
    cid = np.repeat(np.arange(n), n_orders)
    orders = pd.DataFrame({"order_id": np.arange(len(cid)), "customer_id": cid, "product_id": r.randint(0, 20, len(cid)), "amount": r.gamma(2, 40, len(cid))})
    prod = pd.DataFrame({"product_id": np.arange(20), "category": r.choice(list("ABC"), 20), "price": r.uniform(5, 90, 20)})
    paths = []
    (out_dir / "mt").mkdir(exist_ok=True)
    for nm, d in (("customers", cust), ("orders", orders), ("products", prod)):
        p = out_dir / "mt" / f"{nm}.csv"
        d.to_csv(p, index=False)
        paths.append(str(p))
    return {"name": "multitable_churn", "kind": "multitable", "paths": paths,
            "frame": {"analysis_type": "churn", "problem_type": "classification", "target": "churn", "date_column": None, "segment_columns": ["region"],
                      "kpis": _kpi_rate("churn", "Yes")},
            "truth": {"target": "churn", "problem_type": "classification", "drivers": ["orders__"], "base": "customers", "rows": n}}


_POS = ["excellent", "great", "love", "fantastic", "perfect", "recommend", "happy", "amazing"]
_NEG = ["terrible", "broken", "refund", "awful", "disappointed", "waste", "poor", "slow"]
_NEU = ["delivery", "package", "item", "order", "arrived", "box", "color", "size", "price", "store", "week", "used"]


def text_reviews(out_dir: Path, seed: int = 7, n: int = 700) -> Dict[str, Any]:
    r = np.random.RandomState(seed)
    y = r.randint(0, 2, n)
    texts = []
    for lab in y:
        words = list(r.choice(_POS if lab else _NEG, r.randint(2, 4))) + list(r.choice(_NEU, r.randint(4, 8)))
        r.shuffle(words)
        texts.append(" ".join(words))
    df = pd.DataFrame({"review": texts, "stars_given": r.randint(1, 6, n), "verified": r.choice(["yes", "no"], n),
                       "label": np.where(y == 1, "positive", "negative")})
    p = out_dir / "text_reviews.csv"
    df.to_csv(p, index=False)
    return {"name": "text_reviews", "kind": "text", "path": str(p),
            "frame": {"analysis_type": "customer_analysis", "problem_type": "classification", "target": "label", "date_column": None, "segment_columns": ["verified"],
                      "kpis": [{"name": "Records", "agg": "count"}]},
            "truth": {"target": "label", "problem_type": "classification", "drivers": ["review"]}}


def _make_images(folder: Path, labels: np.ndarray, r) -> List[str]:
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i, lab in enumerate(labels):
        base = np.array([200, 60, 60]) if lab else np.array([60, 60, 200])
        arr = np.clip(base + r.normal(0, 25, (24, 24, 3)), 0, 255).astype("uint8")
        p = folder / f"img_{i}.png"
        Image.fromarray(arr).save(p)
        out.append(f"images/img_{i}.png")
    return out


def image_dataset(out_dir: Path, seed: int = 8, n: int = 300) -> Dict[str, Any]:
    r = np.random.RandomState(seed)
    y = r.randint(0, 2, n)
    files = _make_images(out_dir / "images", y, r)
    df = pd.DataFrame({"photo": files, "weight": r.normal(size=n), "kind": np.where(y == 1, "warm", "cool")})
    p = out_dir / "image_dataset.csv"
    df.to_csv(p, index=False)
    return {"name": "image_dataset", "kind": "image", "path": str(p), "image_root": str(out_dir),
            "frame": {"analysis_type": "customer_analysis", "problem_type": "classification", "target": "kind", "date_column": None, "segment_columns": [],
                      "kpis": [{"name": "Records", "agg": "count"}]},
            "truth": {"target": "kind", "problem_type": "classification", "drivers": ["photo"]}}


def multimodal(out_dir: Path, seed: int = 9, n: int = 400) -> Dict[str, Any]:
    r = np.random.RandomState(seed)
    y = r.randint(0, 2, n)
    y_img = np.where(r.uniform(size=n) < 0.8, y, 1 - y)            # each modality is only ~80% reliable, so BOTH are needed
    y_txt = np.where(r.uniform(size=n) < 0.8, y, 1 - y)
    files = _make_images(out_dir / "images", y_img, r)
    texts = [" ".join(list(r.choice(_POS if lab else _NEG, 2)) + list(r.choice(_NEU, 5))) for lab in y_txt]
    df = pd.DataFrame({"photo": files, "comment": texts, "age_band": r.choice(["young", "mid", "old"], n), "score": r.normal(size=n),
                       "outcome": np.where(y == 1, "good", "bad")})
    p = out_dir / "multimodal.csv"
    df.to_csv(p, index=False)
    return {"name": "multimodal", "kind": "multimodal", "path": str(p), "image_root": str(out_dir),
            "frame": {"analysis_type": "customer_analysis", "problem_type": "classification", "target": "outcome", "date_column": None, "segment_columns": ["age_band"],
                      "kpis": [{"name": "Records", "agg": "count"}]},
            "truth": {"target": "outcome", "problem_type": "classification", "drivers": ["photo", "comment"]}}


# uploaded media (no CSV): folders of images / documents
def _png(path: Path, rgb, r, size=40):
    from PIL import Image
    arr = np.clip(np.array(rgb) + r.normal(0, 22, (size, size, 3)), 0, 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def image_folders(out_dir: Path, seed: int = 11, per: int = 20) -> Dict[str, Any]:
    """Folder-per-class images (like an uploaded ZIP). Plants: 3 exact duplicates and 1 unreadable file."""
    r = np.random.RandomState(seed)
    root = out_dir / "photos"
    for cls, base in (("sunsets", (230, 120, 40)), ("forests", (40, 140, 60)), ("oceans", (40, 90, 200))):
        for i in range(per):
            _png(root / cls / f"{cls}_{i}.png", base, r)
    import shutil
    for i in range(3):
        shutil.copy(root / "forests" / f"forests_{i}.png", root / "forests" / f"copy_of_{i}.png")
    (root / "oceans" / "corrupt.png").write_bytes(b"this is not an image")
    return {"name": "image_folders", "kind": "media", "folder": str(root),
            "frame": {"analysis_type": "general_exploration", "problem_type": "classification", "target": "label", "date_column": None,
                      "segment_columns": [], "kpis": [{"name": "Records", "agg": "count"}]},
            "truth": {"target": "label", "problem_type": "classification", "classes": 3, "n": 3 * per + 3, "unreadable": 1, "exact_duplicates": 3}}


def unlabeled_images(out_dir: Path, seed: int = 12, per: int = 25) -> Dict[str, Any]:
    r = np.random.RandomState(seed)
    root = out_dir / "unlabeled"
    lab = []
    for k, base in enumerate([(220, 40, 40), (40, 40, 220), (40, 200, 60)]):
        for i in range(per):
            _png(root / f"img_{k}_{i}.png", base, r)
            lab.append(k)
    return {"name": "unlabeled_images", "kind": "media", "folder": str(root),
            "frame": {"analysis_type": "general_exploration", "problem_type": "clustering", "target": None, "date_column": None,
                      "segment_columns": [], "kpis": [{"name": "Records", "agg": "count"}]},
            "truth": {"problem_type": "clustering", "k": 3, "n": 3 * per, "labels_by_name": {f"img_{k}_{i}.png": k for k in range(3) for i in range(per)}}}


_TOPICS = {
    "finance": "invoice payment budget revenue quarterly profit expenses audit tax balance sheet accounting".split(),
    "health": "patient clinic diagnosis treatment symptoms medication hospital doctor nurse therapy recovery".split(),
    "travel": "flight hotel itinerary passport luggage tourist beach museum booking airport journey".split(),
}
_FILL = "the and for with from this that were have report about during after between into over under".split()


def _doc_text(topic, r, n=90):
    words = list(r.choice(_TOPICS[topic], n // 2)) + list(r.choice(_FILL, n // 2))
    r.shuffle(words)
    return " ".join(words[:n // 5]) + ". " + " ".join(words[n // 5:2 * n // 5]) + ". " + " ".join(words[2 * n // 5:]) + "."


def document_folders(out_dir: Path, seed: int = 13, per: int = 14) -> Dict[str, Any]:
    """Folder-per-topic documents in mixed formats (.txt .docx .pdf .html) + a scanned-style PDF with no text layer."""
    r = np.random.RandomState(seed)
    root = out_dir / "docs"
    from docx import Document
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    for topic in _TOPICS:
        (root / topic).mkdir(parents=True, exist_ok=True)
        for i in range(per):
            t = _doc_text(topic, r)
            kind = i % 4
            if kind == 0:
                (root / topic / f"{topic}_{i}.txt").write_text(t, encoding="utf-8")
            elif kind == 1:
                dd = Document()
                dd.add_paragraph(t)
                dd.save(str(root / topic / f"{topic}_{i}.docx"))
            elif kind == 2:
                c = canvas.Canvas(str(root / topic / f"{topic}_{i}.pdf"), pagesize=A4)
                y = 800
                for line in [t[j:j + 90] for j in range(0, len(t), 90)]:
                    c.drawString(40, y, line)
                    y -= 14
                c.save()
            else:
                (root / topic / f"{topic}_{i}.html").write_text(f"<html><body><script>var x=1;</script><p>{t}</p></body></html>", encoding="utf-8")
    from PIL import Image
    Image.fromarray(np.full((200, 150, 3), 255, "uint8")).save(root / "finance" / "scan.pdf")          # image-only PDF: no text layer
    (root / "health" / "health_dup.txt").write_text((root / "health" / "health_0.txt").read_text(encoding="utf-8"), encoding="utf-8")
    return {"name": "document_folders", "kind": "media", "folder": str(root),
            "frame": {"analysis_type": "general_exploration", "problem_type": "classification", "target": "label", "date_column": None,
                      "segment_columns": [], "kpis": [{"name": "Records", "agg": "count"}]},
            "truth": {"target": "label", "problem_type": "classification", "classes": 3, "n": 3 * per + 2, "empty_docs": 1, "exact_dup_docs": 1}}


GENERATORS = {"planted_classification": planted_classification, "planted_regression": planted_regression, "planted_timeseries": planted_timeseries,
              "planted_clusters": planted_clusters, "leakage_trap": leakage_trap, "constant_and_missing": constant_and_missing,
              "multitable_churn": multitable_churn, "text_reviews": text_reviews, "image_dataset": image_dataset, "multimodal": multimodal, "image_folders": image_folders, "unlabeled_images": unlabeled_images, "document_folders": document_folders}
QUICK = ["planted_classification", "planted_regression", "leakage_trap", "constant_and_missing", "planted_clusters"]
