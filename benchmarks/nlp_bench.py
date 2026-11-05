#!/usr/bin/env python3
"""NLP pipeline benchmark: previous candidate set vs the current one, on standard text-classification corpora.

    python benchmarks/nlp_bench.py --dataset ag_news --n-train 20000 --n-test 5000      # needs network + `pip install datasets`
    python benchmarks/nlp_bench.py --dataset imdb
    python benchmarks/nlp_bench.py --dataset 20ng                                         # scikit-learn downloader
    python benchmarks/nlp_bench.py --dataset csv:path/to/data.csv --text-col text --label-col label
    python benchmarks/nlp_bench.py --dataset synthetic                                    # offline PROXY only, not a standard benchmark

Every model is fitted on the training split only and scored on the untouched test split (accuracy, macro-F1, fit seconds, predict seconds per 1k docs).
Results are written to benchmarks/results_<dataset>.json and printed as a markdown table. Add --embed to include sentence-embedding models if installed.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from nexusml import nlp


def load(name: str, n_train: int, n_test: int, text_col: str, label_col: str, seed: int):
    rs = np.random.RandomState(seed)
    if name == "synthetic":
        return synthetic(n_train, n_test, seed)
    if name.startswith("csv:"):
        df = pd.read_csv(name[4:]).dropna(subset=[text_col, label_col])
        idx = rs.permutation(len(df))
        k = int(len(df) * 0.8)
        tr, te = df.iloc[idx[:k]], df.iloc[idx[k:]]
    elif name in ("ag_news", "imdb"):
        from datasets import load_dataset                         # pip install datasets
        d = load_dataset(name)
        tr, te = d["train"].to_pandas(), d["test"].to_pandas()
        text_col, label_col = "text", "label"
    elif name == "20ng":
        from sklearn.datasets import fetch_20newsgroups
        a, b = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes")), fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
        tr, te = pd.DataFrame({"text": a.data, "label": a.target}), pd.DataFrame({"text": b.data, "label": b.target})
        text_col, label_col = "text", "label"
    else:
        raise SystemExit(f"unknown dataset {name}")
    tr, te = tr.sample(min(n_train, len(tr)), random_state=seed), te.sample(min(n_test, len(te)), random_state=seed)
    return tr[text_col].astype(str).tolist(), tr[label_col].to_numpy(), te[text_col].astype(str).tolist(), te[label_col].to_numpy()


def synthetic(n_train: int, n_test: int, seed: int):
    """Offline proxy: 4 topics with overlapping vocabulary, shared filler words, typos and mislabelled noise. Exists so the script runs anywhere."""
    rs = np.random.RandomState(seed)
    topics = {t: [f"{t}{i}" for i in range(80)] for t in ("sport", "tech", "money", "health")}
    shared = [f"common{i}" for i in range(300)]
    mix = list(topics)

    def doc(label):
        own, other = topics[mix[label]], topics[mix[rs.randint(4)]]
        w = list(rs.choice(own, rs.randint(4, 9))) + list(rs.choice(other, rs.randint(2, 6))) + list(rs.choice(shared, rs.randint(15, 30)))
        rs.shuffle(w)
        w = [x[:-1] + rs.choice(list("abcxyz")) if rs.rand() < 0.08 else x for x in w]
        return " ".join(w)
    n = n_train + n_test
    y = rs.randint(0, 4, n)
    flip = rs.rand(n) < 0.04
    y_obs = np.where(flip, rs.randint(0, 4, n), y)
    docs = [doc(l) for l in y]
    return docs[:n_train], y_obs[:n_train], docs[n_train:], y_obs[n_train:]


def candidates(which: str, embed: bool):
    from nexusml.modeling import _unspec
    C = lambda rep, est: Pipeline([("t", nlp.text_transformer(rep)), ("m", est)])
    old = {"LogReg · word TF-IDF": C("word", LogisticRegression(max_iter=3000, C=4.0)),
           "LogReg · char TF-IDF": C("char", LogisticRegression(max_iter=3000, C=4.0)),
           "LinearSVC · word TF-IDF": C("word", nlp.CalibratedLinearSVC(C=0.6, random_state=42)),
           "ComplementNB · word TF-IDF": C("word", ComplementNB(alpha=0.3))}
    new = {"LogReg · word+char TF-IDF": C("wordchar", LogisticRegression(max_iter=3000, C=4.0)),
           "NB-SVM · word TF-IDF": C("word", nlp.NBLogisticRegression(C=4.0)),
           "LinearSVC · word+char TF-IDF": C("wordchar", nlp.CalibratedLinearSVC(C=0.6, random_state=42))}
    if embed and nlp.embeddings_available():
        new["LogReg · embeddings"] = C("embed", LogisticRegression(max_iter=3000, C=4.0))
    return {"previous": old, "added": new}[which] if which in ("previous", "added") else {**old, **new}


def run(args):
    Xtr, ytr, Xte, yte = load(args.dataset, args.n_train, args.n_test, args.text_col, args.label_col, args.seed)
    print(f"dataset={args.dataset} train={len(Xtr)} test={len(Xte)} classes={len(set(ytr))}", flush=True)
    rows = []
    for group in ("previous", "added"):
        for name, pipe in candidates(group, args.embed).items():
            t0 = time.time()
            pipe.fit(pd.Series(Xtr), ytr)
            fit_s = time.time() - t0
            t0 = time.time()
            pred = pipe.predict(pd.Series(Xte))
            pred_s = (time.time() - t0) / max(len(Xte), 1) * 1000
            rows.append({"set": group, "model": name, "accuracy": round(accuracy_score(yte, pred), 4), "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
                         "fit_s": round(fit_s, 1), "predict_s_per_1k": round(pred_s, 2)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).parent / f"results_{args.dataset.split(':')[0]}.json"
    out.write_text(json.dumps({"dataset": args.dataset, "n_train": len(Xtr), "n_test": len(Xte), "rows": rows}, indent=2))
    best_prev, best_new = df[df.set == "previous"].macro_f1.max(), df[df.set == "added"].macro_f1.max()
    print("\n| set | model | accuracy | macro-F1 | fit s | predict s/1k docs |\n|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['set']} | {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['fit_s']} | {r['predict_s_per_1k']} |")
    print(f"\nbest previous macro-F1 {best_prev:.4f}; best added {best_new:.4f}; difference {best_new - best_prev:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed", action="store_true")
    run(ap.parse_args())
