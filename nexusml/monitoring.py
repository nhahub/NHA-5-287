"""Drift monitoring: compare NEW data with the data a model was trained on (or any reference file).
numeric -> two-sample Kolmogorov-Smirnov + PSI; categorical -> chi-square + PSI. Alerts when PSI >= 0.2 (a common rule of thumb:
<0.1 stable, 0.1-0.2 moderate, >=0.2 major) or the test is significant with a non-trivial shift.

  python -m nexusml.cli monitor reference.csv new.csv [--every 3600] [--out monitoring/]   (or run it from cron / CI)"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def psi(ref: pd.Series, new: pd.Series, bins: int = 10) -> float:
    ref, new = ref.dropna(), new.dropna()
    if len(ref) == 0 or len(new) == 0:
        return float("nan")
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > 10:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref, edges)[0] / len(ref)
        q = np.histogram(new, edges)[0] / len(new)
    else:
        cats = sorted(set(ref.astype(str)) | set(new.astype(str)))
        p = ref.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
        q = new.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def check_drift(reference: pd.DataFrame, new: pd.DataFrame, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
    cols = [c for c in reference.columns if c in new.columns and c not in (exclude or [])]
    rows = []
    for c in cols:
        r, n = reference[c], new[c]
        try:
            if pd.api.types.is_numeric_dtype(r) and r.nunique() > 10:
                ks = stats.ks_2samp(r.dropna(), n.dropna())
                test, p = "KS", float(ks.pvalue)
            else:
                cats = sorted(set(r.dropna().astype(str)) | set(n.dropna().astype(str)))
                tab = np.array([[ (r.astype(str) == k).sum() for k in cats], [(n.astype(str) == k).sum() for k in cats]])
                tab = tab[:, tab.sum(axis=0) > 0]
                p = float(stats.chi2_contingency(tab)[1]) if tab.shape[1] > 1 else 1.0
                test = "chi-square"
        except Exception:
            test, p = "n/a", float("nan")
        v = psi(r, n)
        level = "major" if v >= 0.2 else "moderate" if v >= 0.1 else "stable"
        rows.append({"column": c, "test": test, "p_value": round(p, 6) if p == p else None, "psi": round(v, 4) if v == v else None, "level": level})
    missing = [c for c in reference.columns if c not in new.columns and c not in (exclude or [])]
    new_cols = [c for c in new.columns if c not in reference.columns]
    rows.sort(key=lambda x: -(x["psi"] or 0))
    return {"checked": len(rows), "major": [r["column"] for r in rows if r["level"] == "major"],
            "moderate": [r["column"] for r in rows if r["level"] == "moderate"], "missing_columns": missing, "new_columns": new_cols,
            "alert": bool([r for r in rows if r["level"] == "major"] or missing), "columns": rows,
            "n_reference": int(len(reference)), "n_new": int(len(new)), "ts": time.strftime("%Y-%m-%d %H:%M:%S")}


def to_markdown(rep: Dict[str, Any]) -> str:
    L = [f"# Drift report ({rep['ts']})", "", f"Reference rows: {rep['n_reference']:,}; new rows: {rep['n_new']:,}; columns checked: {rep['checked']}.",
         f"**ALERT**" if rep["alert"] else "No major drift.", ""]
    if rep["missing_columns"]:
        L.append(f"Missing in new data: {rep['missing_columns']}")
    L += ["| column | test | p | PSI | level |", "|---|---|---|---|---|"] + [f"| {r['column']} | {r['test']} | {r['p_value']} | {r['psi']} | {r['level']} |" for r in rep["columns"][:25]]
    return "\n".join(L) + "\n"


def monitor(reference_csv: str, new_csv: str, out_dir: Optional[str] = None, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
    rep = check_drift(pd.read_csv(reference_csv), pd.read_csv(new_csv), exclude)
    out = Path(out_dir or Path(config.artifacts_dir()) / "monitoring")
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"drift_{stamp}.json").write_text(json.dumps(rep, indent=1))
    (out / f"drift_{stamp}.md").write_text(to_markdown(rep))
    try:
        from . import runstore
        runstore.audit("drift_check", None, {"alert": rep["alert"], "major": rep["major"], "new": new_csv})
    except Exception:
        pass
    return rep
