"""Builds the offline benchmark datasets from data that ships with scikit-learn and matplotlib (no download needed), plus Electric_Production.csv if you give its path.

    python benchmarks/make_local_datasets.py [path/to/Electric_Production.csv]
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_wine

out = Path(__file__).resolve().parent / "data"
out.mkdir(exist_ok=True)
load_diabetes(as_frame=True).frame.rename(columns={"target": "progression"}).to_csv(out / "diabetes.csv", index=False)
b = load_breast_cancer(as_frame=True).frame.copy(); b["diagnosis"] = b.pop("target").map({0: "malignant", 1: "benign"}); b.to_csv(out / "breast_cancer.csv", index=False)
w = load_wine(as_frame=True).frame.copy(); w["cultivar"] = w.pop("target").map(lambda i: f"class_{i}"); w.to_csv(out / "wine.csv", index=False)
g = load_digits(as_frame=True).frame.copy(); g["digit"] = g.pop("target").astype(str); g.to_csv(out / "digits.csv", index=False)
import matplotlib
st = pd.read_csv(Path(matplotlib.get_data_path()) / "sample_data" / "Stocks.csv", comment="#", parse_dates=["Date"])
st = st[st.Date.dt.day == 1]
for col, name in (("MSFT", "msft_monthly"), ("^GSPC", "sp500_monthly")):
    st[["Date", col]].dropna().rename(columns={"Date": "date", col: "price"}).to_csv(out / f"{name}.csv", index=False)
if len(sys.argv) > 1:
    e = pd.read_csv(sys.argv[1]); e.columns = ["date", "production"]; e["date"] = pd.to_datetime(e["date"], format="%m/%d/%Y"); e.to_csv(out / "electric.csv", index=False)
print("wrote", sorted(p.name for p in out.glob("*.csv")))
