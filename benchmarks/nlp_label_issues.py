import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,sys.argv[1])
import numpy as np, pandas as pd
from nexusml import nlp
d=pd.read_csv("/home/claude/work/nlp/libdocs_messy.csv"); t=pd.read_csv("/home/claude/work/nlp/libdocs_messy_truth.csv")._noisy.values
docs=d.text.fillna("").astype(str).tolist(); lab=d.library.astype(str).tolist()
r=nlp.label_issues(docs,lab,top=100000); sus=np.array([s["index"] for s in r["suspects"]]); hit=t[sus].sum() if len(sus) else 0
print(f"planted wrong labels={int(t.sum())} | flagged={len(sus)} | true hits={int(hit)} | precision={hit/max(1,len(sus)):.2f} | recall={hit/t.sum():.2f}")
# among CLEAN-text rows only (excluding NaN/empty/junk/1-2 word rows which no model can judge)
ok=np.array([len(x.split())>=8 for x in docs]); tt=t[ok]; ss=[i for i in sus if ok[i]]
print(f"  on judgeable rows (>=8 words): noisy={int(tt.sum())} flagged={len(ss)} hits={int(t[ss].sum())} precision={t[ss].sum()/max(1,len(ss)):.2f} recall={t[ss].sum()/max(1,tt.sum()):.2f}")
for k in (10,12,25,50,100):
    top=sus[:k]; print(f"  precision@{k:3d} = {t[top].sum()/len(top):.2f}")
