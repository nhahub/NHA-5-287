import sys, os, json, tempfile, warnings, time; warnings.filterwarnings("ignore"); sys.path.insert(0,os.environ.get("NEXUS_ROOT","/home/claude/NexusML_v0")); TAG=os.environ.get("TAG","base")
from pathlib import Path
from unittest import mock
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from nexusml import config, modeling
from nexusml.evaluation import harness
W=Path("/home/claude/work/nlp"); KPI=[{"name":"Records","agg":"count"}]
SETS={"libdocs":("text","library"),"licenses":("text","license_family"),"libdocs_messy":("text","library")}
def case(name):
    t,y=SETS[name]
    return {"name":name,"kind":"text","path":str(W/f"{name}.csv"),"frame":{"analysis_type":"general_exploration","problem_type":"classification","target":y,"date_column":None,"segment_columns":[],"kpis":KPI},
            "truth":{"target":y,"problem_type":"classification"}}
def feats(tr_txt):
    w=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2); c=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),sublinear_tf=True,min_df=2,max_features=60000)
    w.fit(tr_txt); c.fit(tr_txt); return lambda t: hstack([w.transform(t),c.transform(t)]).tocsr()
def mine(df,tcol,ycol,tr,te):
    txt=df[tcol].fillna("").astype(str).values; y=df[ycol].astype(str).values
    f=feats(txt[tr]); Xtr,Xte=f(txt[tr]),f(txt[te])
    cands={"LinearSVC":LinearSVC(C=0.5),"LogReg":LogisticRegression(C=20,max_iter=2000),"ComplementNB":ComplementNB(alpha=0.3)}
    vc=pd.Series(y[tr]).value_counts(); k=int(min(5,vc[vc>=2].min() if (vc>=2).any() else 2)); k=max(2,k)
    try:
        cv=StratifiedKFold(k,shuffle=True,random_state=0); sc={n:cross_val_score(m,Xtr,y[tr],cv=cv,scoring="f1_macro").mean() for n,m in cands.items()}
    except Exception: sc={n:0 for n in cands}
    best=max(sc,key=sc.get); m=cands[best].fit(Xtr,y[tr]); p=m.predict(Xte)
    return {"model":best,"acc":accuracy_score(y[te],p),"f1m":f1_score(y[te],p,average="macro")}
def run(name,seeds):
    t,yc=SETS[name]; df=pd.read_csv(W/f"{name}.csv"); out=[]
    real=modeling.make_splits; cap={}
    def spy(d,plan):
        r=real(d,plan); cap["te"]=np.array(r["X_test"].index); cap["tr"]=np.array(r["X_train"].index); return r
    for s in seeds:
        t0=time.time()
        with mock.patch.object(config,"RANDOM_STATE",s), mock.patch.object(modeling,"make_splits",side_effect=spy):
            r=harness.run_case(case(name),Path(tempfile.mkdtemp()),seed=s)
        v=r["values"]; lb=v.get("leaderboard") or {}; top=(lb.get("rows") or [{}])[0]
        row={"seed":s,"secs":round(time.time()-t0),"errors":v.get("errors"),"winner":lb.get("winner"),"metric":lb.get("metric"),"nexus_heldout":top.get("heldout"),"n_test":len(cap["te"]),"n_cand":len(lb.get("rows") or [])}
        row["me"]=mine(df,t,yc,cap["tr"],cap["te"]); out.append(row); json.dump(out,open(W/f"cmp_{TAG}_{name}.json","w"),default=str); print(name,"seed",s,row["nexus_heldout"],row["me"],flush=True)
    return r
if __name__=="__main__":
    name=sys.argv[1]; seeds=[int(x) for x in sys.argv[2:]] or [1]
    r=run(name,seeds)
    json.dump({"evidence":[(e["id"],e["text"]) for e in r["values"]["evidence"]],"audit":r["values"].get("audit"),"warnings":r["values"].get("warnings")},open(W/f"ev_{TAG}_{name}.json","w"),default=str)
