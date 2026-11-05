"""Run the full agentic pipeline (offline, stub LLM) on one real dataset and record train/test metrics.

    python benchmarks/make_local_datasets.py                    # writes the datasets to ./benchmarks/data (needs the Electric_Production.csv you supply, optional)
    python benchmarks/real_data_bench.py <repo root> <dataset> <out.json>      e.g.  python benchmarks/real_data_bench.py . electric out.json
Run it once with this repo and once with an older checkout to compare versions on the same data and split.
"""
import sys,warnings,os,tempfile,json,time; ROOT,DS=sys.argv[1],sys.argv[2]; sys.path.insert(0,ROOT); warnings.filterwarnings('ignore')
from unittest import mock
art=tempfile.mkdtemp(); os.environ.update({"NEXUSML_ARTIFACTS":art,"NEXUSML_PARALLEL":"1","GROQ_API_KEY":"stub-key"})
from nexusml import llm
from nexusml.evaluation.stub_llm import make_stub
from nexusml.runtime import run_to_completion
from nexusml.storage import get_registry
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score
CFG={"electric":("time_series","production","date"),"msft_monthly":("time_series","price","date"),"sp500_monthly":("time_series","price","date"),
     "diabetes":("regression","progression",None),"breast_cancer":("classification","diagnosis",None),"wine":("classification","cultivar",None),"digits":("classification","digit",None)}
pt,target,date=CFG[DS]
frame={"analysis_type":"sales_performance" if pt=="time_series" else "general_exploration","problem_type":pt,"target":target,"date_column":date,"segment_columns":[],
       "kpis":[{"name":"Total","agg":"sum","column":target,"format":"number","group_by":[]}] if pt!="classification" else [{"name":"Cases","agg":"count","column":target,"format":"number","group_by":[]}]}
t0=time.time()
with mock.patch.object(llm,"available",return_value=True), mock.patch.object(llm,"ask_json",side_effect=make_stub(frame)):
    v=run_to_completion(os.path.join(os.environ.get("NEXUSML_BENCH_DATA", os.path.join(ROOT, "benchmarks", "data")), f"{DS}.csv"),f"Predict {target}",auto_approve=True,tune_n_iter=6,max_retries=1,enable_tuning=True,seed=42)
secs=round(time.time()-t0,1)
reg=get_registry(); sp=reg.load(v["splits_key"]); models=reg.load(v["models_key"])["models"]; best=v["best_model_name"]
ytr,yte=np.asarray(sp["y_train"]),np.asarray(sp["y_test"]); est=models[best]
def m(y,p,P=None):
    if pt=="classification":
        r={"acc":accuracy_score(y,p),"f1_macro":f1_score(y,p,average="macro")}
        if P is not None and P.shape[1]==2: r["auc"]=roc_auc_score(y,P[:,1])
        return r
    return {"r2":r2_score(y,p),"rmse":mean_squared_error(y,p)**.5,"mae":mean_absolute_error(y,p)}
def ev(e):
    ptr,pte=e.predict(sp["X_train"]),e.predict(sp["X_test"])
    Ptr=Pte=None
    if pt=="classification" and hasattr(e,"predict_proba"): Ptr,Pte=e.predict_proba(sp["X_train"]),e.predict_proba(sp["X_test"])
    return {"train":m(ytr,ptr,Ptr),"test":m(yte,pte,Pte)}
res=ev(est)
cands={}
for name,e in models.items():
    try: cands[name]=ev(e)
    except Exception as ex: pass
out={"y_test":yte.tolist(),"pred_test":[float(x) for x in est.predict(sp["X_test"])],"dataset":DS,"best":best,"seconds":secs,"n_train":len(ytr),"n_test":len(yte),"families_tried":sorted({c.get("name","").split(" + ")[0] for c in v["candidate_metrics"]}),
     "chosen":res,"candidates":cands,"n_trials":len(getattr(v.get("automl_report",{}),"get",lambda *_:[])("trials",[])) if isinstance(v.get("automl_report"),dict) else None}
json.dump(out,open(sys.argv[3],"w"),default=str,indent=1)
print(DS,best,secs,"s",{k:round(x,3) for k,x in res["test"].items()})
