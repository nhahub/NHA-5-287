import json,os,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
mine=json.load(open(os.path.join(HERE,"results","claude_handbuilt.json")))
def ld(r,ds):
    f=os.path.join(HERE,"results",f"{r}_{ds}.json"); return json.load(open(f)) if os.path.exists(f) else None
rows=[]
print("TIME SERIES (identical last-77-month window; one-step-ahead, true lags)")
for ds in ("electric","msft_monthly","sp500_monthly"):
    a,b,m=ld("v4",ds),ld("v5",ds),mine[ds]
    def sc(y,p): y,p=np.asarray(y)[-77:],np.asarray(p)[-77:]; return r2_score(y,p),mean_squared_error(y,p)**.5,mean_absolute_error(y,p)
    for name,x,pred,y,tr in (("v4",a,a["pred_test"],a["y_test"],a["chosen"]["train"]),("v5",b,b["pred_test"],b["y_test"],b["chosen"]["train"]),("Claude",None,m["pred_test"],m["y_test"],m["train"])):
        r2,rm,ma=sc(y,pred); print(f"{ds:14s} {name:7s} test R2 {r2:6.3f} RMSE {rm:8.3f} MAE {ma:7.3f} | train R2 {tr['r2']:.3f} RMSE {tr['rmse']:.3f}  model: {(x['best'] if x else m['best'])[:42]}")
print("\nTABULAR (identical 80/20 split)")
for ds in ("diabetes","breast_cancer","wine","digits"):
    a,b,m=ld("v4",ds),ld("v5",ds),mine[ds]
    for name,x in (("v4",a),("v5",b),("Claude",m)):
        if x is None: print(f"{ds:14s} {name:7s} (still running)"); continue
        t=x["chosen"] if "chosen" in x else x; te,tr=t["test"],t["train"]
        k=("r2","rmse") if "r2" in te else ("acc","f1_macro")
        print(f"{ds:14s} {name:7s} test {k[0]} {te[k[0]]:.3f} {k[1]} {te[k[1]]:.3f} | train {k[0]} {tr[k[0]]:.3f}  model: {x['best'][:40]}")
