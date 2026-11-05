import sys,os,warnings,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE)); warnings.filterwarnings('ignore')
DATA=os.environ.get('NEXUSML_BENCH_DATA',os.path.join(HERE,'data'))
# My hand-built reference: a careful practitioner's workflow (several sensible models, chosen by cross-validation on TRAIN only, then scored once on the same split the pipeline uses).
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from nexusml import modeling as M
CFG={"diabetes":("regression","progression"),"breast_cancer":("classification","diagnosis"),"wine":("classification","cultivar"),"digits":("classification","digit")}
out={}
for ds,(pt,target) in CFG.items():
    df=pd.read_csv(os.path.join(DATA,f"{ds}.csv")); sp=M.make_splits(df,{"problem_type":pt,"target":target,"drop_columns":[],"datetime_columns":[]})
    Xtr,Xte,ytr,yte=sp["X_train"].to_numpy(float),sp["X_test"].to_numpy(float),np.asarray(sp["y_train"]),np.asarray(sp["y_test"])
    if pt=="regression":
        cands={"RidgeCV":make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,30))),"HistGB":HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_depth=3,random_state=0),
               "RandomForest":RandomForestRegressor(300,min_samples_leaf=3,random_state=0,n_jobs=-1),"SVR":make_pipeline(StandardScaler(),SVR(C=100,epsilon=5,gamma="scale"))}
        cv,sc=KFold(5,shuffle=True,random_state=0),"r2"
    else:
        cands={"LogReg":make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=3000)),"SVC":make_pipeline(StandardScaler(),SVC(C=10,probability=True,random_state=0)),
               "HistGB":HistGradientBoostingClassifier(random_state=0),"RandomForest":RandomForestClassifier(300,random_state=0,n_jobs=-1)}
        cv,sc=StratifiedKFold(5,shuffle=True,random_state=0),"accuracy"
    cvs={k:cross_val_score(m,Xtr,ytr,cv=cv,scoring=sc).mean() for k,m in cands.items()}; best=max(cvs,key=cvs.get); m=cands[best].fit(Xtr,ytr)
    def ev(X,y):
        p=m.predict(X)
        if pt=="regression": return {"r2":r2_score(y,p),"rmse":mean_squared_error(y,p)**.5,"mae":mean_absolute_error(y,p)}
        r={"acc":accuracy_score(y,p),"f1_macro":f1_score(y,p,average="macro")}
        if len(set(ytr))==2: r["auc"]=roc_auc_score(y,m.predict_proba(X)[:,1])
        return r
    out[ds]={"best":best,"cv":{k:round(v,3) for k,v in cvs.items()},"train":ev(Xtr,ytr),"test":ev(Xte,yte)}
    print(ds,best,{k:round(v,3) for k,v in out[ds]["test"].items()},flush=True)
# ---- time series: clean protocol (select on rolling CV inside train only)
def fe(y,dates,idx,L):
    mo=pd.to_datetime(dates).dt.month.values[idx]
    return np.column_stack([L[idx-1],L[idx-2],L[idx-3],L[idx-12],L[idx-13],[L[i-12:i].mean() for i in idx],np.sin(2*np.pi*mo/12),np.cos(2*np.pi*mo/12),idx/len(y)])
for ds in ("electric","msft_monthly","sp500_monthly"):
    d=pd.read_csv(os.path.join(DATA,f"{ds}.csv")); col=[c for c in d.columns if c!="date"][0]; y=d[col].values.astype(float); n=len(y); L=np.log(y)
    NT=77; tr_end=n-NT; te=np.arange(tr_end,n); tr=np.arange(13,tr_end)
    def pred(name,idx,model=None):
        if name=="naive": return y[idx-1]
        if name=="seasonal_naive": return y[idx-12]
        F=fe(y,d.date,idx,L)
        return np.exp(model.predict(F)) if name=="ridge_log" else np.exp(model.predict(F)+L[idx-12])
    def fit(name,idx):
        if name=="ridge_log": return make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,20))).fit(fe(y,d.date,idx,L),L[idx])
        if name=="hgb_sdelta": return HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_depth=3,random_state=0).fit(fe(y,d.date,idx,L),L[idx]-L[idx-12])
    # rolling-origin CV on the training part only (3 folds like the project)
    cvs={}
    for name in ("naive","seasonal_naive","ridge_log","hgb_sdelta"):
        sc=[]
        for a,b in TimeSeriesSplit(3,test_size=max(5,len(tr)//6)).split(tr):
            ia,ib=tr[a],tr[b]; mdl=fit(name,ia) if name in("ridge_log","hgb_sdelta") else None
            sc.append(r2_score(y[ib],pred(name,ib,mdl)))
        cvs[name]=float(np.mean(sc))
    best=max(cvs,key=cvs.get); mdl=fit(best,tr) if best in("ridge_log","hgb_sdelta") else None
    f=lambda idx:{"r2":r2_score(y[idx],pred(best,idx,mdl)),"rmse":mean_squared_error(y[idx],pred(best,idx,mdl))**.5,"mae":mean_absolute_error(y[idx],pred(best,idx,mdl))}
    out[ds]={"best":best,"cv":{k:round(v,3) for k,v in cvs.items()},"train":f(tr),"test":f(te),"y_test":y[te].tolist(),"pred_test":pred(best,te,mdl).tolist()}
    print(ds,best,{k:round(v,3) for k,v in out[ds]["test"].items()},flush=True)
json.dump(out,open(os.path.join(HERE,"results","claude_handbuilt.json"),"w"),default=str)
