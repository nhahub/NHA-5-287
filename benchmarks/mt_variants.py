import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,sys.argv[1] if len(sys.argv)>1 else "/home/claude/NexusML_v0")
import numpy as np, pandas as pd
from nexusml import multitable as MT
U="/mnt/user-data/uploads/"
c0,o0,p0=[pd.read_csv(U+n+".csv") for n in ("customers","orders","products")]
def plan_of(t,**kw):
    p=MT.propose(t,**kw); return p
def show(name,t,**kw):
    print(f"\n### {name}")
    try:
        p=plan_of(t,**kw); print(MT.describe(p))
        p["approved"]=True; m=MT.apply(t,p); print("   -> merged",m.shape, "base rows",len(t[p['base']]))
        return p,m
    except Exception as e:
        print("   !! EXCEPTION",type(e).__name__,str(e)[:200]); return None,None
# V1: every table has its own primary key literally called id
c=c0.rename(columns={"customer_id":"id"}); o=o0.rename(columns={"order_id":"id"}); p=p0.rename(columns={"product_id":"id"})
pl,m=show("V1 primary keys all named 'id'",{"customers":c,"orders":o,"products":p})
if pl: print("   spurious joins (a table joined via id~id):",[j for j in pl["joins"] if j["left_on"]=="id" and j["right_on"]=="id"]); print("   columns:",list(m.columns)[:12])
# V2 dirty keys: string keys with prefix, whitespace and case noise in orders
c=c0.copy(); c["customer_id"]="C"+c.customer_id.astype(str).str.zfill(4); o=o0.copy(); o["customer_id"]=" c"+o.customer_id.astype(str).str.zfill(4)+" "
pl,m=show("V2 keys differ only by whitespace/case ('C0001' vs ' c0001 ')",{"customers":c,"orders":o,"products":p0})
# V3 orphans + NaN keys + customers without orders
o=o0.copy(); r=np.random.RandomState(1); o.loc[r.choice(len(o),250,replace=False),"customer_id"]=99999; o["customer_id"]=o.customer_id.astype(float); o.loc[r.choice(len(o),100,replace=False),"customer_id"]=np.nan
gone=r.choice(700,100,replace=False); o=o[~o.customer_id.isin(gone)]
pl,m=show("V3 orphan keys, NaN keys (float dtype), 100 customers with no orders",{"customers":c0,"orders":o,"products":p0})
if m is not None:
    z=m[m.customer_id.isin(gone)]; print("   customers with no orders: n_rows",z.orders__n_rows.unique()[:3],"| amount__sum values",z.orders__amount__sum.unique()[:3],"| amount__max",z.orders__amount__max.unique()[:3])
# V4 the target lives in a CHILD table (predict order amount)
pl,m=show("V4 target 'amount' is in orders (no target hint available)",{"customers":c0,"orders":o0,"products":p0})
print("   base chosen:",pl["base"] if pl else None,"-> does merged table still contain a row-level 'amount'?", (m is not None and "amount" in m.columns))
# V5 same key, different names
o=o0.rename(columns={"customer_id":"cust_id"})
show("V5 FK named cust_id vs customer_id",{"customers":c0,"orders":o,"products":p0})
# V6 second child table + bridge
tick=pd.DataFrame({"ticket_id":range(400),"customer_id":np.random.RandomState(2).randint(0,700,400),"severity":np.random.RandomState(3).choice(["low","high"],400)})
pl,m=show("V6 second child table (tickets) hanging off customers",{"customers":c0,"orders":o0,"products":p0,"tickets":tick})
if m is not None: print("   ticket cols:",[x for x in m.columns if x.startswith("tickets")])
