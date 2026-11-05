import ast, glob, os, re, random, json
import pandas as pd, numpy as np
rs=random.Random(0); R=np.random.RandomState(0)
SP="/usr/local/lib/python3.12/dist-packages"
if not os.path.isdir(SP+"/numpy"): SP=[p for p in glob.glob("/usr/**/site-packages",recursive=True)+glob.glob("/usr/**/dist-packages",recursive=True) if os.path.isdir(p+"/numpy")][0]
def paras(pkg):
    out=[]
    for f in glob.glob(f"{SP}/{pkg}/**/*.py",recursive=True):
        if "/tests/" in f or "/_vendor" in f: continue
        try: tree=ast.parse(open(f,encoding="utf-8").read())
        except Exception: continue
        for node in ast.walk(tree):
            if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                d=ast.get_docstring(node)
                if not d: continue
                for p in re.split(r"\n\s*\n",d):
                    p=re.sub(r"\s+"," ",p).strip()
                    if 12<=len(p.split())<=220 and not re.match(r"^[-=~^]+$",p): out.append(p)
    return list(dict.fromkeys(out))
rows=[]
for pkg in ["numpy","pandas","scipy","sklearn","matplotlib"]:
    P=paras(pkg); rs.shuffle(P); print(pkg,len(P)); rows+=[(t,pkg) for t in P[:400]]
df=pd.DataFrame(rows,columns=["text","library"]).sample(frac=1,random_state=1).reset_index(drop=True)
df.to_csv("libdocs.csv",index=False); print("libdocs",df.shape,"median words",int(df.text.str.split().str.len().median()))
# licenses
fam={"GPL":"GPL","GPL-1":"GPL","GPL-2":"GPL","GPL-3":"GPL","LGPL":"LGPL","LGPL-2":"LGPL","LGPL-2.1":"LGPL","LGPL-3":"LGPL","Apache-2.0":"Apache","MPL-1.1":"MPL","MPL-2.0":"MPL","BSD":"BSD","GFDL":"GFDL","GFDL-1.2":"GFDL","GFDL-1.3":"GFDL","Artistic":"Artistic","CC0-1.0":"CC0"}
lic=[]
for f,lab in fam.items():
    p="/usr/share/common-licenses/"+f
    if not os.path.exists(p): continue
    txt=open(p,encoding="utf-8",errors="replace").read()
    for para in re.split(r"\n\s*\n",txt):
        para=re.sub(r"\s+"," ",para).strip()
        if len(para.split())>=15: lic.append((para,lab,f))
L=pd.DataFrame(lic,columns=["text","license_family","file"]); print("licenses",L.shape,L.license_family.value_counts().to_dict(),"exact dup texts:",L.text.duplicated().sum())
L.drop(columns="file").sample(frac=1,random_state=2).reset_index(drop=True).to_csv("licenses.csv",index=False)
# messy variant of libdocs
m=df.copy(); n=len(m); idx=R.permutation(n)
def pick(a,b): return idx[int(a*n):int(b*n)]
emo=["😀","🔥","👍","🙏","😂","✅"]
for i in pick(0,.10): m.loc[i,"text"]=f"<p><b>{m.loc[i,'text']}</b><br/>&nbsp;</p>"
for i in pick(.10,.18): m.loc[i,"text"]+=" See https://docs.example.com/page?id=123 or mail help@example.org "+rs.choice(emo)
for i in pick(.18,.22): m.loc[i,"text"]=m.loc[i,"text"].upper()
for i in pick(.22,.25): m.loc[i,"text"]=np.nan
for i in pick(.25,.27): m.loc[i,"text"]=rs.choice(["","   ","n/a","-","?"])
for i in pick(.27,.30): m.loc[i,"text"]=" ".join(m.loc[i,"text"].split()[:2])
for i in pick(.30,.31): m.loc[i,"text"]=" ".join([m.loc[i,"text"]]*40)
for i in pick(.31,.35): m.loc[i,"text"]=m.loc[i,"text"].replace("e","3").replace("a","@") if rs.random()<.5 else re.sub(r"\s","  \t",m.loc[i,"text"])
dups=m.sample(int(.05*n),random_state=3); m=pd.concat([m,dups],ignore_index=True)           # 5% duplicated rows
noisy=R.choice(len(m),int(.05*len(m)),replace=False); labs=sorted(m.library.unique()); m["_noisy"]=0
for i in noisy:
    m.loc[i,"library"]=rs.choice([l for l in labs if l!=m.loc[i,"library"]]); m.loc[i,"_noisy"]=1      # 5% wrong labels
m=m.sample(frac=1,random_state=4).reset_index(drop=True)
m[["_noisy"]].to_csv("libdocs_messy_truth.csv",index=False); m.drop(columns="_noisy").to_csv("libdocs_messy.csv",index=False)
print("messy",m.shape,"nan",int(m.text.isna().sum()),"noisy labels",int(m._noisy.sum()))
