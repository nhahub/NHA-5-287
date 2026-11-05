"""Usage: python bench_agent.py <project_root>  -> JSON. Same scenarios against any NexusML version.
The scripted 'model' is as good as its protocol allows: it batches independent calls only if the protocol advertises "actions"."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
root = sys.argv[1]; sys.path.insert(0, root)
from unittest import mock
import numpy as np, pandas as pd
from nexusml import llm
from nexusml.assistant import agent as A, tools as T

rs = np.random.RandomState(0); n = 400
df = pd.DataFrame({"Total Revenue": rs.gamma(3, 100, n), "region": rs.choice(list("NSEW"), n), "units": rs.poisson(6, n).astype(float),
                   "price": rs.uniform(5, 50, n), "discount": rs.uniform(0, .3, n), "channel": rs.choice(["web", "store", "app"], n),
                   "order_date": pd.date_range("2024-01-01", periods=n, freq="D")})
df.loc[rs.choice(n, 30, replace=False), "discount"] = np.nan
df.loc[:5, "Total Revenue"] *= 9

def call(tool, **a): return {"tool": tool, "args": a}
SCEN = {
 "S1 three independent checks":      ("Overview, outliers and missing values please", [call("overview"), call("outliers"), call("missing")]),
 "S2 revenue by region + drivers":   ("Does revenue differ by region and what drives it?", [call("relationship", a="Total Revenue", b="region"), call("drivers", target="Total Revenue"), call("group_summary", by="region", value="Total Revenue")]),
 "S3 column named loosely":          ("Describe total_revenue", [call("distribution", column="total_revenue")]),
 "S4 model repeats a call":          ("Profile the data", [call("overview"), call("overview"), call("missing")]),
 "S5 six-part health check":         ("Full health check of this data", [call("overview"), call("missing"), call("outliers"), call("correlations"), call("duplicates"), call("distribution", column="units")]),
 "S6 stuck on a failing call":       ("Outliers in margin", [call("outliers", columns=["margin"])] * 8),
}

def make_model(plan):
    def model(pl):
        batched = "actions" in json.dumps(pl.get("output_schema", {}))
        obs = pl.get("observations") or []
        if pl.get("must_reply_now") or len(obs) >= len(plan) and not any("error" in o for o in obs[-1:]) and plan[0]["args"].get("columns") != ["margin"]:
            return {"reply": "Here is what I found; see the tables and charts below.", "followups": []}
        if pl.get("must_reply_now"):
            return {"reply": "I could not check that column.", "followups": []}
        # which planned calls are still unanswered? (obs holds every attempt, in order)
        k = len(obs)
        if obs and "error" in obs[-1] and "Closest matches" in obs[-1]["error"]:   # a model reads the error and fixes the column name
            import re; m = re.search(r"Closest matches: \['([^']+)'", obs[-1]["error"]); a = dict(plan[k-1]["args"]); a["column"] = m.group(1) if m else a.get("column")
            return {"action": {"tool": plan[k-1]["tool"], "args": a}}
        if batched and plan[0]["args"].get("columns") != ["margin"]:      # the stuck-model scenario repeats ONE call per step under either protocol
            rest = plan[k:]
            return {"actions": rest} if rest else {"reply": "Done; see below."}
        if plan[0]["args"].get("columns") == ["margin"]:
            return {"action": plan[0]}
        return {"action": plan[k]} if k < len(plan) else {"reply": "Done; see below."}
    return model

out = {}
for name, (msg, plan) in SCEN.items():
    sent = {"chars": 0, "calls": 0}; runs = {"n": 0}
    model = make_model(plan)
    def fake(system, prompt, purpose="", max_tokens=0, temperature=0):
        sent["chars"] += len(system) + len(prompt); sent["calls"] += 1
        return model(json.loads(prompt)), None, {"purpose": purpose, "model": "scripted", "ok": True, "seconds": 0.0, "tokens": 0}
    real_run = T.run_tool
    def counted(*a, **k): runs["n"] += 1; return real_run(*a, **k)
    t = time.time()
    with mock.patch.object(llm, "ask_json", side_effect=fake), mock.patch.object(T, "run_tool", side_effect=counted):
        r = A.respond(msg, None, T.ToolContext(df.copy(), "sales"))
    answered = not r["text"].startswith(("I ran out of steps", "I wasn't able"))
    out[name] = {"llm_calls": sent["calls"], "tool_runs": runs["n"], "prompt_chars": sent["chars"], "answered": answered, "seconds": round(time.time() - t, 2)}
print(json.dumps(out))
