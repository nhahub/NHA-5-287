"""Multi-table input: several CSV/Excel files (or sheets) -> ONE analysis table, with every join validated by code.

Flow: load_tables -> propose (key candidates + guards; the LLM may only choose among validated candidates) -> user approves the plan
-> apply() builds one row per base-table record. Child tables are AGGREGATED to the base grain (never row-multiplying joins) and the
result is checked: same row count as the base table, otherwise JoinGuardError. Nothing is joined without approval.
"""
import hashlib
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import config

MIN_MATCH = 0.30            # share of child keys found in the parent table; below this the join is rejected
MAX_NEW_COLS = 60


class JoinGuardError(ValueError):
    pass


class ApprovalRequired(RuntimeError):
    pass


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _singular(s: str) -> str:
    s = _norm(s)
    return s[:-1] if s.endswith("s") and len(s) > 3 else s


def load_tables(paths: List[str]) -> Dict[str, pd.DataFrame]:
    """CSV files, Excel workbooks (each sheet = one table), or a directory of such files."""
    files: List[Path] = []
    for p in paths:
        pp = Path(p)
        files += sorted(f for f in pp.iterdir() if f.suffix.lower() in (".csv", ".xlsx", ".xls")) if pp.is_dir() else [pp]
    tables: Dict[str, pd.DataFrame] = {}
    for f in files:
        if f.suffix.lower() == ".csv":
            tables[f.stem] = pd.read_csv(f)
        else:
            for sheet, df in pd.read_excel(f, sheet_name=None).items():
                tables[f"{f.stem}.{sheet}" if len(files) > 1 else str(sheet)] = df
    if len(tables) < 2:
        raise ValueError("multi-table input needs at least two tables")
    return tables


def _stats(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    a, b = a.dropna(), b.dropna()
    if len(a) == 0 or len(b) == 0:
        return {}
    ua, ub = a.nunique() / len(a), b.nunique() / len(b)
    a_unique, b_unique = ua > 0.999, ub > 0.999
    card = "one_to_one" if a_unique and b_unique else "one_to_many" if a_unique else "many_to_one" if b_unique else "many_to_many"
    one, many = (a, b) if a_unique or not b_unique else (b, a)
    match = float(many.drop_duplicates().isin(set(one)).mean())
    return {"cardinality": card, "match_rate": round(match, 3), "a_unique": bool(a_unique), "b_unique": bool(b_unique)}


def key_candidates(tables: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    names, out = list(tables), []
    for i, ta in enumerate(names):
        for tb in names[i + 1:]:
            for ca in tables[ta].columns:
                for cb in tables[tb].columns:
                    same = _norm(ca) == _norm(cb)
                    fk = (_norm(ca) == "id" and _norm(cb) == _singular(ta) + "id") or (_norm(cb) == "id" and _norm(ca) == _singular(tb) + "id")
                    if not (same or fk):
                        continue
                    st = _stats(tables[ta][ca], tables[tb][cb])
                    if not st:
                        continue
                    out.append({"a": ta, "a_col": ca, "b": tb, "b_col": cb, **st})
    return out


def _guard(c: Dict[str, Any]) -> Optional[str]:
    if c["cardinality"] == "many_to_many":
        return "many-to-many key: joining would multiply rows"
    if c["match_rate"] < MIN_MATCH:
        return f"only {c['match_rate']:.0%} of child keys match the parent table (need >= {MIN_MATCH:.0%})"
    return None


def propose(tables: Dict[str, pd.DataFrame], base: Optional[str] = None, choose: Optional[Callable[[List[Dict[str, Any]], List[str]], Optional[str]]] = None) -> Dict[str, Any]:
    """Build a join plan. `choose(candidates, table_names)` (e.g. an LLM) may return the base table name; it is validated, and ignored if invalid."""
    cands = key_candidates(tables)
    ok, rejected = [], []
    for c in cands:
        why = _guard(c)
        (rejected if why else ok).append({**c, "rejected": why} if why else c)
    hub: Dict[str, int] = {}
    for c in ok:                                                      # the table whose key is referenced by the most others = parent/hub
        parent = c["a"] if c["a_unique"] and not c["b_unique"] else c["b"] if c["b_unique"] and not c["a_unique"] else c["a"]
        hub[parent] = hub.get(parent, 0) + 1
    source = "heuristic"
    if base is None and choose is not None:
        try:
            pick = choose(ok, list(tables))
            if pick in tables:
                base, source = pick, "llm"
        except Exception:
            pass
    if base is None:
        base = max(hub, key=lambda k: (hub[k], len(tables[k]))) if hub else max(tables, key=lambda k: len(tables[k]))
    if base not in tables:
        raise ValueError(f"unknown base table '{base}'")
    joins, used = [], {base}
    changed = True
    while changed:                                                    # connect tables reachable from the base through validated keys
        changed = False
        for c in ok:
            for here, there, h_col, t_col in ((c["a"], c["b"], c["a_col"], c["b_col"]), (c["b"], c["a"], c["b_col"], c["a_col"])):
                if here in used and there not in used:
                    child_unique = tables[there][t_col].dropna().nunique() / max(1, tables[there][t_col].notna().sum()) > 0.999
                    joins.append({"table": there, "parent": here, "left_on": h_col, "right_on": t_col, "mode": "lookup" if child_unique else "aggregate",
                                  "cardinality": c["cardinality"], "match_rate": c["match_rate"]})
                    used.add(there)
                    changed = True
    left_out = [t for t in tables if t not in used]
    return {"base": base, "joins": joins, "rejected": rejected, "unused_tables": left_out, "source": source, "approved": False,
            "rows_base": int(len(tables[base]))}


def _aggregate(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    g = df.groupby(key, dropna=True)
    parts = [g.size().rename(f"{prefix}__n_rows")]
    for c in df.columns:
        if c == key or _norm(c).endswith("id") or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() > 0.95 * len(df)):
            continue                                                   # identifiers carry no signal when averaged/summed
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            for fn in ("mean", "sum", "max"):
                parts.append(getattr(g[c], fn)().rename(f"{prefix}__{c}__{fn}"))
        elif s.nunique(dropna=True) <= 50:
            parts.append(g[c].nunique().rename(f"{prefix}__{c}__nunique"))
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def _enrich(tables: Dict[str, pd.DataFrame], plan: Dict[str, Any], name: str, budget: List[int]) -> pd.DataFrame:
    """Table `name` plus everything joined below it (deepest tables are folded in first, so chains like customers > orders > products work)."""
    df = tables[name].copy()
    for j in [x for x in plan["joins"] if x["parent"] == name]:
        child = _enrich(tables, plan, j["table"], budget)
        prefix = re.sub(r"\W+", "_", j["table"])
        if j["mode"] == "aggregate":
            ag = _aggregate(child, j["right_on"], prefix)
        else:
            ag = child.drop_duplicates(j["right_on"]).set_index(j["right_on"]).add_prefix(prefix + "__")
        ag = ag.iloc[:, :max(0, budget[0])]
        budget[0] -= ag.shape[1]
        before = len(df)
        df = df.merge(ag, how="left", left_on=j["left_on"], right_index=True)
        if len(df) != before:
            raise JoinGuardError(f"join with '{j['table']}' changed the row count ({before} -> {len(df)}); refusing")
        nr = f"{prefix}__n_rows"
        if j["mode"] == "aggregate" and nr in df:
            df[nr] = df[nr].fillna(0)
    return df


def apply(tables: Dict[str, pd.DataFrame], plan: Dict[str, Any]) -> pd.DataFrame:
    if not plan.get("approved"):
        raise ApprovalRequired("the join plan must be approved before it is applied")
    result = _enrich(tables, plan, plan["base"], [MAX_NEW_COLS])
    if len(result) != len(tables[plan["base"]]):
        raise JoinGuardError("result row count differs from the base table")
    return result


def prepare(paths: List[str], approve: Optional[Callable[[Dict[str, Any]], bool]] = None, base: Optional[str] = None,
            choose: Optional[Callable] = None) -> Tuple[str, Dict[str, Any]]:
    """Load -> propose -> ask `approve(plan)` -> apply -> write one CSV. Returns (csv_path, plan). Without an approve callback the plan is NOT applied."""
    tables = load_tables(paths)
    plan = propose(tables, base, choose)
    if approve is None or not approve(plan):
        raise ApprovalRequired("join plan not approved; nothing was joined")
    plan["approved"] = True
    merged = apply(tables, plan)
    plan["rows_after"] = int(len(merged))
    plan["columns_after"] = int(merged.shape[1])
    d = Path(config.artifacts_dir()) / "tables"
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1((plan["base"] + str([j["table"] for j in plan["joins"]])).encode()).hexdigest()[:8]
    out = d / f"merged_{plan['base']}_{h}.csv"
    merged.to_csv(out, index=False)
    return str(out), plan


def describe(plan: Dict[str, Any]) -> str:
    lines = [f"Base table: {plan['base']} ({plan['rows_base']:,} rows)  [chosen by {plan['source']}]"]
    for j in plan["joins"]:
        lines.append(f"  + {j['table']}  on {j['parent']}.{j['left_on']} = {j['table']}.{j['right_on']}  ({j['cardinality']}, {j['match_rate']:.0%} keys match; "
                     f"{'one row per key, columns added' if j['mode'] == 'lookup' else 'aggregated to one row per key'})")
    for r in plan["rejected"]:
        lines.append(f"  x rejected {r['a']}.{r['a_col']} ~ {r['b']}.{r['b_col']}: {r['rejected']}")
    for t in plan["unused_tables"]:
        lines.append(f"  - not connected: {t}")
    return "\n".join(lines)
