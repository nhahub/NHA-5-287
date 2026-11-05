"""Multi-table input: several CSV/Excel files (or sheets) -> ONE analysis table, with every join validated by code.

Flow: load_tables -> propose (key candidates + guards; the LLM may only choose among validated candidates) -> user approves the plan
-> apply() builds one row per base-table record. Child tables are AGGREGATED to the base grain (never row-multiplying joins) and the
result is checked: same row count as the base table, otherwise JoinGuardError. Nothing is joined without approval.
"""
import hashlib
import io
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import config

MIN_MATCH = 0.30            # share of child keys found in the parent table; below this the join is rejected
FUZZY_MATCH = 0.90          # a key whose NAME only resembles the parent's (cust_id ~ customer_id) must match at least this much by VALUE
MAX_NEW_COLS = 60
MAX_SHARE_LEVELS = 6        # categorical child columns with at most this many levels become per-level shares; more levels -> a distinct count


class JoinGuardError(ValueError):
    pass


class ApprovalRequired(RuntimeError):
    pass


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _singular(s: str) -> str:
    s = _norm(s)
    return s[:-1] if s.endswith("s") and len(s) > 3 else s


_AR, _LA = "\u0600-\u06ff", "A-Za-z"


def decode_bytes(raw: bytes) -> str:
    """UTF-8 (with or without BOM); otherwise Arabic cp1256 or Western latin-1. cp1256 accepts almost any byte, so it is only believed when it yields real Arabic text:
    Arabic letters glued inside Latin words ('Mلlaga') mean the file was latin-1 ('Málaga')."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    latin = raw.decode("latin-1")
    try:
        ar = raw.decode("cp1256")
    except UnicodeDecodeError:
        return latin
    if re.search(f"[{_LA}][{_AR}]|[{_AR}][{_LA}]", ar) or not re.search(f"[{_AR}]", ar):
        return latin
    return ar


def read_table(path: Path) -> pd.DataFrame:
    """CSV with the usual real-world differences: a BOM, a non-UTF-8 encoding, ';' or tab instead of ','."""
    text = decode_bytes(Path(path).read_bytes())
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if df.shape[1] == 1:                                              # one column: probably the wrong delimiter
        head = text.splitlines()[0] if text else ""
        for sep in (";", "\t", "|"):
            if sep in head:
                alt = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
                if alt.shape[1] > 1:
                    return alt
    return df


def load_tables(paths: List[str]) -> Dict[str, pd.DataFrame]:
    """CSV files, Excel workbooks (each sheet = one table), or a directory of such files."""
    files: List[Path] = []
    for p in paths:
        pp = Path(p)
        files += sorted(f for f in pp.iterdir() if f.suffix.lower() in (".csv", ".xlsx", ".xls")) if pp.is_dir() else [pp]
    tables: Dict[str, pd.DataFrame] = {}
    for f in files:
        if f.suffix.lower() == ".csv":
            tables[f.stem] = read_table(f)
        else:
            for sheet, df in pd.read_excel(f, sheet_name=None).items():
                tables[f"{f.stem}.{sheet}" if len(files) > 1 else str(sheet)] = df
    if len(tables) < 2:
        raise ValueError("multi-table input needs at least two tables")
    return tables


def norm_key(s: pd.Series) -> pd.Series:
    """Join keys compared as text after removing formatting noise: surrounding spaces, letter case, '3.0' vs 3, zero padding ('0007' vs 7).
    Missing keys stay missing. Both sides of every join go through this, so ' C0001' matches 'c0001' and an int column matches a float column."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        f = s.astype("float64")
        out = pd.Series(pd.NA, index=s.index, dtype="string")
        whole = f.notna() & (f % 1 == 0)
        out[whole] = f[whole].astype("int64").astype(str)
        frac = f.notna() & ~whole
        out[frac] = f[frac].astype(str)
        return out
    out = s.astype("string").str.strip().str.casefold()
    out = out.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
    digits = out.str.fullmatch(r"\d+").fillna(False)
    out[digits] = out[digits].str.lstrip("0").replace("", "0")
    return out.mask(out == "")


def _stats(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    a, b = norm_key(a).dropna(), norm_key(b).dropna()
    if len(a) == 0 or len(b) == 0:
        return {}
    ua, ub = a.nunique() / len(a), b.nunique() / len(b)
    a_unique, b_unique = ua > 0.999, ub > 0.999
    card = "one_to_one" if a_unique and b_unique else "one_to_many" if a_unique else "many_to_one" if b_unique else "many_to_many"
    one, many = (a, b) if a_unique or not b_unique else (b, a)
    keys = set(one)
    match = float(many.drop_duplicates().isin(keys).mean())
    orphan_rows = float((~many.isin(keys)).mean())                      # share of child ROWS (not distinct keys) that have no parent: they cannot take part in the analysis
    return {"cardinality": card, "match_rate": round(match, 3), "orphan_row_share": round(orphan_rows, 4), "a_unique": bool(a_unique), "b_unique": bool(b_unique)}


def _id_stem(name: str) -> str:
    n = _norm(name)
    return n[:-2] if n.endswith("id") and len(n) > 4 else ""


def _fuzzy_related(ca: str, cb: str) -> bool:
    """'cust_id' ~ 'customer_id': both look like identifiers and one stem is a prefix of the other (min 3 letters). The VALUES must then also agree (see key_candidates)."""
    sa, sb = _id_stem(ca), _id_stem(cb)
    return bool(sa and sb and sa != sb and min(len(sa), len(sb)) >= 3 and (sa.startswith(sb) or sb.startswith(sa)))


def key_candidates(tables: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    names, out = list(tables), []
    for i, ta in enumerate(names):
        for tb in names[i + 1:]:
            for ca in tables[ta].columns:
                for cb in tables[tb].columns:
                    na, nb = _norm(ca), _norm(cb)
                    if na == "id" and nb == "id":                          # two tables each having their own primary key called 'id': their small integers overlap by coincidence, which is not a relationship
                        continue
                    same = na == nb
                    fk = (na == "id" and nb == _singular(ta) + "id") or (nb == "id" and na == _singular(tb) + "id")
                    fuzzy = not (same or fk) and _fuzzy_related(ca, cb)
                    if not (same or fk or fuzzy):
                        continue
                    st = _stats(tables[ta][ca], tables[tb][cb])
                    if not st:
                        continue
                    if fuzzy and (st["cardinality"] == "many_to_many" or st["match_rate"] < FUZZY_MATCH):
                        continue                                          # a similar NAME alone is weak evidence: the values must agree too
                    out.append({"a": ta, "a_col": ca, "b": tb, "b_col": cb, **st, "name_match": "fuzzy" if fuzzy else "exact"})
    return out


def _guard(c: Dict[str, Any]) -> Optional[str]:
    if c["cardinality"] == "many_to_many":
        return "many-to-many key: joining would multiply rows"
    if c["match_rate"] < MIN_MATCH:
        return f"only {c['match_rate']:.0%} of child keys match the parent table (need >= {MIN_MATCH:.0%})"
    return None


def _target_table(tables: Dict[str, pd.DataFrame], target: Optional[str]) -> Optional[str]:
    """The one table that holds the target column (exact name, else same letters/digits). Several holders -> ambiguous -> None."""
    if not target:
        return None
    holders = [t for t, df in tables.items() if target in df.columns] or [t for t, df in tables.items() if any(_norm(c) == _norm(target) for c in df.columns)]
    return holders[0] if len(holders) == 1 else None


def propose(tables: Dict[str, pd.DataFrame], base: Optional[str] = None, choose: Optional[Callable[[List[Dict[str, Any]], List[str]], Optional[str]]] = None,
            target: Optional[str] = None) -> Dict[str, Any]:
    """Build a join plan. The main table is, in order: the one given (`base`), the single table that holds `target` (the analysis grain must be the grain of what is predicted),
    `choose(candidates, table_names)` (e.g. an LLM; validated, ignored if invalid), else the table referenced by most others."""
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
    if base is None and target:
        t = _target_table(tables, target)
        if t:
            base, source = t, "target column"
    if base is None and choose is not None:
        try:
            pick = choose(ok, list(tables))
            if pick in tables:
                base, source = pick, "llm"
        except Exception:
            pass
    if base is None:
        base = max(hub, key=lambda k: (hub[k], len(tables[k]))) if hub else max(tables, key=lambda k: len(tables[k]))
    elif source == "heuristic":
        source = "chosen by the user"
    if base not in tables:
        raise ValueError(f"unknown base table '{base}'")
    joins, used = [], {base}
    changed = True
    while changed:                                                    # connect tables reachable from the base through validated keys
        changed = False
        for c in ok:
            for here, there, h_col, t_col in ((c["a"], c["b"], c["a_col"], c["b_col"]), (c["b"], c["a"], c["b_col"], c["a_col"])):
                if here in used and there not in used:
                    kt = norm_key(tables[there][t_col]).dropna()
                    child_unique = kt.nunique() / max(1, len(kt)) > 0.999
                    joins.append({"table": there, "parent": here, "left_on": h_col, "right_on": t_col, "mode": "lookup" if child_unique else "aggregate",
                                  "cardinality": c["cardinality"], "match_rate": c["match_rate"], "orphan_row_share": c.get("orphan_row_share", 0.0), "name_match": c.get("name_match", "exact")})
                    used.add(there)
                    changed = True
    left_out = [t for t in tables if t not in used]
    top = max(hub.values()) if hub else 0
    alternatives = [t for t, n in hub.items() if n == top and t != base and t in used] if source == "heuristic" else []
    warnings: List[str] = []
    for t in left_out:
        if len(tables[t]) > len(tables[base]):
            warnings.append(f"Table '{t}' ({len(tables[t]):,} rows) is not connected to '{base}' ({len(tables[base]):,} rows) and is larger than it. If they should be linked, check that the key columns "
                            f"describe the same thing (different names such as 'cust_id' vs 'customer_id' are matched only when the values also agree).")
    if left_out and len(joins) == 0:
        warnings.append("No table could be linked to the main table: the analysis would use that table alone.")
    for j in joins:
        if j["orphan_row_share"] >= 0.02:
            warnings.append(f"{j['orphan_row_share']:.1%} of '{j['table']}' rows have no matching record in '{j['parent']}' and are left out of the analysis.")
        if j["name_match"] == "fuzzy":
            warnings.append(f"'{j['table']}' is linked through columns with different names ({j['parent']}.{j['left_on']} ~ {j['table']}.{j['right_on']}); the values agree ({j['match_rate']:.0%}), please confirm they mean the same thing.")
    if target and _target_table(tables, target) and _target_table(tables, target) != base:
        warnings.append(f"The target '{target}' is in '{_target_table(tables, target)}', but the main table is '{base}': the target will be summarised per record of '{base}', not predicted per row.")
    return {"base": base, "joins": joins, "rejected": rejected, "unused_tables": left_out, "source": source, "alternatives": alternatives, "warnings": warnings,
            "approved": False, "rows_base": int(len(tables[base]))}


_ZERO_FILL = ("__sum", "__nunique")


def _datetime_cols(df: pd.DataFrame, key: str) -> Dict[str, pd.Series]:
    """Columns that are dates (real datetime dtype, or text that parses as dates for >= 90% of non-missing values and whose name says so)."""
    out: Dict[str, pd.Series] = {}
    for c in df.columns:
        if c == key:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            if re.search(r"date|time|day|_at$|timestamp", str(c), re.I) and s.notna().sum() >= 5:
                p = pd.to_datetime(s, errors="coerce", format="mixed")
                if p.notna().sum() >= 0.9 * s.notna().sum():
                    out[c] = p
    return out


def _aggregate(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """One row per key. Numbers -> mean/sum/max; dates -> recency, tenure and span in days (relative to the latest date in the table);
    a categorical with few levels -> the SHARE of rows in each level (the mix), otherwise a distinct count."""
    g = df.groupby(key, dropna=True)
    parts = [g.size().rename(f"{prefix}__n_rows")]
    dates = _datetime_cols(df, key)
    for c, d in dates.items():
        ref = d.max()
        tmp = pd.DataFrame({"k": df[key], "d": d})
        gg = tmp.groupby("k")["d"]
        parts += [((ref - gg.max()).dt.days).rename(f"{prefix}__{c}__recency_days"), ((ref - gg.min()).dt.days).rename(f"{prefix}__{c}__tenure_days"),
                  ((gg.max() - gg.min()).dt.days).rename(f"{prefix}__{c}__span_days")]
    for c in df.columns:
        if c == key or c in dates or _norm(c).endswith("id") or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() > 0.95 * len(df)):
            continue                                                   # identifiers carry no signal when averaged/summed
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            for fn in ("mean", "sum", "max"):
                parts.append(getattr(g[c], fn)().rename(f"{prefix}__{c}__{fn}"))
        else:
            levels = s.dropna().astype(str).value_counts()
            if 2 <= len(levels) <= MAX_SHARE_LEVELS:                  # k-1 shares: the last one is 1 - the others
                sa = s.astype(str).where(s.notna())
                for lv in levels.index[:-1]:
                    parts.append((sa == lv).where(s.notna()).groupby(df[key]).mean().rename(f"{prefix}__{c}__share_{re.sub(r'[^0-9A-Za-z]+', '_', lv)[:20]}"))
            elif len(levels) <= 50:
                parts.append(g[c].nunique().rename(f"{prefix}__{c}__nunique"))
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def _enrich(tables: Dict[str, pd.DataFrame], plan: Dict[str, Any], name: str, budget: List[int]) -> pd.DataFrame:
    """Table `name` plus everything joined below it (deepest tables are folded in first, so chains like customers > orders > products work)."""
    df = tables[name].copy()
    for j in [x for x in plan["joins"] if x["parent"] == name]:
        child = _enrich(tables, plan, j["table"], budget)
        prefix = re.sub(r"\W+", "_", j["table"])
        child = child.assign(__k=norm_key(child[j["right_on"]])).drop(columns=[j["right_on"]])
        child = child[child["__k"].notna()]
        if j["mode"] == "aggregate":
            ag = _aggregate(child, "__k", prefix)
        else:
            ag = child.drop_duplicates("__k").set_index("__k").add_prefix(prefix + "__")
        ag = ag.iloc[:, :max(0, budget[0])]
        budget[0] -= ag.shape[1]
        before = len(df)
        df = df.assign(__k=norm_key(df[j["left_on"]])).merge(ag, how="left", left_on="__k", right_index=True).drop(columns="__k")
        if len(df) != before:
            raise JoinGuardError(f"join with '{j['table']}' changed the row count ({before} -> {len(df)}); refusing")
        if j["mode"] == "aggregate":                                   # a record with no child rows: counts and totals are 0 (not 'unknown'), shares are 0; means/max/dates stay missing
            nr = f"{prefix}__n_rows"
            for c in ag.columns:
                if c == nr or c.endswith(_ZERO_FILL) or "__share_" in c:
                    df[c] = df[c].fillna(0)
    return df


def apply(tables: Dict[str, pd.DataFrame], plan: Dict[str, Any]) -> pd.DataFrame:
    if not plan.get("approved"):
        raise ApprovalRequired("the join plan must be approved before it is applied")
    result = _enrich(tables, plan, plan["base"], [MAX_NEW_COLS])
    if len(result) != len(tables[plan["base"]]):
        raise JoinGuardError("result row count differs from the base table")
    return result


def prepare(paths: List[str], approve: Optional[Callable[[Dict[str, Any]], bool]] = None, base: Optional[str] = None,
            choose: Optional[Callable] = None, target: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Load -> propose -> ask `approve(plan)` -> apply -> write one CSV. Returns (csv_path, plan). Without an approve callback the plan is NOT applied."""
    tables = load_tables(paths)
    plan = propose(tables, base, choose, target)
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
    if plan.get("alternatives"):
        lines.append(f"  ? other tables that could be the main one: {', '.join(plan['alternatives'])} (pick it as 'Main table' if the analysis should be one row per record of that table)")
    for w in plan.get("warnings", []):
        lines.append(f"  ! {w}")
    return "\n".join(lines)
