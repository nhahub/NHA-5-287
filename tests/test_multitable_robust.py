"""Multi-table joins on realistic, messier inputs. Each case failed (silently) before: see CHANGES.md."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nexusml import multitable as MT


def base_tables(n=300, seed=0):
    r = np.random.RandomState(seed)
    cust = pd.DataFrame({"customer_id": range(n), "region": r.choice(list("NSEW"), n), "churn": r.choice(["Yes", "No"], n, p=[.2, .8])})
    k = r.poisson(6, n) + 1
    orders = pd.DataFrame({"order_id": range(k.sum()), "customer_id": np.repeat(np.arange(n), k), "product_id": r.randint(0, 20, k.sum()), "amount": r.gamma(2, 40, k.sum())})
    prod = pd.DataFrame({"product_id": range(20), "category": r.choice(list("ABC"), 20), "price": r.uniform(5, 90, 20)})
    return cust, orders, prod


def merged(tables, **kw):
    plan = MT.propose(tables, **kw); plan["approved"] = True
    return plan, MT.apply(tables, plan)


class TestJoinCorrectness(unittest.TestCase):
    def test_tables_that_each_have_a_primary_key_called_id_are_not_joined_to_each_other(self):
        c, o, p = base_tables(); c, o, p = c.rename(columns={"customer_id": "id"}), o.rename(columns={"order_id": "id"}), p.rename(columns={"product_id": "id"})
        plan, m = merged({"customers": c, "orders": o, "products": p})
        self.assertFalse([j for j in plan["joins"] if j["left_on"] == "id" and j["right_on"] == "id"])
        by = {j["table"]: (j["left_on"], j["right_on"]) for j in plan["joins"]}
        self.assertEqual(by["orders"], ("id", "customer_id")); self.assertEqual(by["products"], ("product_id", "id"))
        self.assertEqual(len(m), len(c))

    def test_keys_that_differ_only_in_formatting_still_match(self):
        c, o, p = base_tables()
        c["customer_id"] = "C" + c.customer_id.astype(str).str.zfill(4); o["customer_id"] = " c" + o.customer_id.astype(str).str.zfill(4) + " "
        plan, m = merged({"customers": c, "orders": o, "products": p})
        self.assertEqual(plan["base"], "customers"); self.assertEqual({j["table"] for j in plan["joins"]}, {"orders", "products"}); self.assertEqual(plan["unused_tables"], [])
        self.assertGreater(m["orders__n_rows"].min(), 0)

    def test_int_float_and_zero_padded_keys_match(self):
        a = pd.Series([1, 2, 3, 4]); self.assertEqual(list(MT.norm_key(a)), ["1", "2", "3", "4"])
        self.assertEqual(list(MT.norm_key(pd.Series([1.0, 2.0, None]))[:2]), ["1", "2"]); self.assertTrue(pd.isna(MT.norm_key(pd.Series([1.0, None]))[1]))
        self.assertEqual(list(MT.norm_key(pd.Series(["0007", "7", " 7.0 "]))), ["7", "7", "7"])

    def test_missing_keys_never_match_each_other(self):
        c, o, p = base_tables(); c.loc[0, "customer_id"] = np.nan; o.loc[:20, "customer_id"] = np.nan
        plan, m = merged({"customers": c, "orders": o, "products": p})
        self.assertEqual(len(m), len(c)); self.assertEqual(m.loc[0, "orders__n_rows"], 0)

    def test_similar_names_link_only_when_the_values_agree(self):
        c, o, p = base_tables(); o2 = o.rename(columns={"customer_id": "cust_id"})
        plan = MT.propose({"customers": c, "orders": o2, "products": p})
        j = next(j for j in plan["joins"] if j["table"] == "orders"); self.assertEqual(j["name_match"], "fuzzy"); self.assertTrue(any("different names" in w for w in plan["warnings"]))
        o3 = o2.copy(); o3["cust_id"] = np.random.RandomState(1).randint(10_000, 20_000, len(o3))         # same-looking name, unrelated values
        plan3 = MT.propose({"customers": c, "orders": o3, "products": p})
        self.assertNotIn("orders", {j["table"] for j in plan3["joins"] if j["parent"] == "customers"})

    def test_an_unconnected_larger_table_is_called_out(self):
        c, o, p = base_tables(); o = o.rename(columns={"customer_id": "buyer_ref"})
        plan = MT.propose({"customers": c, "orders": o, "products": p})
        self.assertTrue(any("not connected" in w and "larger" in w for w in plan["warnings"]), plan["warnings"])


class TestGrainAndAggregation(unittest.TestCase):
    def test_the_table_holding_the_target_is_the_main_table(self):
        c, o, p = base_tables()
        plan, m = merged({"customers": c, "orders": o, "products": p}, target="amount")
        self.assertEqual(plan["base"], "orders"); self.assertEqual(plan["source"], "target column"); self.assertEqual(len(m), len(o)); self.assertIn("amount", m.columns)
        self.assertTrue({"customers__region", "products__price"} <= set(m.columns))
        self.assertEqual(MT.propose({"customers": c, "orders": o, "products": p}, target="churn")["base"], "customers")

    def test_an_explicit_base_beats_the_target(self):
        c, o, p = base_tables()
        self.assertEqual(MT.propose({"customers": c, "orders": o, "products": p}, base="customers", target="amount")["base"], "customers")

    def test_records_without_children_get_zero_totals_not_missing(self):
        c, o, p = base_tables(); gone = list(range(30)); o = o[~o.customer_id.isin(gone)]
        _, m = merged({"customers": c, "orders": o, "products": p})
        z = m[m.customer_id.isin(gone)]
        self.assertTrue((z["orders__n_rows"] == 0).all()); self.assertTrue((z["orders__amount__sum"] == 0).all()); self.assertTrue(z["orders__amount__max"].isna().all())

    def test_orphan_child_rows_are_reported(self):
        c, o, p = base_tables(); o.loc[:int(.1 * len(o)), "customer_id"] = 99999
        plan = MT.propose({"customers": c, "orders": o, "products": p})
        self.assertTrue(any("have no matching record" in w for w in plan["warnings"]))

    def test_categorical_columns_become_shares_of_the_mix(self):
        c, o, p = base_tables(); _, m = merged({"customers": c, "orders": o, "products": p})
        sh = [x for x in m.columns if "category__share_" in x]; self.assertEqual(len(sh), 2)             # 3 levels -> 2 shares
        self.assertTrue(m[sh].apply(lambda s: s.between(0, 1).all()).all())
        self.assertFalse(any(x.endswith("category__nunique") for x in m.columns))

    def test_date_columns_give_recency_tenure_and_span(self):
        c, o, p = base_tables(n=50); o["order_date"] = (pd.Timestamp("2024-01-01") + pd.to_timedelta(np.random.RandomState(3).randint(0, 365, len(o)), unit="D")).astype(str)
        _, m = merged({"customers": c, "orders": o, "products": p})
        for k in ("recency_days", "tenure_days", "span_days"):
            self.assertIn(f"orders__order_date__{k}", m.columns)
        self.assertTrue((m["orders__order_date__recency_days"] >= 0).all()); self.assertTrue((m["orders__order_date__span_days"] <= m["orders__order_date__tenure_days"]).all())

    def test_row_count_always_equals_the_base_table(self):
        c, o, p = base_tables()
        for t in (None, "amount", "churn"):
            plan, m = merged({"customers": c, "orders": o, "products": p}, target=t); self.assertEqual(len(m), len(({"customers": c, "orders": o, "products": p})[plan["base"]]))


class TestReading(unittest.TestCase):
    def test_bom_semicolon_and_latin1_files_are_read(self):
        d = Path(tempfile.mkdtemp())
        (d / "a.csv").write_bytes("\ufeffcustomer_id;name\n1;Zoë\n2;Ana\n".encode("utf-8"))
        (d / "b.csv").write_bytes("customer_id,city\n1,Málaga\n2,Cádiz\n".encode("latin-1"))
        a, b = MT.read_table(d / "a.csv"), MT.read_table(d / "b.csv")
        self.assertEqual(list(a.columns), ["customer_id", "name"]); self.assertEqual(a.name[0], "Zoë"); self.assertEqual(b.city[0], "Málaga")
        self.assertEqual(set(MT.load_tables([str(d)])), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
