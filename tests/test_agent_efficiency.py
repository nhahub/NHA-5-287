"""Tool-calling efficiency of the assistant: batching, repeat suppression, forced final answer, column-name repair, lean payloads."""
import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from nexusml import llm
from nexusml.assistant import agent as A, tools as T
from nexusml.evaluation.stub_llm import make_stub

FRAME = {"analysis_type": "general_exploration", "problem_type": "regression", "target": "Total Revenue", "date_column": None, "segment_columns": [], "kpis": []}


def frame():
    r = np.random.RandomState(1); n = 120
    return pd.DataFrame({"Total Revenue": r.gamma(3, 100, n), "region": r.choice(list("NSEW"), n), "units": r.poisson(6, n).astype(float), "price": r.uniform(5, 50, n)})


def run(msg, step_fn, df=None, **kw):
    seen = []

    def wrapped(pl):
        seen.append(pl)
        return step_fn(pl)
    with mock.patch.object(llm, "ask_json", side_effect=make_stub(FRAME, {"assistant_step": wrapped})):
        r = A.respond(msg, None, T.ToolContext(frame() if df is None else df, "sales"), **kw)
    return r, seen


def act(tool, **a):
    return {"tool": tool, "args": a}


class TestBatching(unittest.TestCase):
    def test_independent_tools_run_in_one_round_trip(self):
        step = lambda pl: {"reply": "See below."} if pl["observations"] else {"actions": [act("overview"), act("missing"), act("outliers")]}
        r, seen = run("health check", step)
        self.assertEqual(r["calls"], 2)                                                  # one batch + the answer (was 4)
        self.assertEqual([s["tool"] for s in r["steps"]], ["overview", "missing", "outliers"])
        self.assertEqual(r["stats"]["tool_runs"], 3); self.assertEqual(r["stats"]["tool_batches"], 1)

    def test_batch_is_capped_and_the_model_is_told(self):
        calls = [act("distribution", column=c) for c in ("Total Revenue", "units", "price", "region", "units")]
        step = lambda pl: {"reply": "ok"} if pl["observations"] else {"actions": calls + [act("overview")]}
        r, seen = run("x", step)
        self.assertEqual(r["stats"]["tool_runs"], A.MAX_BATCH)
        self.assertLessEqual(len(r["steps"]), A.MAX_BATCH)
        step2 = lambda pl: {"reply": "ok"} if "notes" in pl else {"actions": calls + [act("overview")]} if not pl["observations"] else {"action": act("missing")}
        r2, seen2 = run("x", step2)
        self.assertTrue(any("Only the first" in n for p in seen2 for n in p.get("notes", [])))

    def test_single_action_protocol_still_works(self):
        step = lambda pl: {"reply": "ok"} if pl["observations"] else {"action": act("overview")}
        r, _ = run("overview", step)
        self.assertEqual(r["calls"], 2); self.assertEqual([s["tool"] for s in r["steps"]], ["overview"])

    def test_action_and_reply_together_is_rejected_with_a_reason(self):
        with self.assertRaises(ValueError):
            A.validate_step({"actions": [act("overview")], "reply": "hi"})
        with self.assertRaises(ValueError):
            A.validate_step({"actions": "overview"})
        self.assertEqual(A.validate_step({"action": act("overview"), "actions": [act("missing")]})["actions"][0]["tool"], "overview")


class TestWaste(unittest.TestCase):
    def test_a_repeated_successful_call_is_not_run_again(self):
        step = lambda pl: {"reply": "ok"} if pl["observations"] else {"actions": [act("overview"), act("overview"), act("missing")]}
        r, _ = run("x", step)
        self.assertEqual(r["stats"]["tool_runs"], 2); self.assertEqual(r["stats"]["skipped_repeats"], 1)
        self.assertEqual(len(r["artifacts"]), len({a["title"] for a in r["artifacts"]}))      # the person is not shown the same table twice

    def test_equivalent_spellings_of_a_call_count_as_the_same_call(self):
        self.assertEqual(T.call_key("outliers", {"top": 5, "method": "IQR"}), T.call_key("outliers", {"method": "iqr", "top": 5}))
        self.assertNotEqual(T.call_key("outliers", {"top": 5}), T.call_key("outliers", {"top": 6}))

    def test_a_failing_call_is_not_retried_and_the_model_must_answer(self):
        step = lambda pl: {"reply": "I could not check that."} if pl.get("must_reply_now") else {"action": act("outliers", columns=["ghost"])}
        r, seen = run("x", step)
        self.assertLessEqual(r["calls"], 3); self.assertEqual(r["text"], "I could not check that.")      # the old loop used all 6 and gave no answer
        self.assertTrue(seen[-1].get("must_reply_now")); self.assertEqual(seen[-1]["tools"], [])

    def test_last_step_is_reply_only_and_never_runs_a_tool(self):
        names = ["overview", "missing", "outliers", "correlations", "duplicates", "value_counts"]
        counter = {"i": 0}

        def step(pl):
            if pl.get("must_reply_now"):
                return {"reply": "Here is what I found."}
            counter["i"] += 1
            t = names[counter["i"] - 1]
            return {"action": act(t, column="region") if t == "value_counts" else act(t)}
        r, seen = run("x", step, max_steps=4)
        self.assertEqual(r["text"], "Here is what I found."); self.assertEqual(r["stats"]["tool_runs"], 3)    # 3 tool steps + a forced answer, nothing thrown away
        self.assertTrue(seen[-1]["must_reply_now"] and seen[-1]["steps_left"] == 1)

    def test_a_model_that_ignores_the_reply_only_step_gets_a_summary_not_a_wasted_tool_run(self):
        r, _ = run("x", lambda pl: {"action": act("overview" if not pl["observations"] else "missing" if len(pl["observations"]) == 1 else "outliers")}, max_steps=3)
        self.assertEqual(r["stats"]["tool_runs"], 2); self.assertIn("here is what I did find", r["text"])


class TestColumnRepair(unittest.TestCase):
    def setUp(self):
        self.ctx = T.ToolContext(frame())

    def test_punctuation_and_spacing_differences_are_repaired_and_reported(self):
        for name in ("total_revenue", "totalrevenue", "Total-Revenue", "total.revenue"):
            res = T.run_tool(self.ctx, "distribution", {"column": name})
            self.assertEqual(res.summary["column_names_corrected"], [f"'{name}' -> 'Total Revenue'"]); self.assertTrue(any("Corrected" in n for n in res.notes))

    def test_case_only_differences_resolve_silently_as_before(self):
        self.assertNotIn("column_names_corrected", T.run_tool(self.ctx, "distribution", {"column": "TOTAL REVENUE"}).summary)

    def test_a_typo_in_a_long_name_is_repaired_but_short_names_are_not_guessed(self):
        res = T.run_tool(self.ctx, "distribution", {"column": "Total Revnue"}); self.assertIn("'Total Revnue' -> 'Total Revenue'", res.summary["column_names_corrected"])
        with self.assertRaises(T.ToolError):                      # 'prise' is too short to guess safely: it goes back to the model with the closest matches
            T.run_tool(self.ctx, "distribution", {"column": "prise"})

    def test_unknown_ambiguous_and_hostile_names_are_still_errors(self):
        df = pd.DataFrame({"unit_price": np.arange(30.0), "unit_prize": np.arange(30.0) % 7, "g": list("ab") * 15})
        for bad in ("ghost", "price; import os", "unit_pri"):
            with self.assertRaises(T.ToolError, msg=bad):
                T.run_tool(T.ToolContext(df), "distribution", {"column": bad})

    def test_exact_names_report_nothing(self):
        self.assertNotIn("column_names_corrected", T.run_tool(self.ctx, "distribution", {"column": "units"}).summary)


class TestLeanPayload(unittest.TestCase):
    def test_compact_catalog_is_much_smaller_and_complete(self):
        full, small = json.dumps(T.catalog(True)), json.dumps(T.catalog_compact(True))
        self.assertLess(len(small), 0.65 * len(full)); self.assertEqual(len(T.catalog_compact(True)), len(T.catalog(True)))
        self.assertTrue(all(any(t.startswith(n["name"] + "(") for t in T.catalog_compact(True)) for n in T.catalog(True)))

    def test_timing_noise_is_not_sent_and_old_observations_are_shortened(self):
        big = {"tool": "overview", "args": {}, "title": "t", "summary": {"rows": 5, "blob": "x" * 5000, "_seconds": 0.2}, "step": 1}
        old = A._obs_payload([big], newest_from=2)[0]["summary"]; new = A._obs_payload([big], newest_from=1)[0]["summary"]
        self.assertTrue(old.get("truncated")); self.assertNotIn("_seconds", new); self.assertEqual(new["rows"], 5)

    def test_payload_has_no_rows(self):
        _, seen = run("hi", lambda pl: {"reply": "hello"})
        self.assertNotIn("rows_data", json.dumps(seen[0])); self.assertIsInstance(seen[0]["tools"][0], str)


if __name__ == "__main__":
    unittest.main()
