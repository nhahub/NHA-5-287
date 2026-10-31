"""Tests for the v4 features. Run: python -m unittest discover -s tests   (use NEXUSML_VALIDATE=1 to also type-check every node output)"""
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ["NEXUSML_BACKEND"] = "builtin"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nexusml import (claims, config, drilldown, engine, fairness, featurizers, llm, modelio, monitoring, multitable, rca, runlog, runstore,
                      schemas, stats_tests, targets, tracking, uncertainty)
from nexusml.agents import bus, supervisor
from nexusml.evaluation import datasets as D
from nexusml.evaluation import harness as H
from nexusml.evaluation.stub_llm import make_stub

HR = Path(__file__).resolve().parent / "fixtures"


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": self.d})
        self.env.start()

    def tearDown(self):
        self.env.stop()


class TestEngineParallelism(unittest.TestCase):
    def build(self, par):
        import operator
        from typing import Annotated, List, TypedDict

        def _last(a, b):
            return b if b is not None else a

        class S(TypedDict, total=False):
            log: Annotated[List[str], operator.add]
            status: Annotated[str, _last]

        g = engine.StateGraph(S)
        for n in ("prep", "a", "b", "c", "join", "loop", "end"):
            def f(st, n=n):
                time.sleep(0.15 if n in "abc" else 0)
                return {"log": [n], "status": n}
            g.add_node(n, f)
        g.add_edge(engine.START, "prep")
        for x in "abc":
            g.add_edge("prep", x)
        g.add_edge(["a", "b", "c"], "join")
        g.add_edge("join", "loop")
        g.add_conditional_edges("loop", lambda st: "again" if st["log"].count("loop") < 2 else "done", {"again": "loop", "done": "end"})
        g.add_edge("end", engine.END)
        return g.compile(checkpointer=engine.MemorySaver())

    def run_(self, par):
        with mock.patch.dict(os.environ, {"NEXUSML_PARALLEL": "1" if par else "0", "NEXUSML_ARTIFACTS": tempfile.mkdtemp()}):
            cg = self.build(par)
            cfg = {"configurable": {"thread_id": "t"}}
            t0 = time.time()
            list(cg.stream({"log": []}, config=cfg))
            return cg.get_state(cfg).values["log"], time.time() - t0

    def test_fan_out_join_loop_identical_and_faster(self):
        a, ta = self.run_(True)
        b, tb = self.run_(False)
        self.assertEqual(a, b)
        self.assertEqual(a, ["prep", "a", "b", "c", "join", "loop", "loop", "end"])
        self.assertLess(ta, tb - 0.15)


class TestRunLog(Tmp):
    def test_seed_log_and_replay(self):
        runlog.start("r1", 123, {"x": 1})
        self.assertEqual(config.RANDOM_STATE, 123)
        runlog.llm_event("p", "h1", {"a": 1}, None, {"model": "m", "seconds": 1, "tokens": 5})
        runlog.decision("route_x", "ok")
        s = runlog.summary("r1")
        self.assertEqual((s["llm_calls"], s["seed"], s["tokens"]), (1, 123, 5))
        self.assertEqual(s["routes"], [("route_x", "ok")])
        with mock.patch.dict(os.environ, {"NEXUSML_REPLAY": str(runlog.path("r1"))}):
            runlog.start("r2", 1)
            self.assertEqual(runlog.replay_lookup("p", "h1"), {"a": 1})
            with self.assertRaises(RuntimeError):
                runlog.replay_lookup("p", "unknown")
            runlog.end_run()
            self.assertFalse(runlog.replay_active())                  # replay must not leak into later calls
        config.RANDOM_STATE = 42


class TestLLMLayer(Tmp):
    def test_fallback_provider_and_cache(self):
        calls = []

        def post(body, provider=None):
            calls.append(provider["name"] if provider else "primary")
            if provider is None:
                raise OSError("down")
            return {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {"total_tokens": 3}}
        env = {"GROQ_API_KEY": "k", "OTHER_KEY": "k2", "NEXUSML_LLM_CACHE": "1",
               "NEXUSML_FALLBACK_PROVIDERS": json.dumps([{"name": "other", "base_url": "https://x/v1", "api_key_env": "OTHER_KEY", "model": "m"}])}
        with mock.patch.dict(os.environ, env), mock.patch.object(llm, "_post", side_effect=post), mock.patch.object(config, "LLM_CACHE", True), \
                mock.patch.object(config, "LLM_MAX_RETRIES", 0), mock.patch("time.sleep"):
            d, err, rec = llm.ask_json("s", "p", "t")
            self.assertEqual((d, err), ({"ok": True}, None))
            self.assertIn("other", calls)
            n = len(calls)
            d2, _, rec2 = llm.ask_json("s", "p", "t")                 # identical call: served from the cache
            self.assertTrue(rec2.get("cached"))
            self.assertEqual(len(calls), n)

    def test_concurrency_gate_limits_parallel_calls(self):
        active, peak, lock = [0], [0], threading.Lock()

        def post(body, provider=None):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return {"choices": [{"message": {"content": "{}"}}]}
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}), mock.patch.object(llm, "_post", side_effect=post), mock.patch.object(config, "LLM_CONCURRENCY", 2):
            llm._SEM["sem"] = None
            ts = [threading.Thread(target=llm.ask_json, args=("s", f"p{i}", "t")) for i in range(6)]
            [t.start() for t in ts]
            [t.join() for t in ts]
        self.assertLessEqual(peak[0], 2)

    def test_images_never_sent_by_default(self):
        self.assertFalse(llm.supports_images())


class TestClaims(unittest.TestCase):
    EV = [{"id": "M01", "text": "Headline held-out roc_auc = 0.8116 (95% CI 0.73-0.88).", "data": {"v": 0.8116}},
          {"id": "D01", "text": "Overtime: 30.5% attrition vs 10.4% without. Lift 4.1x.", "data": {"a": 0.305, "b": 0.104}}]

    def chk(self, text, ids, roi=False):
        return claims.check_text(text, [e for e in self.EV if e["id"] in ids], self.EV, roi)

    def test_supported_and_derived_numbers_pass(self):
        self.assertEqual(self.chk("AUC 0.81", ["M01"]), [])
        self.assertEqual(self.chk("30.5% vs 10.4%, a 20.1-point gap", ["D01"]), [])
        self.assertEqual(self.chk("about 4x the base rate", ["D01"]), [])

    def test_unsupported_number_and_roi_wording_rejected(self):
        self.assertTrue(self.chk("AUC is 0.92", ["M01"]))
        self.assertTrue(self.chk("this will maximize ROI", ["D01"]))
        self.assertEqual(self.chk("this will maximize ROI", ["D01"], roi=True), [])
        self.assertTrue(self.chk("this is guaranteed to work", ["D01"]))

    def test_gate_is_strict_once_then_annotates(self):
        g = claims.Gate()
        items = [("a", ["M01"], "AUC is 0.99")]
        with self.assertRaises(ValueError):
            g.check(items, self.EV, False)
        self.assertIn("a", g.check(items, self.EV, False))           # second call: annotate, do not kill the run


class TestStatsNarrative(unittest.TestCase):
    def test_sentence_follows_the_flag(self):
        rng = np.random.RandomState(1)
        df = pd.DataFrame({"x": rng.normal(size=400), "g": rng.choice(list("ab"), 400), "y": rng.choice(["Y", "N"], 400)})
        rs = stats_tests.finalize([stats_tests.execute({"id": "T1", "test": "group_comparison", "params": {"measure": "x", "group": "g"}}, df),
                                   stats_tests.execute({"id": "T2", "test": "chi_square", "params": {"a": "g", "b": "y"}}, df)])
        for r in rs:
            self.assertFalse(r["significant"])
            self.assertNotIn("are associated", r["interpretation"])
            self.assertNotIn(" differs", r["interpretation"])
        df.loc[df.g == "a", "x"] += 1.5
        r = stats_tests.finalize([stats_tests.execute({"id": "T1", "test": "group_comparison", "params": {"measure": "x", "group": "g"}}, df)])[0]
        self.assertTrue(r["significant"])
        self.assertIn("differs", r["interpretation"])


class TestUncertaintyAndCost(unittest.TestCase):
    def test_bootstrap_ci_contains_point(self):
        r = np.random.RandomState(0)
        y = r.randint(0, 2, 300)
        p = np.clip(y * 0.4 + r.uniform(size=300) * 0.6, 0, 1)
        ci = uncertainty.bootstrap_ci("roc_auc", y, (p > .5).astype(int), p)
        self.assertLessEqual(ci["lo"], ci["value"])
        self.assertLessEqual(ci["value"], ci["hi"])

    def test_cost_threshold_moves_with_costs(self):
        from sklearn.linear_model import LogisticRegression
        r = np.random.RandomState(0)
        X = pd.DataFrame({"a": r.normal(size=600)})
        y = (X.a + r.normal(0, 1, 600) > 0.8).astype(int).values
        est = LogisticRegression().fit(X[:400], y[:400])
        hi = uncertainty.cost_threshold(est, X[:400], y[:400], X[400:], y[400:], 20, 1)
        lo = uncertainty.cost_threshold(est, X[:400], y[:400], X[400:], y[400:], 1, 20)
        self.assertLess(hi["threshold"], lo["threshold"])           # expensive misses -> flag more -> lower cutoff


class TestAnalysisDepth(unittest.TestCase):
    def test_interactions_and_driver_groups_on_planted_data(self):
        c = D.planted_classification(Path(tempfile.mkdtemp()))
        df = pd.read_csv(c["path"])
        m = rca.measure_vector(df, "attrition", "classification", "Yes", None)
        res = rca.interaction_segments(df, m, ["overtime", "tenure", "income", "region", "age"])
        self.assertTrue(any({"overtime", "tenure"} == {r["a"], r["b"]} for r in res["items"]))
        df["income2"] = df["income"] * 0.9 + np.random.RandomState(1).normal(0, 0.1, len(df))
        g = rca.driver_groups(df, ["income", "income2", "age"])
        self.assertTrue(any({"income", "income2"} <= set(x["members"]) for x in g))

    def test_zero_event_group_is_not_called_significant(self):
        r = np.random.RandomState(2)
        df = pd.DataFrame({"a": r.choice(["x", "y"], 600), "b": r.choice(["p", "q"], 600), "y": (r.uniform(size=600) < 0.15).astype(int)})
        df.loc[(df.a == "x") & (df.b == "p") & (df.index % 3 == 0), "y"] = 0
        m = rca.measure_vector(df.assign(y=df.y.map({1: "Yes", 0: "No"})), "y", "classification", "Yes", None)
        self.assertEqual(rca.interaction_segments(df.assign(y=df.y.map({1: "Yes", 0: "No"})), m, ["a", "b"])["items"], [])


class TestFairness(unittest.TestCase):
    def test_detection(self):
        got = fairness.detect_sensitive(["Age", "Gender", "MaritalStatus", "page_views", "usage", "trace_id", "Income", "birth_date"])
        self.assertEqual(sorted(got), ["Age", "Gender", "MaritalStatus", "birth_date"])

    def test_group_audit_flags_disparity(self):
        raw = pd.DataFrame({"g": ["a"] * 100 + ["b"] * 100})
        y = np.array([1, 0] * 100)
        pred = np.array([1] * 100 + [0] * 90 + [1] * 10)
        a = fairness.group_audit(raw, y, pred, ["g"], {"g": True})[0]
        self.assertTrue(a["flag"])


class TestTargets(unittest.TestCase):
    def test_parse_and_threshold_rules(self):
        s = targets.parse({"metric": "recall", "min": 0.6, "subject_to": {"metric": "precision", "min": 0.5}}, "classification")
        self.assertEqual(s["secondary"]["metric"], "precision")
        with self.assertRaises(ValueError):
            targets.parse({"metric": "r2", "min": 0.5}, "classification")
        with self.assertRaises(ValueError):
            targets.parse({"metric": "rmse"}, "regression")                  # needs a max

    def test_infeasible_target_reports_not_met(self):
        r = np.random.RandomState(0)
        y = r.randint(0, 2, 400)
        p = np.clip(y * 0.3 + r.uniform(size=400) * 0.7, 0, 1)
        s = targets.parse({"metric": "recall", "min": 0.99, "subject_to": {"metric": "precision", "min": 0.99}}, "classification")
        thr = targets.choose_threshold(s, y, p)
        self.assertFalse(targets.evaluate(s, y, p, thr)["met"])


class TestMessageBusAndSupervisor(Tmp):
    def test_bus_runs_service_and_enforces_cap(self):
        c = D.planted_classification(Path(self.d))
        st = {"dataset_path": c["path"], "messages": []}
        msgs, res, used = bus.request(st, "Root Cause Analyst", "Statistician", "statistician/test_association", {"a": "overtime", "b": "attrition"}, 0)
        self.assertEqual(used, 1)
        self.assertIn("overtime", res["summary"])
        self.assertEqual([m["kind"] for m in msgs], ["request", "response"])
        _, res2, used2 = bus.request(st, "x", "y", "statistician/test_association", {"a": "overtime", "b": "attrition"}, config.MAX_AGENT_REQUESTS)
        self.assertIsNone(res2)
        self.assertEqual(used2, config.MAX_AGENT_REQUESTS)

    def test_supervisor_default_when_not_narrow_or_llm_fails(self):
        st = {"user_request": "Why do customers churn and what should we do?", "flexible": True}
        self.assertEqual(supervisor.plan_phases(st, "classification")["skip_phases"], [])
        st2 = {"user_request": "just check outliers in sales", "flexible": True}
        with mock.patch.object(supervisor, "ask", side_effect=RuntimeError("llm down")):
            r = supervisor.plan_phases(st2, "descriptive")
        self.assertEqual((r["skip_phases"], r["source"]), ([], "default"))

    def test_supervisor_skips_by_majority_and_code_overrides(self):
        st = {"user_request": "just show the trend, no model", "flexible": True}
        out = {"run": ["trends"], "rationale": "trend only"}
        with mock.patch.object(supervisor, "ask", return_value=(out, None, {})):
            r = supervisor.plan_phases(st, "descriptive")
        self.assertIn("modeling", r["skip_phases"])
        st["user_request"] = "just predict who will leave"                 # code rule: prediction requests keep modeling
        with mock.patch.object(supervisor, "ask", return_value=({"run": ["kpis"], "rationale": "x"}, None, {})):
            r = supervisor.plan_phases(st, "classification")
        self.assertNotIn("modeling", r["skip_phases"])


class TestSchemas(unittest.TestCase):
    def test_validation(self):
        schemas.validate_update("n", {"evidence": [{"id": "S01", "source": "s", "text": "t", "data": {}}]})
        with self.assertRaises(schemas.SchemaError):
            schemas.validate_update("n", {"not_a_key": 1})
        with self.assertRaises(schemas.SchemaError):
            schemas.validate_update("n", {"evidence": [{"id": "S01", "source": "s", "text": "t", "data": {}}] * 2})
        with self.assertRaises(schemas.SchemaError):
            schemas.validate_update("n", {"messages": [{"kind": "oops"}]})


class TestMultiTable(Tmp):
    def tables(self):
        r = np.random.RandomState(0)
        n = 300
        cust = pd.DataFrame({"customer_id": range(n), "region": r.choice(list("NSEW"), n)})
        orders = pd.DataFrame({"order_id": range(1500), "customer_id": r.randint(0, n, 1500), "product_id": r.randint(0, 20, 1500), "amount": r.gamma(2, 40, 1500)})
        prod = pd.DataFrame({"product_id": range(20), "price": r.uniform(5, 90, 20)})
        junk = pd.DataFrame({"customer_id": r.randint(5000, 6000, 100), "n": 1})
        dup = pd.DataFrame({"region": r.choice(list("NSEW"), 80), "x": r.rand(80)})
        return {"customers": cust, "orders": orders, "products": prod, "junk": junk, "regions": dup}

    def test_guards_and_chained_join_keep_row_count(self):
        t = self.tables()
        plan = multitable.propose(t)
        self.assertEqual(plan["base"], "customers")
        self.assertEqual({j["table"] for j in plan["joins"]}, {"orders", "products"})
        reasons = " ".join(r["rejected"] for r in plan["rejected"])
        self.assertIn("many-to-many", reasons)
        self.assertIn("match", reasons)
        with self.assertRaises(multitable.ApprovalRequired):
            multitable.apply(t, plan)
        plan["approved"] = True
        m = multitable.apply(t, plan)
        self.assertEqual(len(m), len(t["customers"]))
        self.assertTrue(any(c.startswith("orders__") for c in m.columns))
        self.assertFalse(any(c.endswith("order_id__mean") for c in m.columns))      # identifiers are not aggregated

    def test_prepare_requires_approval(self):
        c = D.multitable_churn(Path(self.d))
        with self.assertRaises(multitable.ApprovalRequired):
            multitable.prepare(c["paths"], approve=lambda p: False)
        path, plan = multitable.prepare(c["paths"], approve=lambda p: True)
        self.assertEqual(len(pd.read_csv(path)), c["truth"]["rows"])


class TestModalities(Tmp):
    def test_detection_and_pipeline_features(self):
        c = D.multimodal(Path(self.d))
        df = pd.read_csv(c["path"])
        mods = featurizers.modalities(df.drop(columns=["outcome"]))
        self.assertEqual(mods, {"text": ["comment"], "image": ["photo"]})
        featurizers.set_image_root(self.d)
        from nexusml import modeling
        pre = modeling.build_preprocessor(df.drop(columns=["outcome"]))
        Xt = pre.fit_transform(df.drop(columns=["outcome"]))
        self.assertIsInstance(Xt, np.ndarray)
        self.assertGreater(Xt.shape[1], 30)

    def test_duplicate_leakage_detected(self):
        a = pd.DataFrame({"t": [f"great product number {i} works well for me" for i in range(60)]})
        rep = featurizers.duplicate_report(a, a.iloc[:20], {"text": ["t"], "image": []})
        self.assertGreater(rep["t"]["exact_dup_share"], 0.9)

    def test_unreadable_image_becomes_missing_not_a_crash(self):
        f = featurizers.ImageFeaturizer().fit(pd.DataFrame({"p": ["nope.png"]}))
        self.assertTrue(np.isnan(f.transform(pd.DataFrame({"p": ["nope.png"]}))).all())


class TestDrilldown(Tmp):
    def test_whitelist_and_execution(self):
        c = D.planted_classification(Path(self.d))
        cols = list(pd.read_csv(c["path"]).columns)
        acts = drilldown.validate_actions([{"action": "segment_profile", "column": "region"}, {"action": "drop_table", "column": "x"},
                                           {"action": "test", "a": "overtime", "b": "nope"}], cols)
        self.assertEqual(acts, [{"action": "segment_profile", "column": "region"}])
        st = {"dataset_path": c["path"], "analysis_plan": {"target": "attrition", "problem_type": "classification"}, "evidence": [], "kpi_results": {},
              "data_summary": {"positive_class": "Yes"}}
        ev = drilldown.run(st, acts)
        self.assertEqual(ev[0]["id"], "F01")
        self.assertIn("region", ev[0]["text"])


class TestMonitoringAndStores(Tmp):
    def test_drift_detected(self):
        r = np.random.RandomState(0)
        ref = pd.DataFrame({"a": r.normal(0, 1, 800), "b": r.choice(list("xyz"), 800)})
        same = pd.DataFrame({"a": r.normal(0, 1, 800), "b": r.choice(list("xyz"), 800)})
        moved = pd.DataFrame({"a": r.normal(2, 1, 800), "b": r.choice(list("xyz"), 800)})
        self.assertFalse(monitoring.check_drift(ref, same)["alert"])
        rep = monitoring.check_drift(ref, moved)
        self.assertTrue(rep["alert"])
        self.assertEqual(rep["major"], ["a"])

    def test_audit_log_and_runs(self):
        runstore.audit("plan_approved", "r9", {"drop": ["x"]}, user="ann")
        log = runstore.audit_log("r9")
        self.assertEqual((log[0]["user"], log[0]["action"]), ("ann", "plan_approved"))
        runstore.record_run({"experiment_id": "e1", "run_id": "e1", "dataset_path": "d.csv", "analysis_plan": {"problem_type": "classification", "target": "t"},
                             "metric_ci": {"value": 0.8, "lo": 0.7, "hi": 0.9}, "quality_passed": True, "seed": 1})
        self.assertEqual(runstore.list_runs()[0]["heldout"], 0.8)

    def test_tampered_model_file_is_refused(self):
        p = Path(self.d) / "m.joblib"
        modelio.dump({"a": 1}, p)
        self.assertEqual(modelio.load(p), {"a": 1})
        with open(p, "ab") as f:
            f.write(b"x")
        with self.assertRaises(modelio.UntrustedModelError):
            modelio.load(p)
        q = Path(self.d) / "naked.joblib"
        import joblib
        joblib.dump({"b": 2}, q)
        with self.assertRaises(modelio.UntrustedModelError):
            modelio.load(q)
        self.assertEqual(modelio.load(q, trusted=True), {"b": 2})


class FakeMlflow:
    """A small stand-in that records the calls our tracking code makes (the real MLflow cannot be installed in the test sandbox)."""
    def __init__(self):
        self.calls, self.versions, self.aliases, self.tags = [], 0, {}, {}
        self._stack = []
        m = types.ModuleType("mlflow")
        m.__version__ = "fake"
        m.set_tracking_uri = lambda u: self.calls.append(("uri", u))
        m.log_params = lambda d: self.calls.append(("params", len(d)))
        m.log_metrics = lambda d: self.calls.append(("metrics", len(d)))
        m.log_metric = lambda k, v, step=None: self.calls.append(("metric_step", k, step))
        m.set_tags = lambda d: self.calls.append(("tags", len(d)))
        m.log_artifact = lambda p: self.calls.append(("artifact", Path(p).name))
        outer = self

        class Run:
            def __init__(s, nested):
                s.info = types.SimpleNamespace(run_id=f"run{len(outer.calls)}")
                s.nested = nested

            def __enter__(s):
                outer.calls.append(("start_run", "nested" if s.nested else "root"))
                return s

            def __exit__(s, *a):
                return False
        m.start_run = lambda experiment_id=None, run_name=None, nested=False: Run(nested)

        def reg(uri, name):
            outer.versions += 1
            outer.calls.append(("register", name, outer.versions, uri))
            return types.SimpleNamespace(version=outer.versions)
        m.register_model = reg
        sk = types.ModuleType("mlflow.sklearn")
        self.log_kwargs = []
        sk.log_model = lambda **k: (outer.calls.append(("log_model", sorted(k))), outer.log_kwargs.append(k), types.SimpleNamespace(model_uri="models:/m-abc123"))[2]
        m.sklearn = sk
        tr = types.ModuleType("mlflow.tracking")

        class Client:
            def __init__(s, *a, **k): pass                                    # the real MlflowClient accepts tracking_uri=...
            def get_experiment_by_name(s, n): return None
            def create_experiment(s, n, artifact_location=None): outer.calls.append(("create_experiment", n)); return "1"
            def set_model_version_tag(s, name, v, k, val): outer.tags[(name, v, k)] = val
            def set_registered_model_alias(s, name, alias, v): outer.aliases[(name, alias)] = v
            def search_model_versions(s, q): return [types.SimpleNamespace(version=1, status="READY", aliases=["staging"], run_id="r1", tags={})]
            def search_experiments(s): return [types.SimpleNamespace(experiment_id="1", name="churn | Attrition")]
            def search_runs(s, ids, max_results=1000): return [1, 2, 3]
            def search_logged_models(s, experiment_ids=None, max_results=200):
                return [types.SimpleNamespace(name="model", status=types.SimpleNamespace(name="READY"), model_id="m-1")]
            def search_registered_models(s, max_results=200): return [types.SimpleNamespace(name="churn-Attrition")]
        tr.MlflowClient = Client
        self.modules = {"mlflow": m, "mlflow.sklearn": sk, "mlflow.tracking": tr}


class TestMlflowTracking(Tmp):
    def test_logs_runs_model_and_registry_alias(self):
        from sklearn.linear_model import LogisticRegression
        fake = FakeMlflow()
        state = {"analysis_plan": {"problem_type": "classification", "target": "t", "drop_columns": []}, "analysis_type": "churn",
                 "dataset_profile": {"n_rows": 10, "quality": {"score": 90}}, "candidate_metrics": [
                     {"name": "LR", "tuned": False, "select_source": "cv", "select_score": 0.8, "cv_scores": [0.8, 0.81], "roc_auc": 0.8}],
                 "best_model_name": "LR", "primary_metric": "roc_auc", "experiment_id": "e_test", "user_request": "q", "evidence": [], "charts": [], "llm_calls": [],
                 "quality_passed": True}
        X = pd.DataFrame({"a": [0, 1, 0, 1]})
        modelio_path = Path(self.d) / "b.joblib"
        modelio.dump({"pipeline": LogisticRegression().fit(X, [0, 1, 0, 1])}, modelio_path)
        with mock.patch.dict(sys.modules, fake.modules):
            self.assertTrue(tracking.mlflow_status()["active"])
            info = tracking.log_analysis(state, "models/x.joblib")
            out = tracking.mirror_mlflow(info["run_id"], modelio_path, info["registered"]["model_name"], "Staging")
        self.assertTrue(out["ok"], out)
        kinds = [c[0] for c in fake.calls]
        for k in ("create_experiment", "start_run", "params", "metrics", "log_model", "register"):
            self.assertIn(k, kinds)
        # v5: every compared model is its OWN top-level run (sortable in the MLflow table): the summary run + 1 candidate run
        self.assertEqual(fake.calls.count(("start_run", "root")), 2)
        name = info["registered"]["model_name"]
        self.assertEqual(fake.aliases[(name, "staging")], "1")
        self.assertEqual(fake.tags[(name, "1", "stage")], "Staging")
        self.assertIn("mlflow_run_id", tracking.get_run(info["run_id"]))

    def test_failure_is_reported_precisely_not_silently(self):
        fake = FakeMlflow()
        fake.modules["mlflow"].start_run = mock.Mock(side_effect=PermissionError("read-only store"))
        with mock.patch.dict(sys.modules, fake.modules):
            tracking.new_run("exp", "r", {}, {})
            rid = tracking.list_runs()[0]["run_id"]
            out = tracking.mirror_mlflow(rid, None)
        self.assertFalse(out["ok"])
        self.assertIn("PermissionError", out["message"])

    def state_and_bundle(self):
        from sklearn.linear_model import LogisticRegression
        state = {"analysis_plan": {"problem_type": "classification", "target": "t", "drop_columns": []}, "analysis_type": "churn",
                 "dataset_profile": {"n_rows": 10, "quality": {"score": 90}}, "candidate_metrics": [
                     {"name": "LR", "tuned": False, "select_source": "cv", "select_score": 0.8, "cv_scores": [0.8, 0.81], "roc_auc": 0.8}],
                 "best_model_name": "LR", "primary_metric": "roc_auc", "experiment_id": "e_step", "user_request": "q", "evidence": [], "charts": [], "llm_calls": [],
                 "quality_passed": True}
        X = pd.DataFrame({"a": [0, 1, 0, 1]})
        p = Path(self.d) / "b2.joblib"
        modelio.dump({"pipeline": LogisticRegression().fit(X, [0, 1, 0, 1])}, p)
        return state, p

    def test_model_is_registered_through_the_uri_mlflow_returns(self):
        """MLflow 3 keeps models as 'logged models', so runs:/<id>/model is not reliable; the URI returned by log_model must be registered."""
        fake = FakeMlflow()
        state, bundle = self.state_and_bundle()
        with mock.patch.dict(sys.modules, fake.modules):
            info = tracking.log_analysis(state, "models/x.joblib")
            out = tracking.mirror_mlflow(info["run_id"], bundle, info["registered"]["model_name"])
        self.assertTrue(out["ok"], out)
        reg = [c for c in fake.calls if c[0] == "register"][0]
        self.assertEqual(reg[3], "models:/m-abc123")

    def test_failure_names_the_step_and_is_saved_on_the_run_then_cleared_on_success(self):
        fake = FakeMlflow()
        state, bundle = self.state_and_bundle()
        info = tracking.log_analysis(state, "models/x.joblib")
        rid = info["run_id"]
        good = fake.modules["mlflow"].log_params
        fake.modules["mlflow"].log_params = mock.Mock(side_effect=ValueError("bad param"))
        with mock.patch.dict(sys.modules, fake.modules):
            out = tracking.mirror_mlflow(rid, bundle, "m")
        self.assertFalse(out["ok"])
        self.assertIn("while logging parameters, metrics and tags", out["message"])
        self.assertIn("ValueError", tracking.get_run(rid)["mlflow_error"])
        fake.modules["mlflow"].log_params = good
        with mock.patch.dict(sys.modules, fake.modules):
            out = tracking.mirror_mlflow(rid, bundle, "m")                         # the retry the page offers
        self.assertTrue(out["ok"])
        r = tracking.get_run(rid)
        self.assertEqual(r["mlflow_error"], "")
        self.assertTrue(r["mlflow_run_id"])

    def test_a_model_that_cannot_be_logged_does_not_lose_the_run(self):
        fake = FakeMlflow()
        fake.modules["mlflow.sklearn"].log_model = mock.Mock(side_effect=RuntimeError("cannot pickle"))
        state, bundle = self.state_and_bundle()
        with mock.patch.dict(sys.modules, fake.modules):
            info = tracking.log_analysis(state, "models/x.joblib")
            out = tracking.mirror_mlflow(info["run_id"], bundle, info["registered"]["model_name"])
        self.assertTrue(out["ok"])
        self.assertIn("the model could not be logged", out["message"])
        r = tracking.get_run(info["run_id"])
        self.assertTrue(r["mlflow_run_id"])
        self.assertIn("could not be logged", r["mlflow_notes"])

    def test_model_is_logged_with_cloudpickle_so_the_skops_trust_check_cannot_fail_it(self):
        """Real MLflow 3 failure: its default skops format rejected our custom featurizer types, leaving the logged model with status 'Failed'."""
        fake = FakeMlflow()
        state, bundle = self.state_and_bundle()
        with mock.patch.dict(sys.modules, fake.modules):
            info = tracking.log_analysis(state, "models/x.joblib")
            out = tracking.mirror_mlflow(info["run_id"], bundle, info["registered"]["model_name"])
        self.assertTrue(out["ok"], out)
        self.assertEqual(fake.log_kwargs[0]["serialization_format"], "cloudpickle")
        self.assertEqual(len(fake.log_kwargs), 1)

    def test_falls_back_to_skops_with_exactly_the_pipelines_own_types(self):
        fake = FakeMlflow()
        fake.modules["mlflow.sklearn"].log_model = mock.Mock(side_effect=[RuntimeError("cloudpickle blocked"), types.SimpleNamespace(model_uri="models:/m-2")])
        skops = types.ModuleType("skops")
        sio = types.ModuleType("skops.io")
        sio.dumps = lambda est: b"data"
        sio.get_untrusted_types = lambda data=None: ["nexusml.featurizers.TextFeaturizer"]
        skops.io = sio
        state, bundle = self.state_and_bundle()
        with mock.patch.dict(sys.modules, {**fake.modules, "skops": skops, "skops.io": sio}):
            info = tracking.log_analysis(state, "models/x.joblib")
            out = tracking.mirror_mlflow(info["run_id"], bundle, info["registered"]["model_name"])
        self.assertTrue(out["ok"], out)
        second = fake.modules["mlflow.sklearn"].log_model.call_args_list[1].kwargs
        self.assertEqual(second["serialization_format"], "skops")
        self.assertEqual(second["skops_trusted_types"], ["nexusml.featurizers.TextFeaturizer"])

    def test_when_every_format_fails_the_message_gives_each_reason(self):
        fake = FakeMlflow()
        fake.modules["mlflow.sklearn"].log_model = mock.Mock(side_effect=RuntimeError("Untrusted types found in the file"))
        state, bundle = self.state_and_bundle()
        with mock.patch.dict(sys.modules, fake.modules):
            info = tracking.log_analysis(state, "models/x.joblib")
            out = tracking.mirror_mlflow(info["run_id"], bundle, info["registered"]["model_name"])
        self.assertTrue(out["ok"])                                                  # the run itself is still safely in MLflow
        self.assertIn("cloudpickle:", out["message"])
        self.assertIn("skops:", out["message"])
        self.assertIn("Untrusted types", out["message"])

    def test_every_compared_model_is_logged_as_a_model_not_only_the_winner(self):
        from sklearn.linear_model import LogisticRegression
        fake = FakeMlflow()
        state, bundle = self.state_and_bundle()
        X = pd.DataFrame({"a": [0, 1, 0, 1]})
        cand = {"LR": LogisticRegression().fit(X, [0, 1, 0, 1])}
        with mock.patch.dict(sys.modules, fake.modules):
            info = tracking.log_analysis(state, "models/x.joblib")
            out = tracking.mirror_mlflow(info["run_id"], bundle, info["registered"]["model_name"], candidates=cand)
        self.assertTrue(out["ok"], out)
        names = [k.get("name") for k in fake.log_kwargs]
        self.assertEqual(sorted(names), ["LR", "model"])                           # the winner (registered) + the compared candidate
        with mock.patch.dict(sys.modules, fake.modules), mock.patch.dict(os.environ, {"NEXUSML_MLFLOW_ALL_MODELS": "0"}):
            fake.log_kwargs.clear()
            info = tracking.log_analysis({**state, "experiment_id": "e_two"}, "models/x.joblib")
            tracking.mirror_mlflow(info["run_id"], bundle, info["registered"]["model_name"], candidates=cand)
        self.assertEqual([k.get("name") for k in fake.log_kwargs], ["model"])      # switch NEXUSML_MLFLOW_ALL_MODELS=0 keeps only the winner

    def test_the_message_says_exactly_what_was_logged(self):
        """So the user can match the app's numbers against what MLflow's own screens show."""
        from sklearn.linear_model import LogisticRegression
        fake = FakeMlflow()
        state, bundle = self.state_and_bundle()
        X = pd.DataFrame({"a": [0, 1, 0, 1]})
        cand = {"LR": LogisticRegression().fit(X, [0, 1, 0, 1])}
        with mock.patch.dict(sys.modules, fake.modules):
            info = tracking.log_analysis(state, "models/x.joblib")
            out = tracking.mirror_mlflow(info["run_id"], bundle, info["registered"]["model_name"], candidates=cand)
        self.assertEqual(out["counts"]["candidate_runs"], 1)
        self.assertEqual(out["counts"]["models"], 2)                               # the winner + the compared candidate
        self.assertIn("1 compared-model run(s)", out["message"])
        self.assertIn("2 model(s)", out["message"])
        self.assertIn("1 compared-model run(s)", tracking.get_run(info["run_id"])["mlflow_summary"])

    def test_a_candidate_that_cannot_be_logged_is_reported_but_the_run_and_winner_survive(self):
        from sklearn.linear_model import LogisticRegression
        fake = FakeMlflow()
        state, bundle = self.state_and_bundle()
        X = pd.DataFrame({"a": [0, 1, 0, 1]})
        cand = {"LR": LogisticRegression().fit(X, [0, 1, 0, 1])}
        ok = types.SimpleNamespace(model_uri="models:/m-ok")
        fake.modules["mlflow.sklearn"].log_model = mock.Mock(side_effect=[RuntimeError("blocked a"), RuntimeError("blocked b"), ok, ok, ok, ok])
        with mock.patch.dict(sys.modules, fake.modules):
            info = tracking.log_analysis(state, "models/x.joblib")
            out = tracking.mirror_mlflow(info["run_id"], bundle, info["registered"]["model_name"], candidates=cand)
        self.assertTrue(out["ok"])
        self.assertIn("model LR not logged", out["message"])
        self.assertTrue(tracking.get_run(info["run_id"])["mlflow_run_id"])

    def test_inventory_reports_registered_models_and_logged_models(self):
        fake = FakeMlflow()
        with mock.patch.dict(sys.modules, fake.modules):
            inv = tracking.mlflow_inventory()
        self.assertTrue(inv["active"])
        self.assertEqual(inv["registered"][0]["name"], "churn-Attrition")
        self.assertEqual(inv["registered"][0]["versions"][0]["aliases"], ["staging"])
        self.assertEqual(inv["experiments"][0]["models"][0]["status"], "READY")
        self.assertEqual(inv["experiments"][0]["runs"], 3)
        with mock.patch.dict(os.environ, {"NEXUSML_TRACKING": "json"}):
            self.assertFalse(tracking.mlflow_inventory()["active"])


    def test_none_and_nan_metrics_are_dropped_instead_of_failing_the_whole_log(self):
        self.assertEqual(tracking._finite({"a": 1, "b": None, "c": float("nan"), "d": float("inf"), "e": "x", "f": True, "g": 2.5}), {"a": 1.0, "g": 2.5})



    def test_json_only_mode_says_why(self):
        with mock.patch.dict(os.environ, {"NEXUSML_TRACKING": "json"}):
            self.assertFalse(tracking.mlflow_status()["active"])


class TestPackageAndKit(unittest.TestCase):
    def test_deployable_templates_verify_hash_and_vendor_new_modules(self):
        from nexusml import export
        for f in ("featurizers.py", "modelio.py"):
            self.assertIn(f, export.VENDOR)
        self.assertIn("modelio.load", export.PREDICT)
        self.assertIn("modelio.load", export.SERVE)
        self.assertIn("NEXUSML_REPLAY", export.REPRODUCE)


class TestMediaIngestion(Tmp):
    def zip_of(self, folder):
        import shutil
        return shutil.make_archive(str(Path(self.d) / "upload"), "zip", str(Path(folder).parent), Path(folder).name)

    def test_zip_is_extracted_safely_and_folders_become_labels(self):
        import io, zipfile
        from nexusml import media
        c = D.image_folders(Path(self.d))
        buf = io.BytesIO(Path(self.zip_of(c["folder"])).read_bytes())
        with zipfile.ZipFile(buf, "a") as z:
            z.writestr("../escape.txt", "x")                 # path traversal
            z.writestr("photos/tool.exe", "MZ")              # not a whitelisted type
        info = media.prepare([("photos.zip", buf.getvalue())], Path(self.d) / "work")
        self.assertEqual((info["kind"], info["n"], len(info["classes"])), ("images", 63, 3))
        self.assertEqual(len(info["unreadable"]), 1)
        self.assertEqual(info["suggest"]["problem_type"], "classification")
        self.assertFalse((Path(self.d) / "escape.txt").exists())
        self.assertFalse(list((Path(self.d) / "work").rglob("*.exe")))

    def test_flat_images_are_unlabeled_clustering(self):
        from nexusml import media
        c = D.unlabeled_images(Path(self.d))
        info = media.prepare_paths([c["folder"]], Path(self.d) / "work")
        self.assertIsNone(info["classes"])
        self.assertEqual(info["suggest"]["problem_type"], "clustering")

    def test_documents_in_four_formats_and_scanned_pdf_flagged(self):
        from nexusml import media
        c = D.document_folders(Path(self.d))
        info = media.prepare_paths([c["folder"]], Path(self.d) / "work")
        df = pd.read_csv(info["csv"])
        self.assertEqual(info["kind"], "documents")
        self.assertEqual(set(df["document"].map(lambda p: p.rsplit(".", 1)[-1])), {"txt", "docx", "pdf", "html"})
        scan = df[df["document"].str.endswith("scan.pdf")].iloc[0]
        self.assertEqual(scan["has_text"], 0)
        self.assertTrue(any("scan" in x or "text layer" in x for x in info["text_problems"]))
        html = df[df["document"].str.endswith(".html")].iloc[0]["text"]
        self.assertNotIn("var x", html)                       # scripts are not text

    def test_mixed_upload_analyses_majority_and_reports_set_aside(self):
        from nexusml import media
        c = D.image_folders(Path(self.d))
        (Path(c["folder"]) / "notes.txt").write_text("a lone note " * 10)
        info = media.prepare_paths([c["folder"]], Path(self.d) / "work")
        self.assertEqual(info["kind"], "images")
        self.assertEqual(info["set_aside"], {"documents": 1})

    def test_arabic_text_is_detected_and_not_scored_as_english(self):
        from nexusml import mediamining
        s = mediamining.script_share(pd.Series(["مرحبا بكم في هذا التقرير عن المبيعات والأرباح"] * 5))
        self.assertGreater(s["arabic"], 0.9)
        from nexusml import media
        self.assertIsNone(media.text_stats("هذا نص عربي طويل " * 20)["readability"])

    def test_row_for_new_file_matches_training_columns(self):
        from nexusml import media
        c = D.image_folders(Path(self.d))
        info = media.prepare_paths([c["folder"]], Path(self.d) / "work")
        cols = [x for x in pd.read_csv(info["csv"]).columns if x != "label"]
        row = media.row_for_file(Path(c["folder"]) / "oceans" / "oceans_0.png", cols, Path(c["folder"]))
        self.assertEqual(list(row), cols)
        with self.assertRaises(media.MediaError):
            media.row_for_file(Path(c["folder"]) / "oceans" / "corrupt.png", cols, Path(c["folder"]))


class TestMediaPipeline(unittest.TestCase):
    def run_case(self, name):
        work = Path(tempfile.mkdtemp())
        return D.GENERATORS[name](work), work

    def test_labelled_images_find_duplicates_shortcut_and_classify(self):
        case, work = self.run_case("image_folders")
        res = H.run_case(case, work)
        s = res["scores"]
        self.assertEqual(s["errors"], [])
        xs = " ".join(s["media"]["xs"])
        self.assertIn("byte-identical", xs)
        self.assertIn("shortcut risk", xs)
        self.assertGreaterEqual(res["values"]["candidate_metrics"][0]["accuracy"], 0.9)

    def test_documents_themes_match_folders_and_exact_duplicate_found(self):
        case, work = self.run_case("document_folders")
        res = H.run_case(case, work)
        xs = " ".join(res["scores"]["media"]["xs"])
        self.assertIn("natural themes", xs)
        self.assertIn("identical text", xs)
        self.assertIn("no extractable text", xs.replace("almost no extractable text", "no extractable text"))
        self.assertGreaterEqual(res["values"]["candidate_metrics"][0]["accuracy"], 0.9)

    def test_unlabeled_images_are_grouped_correctly(self):
        case, work = self.run_case("unlabeled_images")
        res = H.run_case(case, work)
        self.assertGreaterEqual(res["scores"]["clustering"]["ari"], 0.9)

    def test_document_text_never_reaches_the_llm_by_default(self):
        case, work = self.run_case("document_folders")
        seen = []
        stub = make_stub(case["frame"])

        def spy(system, prompt, purpose="", max_tokens=0, temperature=0):
            seen.append(prompt)
            return stub(system, prompt, purpose, max_tokens, temperature)
        from nexusml import media
        from nexusml.runtime import run_to_completion
        info = media.prepare_paths([case["folder"]], work)
        secret = pd.read_csv(info["csv"])["text"].iloc[0][:60]
        with mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": tempfile.mkdtemp(), "GROQ_API_KEY": "k"}), mock.patch.object(llm, "available", return_value=True), \
                mock.patch.object(llm, "ask_json", side_effect=spy):
            run_to_completion(info["csv"], "themes?", auto_approve=True, enable_tuning=False, max_retries=1, target_column="label", problem_type="classification")
        self.assertTrue(seen)
        self.assertFalse(any(secret in p for p in seen), "raw document text must not be sent to the LLM")
        self.assertFalse(any("name_doc_topics" in p for p in seen))




class TestOptionsCardAndTheme(unittest.TestCase):
    """The analysis options are proposed BY THE ASSISTANT in the chat (not hidden in the sidebar), and the app has a designed navy/blue theme."""

    def hr_like(self):
        r = np.random.RandomState(0)
        n = 400
        df = pd.DataFrame({"employee_id": range(n), "Age": r.randint(20, 60, n), "Gender": r.choice(["F", "M"], n), "tenure": r.normal(size=n),
                           "overtime": r.choice(["Yes", "No"], n), "Attrition": r.choice(["Yes", "No"], n, p=[.2, .8])})
        p = Path(tempfile.mkdtemp()) / "staff.csv"
        df.to_csv(p, index=False)
        return str(p)








    def test_suggestions_scan_is_local_and_handles_forecast_and_regression(self):
        from nexusml import suggest
        c = D.planted_regression(Path(tempfile.mkdtemp()))
        s = suggest.suggest_options(c["path"], "what drives price?")
        self.assertEqual((s["target"], s["problem_type"]), ("price", "regression"))
        t = D.planted_timeseries(Path(tempfile.mkdtemp()))
        s = suggest.suggest_options(t["path"], "forecast sales for next month")
        self.assertEqual(s["problem_type"], "time_series")


class TestSavedChats(unittest.TestCase):
    """Conversations are saved like ChatGPT / Gemini: they survive 'New chat', can be reopened, renamed, pinned, deleted, and are per user."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": self.d, "GROQ_API_KEY": "k"})
        self.env.start()
        self.llm = [mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=make_stub(
            {"analysis_type": "churn", "problem_type": "classification", "target": "attrition", "date_column": None, "segment_columns": ["overtime"],
             "kpis": [{"name": "Records", "agg": "count"}]}))]
        [p.start() for p in self.llm]

    def tearDown(self):
        [p.stop() for p in self.llm]
        self.env.stop()


    def table(self):
        c = D.planted_classification(Path(self.d))
        return c["path"]









class TestErrorPanelAndHardening(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": self.d, "GROQ_API_KEY": "k"})
        self.env.start()

    def tearDown(self):
        self.env.stop()


    def test_damaged_run_files_and_old_registry_do_not_crash_tracking(self):
        from nexusml import tracking
        tracking.new_run("exp", "good", {"a": 1}, {"m": 0.5})
        runs_dir = tracking._root() / "runs"
        (runs_dir / "junk.json").write_text(json.dumps({"hello": "not a run"}))
        (runs_dir / "partial.json").write_text(json.dumps({"run_id": "partial", "experiment": "exp"}))       # missing start/params/...
        (tracking._root() / "registry.json").write_text(json.dumps({"something_old": 1}))
        rows = tracking.runs_table(None)
        self.assertIn("good", [r["name"] for r in rows])
        self.assertEqual(tracking.registry()["models"], {})


FAKE_MLFLOW_SERVER = (
    "import sys, http.server\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        self.send_response(200 if self.path == '/health' else 404); self.end_headers()\n"
    "        if self.path == '/health': self.wfile.write(b'OK')\n"
    "    def log_message(self, *a): pass\n"
    "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n")


class TestOpenMlflowFromTheApp(unittest.TestCase):
    """The 'Open MLflow' button starts MLflow in the background (no terminal), reuses a running one, never duplicates, and stops only its own server.
    A stand-in web server plays the role of `mlflow server` (real MLflow cannot be installed in the build sandbox)."""

    def setUp(self):
        import socket
        from nexusml import mlflow_ui
        self.M = mlflow_ui
        self.d = tempfile.mkdtemp()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.env = mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": self.d, "NEXUSML_MLFLOW_UI_PORT": str(self.port)})
        self.env.start()
        self.ml = [mock.patch.object(tracking, "mlflow_status", return_value={"active": True, "version": "3.17.0", "uri": "sqlite:///x"}),
                   mock.patch.object(tracking, "tracking_uri", return_value="sqlite:///C:/Users/bad nerd/x/mlflow.db")]
        [p.start() for p in self.ml]
        self.cands = [[sys.executable, "-c", FAKE_MLFLOW_SERVER, "{port}"]]

    def tearDown(self):
        self.M.stop()
        [p.stop() for p in self.ml]
        self.env.stop()

    def test_command_passes_the_database_as_one_argument_so_spaces_need_no_quotes(self):
        c = self.M.default_candidates("sqlite:///C:/Users/bad nerd/Desktop/x/mlflow.db", 5000)
        self.assertIn("sqlite:///C:/Users/bad nerd/Desktop/x/mlflow.db", c[0])
        self.assertEqual(c[0][-2:], ["--workers", "1"])                       # one worker: avoids the Windows / Python 3.14 WinError 10022 noise
        self.assertNotIn("--workers", c[1])                                   # plain `mlflow ui` is the fallback

    def test_start_runs_in_the_background_and_a_second_press_reuses_it(self):
        r = self.M.start(candidates=self.cands)
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["ours"])
        self.assertTrue(self.M.is_mlflow(self.port))
        pid = self.M.status()["pid"]
        r2 = self.M.start(candidates=self.cands)                              # pressing the button again must not start a second server
        self.assertTrue(r2["reused"])
        self.assertEqual(self.M.status()["pid"], pid)

    def test_stop_ends_only_the_server_the_app_started(self):
        self.M.start(candidates=self.cands)
        self.assertTrue(self.M.stop())
        import time
        for _ in range(20):
            if not self.M.is_mlflow(self.port):
                break
            time.sleep(0.2)
        self.assertFalse(self.M.is_mlflow(self.port))
        self.assertFalse(self.M.status()["running"])
        self.assertFalse(self.M.stop())                                        # nothing left to stop

    def test_an_mlflow_you_started_yourself_is_reused_and_never_killed(self):
        import subprocess
        mine = subprocess.Popen([sys.executable, "-c", FAKE_MLFLOW_SERVER, str(self.port)])
        try:
            import time
            for _ in range(30):
                if self.M.is_mlflow(self.port):
                    break
                time.sleep(0.2)
            r = self.M.start(candidates=self.cands)
            self.assertTrue(r["ok"] and r["reused"] and not r["ours"], r)
            self.assertFalse(self.M.stop())                                    # not ours: left alone
            self.assertTrue(self.M.is_mlflow(self.port))
        finally:
            mine.terminate()
            mine.wait(timeout=10)

    def test_a_non_mlflow_program_on_the_port_is_not_mistaken_for_mlflow(self):
        import socket
        import threading
        srv = socket.socket()
        srv.bind(("127.0.0.1", self.port))
        srv.listen(5)
        t = threading.Thread(target=lambda: [srv.accept() for _ in range(0)], daemon=True)
        t.start()
        try:
            self.assertFalse(self.M.is_mlflow(self.port))
            r = self.M.start(candidates=self.cands)
            self.assertTrue(r["ok"], r)
            self.assertNotEqual(int(r["url"].rsplit(":", 1)[1]), self.port)   # moved to the next free port
        finally:
            srv.close()

    def test_falls_back_to_the_next_command_when_the_first_one_exits(self):
        bad = [sys.executable, "-c", "import sys; print('no such option --workers'); sys.exit(2)"]
        r = self.M.start(candidates=[bad] + self.cands)
        self.assertTrue(r["ok"], r)

    def test_when_every_command_fails_the_message_shows_the_log(self):
        bad = [sys.executable, "-c", "print('boom-marker-123'); import sys; sys.exit(3)"]
        r = self.M.start(candidates=[bad], wait=5)
        self.assertFalse(r["ok"])
        self.assertIn("could not be started", r["message"])
        self.assertIn("boom-marker-123", r["message"])

    def test_not_installed_gives_the_reason_instead_of_a_crash(self):
        with mock.patch.object(tracking, "mlflow_status", return_value={"active": False, "reason": "mlflow is not installed (pip install mlflow)"}):
            r = self.M.start(candidates=self.cands)
        self.assertFalse(r["ok"])
        self.assertIn("pip install mlflow", r["message"])




class TestHarnessEndToEnd(unittest.TestCase):
    def test_quick_case_with_validation_on(self):
        with mock.patch.dict(os.environ, {"NEXUSML_VALIDATE": "1"}):
            work = Path(tempfile.mkdtemp())
            case = D.leakage_trap(work)
            res = H.run_case(case, work)
        s = res["scores"]
        self.assertEqual(s["errors"], [])
        self.assertTrue(s["framing"]["target_ok"])
        self.assertTrue(s["cleaning"]["dropped_all"])                      # the leaky column is removed
        self.assertTrue(s["ci"]["contains_point"])




class TestSearchAndFormats(Tmp):
    def test_document_search_similarity_and_grounded_answer(self):
        from nexusml import media, media_search
        c = D.document_folders(Path(self.d))
        df = pd.read_csv(media.prepare_paths([c["folder"]], Path(self.d) / "w")["csv"])
        ix = media_search.DocIndex(df)
        hits = ix.search("hospital doctor medication", 4)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["label"], "health")
        self.assertEqual(ix.search("zzzunknownword"), [])
        sim = ix.similar_to_text(df.iloc[0]["text"], 3, exclude=df.iloc[0]["document"])
        self.assertTrue(all(s["document"] != df.iloc[0]["document"] for s in sim))
        sent = []
        stub = make_stub({})

        def spy(system, prompt, purpose="", max_tokens=0, temperature=0):
            sent.append(prompt)
            return stub(system, prompt, purpose, max_tokens, temperature)
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}), mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=spy):
            out = media_search.answer({}, "which medication?", hits)
        self.assertIn(out["sources"][0], [h["document"] for h in hits])
        self.assertLessEqual(len(sent[0]), 4000)                      # only the few retrieved passages were sent

    def test_image_similarity_returns_same_class_first(self):
        from nexusml import media, media_search
        c = D.image_folders(Path(self.d))
        info = media.prepare_paths([c["folder"]], Path(self.d) / "w")
        res = media_search.image_similarity(pd.read_csv(info["csv"]), "image", Path(info["root"]), Path(c["folder"]) / "oceans" / "oceans_0.png", 4)
        self.assertTrue(all("oceans" in r["image"] for r in res))

    def test_more_table_formats_become_csv(self):
        from nexusml import ingest
        self.assertEqual(ingest.to_csv_bytes("a.jsonl", b'{"a":1,"b":{"c":2}}\n{"a":3,"b":{"c":4}}\n').decode().split()[0], "a,b.c")
        self.assertIn("x", ingest.to_csv_bytes("a.json", b'{"data":[{"x":1},{"x":2}]}').decode())
        self.assertIn("a", ingest.to_csv_bytes("a.tsv", b"a\tb\n1\t2\n").decode())
        self.assertIn("x", ingest.to_csv_bytes("a.xml", b"<r><i><x>1</x></i><i><x>2</x></i></r>").decode())
        with self.assertRaises(ValueError):
            ingest.to_csv_bytes("bad.json", b"{not json")


class TestConciergeAndVision(Tmp):
    def test_routing_general_vs_followup_vs_needs_data(self):
        from nexusml import concierge
        self.assertEqual(concierge.route("What is the difference between bagging and boosting?", False), "general")
        self.assertEqual(concierge.route("analyze my sales data", False), "needs_data")
        self.assertEqual(concierge.route("how reliable is this model?", True), "followup")
        self.assertEqual(concierge.route("what does overfitting mean?", True), "general")
        self.assertEqual(concierge.route("which tenure bands matter?", True, ["tenure"]), "followup")

    def test_general_answer_needs_no_dataset(self):
        from nexusml import concierge
        stub = make_stub({})
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}), mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=stub):
            r = concierge.answer({}, "Explain precision vs recall", [("hi", "hello")])
            r2 = concierge.answer({}, "analyze my sales", [], "needs_data")
        self.assertTrue(r["answer"])
        self.assertTrue(r2["needs_data"])

    def test_vision_request_shape_and_error_mapping(self):
        sent = {}

        def post(body, provider=None):
            sent.update(body)
            return {"choices": [{"message": {"content": '{"main_subject": "dog", "confidence": "high", "summary": "A dog on grass."}'}}], "usage": {"total_tokens": 9}}
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}), mock.patch.object(llm, "_post", side_effect=post):
            d, err, rec = llm.ask_vision("sys", "what is it?", b"\xff\xd8fake", "image/jpeg")
        self.assertIsNone(err)
        self.assertEqual(d["main_subject"], "dog")
        self.assertEqual(sent["model"], "meta-llama/llama-4-scout-17b-16e-instruct")
        part = sent["messages"][1]["content"][1]
        self.assertTrue(part["image_url"]["url"].startswith("data:image/jpeg;base64,"))

        import io, urllib.error
        e404 = urllib.error.HTTPError("u", 404, "nf", {}, io.BytesIO(b'{"error":{"message":"The model `x` does not exist"}}'))
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}), mock.patch.object(llm, "_post", side_effect=e404):
            d, err, _ = llm.ask_vision("s", "p", b"x")
        self.assertIsNone(d)
        self.assertIn("NEXUSML_VISION_MODEL", err)

    def test_describe_is_honest_without_a_vision_model_and_names_subject_with_one(self):
        from nexusml import vision
        from PIL import Image
        p = Path(self.d) / "pet.png"
        Image.fromarray((np.random.RandomState(0).rand(64, 64, 3) * 255).astype("uint8")).save(p)
        with mock.patch.object(vision, "local_classify", return_value=None):
            d = vision.describe(p, send_to_ai=False)
            txt = vision.format_answer(d)
        self.assertEqual(d["source"], "none")
        self.assertIn("cannot tell what this image shows", txt)
        self.assertIn("px", txt)                                          # local measurements are still shown
        good = ({"main_subject": "dog", "confidence": "high", "summary": "A dog.", "objects": [{"label": "grass", "confidence": "medium"}], "details": ["brown fur"]}, None, {})
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}), mock.patch.object(llm, "ask_vision", return_value=good) as av:
            d = vision.describe(p, question="what is it?", send_to_ai=True)
        self.assertEqual(d["subject"], "dog")
        self.assertIn("dog", vision.format_answer(d))
        self.assertIn("what is it?", av.call_args[0][1])                  # the user's question reached the model

    def test_corrupt_image_gives_a_clear_error(self):
        from nexusml import media, vision
        p = Path(self.d) / "bad.png"
        p.write_bytes(b"nope")
        with self.assertRaises(media.MediaError):
            vision.local_facts(p)


class TestDocumentUnderstanding(Tmp):
    def doc(self, name="report.docx", paras=None):
        from docx import Document
        d = Document()
        for p in paras or ["Quarterly review", "Revenue grew 12% to $4.2 million in the third quarter of 2025 driven by strong subscription renewals.",
                           "Customer churn fell from 6.1% to 4.8% after the onboarding redesign launched on 2025-07-15.",
                           "The main risk is supplier concentration, so procurement will add two new vendors before March.",
                           "Management recommends increasing the marketing budget by 10% and hiring three support engineers."]:
            d.add_paragraph(p)
        p = Path(self.d) / name
        d.save(str(p))
        return p

    def test_local_analysis_and_extractive_summary(self):
        from nexusml import docchat
        a = docchat.analyze(self.doc())
        self.assertGreater(a["stats"]["n_words"], 40)
        self.assertTrue(a["extractive_summary"])
        self.assertTrue(any("$4.2" in x for x in a["amounts"]))
        self.assertTrue(any("2025-07-15" in x for x in a["dates"]))
        self.assertIn("Key terms", docchat.format_local(a))

    def test_empty_or_scanned_document_gives_reason(self):
        from nexusml import docchat, media
        from PIL import Image
        p = Path(self.d) / "scan.pdf"
        Image.fromarray(np.full((120, 90, 3), 255, "uint8")).save(p)
        with self.assertRaises(media.MediaError) as cm:
            docchat.analyze(p)
        self.assertIn("text", str(cm.exception).lower())

    def test_short_doc_one_call_long_doc_map_reduce_and_says_coverage(self):
        from nexusml import docchat
        calls = []
        stub = make_stub({})

        def spy(system, prompt, purpose="", max_tokens=0, temperature=0):
            calls.append(purpose)
            return stub(system, prompt, purpose, max_tokens, temperature)
        a = docchat.analyze(self.doc())
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}), mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=spy):
            r = docchat.summarize_with_ai({}, a)
            self.assertEqual(calls, ["summarize_document"])
            calls.clear()
            big = {**a, "text": " ".join(f"Sentence number {i} talks about topic {i % 7} in some detail." for i in range(4000))}
            r2 = docchat.summarize_with_ai({}, big)
        self.assertEqual(calls.count("summarize_document"), 1)
        self.assertEqual(calls.count("summarize_chunk"), docchat.MAX_CHUNKS)
        self.assertIn("evenly spaced", r2["covered"])
        self.assertIn("Summary", docchat.format_ai(r2))

    def test_pdf_and_arabic_documents(self):
        from nexusml import docchat
        from reportlab.pdfgen import canvas
        p = Path(self.d) / "a.pdf"
        c = canvas.Canvas(str(p))
        for i, line in enumerate(["Annual report of the logistics division.", "Deliveries increased by 8 percent compared with last year.",
                                  "The warehouse in Cairo was expanded and staffing was increased.", "Customer satisfaction improved across all regions."]):
            c.drawString(50, 780 - 20 * i, line)
        c.save()
        a = docchat.analyze(p)
        self.assertGreaterEqual(a["pages"], 1)
        ar = Path(self.d) / "ar.txt"
        ar.write_text("هذا تقرير عن المبيعات في الربع الثالث. ارتفعت الإيرادات بنسبة عشرة بالمئة مقارنة بالعام الماضي. تحسن رضا العملاء في جميع المناطق. " * 3, encoding="utf-8")
        b = docchat.analyze(ar)
        self.assertGreater(b["script"]["arabic"], 0.5)
        self.assertIn("Arabic", docchat.format_local(b))


class TestDocumentReadingWithoutOptionalLibraries(Tmp):
    """Regression: a machine that only ran `pip install -r requirements.txt` of an older release had no pypdf / python-docx and could read nothing."""
    NO_LIBS = {"pypdf": None, "docx": None, "bs4": None, "odf": None}

    def files(self):
        from docx import Document
        from reportlab.pdfgen import canvas
        d = Path(self.d)
        doc = Document()
        doc.add_heading("Quarterly review", 1)
        doc.add_paragraph("Revenue grew twelve percent and churn fell to 4.8 percent after the onboarding redesign launched in July.")
        tb = doc.add_table(rows=1, cols=2)
        tb.cell(0, 0).text, tb.cell(0, 1).text = "Cairo", "14%"
        doc.sections[0].header.paragraphs[0].text = "CONFIDENTIAL header text"
        doc.save(str(d / "r.docx"))
        c = canvas.Canvas(str(d / "r.pdf"))
        c.drawString(50, 780, "Annual logistics report. Deliveries increased by eight percent compared with last year.")
        c.save()
        (d / "p.html").write_text("<html><body><script>var secret=1;</script><p>Plain html paragraph with enough words to count as text.</p></body></html>")
        (d / "bad.docx").write_bytes(b"an old binary doc renamed")
        return d

    def test_docx_and_html_read_with_no_extra_packages(self):
        from nexusml import media
        d = self.files()
        with mock.patch.dict(sys.modules, self.NO_LIBS):
            t, _, note = media.extract_text(d / "r.docx")
            h, _, hn = media.extract_text(d / "p.html")
        self.assertEqual(note, "")
        for s in ("Quarterly review", "twelve percent", "Cairo", "CONFIDENTIAL header text"):
            self.assertIn(s, t)                                           # body, table AND header text
        self.assertIn("Plain html paragraph", h)
        self.assertNotIn("secret", h)

    def test_pdf_uses_pdftotext_when_pypdf_is_missing_else_says_what_to_install(self):
        import shutil as _sh
        from nexusml import media
        d = self.files()
        with mock.patch.dict(sys.modules, self.NO_LIBS):
            if _sh.which("pdftotext"):
                t, n, note = media.extract_text(d / "r.pdf")
                self.assertIn("Deliveries increased", t)
                self.assertGreaterEqual(n, 1)
            with mock.patch.object(media.shutil, "which", return_value=None):
                t, n, note = media.extract_text(d / "r.pdf")
        self.assertEqual(t.strip(), "")
        self.assertIn("pip install pypdf", note)

    def test_damaged_or_old_doc_gets_a_friendly_reason(self):
        from nexusml import media
        d = self.files()
        for libs in ({}, self.NO_LIBS):
            with mock.patch.dict(sys.modules, libs):
                note = media.extract_text(d / "bad.docx")[2]
            self.assertIn("re-saved as .docx", note)

    def test_single_docx_flow_works_end_to_end_without_libraries(self):
        from nexusml import docchat
        d = self.files()
        with mock.patch.dict(sys.modules, self.NO_LIBS):
            a = docchat.analyze(d / "r.docx")
        self.assertGreater(a["stats"]["n_words"], 15)
        self.assertTrue(a["extractive_summary"])

    def test_doctor_reports_each_file_type_and_requirements_list_the_libraries(self):
        from nexusml import media
        st = media.dependency_status()
        for k in ("PDF", "Word .docx", "HTML", "Images", "Excel"):
            self.assertIn(k, st)
        with mock.patch.dict(sys.modules, self.NO_LIBS), mock.patch.object(media.shutil, "which", return_value=None):
            self.assertFalse(media.dependency_status()["PDF"]["ok"])
            self.assertIn("pypdf", media.dependency_status()["PDF"]["fix"])
        req = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text()
        for lib in ("pypdf", "python-docx", "beautifulsoup4", "pillow", "openpyxl"):
            self.assertIn(lib, req)


class TestAppSingleItemsAndChat(unittest.TestCase):

    def png(self, name="pet.png"):
        from PIL import Image
        p = Path(tempfile.mkdtemp()) / name
        Image.fromarray((np.random.RandomState(1).rand(48, 48, 3) * 255).astype("uint8")).save(p)
        return str(p)








if __name__ == "__main__":
    unittest.main()
