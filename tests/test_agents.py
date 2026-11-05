import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["NEXUSML_BACKEND"] = "builtin"
os.environ.pop("GROQ_API_KEY", None)

import numpy as np
import pandas as pd

from nexusml import catalog, kpi, llm, modeling, profiling, rca, stats_tests, trends
from nexusml.runtime import run_to_completion
from nexusml.storage import get_registry

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures"
FAST = dict(auto_approve=True, tune_n_iter=2, max_retries=1)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._env = mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": self.dir})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


# ------------------------------------------------------------------ catalog
class TestCatalog(unittest.TestCase):
    def test_business_vs_analytical_separation(self):
        self.assertEqual(catalog.norm_analysis_type("Sales Performance"), "sales_performance")
        self.assertEqual(catalog.norm_problem_type("forecasting"), "time_series")
        self.assertIsNone(catalog.norm_problem_type("sales_performance"))   # business term is NOT a problem type
        self.assertIsNone(catalog.norm_analysis_type("classification"))     # analytical term is NOT a business type


# ------------------------------------------------------------- statistics
class TestStats(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.big = pd.DataFrame({"x": np.r_[rng.normal(0, 1, 400), rng.normal(0.6, 1, 400)], "g": ["a"] * 400 + ["b"] * 400})
        self.small = pd.DataFrame({"x": np.r_[rng.exponential(1, 12), rng.exponential(3, 12)], "g": ["a"] * 12 + ["b"] * 12})

    def test_large_groups_use_clt_not_shapiro(self):
        r = stats_tests.run_group_comparison(self.big, "x", "g")
        self.assertEqual(r["method"], "Welch t-test")
        self.assertFalse(any("Shapiro" in a for a in r["assumptions"]))   # not needed for n >= 200

    def test_small_skewed_groups_use_shapiro_then_nonparametric(self):
        r = stats_tests.run_group_comparison(self.small, "x", "g")
        self.assertTrue(any("Shapiro" in a for a in r["assumptions"]))
        self.assertEqual(r["method"], "Mann-Whitney U")

    def test_ks_drift_only_offered_with_reference(self):
        df = self.big.assign(y=self.big["x"] * 2 + 1)
        prof = profiling.profile_dataset(df)
        c, skipped = stats_tests.enumerate_candidates(df, prof, "y", "regression", ["x", "g"], [], False, False)
        self.assertFalse(any(x["test"] == "ks_drift" for x in c))
        self.assertTrue(any("Kolmogorov" in s["test"] for s in skipped))
        self.assertTrue(any("Shapiro" in s["test"] for s in skipped))      # never run standalone
        c2, _ = stats_tests.enumerate_candidates(df, prof, "y", "regression", ["x", "g"], [], True, False)
        self.assertTrue(any(x["test"] == "ks_drift" for x in c2))

    def test_ks_detects_real_shift_only(self):
        rng = np.random.RandomState(1)
        a = pd.DataFrame({"f": rng.normal(0, 1, 500), "h": rng.normal(0, 1, 500)})
        b = pd.DataFrame({"f": rng.normal(1.5, 1, 500), "h": rng.normal(0, 1, 500)})
        r = stats_tests.run_ks_drift(a, b, ["f", "h"], "train_vs_test")
        self.assertTrue(r["significant"])
        self.assertEqual([t["feature"] for t in r["table"] if t["drifted"]], ["f"])

    def test_sparse_chi_square_refused(self):
        df = pd.DataFrame({"a": list("ab" * 6) + ["c"] * 3, "b": list("xy" * 6) + ["z"] * 3})
        self.assertIn("skipped", stats_tests.run_chi_square(df, "a", "b"))

    def test_bh_adjustment_monotone(self):
        adj = stats_tests.benjamini_hochberg([0.001, 0.02, 0.03, 0.5])
        self.assertTrue(all(a >= p for a, p in zip(adj, [0.001, 0.02, 0.03, 0.5])))


# --------------------------------------------------------------- kpi / trends / rca
class TestBusinessMath(unittest.TestCase):
    def test_kpi_validation_rejects_invented_columns(self):
        df = pd.DataFrame({"rev": [1.0, 2.0], "seg": ["a", "b"]})
        self.assertIsNone(kpi.validate_spec({"agg": "sum", "column": "ghost"}, df)[0])
        self.assertIsNone(kpi.validate_spec({"agg": "sum", "column": "seg"}, df)[0])       # not numeric
        self.assertIsNone(kpi.validate_spec({"agg": "exec", "column": "rev"}, df)[0])
        self.assertIsNotNone(kpi.validate_spec({"agg": "sum", "column": "rev", "group_by": ["seg", "ghost"]}, df)[0])

    def test_partial_edge_periods_are_dropped(self):
        d = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=200), "v": 1.0})   # starts on a Sunday: 1-day first week
        s = trends.period_series(d, "date", {"agg": "sum", "column": "v"})
        self.assertTrue((s == 7).all())

    def test_anomaly_found(self):
        rng = np.random.RandomState(0)
        s = pd.Series(100 + rng.normal(0, 2, 60), index=pd.date_range("2023-01-01", periods=60, freq="D"))
        s.iloc[30] = 300
        r = trends.analyze_series(s, "m")
        self.assertIn(str(s.index[30].date()), [a["period"] for a in r["anomalies"]])

    def test_level_shift_not_faked_by_trend_but_found_when_real(self):
        idx, rs = pd.date_range("2023-01-01", periods=400), np.random.RandomState(0)
        pure = pd.Series(np.arange(400) * 0.5 + rs.normal(0, 3, 400), index=idx)
        self.assertIsNone(trends.analyze_series(pure, "m").get("change_point"))          # trend alone is not a break
        jump = pd.Series(np.arange(400) * 0.5 + np.r_[np.zeros(250), np.full(150, 40.0)] + rs.normal(0, 3, 400), index=idx)
        cp = trends.analyze_series(jump, "m")["change_point"]
        self.assertEqual(cp["period"], str(idx[250].date()))
        self.assertAlmostEqual(cp["shift_vs_trend"], 40, delta=3)

    def test_flat_series_is_not_decomposed(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"d": pd.date_range("2023-01-01", periods=400), "y": rng.binomial(1, 0.3, 400), "s": rng.choice(["a", "b"], 400)})
        m = {"y": df["y"].astype(float), "name": "rate", "kind": "rate", "additive": False, "fmt": "pct"}
        ch = rca.contribution_change(df, "d", m, ["s"])
        ev = rca.evidence_from([], {**ch, "total_current": 0.3, "total_base": 0.3}, m, [])
        self.assertEqual(len(ev), 1)      # only the headline; no nonsense "350% of change" items


# ------------------------------------------------------------- profiling leaks
class TestLeakage(unittest.TestCase):
    def test_nonlinear_single_feature_leak_caught(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(-3, 3, 600)
        df = pd.DataFrame({"x": x, "noise": rng.normal(size=600), "leak": x ** 2, "y": x ** 2 + rng.normal(0, 0.01, 600)})
        prof = profiling.profile_dataset(df)
        issues = profiling.audit_columns(df, prof, "y", "regression")
        self.assertIn("leak", [i["column"] for i in issues if i["issue"] == "leakage_suspect"])


# ------------------------------------------------------------------ tuning
class TestTuning(unittest.TestCase):
    def test_space_is_clipped_and_whitelisted(self):
        sp, notes = modeling.sanitize_space("Ridge", {"alpha": {"type": "loguniform", "low": 1e-12, "high": 1e12}, "evil": {"type": "choice", "values": [1]}})
        self.assertEqual(sp["alpha"]["low"], modeling.PARAM_BOUNDS["alpha"][0])
        self.assertEqual(sp["alpha"]["high"], modeling.PARAM_BOUNDS["alpha"][1])
        self.assertNotIn("evil", sp)
        sp2, n2 = modeling.sanitize_space("RandomForest", {"max_depth": {"type": "choice", "values": [3, "rm -rf", 9999]}, "n_estimators": "garbage"})
        self.assertEqual(sp2["max_depth"]["values"], [3])
        self.assertEqual(sp2["n_estimators"], modeling.DEFAULT_SPACES["RandomForest"]["n_estimators"])
        self.assertTrue(n2)

    def test_nested_cv_gates_tuned_models(self):
        df = pd.read_csv(SAMPLE / "house_prices.csv").head(500)
        plan = dict(problem_type="regression", target="price", drop_columns=["listing_id"], datetime_columns=[], date_column=None)
        sp = modeling.make_splits(df, plan)
        out = modeling.fit_default("regression", sp, {})
        rep = modeling.tune_candidates("regression", sp, {}, out["candidates"], out["models"], None, n_iter=3, top_k=1)
        (name, e), = rep["models"].items()
        self.assertEqual(len(e["trials"]), 3)
        self.assertEqual(len(e["nested_scores"]), rep["outer_cv"])
        c = next(c for c in out["candidates"] if c["name"] == name)
        if e["used_tuned"]:
            self.assertEqual(c["select_score"], round(e["nested_mean"], 4))
            self.assertEqual(c["select_source"], "nested CV (tuned)")
        else:
            self.assertEqual(c["select_score"], c["default_cv"])
        self.assertEqual(modeling.pick_best(out["candidates"]), max(out["candidates"], key=lambda r: r["select_score"])["name"])


# --------------------------------------------------------------------- llm client
class TestLLMClient(unittest.TestCase):
    def test_no_key_returns_error_not_exception(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            data, err, _ = llm.ask_json("s", "p", "t")
        self.assertIsNone(data)
        self.assertIn("GROQ_API_KEY", err)

    def test_success_parses_fenced_json_and_never_leaks_key(self):
        resp = {"choices": [{"message": {"content": "```json\n{\"a\": 1}\n```"}}], "usage": {"total_tokens": 7}}
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_SECRET"}), mock.patch.object(llm, "_post", return_value=resp):
            data, err, rec = llm.ask_json("s", "p", "t")
        self.assertEqual(data, {"a": 1})
        self.assertEqual(rec["tokens"], 7)
        self.assertNotIn("gsk_SECRET", json.dumps(rec))

    def test_reasoning_model_gets_headroom_and_think_blocks_are_stripped(self):
        sent = []
        resp = {"choices": [{"message": {"content": "<think>hmm</think>{\"a\": 2}"}}], "usage": {"total_tokens": 5}}
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}), mock.patch.object(llm.config, "LLM_MODEL", "openai/gpt-oss-120b"), \
                mock.patch.object(llm, "_post", side_effect=lambda b: sent.append(dict(b)) or resp):
            data, err, _ = llm.ask_json("s", "p", "t", max_tokens=1000)
        self.assertEqual(data, {"a": 2})
        self.assertEqual(sent[0]["reasoning_effort"], "low")
        self.assertGreater(sent[0]["max_tokens"], 1000)

    def test_unsupported_params_are_dropped_on_400_then_succeeds(self):
        import urllib.error
        calls = []

        def post(body):
            calls.append(sorted(body))
            if "reasoning_effort" in body or "include_reasoning" in body:
                raise urllib.error.HTTPError("u", 400, "bad", {}, None)
            return {"choices": [{"message": {"content": "{\"ok\": 1}"}}], "usage": {}}
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}), mock.patch.object(llm.config, "LLM_MODEL", "openai/gpt-oss-20b"), \
                mock.patch.object(llm, "_post", side_effect=post):
            data, err, _ = llm.ask_json("s", "p", "t")
        self.assertEqual(data, {"ok": 1})
        self.assertNotIn("reasoning_effort", calls[-1])

    def test_404_preflight_lists_usable_models(self):
        import urllib.error
        e = urllib.error.HTTPError("u", 404, "nf", {}, None)
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}), mock.patch.object(llm, "_post", side_effect=e), \
                mock.patch.object(llm, "list_models", return_value=["openai/gpt-oss-20b", "qwen/qwen3-32b"]):
            with self.assertRaises(llm.LLMRequired) as cm:
                llm.preflight()
        msg = str(cm.exception)
        self.assertIn("openai/gpt-oss-20b", msg)
        self.assertIn("NEXUSML_LLM_MODEL", msg)

    def test_http_error_is_reported(self):
        import urllib.error
        e = urllib.error.HTTPError("u", 401, "bad", {}, None)
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_SECRET"}), mock.patch.object(llm, "_post", side_effect=e):
            data, err, _ = llm.ask_json("s", "p", "t")
        self.assertIsNone(data)
        self.assertEqual(err, "HTTP 401")
        self.assertNotIn("SECRET", err)


# ------------------------------------------------------------ cooperative stub LLM
KPI_CHURN = [{"name": "Churn rate", "agg": "rate", "column": "churn", "positive": "Yes", "format": "percent", "higher_is_better": False, "group_by": ["contract"]},
             {"name": "Customers", "agg": "count"}]


def frame_for(case):
    def f(pl):
        ov = pl.get("user_choices_you_must_respect", {})
        base = {
            "churn": {"analysis_type": "churn", "problem_type": "classification", "target": "churn", "date_column": "signup_date",
                      "segment_columns": ["contract", "region"], "kpis": KPI_CHURN},
            "house": {"analysis_type": "profitability", "problem_type": "regression", "target": "price", "date_column": None,
                      "segment_columns": [], "kpis": [{"name": "Avg price", "agg": "mean", "column": "price", "format": "currency"}]},
            "sales": {"analysis_type": "sales_performance", "problem_type": "time_series", "target": "sales", "date_column": "date",
                      "segment_columns": ["promo", "ghost"], "kpis": [{"name": "Total sales", "agg": "sum", "column": "sales", "format": "currency", "group_by": ["promo"]},
                                                                      {"name": "Invented", "agg": "sum", "column": "nonexistent"}]},
            "cluster": {"analysis_type": "customer_analysis", "problem_type": "clustering", "target": None, "date_column": "signup_date",
                        "segment_columns": ["contract"], "kpis": [{"name": "Avg tenure", "agg": "mean", "column": "tenure_months"}]},
            "descriptive": {"analysis_type": "customer_analysis", "problem_type": "descriptive", "target": None, "date_column": "signup_date",
                            "segment_columns": ["contract"], "kpis": [{"name": "Customers", "agg": "count"}]},
            "ambiguous": {"analysis_type": "churn", "problem_type": "classification", "target": None, "date_column": None,
                          "segment_columns": ["contract"], "kpis": [{"name": "Customers", "agg": "count"}]},
        }[case]
        out = {**base, "business_question": "Q?", "hypotheses": ["H1"]}
        if ov.get("target_column"):
            out["target"] = ov["target_column"]
        if ov.get("problem_type"):
            out["problem_type"] = ov["problem_type"]
        return out
    return f


def coop(case, overrides=None):
    """A well-behaved Groq stand-in, built from each prompt's own payload. `overrides` = {purpose: fn(payload)->dict}."""
    overrides = overrides or {}

    def fake(system, prompt, purpose="", max_tokens=0, temperature=0):
        rec = {"purpose": purpose, "model": "fake", "ok": True, "seconds": 0.0, "tokens": 10}
        pl = json.loads(prompt)
        ev = [e["id"] for e in pl.get("evidence", [])]
        if purpose in overrides:
            return overrides[purpose](pl), None, rec
        d = {
            "preflight": {"ok": True},
            "frame_business_problem": frame_for(case)(pl),
            "select_statistical_tests": {"selected": [{"id": c["id"], "reason": "relevant"} for c in pl.get("candidates", [])[:6]]},
            "review_column_drops": {"reviews": [{"column": a["column"], "verdict": "drop", "why": "not useful"} for a in pl.get("proposed_drops", [])]},
            "advise_hyperparameter_ranges": {"spaces": {}, "rationale": "defaults suit this sample size"},
            "choose_retry_strategy": {"levers": list(pl.get("available_levers", {}))[:1], "rationale": "address the failure"},
            "name_segments": {"segments": [{"cluster": s["cluster"], "name": f"Segment {s['cluster']}", "description": "desc"} for s in pl.get("segments", [])]},
            "root_cause_analysis": {"root_causes": [{"finding": "f", "hypothesis": "h", "evidence_ids": ev[:2], "confidence": "medium", "validation_step": "pilot"}]},
            "business_insights": {"headline": "Headline", "executive_summary": "Summary.", "key_insights": [{"title": "T", "insight": "I", "so_what": "S", "evidence_ids": ev[:2]}], "caveats": []},
            "recommendations": {"recommendations": [{"action": "Act", "rationale": "r", "evidence_ids": ev[:1], "priority": "high", "effort": "low",
                                                     "expected_impact": "i", "kpi_to_track": "k", "risk": "r", "owner": "o"}]},
            "powerbi_report_layout": {"pages": [{"title": "Overview", "visuals": [{"type": "card", "table": "KPI_Summary", "fields": ["kpi", "value"], "title": "KPIs"}]}]},
        }
        if purpose not in d:
            from nexusml.evaluation.stub_llm import v6_answer
            v6 = v6_answer(purpose, pl)
            if v6 is None:
                raise KeyError(purpose)
            return v6, None, rec
        return d[purpose], None, rec
    return fake


class LLMCase(Tmp):
    case = "churn"
    overrides = None

    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(llm, "available", return_value=True),
                  mock.patch.object(llm, "ask_json", side_effect=coop(self.case, self.overrides)),
                  mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_SECRET"})):
            p.start()
            self.addCleanup(p.stop)


class TestPipeline(LLMCase):
    def test_churn_full_run(self):
        v = run_to_completion(str(SAMPLE / "churn.csv"), "Why do customers churn?", powerbi_enabled=True, **FAST)
        self.assertEqual(v["errors"], [])
        self.assertEqual((v["analysis_type"], v["problem_type"]), ("churn", "classification"))
        self.assertIn("account_closed_flag", v["analysis_plan"]["drop_columns"])           # leak removed
        self.assertTrue(v["stat_results"] and v["kpi_results"] and v["trend_results"])
        sections = {c["section"] for c in v["charts"]}
        self.assertTrue({"distributions", "outliers", "correlations", "modeling", "explain"} <= sections)
        self.assertTrue(any(c["id"] == "eda_box" for c in v["charts"]) and any(c["id"] == "eda_corr" for c in v["charts"]))
        for c in v["charts"]:
            self.assertTrue(c["key"] or c["spec_key"])
            for k in (c["key"], c["spec_key"]):
                if k:
                    self.assertTrue(get_registry().path_of(k).stat().st_size > 200)
        self.assertTrue(sum(1 for c in v["charts"] if c["spec_key"]) >= 10)                  # interactive charts exist
        self.assertTrue(v["tuning_report"]["enabled"])
        self.assertEqual(v["tuning_report"]["engine"], "automl")                              # v6: the AutoML agent searches; its proposals are validated and clipped
        self.assertIn("validated", v["tuning_report"]["space_source"])
        ids = {e["id"] for e in v["evidence"]}
        self.assertEqual(len(ids), len(v["evidence"]))
        for r in v["root_causes"] + v["recommendations"]:
            self.assertTrue(set(r["evidence_ids"]) <= ids)
        html = get_registry().path_of(v["report_html_key"]).read_text()
        self.assertIn("data:image/png;base64", html)
        for h in ("Statistical analysis", "Goal-driven AutoML", "Root-cause analysis", "Recommendations", "Evidence register"):
            self.assertIn(h, v["report_markdown"])
        self.assertTrue(Path(v["powerbi_result"]["package_dir"], "measures.dax").exists())
        self.assertEqual(v["powerbi_result"]["push"]["status"], "not_configured")
        self.assertEqual(v["powerbi_result"]["layout_source"], "LLM")
        self.assertTrue(all(c["ok"] for c in v["llm_calls"]))
        self.assertNotIn("gsk_SECRET", json.dumps(v, default=str))                          # secrets never in state

    def test_clarification_flow_and_persistence(self):
        from nexusml.graph import build_graph
        from nexusml.runtime import new_thread, snapshot
        from nexusml.state import initial_state
        self.__class__.case = "ambiguous"
        with mock.patch.object(llm, "ask_json", side_effect=coop("ambiguous")):
            g, cfg = build_graph(), new_thread()
            list(g.stream(initial_state(str(SAMPLE / "churn.csv"), "Predict the outcome that matters", thread_id=cfg["configurable"]["thread_id"], **FAST), config=cfg))
            values, nxt = snapshot(g, cfg)
            self.assertIn("await_clarification", nxt)
            values2, nxt2 = snapshot(build_graph(), cfg)                                    # state restored from disk
            self.assertEqual(nxt2, nxt)
            self.assertTrue(values2["dataset_profile"])
            g.update_state(cfg, {"target_column": "churn"})                                 # the human answers
            list(g.stream(None, config=cfg))
            self.assertEqual(snapshot(g, cfg)[0]["target_column"], "churn")
        self.__class__.case = "churn"


class TestHouse(LLMCase):
    case = "house"

    def test_tuning_can_be_disabled(self):
        v = run_to_completion(str(SAMPLE / "house_prices.csv"), "Predict price", enable_tuning=False, auto_approve=True)
        self.assertFalse(v["tuning_report"]["enabled"])
        self.assertTrue(all(not c["tuned"] for c in v["candidate_metrics"]))

    def test_drift_runs_with_reference_data(self):
        full = pd.read_csv(SAMPLE / "house_prices.csv")
        cur = full.iloc[600:].copy()
        cur["sqft"] = cur["sqft"] * 1.5
        cur.to_csv(Path(self.dir, "shifted.csv"), index=False)
        full.head(600).to_csv(Path(self.dir, "ref.csv"), index=False)
        v = run_to_completion(str(Path(self.dir, "shifted.csv")), "Predict price", reference_path=str(Path(self.dir, "ref.csv")), enable_tuning=False, auto_approve=True)
        ks = [r for r in v["stat_results"] if r["test"] == "ks_drift"]
        self.assertTrue(ks and ks[0]["significant"])


class TestSales(LLMCase):
    case = "sales"

    def test_time_series_and_guards_on_llm_output(self):
        v = run_to_completion(str(SAMPLE / "daily_sales.csv"), "What drives sales?", **FAST)
        self.assertEqual((v["analysis_type"], v["problem_type"]), ("sales_performance", "time_series"))   # business != analytical
        self.assertEqual(v["segment_columns"], ["promo"])                                                  # invented 'ghost' dropped
        self.assertEqual(list(v["kpi_results"]), ["Total sales"])                                          # invalid KPI dropped
        self.assertEqual(v["data_summary"]["split"], "chronological")
        self.assertIn("lags", v["data_summary"]["time_series"])
        self.assertTrue(any(k in v["baseline_metrics"]["name"] for k in ("last value", "seasonal naive")))      # v5: the bar also includes "same period last season"


class TestSegments(Tmp):
    def _run(self, case, **kw):
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=coop(case)), \
                mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            return run_to_completion(str(SAMPLE / "churn.csv"), "x", **FAST, **kw)

    def test_clustering(self):
        v = self._run("cluster")
        self.assertEqual(v["errors"], [])
        self.assertTrue(v["cluster_results"]["k"] >= 2)
        self.assertTrue(all(p["name"].startswith("Segment") for p in v["cluster_results"]["profiles"]))
        out = modeling.predict_with_bundle(get_registry().load(v["model_key"]), pd.read_csv(SAMPLE / "churn.csv").head(5))
        self.assertEqual(len(out), 5)

    def test_descriptive_has_no_model_and_no_drift(self):
        v = self._run("descriptive")
        self.assertEqual(v["errors"], [])
        self.assertIsNone(v["model_key"])
        self.assertFalse(any(r["test"] == "ks_drift" for r in v["stat_results"]))


# ---------------------------------------------- no fallback: failures stop the run loudly
class TestNoFallback(Tmp):
    def test_missing_key_stops_before_any_work(self):
        os.environ.pop("GROQ_API_KEY", None)
        with self.assertRaises(llm.LLMRequired) as cm:
            run_to_completion(str(SAMPLE / "churn.csv"), "x", **FAST)
        self.assertIn("GROQ_API_KEY", str(cm.exception))

    def test_groq_rejection_stops_run_with_actionable_message(self):
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", return_value=(None, "HTTP 401", {"ok": False})):
            with self.assertRaises(llm.LLMRequired) as cm:
                run_to_completion(str(SAMPLE / "churn.csv"), "x", **FAST)
        self.assertIn("HTTP 401", str(cm.exception))
        self.assertIn("new key", str(cm.exception))

    def test_invalid_answer_gets_one_corrective_retry_then_succeeds(self):
        seen = []

        def frame(pl):
            seen.append(pl)
            f = frame_for("churn")(pl)
            if len(seen) == 1:
                f["analysis_type"] = "not_a_business_type"            # first answer invalid
            return f
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=coop("churn", {"frame_business_problem": frame})), \
                mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            v = run_to_completion(str(SAMPLE / "churn.csv"), "x", enable_tuning=False, auto_approve=True)
        self.assertEqual(len(seen), 2)
        self.assertIn("your_previous_answer_was_rejected_because", seen[1])
        self.assertEqual(v["analysis_type"], "churn")

    def test_persistently_invalid_answer_stops_run(self):
        bad = lambda pl: {**frame_for("churn")(pl), "problem_type": "sales_performance"}    # business term in the analytical field
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=coop("churn", {"frame_business_problem": bad})), \
                mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with self.assertRaises(llm.LLMRequired) as cm:
                run_to_completion(str(SAMPLE / "churn.csv"), "x", **FAST)
        self.assertIn("unusable output", str(cm.exception))

    def test_ungrounded_claims_are_rejected(self):
        ungrounded = lambda pl: {"root_causes": [{"finding": "made up", "hypothesis": "h", "evidence_ids": ["ZZ99"], "confidence": "high"}]}
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=coop("churn", {"root_cause_analysis": ungrounded})), \
                mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with self.assertRaises(llm.LLMRequired):
                run_to_completion(str(SAMPLE / "churn.csv"), "x", enable_tuning=False, auto_approve=True)

    def test_out_of_range_hyperparameters_are_clipped_not_trusted(self):
        """v6: the AutoML agent may propose a narrower search range (space_hint); it is clipped to safe bounds, never trusted."""
        wild = lambda pl: {"diagnosis": "widen the forest", "actions": [{"type": "search_family", "params": {"family": "RandomForest", "n_trials": 3,
                           "space_hint": {"n_estimators": {"type": "randint", "low": 1, "high": 10 ** 9}}}, "why": "x"}], "stop": None}
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=coop("house", {"automl_round": wild})), \
                mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            v = run_to_completion(str(SAMPLE / "house_prices.csv"), "x", **FAST)
        rf = [t["params"]["n_estimators"] for t in v["automl_report"]["trials"] if t["family"] == "RandomForest" and "n_estimators" in t["params"]]
        self.assertTrue(rf)
        self.assertLessEqual(max(rf), 500)                                       # 10**9 was clipped to the hard bound


class TestPowerBI(unittest.TestCase):
    def test_dax_generation_and_push_without_credentials(self):
        from nexusml import powerbi
        self.assertEqual(powerbi.dax_for({"agg": "sum", "column": "Rev"}), "SUM('Fact'[Rev])")
        self.assertIn("DIVIDE(CALCULATE(COUNTROWS", powerbi.dax_for({"agg": "rate", "column": "c", "positive": "Yes"}))
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("POWERBI_PUSH_URL", "POWERBI_DATASET_ID", "POWERBI_ACCESS_TOKEN"):
                os.environ.pop(k, None)
            self.assertEqual(powerbi.push({})["status"], "not_configured")


if __name__ == "__main__":
    unittest.main()
