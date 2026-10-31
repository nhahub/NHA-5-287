import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["NEXUSML_BACKEND"] = "builtin"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from nexusml import diagnostics, external, llm, plotly_viz as pv, tracking
from nexusml.runtime import run_to_completion
from nexusml.storage import get_registry
from test_agents import FAST, SAMPLE, LLMCase, Tmp, coop


# --------------------------------------------------------------- interactive specs
class TestPlotlySpecs(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "g": rng.choice(["x", "y", "z"], 200),
                                "t": pd.date_range("2024-01-01", periods=200), "y": rng.binomial(1, 0.3, 200)})

    def _ok(self, spec):
        json.dumps(spec)                                         # must be pure JSON
        self.assertIn("data", spec)
        self.assertIn("layout", spec)
        return spec

    def test_dropdown_shows_exactly_one_option_at_a_time(self):
        s = self._ok(pv.hist(self.df, ["a", "b"], "g"))
        btn = s["layout"]["updatemenus"][0]["buttons"]
        self.assertEqual(len(btn), 2)
        for b in btn:
            vis = b["args"][0]["visible"]
            self.assertEqual(len(vis), len(s["data"]))
            self.assertEqual(sum(vis), 3)                        # 3 group traces for the chosen variable
        self.assertEqual(sum(1 for t in s["data"] if t["visible"]), 3)

    def test_explorer_all_chart_types_and_filters(self):
        for t in pv.EXPLORE_TYPES:
            spec = {"type": t, "x": "g", "y": "a", "color": "g" if t in ("Bar", "Line") else None, "agg": "mean"}
            if t == "Line":
                spec["x"] = "t"
            if t == "Scatter":
                spec["x"] = "b"
            s = self._ok(pv.explore(self.df, spec))
            self.assertTrue(s["data"], t)
        f = pv.explore(self.df, {"type": "Bar", "x": "g", "agg": "count", "filters": {"g": ("in", ["x"])}})
        self.assertEqual(f["data"][0]["x"], ["x"])
        self.assertEqual(pv.explore(self.df, {"type": "Bar", "x": "g", "filters": {"a": ("range", 99, 100)}})["data"], [])

    def test_parallel_coordinates_handles_log_and_categorical_params(self):
        trials = [{"rank": i + 1, "params": {"C": 10 ** (i - 3), "max_features": ["sqrt", 0.5, "sqrt", 0.5, "sqrt"][i], "d": None if i == 0 else i}, "mean_score": 0.9 - i * 0.01, "std_score": 0.01}
                  for i in range(5)]
        s = self._ok(pv.parallel({"RandomForest": {"trials": trials}}))
        labels = [d["label"] for d in s["data"][0]["dimensions"]]
        self.assertIn("log10 C", labels)
        self.assertEqual(labels[-1], "CV score")

    def test_cost_curve_picks_cheapest_threshold(self):
        tab = diagnostics.threshold_table(np.array([0, 0, 1, 1, 1, 0] * 20), np.array([0.1, 0.4, 0.6, 0.9, 0.35, 0.7] * 20))
        cheap_fn = pv.cost_curve(tab, c_fp=1, c_fn=50)["data"][1]["x"][0]
        cheap_fp = pv.cost_curve(tab, c_fp=50, c_fn=1)["data"][1]["x"][0]
        self.assertLess(cheap_fn, cheap_fp)                      # costly misses => lower threshold


# --------------------------------------------------------------- diagnostics
class TestDiagnostics(unittest.TestCase):
    def test_calibration_threshold_and_pdp(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"f1": rng.normal(size=400), "f2": rng.normal(size=400)})
        y = (X["f1"] + rng.normal(0, 0.5, 400) > 0).astype(int).to_numpy()
        m = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
        p = m.predict_proba(X)[:, 1]
        cal = diagnostics.calibration_data(y, p)
        self.assertTrue(0 <= cal["brier"] <= 1 and len(cal["fraction_positive"]) >= 3)
        tab = diagnostics.threshold_table(y, p)
        self.assertEqual(tab[0]["tp"] + tab[0]["fn"], int(y.sum()))
        curves = diagnostics.pdp_curves(m, X, ["f1", "f2"], 2)
        self.assertEqual(set(curves), {"f1", "f2"})
        self.assertGreater(curves["f1"]["y"][-1], curves["f1"]["y"][0])      # outcome rises with f1


# --------------------------------------------------------------- tracking / registry
class TestTracking(Tmp):
    def _state(self):
        return {"analysis_plan": {"problem_type": "classification", "target": "churn", "date_column": None, "drop_columns": ["id"]},
                "analysis_type": "churn", "experiment_id": "EXP_1", "dataset_profile": {"n_rows": 100, "quality": {"score": 90}},
                "candidate_metrics": [{"name": "RF", "cv_mean": 0.7, "cv_std": 0.01, "default_cv": 0.7, "select_score": 0.75, "select_source": "nested CV (tuned)", "tuned": True,
                                       "roc_auc": 0.74, "fit_seconds": 1.0, "cv_scores": [0.7, 0.71, 0.69],
                                       "tuning": {"best_params": {"max_depth": 8}, "nested_scores": [0.74, 0.76, 0.75],
                                                  "trials": [{"rank": 1, "params": {"max_depth": 8}, "mean_score": 0.75, "std_score": 0.01, "fit_seconds": 0.3},
                                                             {"rank": 2, "params": {"max_depth": 4}, "mean_score": 0.72, "std_score": 0.02, "fit_seconds": 0.2}]}},
                                      {"name": "LR", "cv_mean": 0.74, "cv_std": 0.01, "default_cv": 0.74, "select_score": 0.74, "select_source": "default CV", "tuned": False,
                                       "roc_auc": 0.73, "fit_seconds": 0.1, "cv_scores": [0.74, 0.73, 0.75], "tuning": None}],
                "best_model_name": "RF", "baseline_metrics": {"cv_score": 0.5}, "data_summary": {"n_features": 5, "split": "random"},
                "tuning_report": {"enabled": True, "n_iter": 2, "inner_cv": 3, "outer_cv": 3, "space_source": "x"}, "strategies_applied": [],
                "primary_metric": "roc_auc", "quality_passed": True, "evidence": [], "charts": [], "llm_calls": [], "user_request": "q"}

    def test_logs_runs_children_trials_and_registers(self):
        info = tracking.log_analysis(self._state(), "models/x.joblib")
        parent = tracking.get_run(info["run_id"])
        self.assertEqual(parent["metrics"]["best_select_score"], 0.75)
        kids = tracking.list_runs(parent=info["run_id"])
        self.assertEqual({k["name"] for k in kids}, {"RF", "LR"})
        rf = next(k for k in kids if k["name"] == "RF")
        self.assertEqual(rf["history"]["cv_fold_score"], [0.7, 0.71, 0.69])
        self.assertEqual(len(tracking.list_runs(parent=rf["run_id"])), 2)               # one run per search trial
        tm = tracking.trial_models(info["run_id"])
        self.assertEqual(tm["RF"]["trials"][0]["params"], {"max_depth": 8})
        self.assertEqual(info["registered"]["stage"], "Staging")
        self.assertEqual(tracking.runs_table()[0]["best_model"], "RF")

    def test_only_one_production_version(self):
        a = tracking.register_model("m", "r1", {"x": 1.0}, "k1")
        b = tracking.register_model("m", "r2", {"x": 2.0}, "k2")
        tracking.set_stage("m", a["version"], "Production")
        tracking.set_stage("m", b["version"], "Production")
        stages = {v["version"]: v["stage"] for v in tracking.registry()["models"]["m"]["versions"]}
        self.assertEqual(stages, {1: "Archived", 2: "Production"})
        with self.assertRaises(ValueError):
            tracking.set_stage("m", 1, "Nonsense")

    def test_mlflow_mirror_is_optional(self):
        info = tracking.log_analysis(self._state(), None)
        with mock.patch.dict(sys.modules, {"mlflow": None}):
            out = tracking.mirror_mlflow(info["run_id"], None)
            self.assertFalse(out["ok"])
            self.assertIn("not installed", out["message"])


# --------------------------------------------------------------- external context
class TestExternal(unittest.TestCase):
    def test_search_parses_and_never_leaks_key(self):
        body = json.dumps({"results": [{"title": "T", "url": "https://www.example.com/a", "content": "x " * 300, "score": 0.9}]}).encode()
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = body
        sent = {}

        def fake_open(req, timeout=0):
            sent["auth"], sent["body"] = req.headers.get("Authorization"), json.loads(req.data)
            return resp
        with mock.patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-SECRET"}), mock.patch("urllib.request.urlopen", side_effect=fake_open):
            r = external.search("industry trends")
        self.assertEqual(sent["auth"], "Bearer tvly-SECRET")
        self.assertEqual(sent["body"]["search_depth"], "basic")
        self.assertEqual(r[0]["domain"], "example.com")
        self.assertLessEqual(len(r[0]["snippet"]), 220)
        self.assertNotIn("SECRET", json.dumps(r))

    def test_missing_key_raises(self):
        os.environ.pop("TAVILY_API_KEY", None)
        with self.assertRaises(external.ExternalError):
            external.search("q")


class TestContextInPipeline(LLMCase):
    case = "churn"
    overrides = {"plan_context_queries": lambda pl: {"queries": ["customer churn drivers telecom industry trends", "ab"]}}

    def test_web_context_becomes_labelled_evidence(self):
        fake_results = [{"title": "Churn rises", "url": "https://x.org/a", "domain": "x.org", "snippet": "Industry churn is rising.", "score": 1}]
        with mock.patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-k"}), mock.patch.object(external, "search", return_value=fake_results):
            v = run_to_completion(str(SAMPLE / "churn.csv"), "Why churn?", enable_tuning=False, auto_approve=True)
        w = [e for e in v["evidence"] if e["id"].startswith("W")]
        self.assertEqual(len(w), 1)
        self.assertIn("EXTERNAL", w[0]["text"])
        self.assertEqual(v["external_context"]["queries"], ["customer churn drivers telecom industry trends"])       # 'ab' too short: rejected
        self.assertIn("External web context", v["report_markdown"])

    def test_skipped_without_key(self):
        os.environ.pop("TAVILY_API_KEY", None)
        v = run_to_completion(str(SAMPLE / "churn.csv"), "x", enable_tuning=False, auto_approve=True)
        self.assertEqual(v["external_context"]["status"], "skipped")


class TestArtifactsAndReport(LLMCase):
    def test_interactive_report_tracking_and_state_snapshot(self):
        v = run_to_completion(str(SAMPLE / "churn.csv"), "Why churn?", **{**FAST, "tune_n_iter": 4})
        ids = {c["id"] for c in v["charts"] if c["spec_key"]}
        for want in ("eda_hist", "eda_box", "eda_corr", "model_automl_trials", "model_calibration", "model_threshold", "model_learning", "model_pdp", "model_lift", "model_roc"):
            self.assertIn(want, ids, want)
        html = get_registry().path_of(v["report_html_key"]).read_text()
        self.assertIn("cdn.plot.ly", html)
        self.assertIn("Plotly.newPlot", html)
        self.assertIn("Model card", v["report_markdown"])
        self.assertIn("Experiment tracking", v["report_markdown"])
        self.assertTrue(v["tracking"]["registered"]["version"] >= 1)
        run = tracking.get_run(v["tracking"]["run_id"])
        self.assertEqual(run["tags"]["best_model"], v["best_model_name"])
        self.assertTrue(any(r["name"].startswith("RandomForest") for r in tracking.list_runs(parent=run["run_id"])))      # v6 names a candidate after the recipe that made it win
        snap = json.loads(get_registry().path_of(f"runs/{v['experiment_id']}/state.json").read_text())      # app can reopen this
        self.assertEqual(snap["experiment_id"], v["experiment_id"])
        self.assertTrue(get_registry().list_runs()[-1].get("state_key"))


class TestFollowUp(LLMCase):
    overrides = {"followup_question": lambda pl: {"answer": "Month-to-month contracts churn most.", "evidence_ids": [pl["evidence"][0]["id"], "BOGUS"], "follow_ups": ["Which regions?"]}}

    def test_answers_are_grounded(self):
        from nexusml import qa
        v = run_to_completion(str(SAMPLE / "churn.csv"), "Why churn?", enable_tuning=False, auto_approve=True)
        a = qa.answer(v, "Which contracts churn?", [])
        self.assertEqual(len(a["evidence_ids"]), 1)
        self.assertNotIn("BOGUS", a["evidence_ids"])
        self.assertTrue(a["answer"])


if __name__ == "__main__":
    unittest.main()
