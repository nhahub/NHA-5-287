"""v9 (NexusML v5): insight-driven model advisor, seasonal time-series models, neural nets, exclusion audit, link ingestion."""
import io
import json
import os
import pickle
import socketserver
import sys
import tempfile
import threading
import unittest
import warnings
import zipfile
import http.server
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from nexusml import advisor, estimators as E, linkfetch, modeling as M


def monthly(n=240, seed=0, trend=0.05, amp=0.25, noise=0.02, walk=False):
    rs = np.random.RandomState(seed)
    t = np.arange(n)
    if walk:
        y = 100 * np.exp(np.cumsum(rs.randn(n) * 0.04))
    else:
        y = 100 * np.exp(trend * t / 12) * (1 + amp * np.sin(2 * np.pi * t / 12)) * np.exp(rs.randn(n) * noise)
    return pd.DataFrame({"date": pd.date_range("2000-01-01", periods=n, freq="MS"), "value": y})


class SeriesInsights(unittest.TestCase):
    def test_finds_season_trend_and_log_scale(self):
        d = monthly()
        i = advisor.series_insights(d.value.values, d.date)
        self.assertEqual(i["period"], 12)
        self.assertGreater(i["seasonal_strength"], 0.8)
        self.assertGreater(i["trend_strength"], 0.5)
        self.assertTrue(i["log_scale"])                       # the swings grow with the level

    def test_random_walk_has_no_season(self):
        d = monthly(walk=True, n=300)
        i = advisor.series_insights(d.value.values, d.date)
        self.assertEqual(i["period"], 0)

    def test_seasonal_lags_added_and_bounded(self):
        lags, rolls = advisor.ts_lags({"period": 12}, [1, 2, 3], [3], n=240)
        self.assertIn(12, lags)
        self.assertIn(12, rolls)
        lags, _ = advisor.ts_lags({"period": 52}, [1, 2], [], n=100)
        self.assertNotIn(52, lags)                            # would eat more than a fifth of a 100-row series


class Estimators(unittest.TestCase):
    def setUp(self):
        d = monthly()
        sp = M.make_splits(d, {"problem_type": "time_series", "target": "value", "date_column": "date", "drop_columns": [], "datetime_columns": []})
        self.sp = sp
        self.st = M.enrich_strategy("time_series", sp, {**M.DEFAULT_STRATEGY})

    def test_prepare_adds_position_phase_and_seasonal_lags(self):
        X = self.sp["X_train"]
        for c in ("ts_t", "ts_phase", "ts_sin", "value_lag12"):
            self.assertIn(c, X.columns)
        self.assertEqual(self.st["ts_period"], 12)
        self.assertEqual(self.st["ts_ref"], "value_lag12")

    def test_season_measured_on_training_part_only(self):
        self.assertLessEqual(self.sp["ts_info"]["insights"]["n"], int(len(monthly()) * 0.81))

    def test_trend_seasonal_extrapolates(self):
        m = E.TrendSeasonalRegressor(period=12, degree=1, log=True).fit(self.sp["X_train"], self.sp["y_train"])
        self.assertGreater(r2_score(self.sp["y_test"], m.predict(self.sp["X_test"])), 0.9)

    def test_seasonal_delta_lets_trees_forecast_a_trend(self):
        X, Xt, y, yt = self.sp["X_train"], self.sp["X_test"], self.sp["y_train"], self.sp["y_test"]
        raw = HistGradientBoostingRegressor(random_state=0).fit(X.drop(columns=["date"], errors="ignore"), y)
        wrapped = E.SeasonalDeltaRegressor(HistGradientBoostingRegressor(random_state=0), col="value_lag12", log=True).fit(X.drop(columns=["date"], errors="ignore"), y)
        r_raw = r2_score(yt, raw.predict(Xt.drop(columns=["date"], errors="ignore")))
        r_w = r2_score(yt, wrapped.predict(Xt.drop(columns=["date"], errors="ignore")))
        self.assertGreater(r_w, r_raw)
        self.assertGreater(r_w, 0.9)

    def test_pickle_and_clone(self):
        for m in (E.SeasonalNaive(col="value_lag12"), E.TrendSeasonalRegressor(), E.SeasonalDeltaRegressor(HistGradientBoostingRegressor(), col="value_lag12"),
                  E.MLPRegressorNet(), E.MLPClassifierNet()):
            clone(m)
            pickle.dumps(m)

    def test_mlp_fits_nonlinear_regression_and_classification(self):
        rs = np.random.RandomState(0)
        X = rs.randn(600, 4)
        y = np.sin(X[:, 0] * 2) + X[:, 1] ** 2 + 0.05 * rs.randn(600)
        z = X[:, 0] * X[:, 1] + 0.05 * rs.randn(600)                  # XOR-like: no linear model can separate it
        r = E.MLPRegressorNet(h1=32, h2=16, random_state=0).fit(X[:450], y[:450])
        self.assertGreater(r2_score(y[450:], r.predict(X[450:])), 0.7)
        c = E.MLPClassifierNet(random_state=0).fit(X[:450], (z[:450] > 0).astype(int))
        self.assertGreater((c.predict(X[450:]) == (z[450:] > 0)).mean(), 0.85)
        self.assertEqual(c.predict_proba(X[:3]).shape, (3, 2))


class Advisor(unittest.TestCase):
    def _plan(self, X, y, task="regression", k=0):
        avail = ["Ridge", "DecisionTree", "KNN", "RandomForest", "HistGradientBoosting", "MLP"] if task == "regression" else \
                ["LogisticRegression", "DecisionTree", "KNN", "RandomForest", "HistGradientBoosting", "MLP"]
        return advisor.build_plan(task, {"X_train": X, "y_train": y, "classes": list(range(k)) if k else None}, avail)

    def test_linear_data_gets_linear_shortlist_with_reasons(self):
        rs = np.random.RandomState(0)
        X = pd.DataFrame(rs.randn(800, 6), columns=list("abcdef"))
        y = X.a * 3 - X.b * 2 + rs.randn(800) * 0.3
        p = self._plan(X, y.values)
        names = [e["family"] for e in p["shortlist"]]
        self.assertIn("Ridge", names)
        self.assertNotIn("MLP", names)
        self.assertTrue(all(e["why"] for e in p["shortlist"] + p["excluded"]))
        self.assertIn("MLP", [e["family"] for e in p["excluded"]])

    def test_nonlinear_data_gets_nonlinear_models(self):
        rs = np.random.RandomState(1)
        X = pd.DataFrame(rs.randn(1200, 5), columns=list("abcde"))
        y = np.sin(X.a * 2) * X.b + (X.c > 0) * 2 + rs.randn(1200) * 0.1
        names = [e["family"] for e in self._plan(X, y.values)["shortlist"]]
        self.assertIn("HistGradientBoosting", names)
        self.assertIn("RandomForest", names)

    def test_tiny_data_avoids_data_hungry_models(self):
        rs = np.random.RandomState(2)
        X = pd.DataFrame(rs.randn(120, 4), columns=list("abcd"))
        p = self._plan(X, (X.a + rs.randn(120) * .5).values)
        self.assertNotIn("HistGradientBoosting", [e["family"] for e in p["shortlist"]])
        self.assertGreaterEqual(len(p["shortlist"]), 2)

    def test_series_plan_adds_seasonal_families_and_drops_trees(self):
        ins = {"period": 12, "seasonal_strength": 0.9, "trend_strength": 0.7}
        avail = ["Ridge", "DecisionTree", "KNN", "RandomForest", "HistGradientBoosting", "Naive (last value)", "Seasonal naive", "TrendSeasonal",
                 "Ridge · seasonal Δ", "HistGradientBoosting · seasonal Δ"]
        p = advisor.plan_series(ins, avail)
        keep = [e["family"] for e in p["shortlist"]]
        self.assertTrue({"Seasonal naive", "TrendSeasonal", "Ridge · seasonal Δ"} <= set(keep))
        self.assertIn("RandomForest", [e["family"] for e in p["excluded"]])
        rw = advisor.plan_series({"period": 0, "seasonal_strength": 0.0, "trend_strength": 0.1, "random_walk": True}, avail)
        self.assertIn("TrendSeasonal", [e["family"] for e in rw["excluded"]])


class SessionIntegration(unittest.TestCase):
    def test_portfolio_runs_only_shortlist_and_audit_verifies_exclusions(self):
        from nexusml import automl_run
        from nexusml.automl_core import Goal
        d = monthly()
        sp = M.make_splits(d, {"problem_type": "time_series", "target": "value", "date_column": "date", "drop_columns": [], "datetime_columns": []})
        sess = automl_run.build_session("time_series", sp, {"tune": False}, Goal.from_spec(None, "time_series"), 42, 1, 90, lambda m: None)
        fams = sess.families()
        self.assertLess(len(fams), len(sess._pool))
        self.assertIn("TrendSeasonal", fams)
        sess.portfolio()
        tried = {t.cfg.family for t in sess.trials}
        self.assertTrue(tried <= set(fams))
        audit = sess.audit_exclusions(max_families=2)
        self.assertTrue(audit and all("reason_excluded" in a for a in audit))
        best = sess.best()
        self.assertIn(best.cfg.family, ("TrendSeasonal", "Seasonal naive", "Ridge", "Ridge · seasonal Δ", "HistGradientBoosting · seasonal Δ"))

    def test_fit_default_forecasts_a_trending_seasonal_series_better_than_naive(self):
        d = monthly()
        sp = M.make_splits(d, {"problem_type": "time_series", "target": "value", "date_column": "date", "drop_columns": [], "datetime_columns": []})
        out = M.fit_default("time_series", sp, {**M.DEFAULT_STRATEGY})
        best = max(out["candidates"], key=lambda c: c["select_score"])
        self.assertGreater(best["r2"], out["baseline"]["r2"])
        self.assertIn("seasonal", out["baseline"]["name"])    # the bar to beat includes 'same period last season'


class Links(unittest.TestCase):
    CSV = b"date,sales\n" + b"\n".join(f"2024-01-{i % 28 + 1:02d},{i * 3}".encode() for i in range(60))

    @classmethod
    def setUpClass(cls):
        csv = cls.CSV
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr("train.csv", csv)
            z.writestr("../evil.csv", csv)
            z.writestr("run.exe", b"x")
        zipped = b.getvalue()

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                def send(body, ct, fn=None):
                    self.send_response(200)
                    self.send_header("Content-Type", ct)
                    self.send_header("Content-Length", str(len(body)))
                    if fn:
                        self.send_header("Content-Disposition", f'attachment; filename="{fn}"')
                    self.end_headers()
                    self.wfile.write(body)
                if self.path.startswith("/api/v1") and not self.headers.get("Authorization"):
                    self.send_response(401)
                    self.end_headers()
                elif self.path == "/api/v1/datasets/view/o/s":
                    send(json.dumps({"totalBytes": len(zipped)}).encode(), "application/json")
                elif self.path == "/api/v1/datasets/view/o/big":
                    send(json.dumps({"totalBytes": 10 ** 9}).encode(), "application/json")
                elif self.path == "/api/v1/datasets/download/o/s":
                    send(zipped, "application/zip", "s.zip")
                elif self.path == "/f.csv":
                    send(csv, "text/csv", "f.csv")
                elif self.path == "/page":
                    send(b"<html></html>", "text/html")
                else:
                    self.send_response(404)
                    self.end_headers()
        cls.srv = socketserver.TCPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.saved = dict(os.environ)
        os.environ.update({"NEXUSML_ALLOW_PRIVATE_URLS": "1", "NEXUSML_KAGGLE_API": cls.base + "/api/v1", "KAGGLE_API_TOKEN": "dummy", "KAGGLE_CONFIG_DIR": "/nonexistent",
                           "NEXUSML_ARTIFACTS": tempfile.mkdtemp()})

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        os.environ.clear()
        os.environ.update(cls.saved)

    def test_parse(self):
        self.assertEqual(linkfetch.parse("https://www.kaggle.com/datasets/o/n")["kind"], "kaggle_dataset")
        self.assertEqual(linkfetch.parse("https://www.kaggle.com/datasets/o/n?select=a.csv")["select"], "a.csv")
        self.assertEqual(linkfetch.parse("https://www.kaggle.com/competitions/titanic")["kind"], "kaggle_competition")
        self.assertIn("raw.githubusercontent.com", linkfetch.parse("https://github.com/u/r/blob/main/x.csv")["url"])
        for bad in ("not a link", "https://www.kaggle.com/code/a/b", "https://www.kaggle.com/"):
            with self.assertRaises(linkfetch.LinkError):
                linkfetch.parse(bad)

    def test_find_urls(self):
        self.assertEqual(linkfetch.find_urls("look at https://kaggle.com/datasets/a/b, please"), ["https://kaggle.com/datasets/a/b"])

    def test_missing_kaggle_token_is_explained(self):
        tok = os.environ.pop("KAGGLE_API_TOKEN")
        try:
            with self.assertRaises(linkfetch.LinkError) as cm:
                linkfetch.fetch("https://www.kaggle.com/datasets/o/s")
            self.assertIn("API token", str(cm.exception))
        finally:
            os.environ["KAGGLE_API_TOKEN"] = tok

    def test_kaggle_dataset_unzips_safely(self):
        names = [n for n, _ in linkfetch.fetch("https://www.kaggle.com/datasets/o/s")["files"]]
        self.assertEqual(names, ["train.csv"])                # path-traversal member and the .exe were dropped

    def test_size_limits(self):
        with self.assertRaises(linkfetch.LinkError) as cm:
            linkfetch.fetch("https://www.kaggle.com/datasets/o/big")
        self.assertIn("select=", str(cm.exception))
        os.environ["NEXUSML_MAX_DOWNLOAD_MB"] = "0.00001"
        try:
            with self.assertRaises(linkfetch.LinkError):
                linkfetch.download(self.base + "/f.csv")
        finally:
            os.environ.pop("NEXUSML_MAX_DOWNLOAD_MB")

    def test_internal_addresses_blocked_by_default(self):
        os.environ["NEXUSML_ALLOW_PRIVATE_URLS"] = "0"
        os.environ["NEXUSML_ALLOW_HTTP"] = "1"
        try:
            with self.assertRaises(linkfetch.LinkError):
                linkfetch.download(self.base + "/f.csv")
            with self.assertRaises(linkfetch.LinkError):
                linkfetch._check_host("169.254.169.254", "https")
        finally:
            os.environ["NEXUSML_ALLOW_PRIVATE_URLS"] = "1"
            os.environ.pop("NEXUSML_ALLOW_HTTP")

    def test_html_page_rejected_and_direct_file_added(self):
        with self.assertRaises(linkfetch.LinkError):
            linkfetch.fetch(self.base + "/page")
        r = linkfetch.add_from_link(self.base + "/f.csv")
        self.assertEqual(r["primary"]["rows"], 60)

    def test_materialize_for_cli(self):
        p = linkfetch.materialize(self.base + "/f.csv", tempfile.mkdtemp())
        self.assertTrue(p.endswith("f.csv") and Path(p).read_bytes() == self.CSV)

    def test_export_bundle_carries_new_modules(self):
        src = (ROOT / "nexusml" / "export.py").read_text()
        self.assertIn('"estimators.py"', src)


if __name__ == "__main__":
    unittest.main()
