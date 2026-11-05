"""v8: PDF ingestion, provider selection, SMOTE/ADASYN, threshold tuning, segmentation, NB-SVM, CI syntax gate."""
import importlib
import os
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nexusml import config, imbalance, nlp, pdfparse, segmentation, threshold

try:
    import reportlab  # noqa: F401
    from tests.fixtures import make_pdfs
    HAVE_RL = True
except Exception:
    HAVE_RL = False


class SyntaxGate(unittest.TestCase):
    def test_py311_compatible_and_compiles(self):
        import check_py311
        self.assertEqual(check_py311.main(), 0)

    def test_checker_catches_nested_quotes(self):
        import tempfile
        import check_py311
        f = Path(tempfile.mkdtemp()) / "x.py"
        f.write_text('d = {"a": 1}\nt = f"{d.get("a")}"\n')
        if sys.version_info >= (3, 12):
            self.assertTrue(check_py311.scan(f))


class Providers(unittest.TestCase):
    def test_switch_and_status_never_expose_keys(self):
        old = os.environ.get("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = "secret-value-123"
        try:
            self.assertEqual(config.set_provider("groq"), "groq")
            self.assertEqual(config.LLM_KEY_ENV, "GROQ_API_KEY")
            st = config.provider_status()
            self.assertNotIn("secret-value-123", repr(st))
            self.assertTrue(next(r for r in st if r["id"] == "groq")["configured"])
            config.set_provider("gemini")
            self.assertEqual(config.LLM_KEY_ENV in ("GEMINI_API_KEY", "GOOGLE_API_KEY"), True)
            with self.assertRaises(ValueError):
                config.set_provider("nope")
        finally:
            config.set_provider("groq")
            os.environ.pop("GROQ_API_KEY", None) if old is None else os.environ.__setitem__("GROQ_API_KEY", old)

    def test_no_keys_in_source(self):
        import re
        pat = re.compile(r"(gsk_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9]{30,})")
        for p in list((ROOT / "nexusml").rglob("*.py")) + [ROOT / ".env.example"]:
            self.assertIsNone(pat.search(p.read_text(encoding="utf-8", errors="ignore")), str(p))


@unittest.skipUnless(HAVE_RL and "pdfplumber" in pdfparse.available_backends(), "needs reportlab + pdfplumber")
class Pdf(unittest.TestCase):
    def test_multipage_table_is_stitched_and_typed(self):
        r = pdfparse.parse(make_pdfs.table_pdf(70))
        self.assertEqual(len(r["tables"]), 1)
        t = r["tables"][0]
        self.assertEqual(t.shape, (70, 4))
        self.assertEqual(list(t.columns), ["Region", "Units", "Revenue", "Margin"])
        self.assertTrue(pd.api.types.is_numeric_dtype(t["Revenue"]))
        self.assertGreaterEqual(t.attrs["n_pages"], 2)

    def test_text_blocks_cleaned(self):
        b = pdfparse.parse(make_pdfs.text_pdf(4))["blocks"]
        txt = " ".join(b["text"])
        self.assertNotIn("ACME Corp Quarterly Report", txt)          # running header removed
        self.assertNotIn("Page 2 of 4", txt)                          # page number removed
        self.assertIn("customer service", txt)
        self.assertNotIn("cus- tomer", txt)                           # hyphenation re-joined
        self.assertIn("heading", set(b["kind"]))

    def test_mixed_and_ingest(self):
        from nexusml import ingest
        raw = make_pdfs.mixed_pdf()
        self.assertEqual(len(pdfparse.parse(raw)["tables"]), 1)
        csv = ingest.to_csv_bytes("p.pdf", raw)
        self.assertEqual(pd.read_csv(__import__("io").BytesIO(csv)).shape[0], 12)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            pdfparse.parse(b"not a pdf")


class Resampling(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=1200, n_features=8, n_informative=4, weights=[.92, .08], random_state=0)

    def test_balances_and_keeps_originals(self):
        for m in imbalance.METHODS:
            Xr, yr, info = imbalance.resample(self.X, self.y, m, random_state=1)
            c = np.bincount(yr)
            self.assertLessEqual(abs(int(c[0]) - int(c[1])), 2 if m == "adasyn" else 0, m)
            self.assertTrue(np.array_equal(Xr[:len(self.X)], self.X))
            self.assertTrue(np.array_equal(Xr, imbalance.resample(self.X, self.y, m, random_state=1)[0]))

    def test_smote_points_inside_minority_box(self):
        Xr, _, _ = imbalance.resample(self.X, self.y, "smote", random_state=0)
        m = self.X[self.y == 1]
        syn = Xr[len(self.X):]
        self.assertTrue(((syn >= m.min(0) - 1e-9) & (syn <= m.max(0) + 1e-9)).all())

    def test_only_training_folds_are_resampled(self):
        pipe = Pipeline([("s", StandardScaler()), ("model", imbalance.ResamplingClassifier(LogisticRegression(max_iter=500), "smote"))])
        self.assertEqual(len(cross_val_score(pipe, self.X, self.y, cv=3, scoring="f1")), 3)
        pipe.fit(self.X, self.y)
        self.assertEqual(pipe.predict(self.X[:5]).shape, (5,))                   # predicting never resamples

    def test_multiclass_tiny_class_and_ratio(self):
        X3, y3 = make_classification(n_samples=600, n_features=8, n_informative=5, n_classes=3, weights=[.8, .17, .03], random_state=0)
        _, yr, _ = imbalance.resample(X3, y3, "adasyn", random_state=0)
        self.assertGreater(np.bincount(yr).min(), np.bincount(y3).min())
        y1 = np.zeros(100, int)
        y1[0] = 1
        self.assertEqual(np.bincount(imbalance.resample(self.X[:100], y1, "smote")[1])[1], 99)       # lone sample is duplicated
        self.assertAlmostEqual(np.bincount(imbalance.resample(self.X, self.y, "smote", sampling_strategy=0.5)[1])[1] / np.bincount(self.y)[0], 0.5, places=2)


class Thresholds(unittest.TestCase):
    def test_tuned_threshold_improves_oof_f1_on_imbalanced(self):
        X, y = make_classification(n_samples=2500, n_features=10, n_informative=5, weights=[.94, .06], flip_y=.02, class_sep=.8, random_state=1)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=0)
        est = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=500))]).fit(Xtr, ytr)
        out = threshold.tune(est, Xtr, ytr, Xte, yte)
        self.assertTrue(out["adopted"])
        self.assertLess(out["threshold"], 0.5)
        self.assertGreater(out["report"]["oof_gain"], 0.005)
        self.assertEqual(out["estimator"].predict(Xte).shape, yte.shape)
        self.assertIn("heldout_at_tuned", out["report"])

    def test_balanced_data_keeps_default(self):
        X, y = make_classification(n_samples=2000, n_features=10, n_informative=6, random_state=3)
        Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
        est = Pipeline([("s", StandardScaler()), ("m", LogisticRegression())]).fit(Xtr, ytr)
        self.assertFalse(threshold.tune(est, Xtr, ytr, Xte, yte, min_gain=0.02)["adopted"])

    def test_objectives_and_cost(self):
        y = np.array([0] * 90 + [1] * 10)
        p = np.concatenate([np.linspace(0, .6, 90), np.linspace(.3, 1, 10)])
        for o in ("f1", "pr_distance", "youden"):
            self.assertTrue(0 <= threshold.best_threshold(y, p, o)["threshold"] <= 1)
        cheap_fp = threshold.best_threshold(y, p, "cost", cost_fn=20, cost_fp=1)["threshold"]
        dear_fp = threshold.best_threshold(y, p, "cost", cost_fn=1, cost_fp=20)["threshold"]
        self.assertLess(cheap_fp, dear_fp)

    def test_pipeline_end_to_end(self):
        from nexusml import modeling as M
        X, y = make_classification(n_samples=900, n_features=8, n_informative=4, weights=[.92, .08], random_state=2)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        df["y"] = np.where(y == 1, "yes", "no")
        plan = {"problem_type": "classification", "target": "y", "drop_columns": [], "datetime_columns": []}
        sp = M.make_splits(df, plan)
        st = {**M.DEFAULT_STRATEGY, "tune": False, "imbalance": "smote"}
        out = M.fit_default("classification", sp, st)
        rep = M.finalize_threshold("classification", sp, out["models"], out["candidates"], st)
        self.assertTrue(rep["enabled"])
        best = out["models"][M.pick_best(out["candidates"])]
        self.assertEqual(len(best.predict(sp["X_test"])), len(sp["X_test"]))
        from nexusml import leaderboard
        self.assertNotIn(type(leaderboard.final_model(best)).__name__, ("ThresholdedClassifier", "ResamplingClassifier"))


class Segmentation(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=1, weights=[.8, .2], random_state=1)
        rs = np.random.RandomState(0)
        self.df = pd.DataFrame(X, columns=list("abcde"))
        self.df["age"] = (40 + 10 * self.df.a + rs.randn(1000) * 5).round()
        self.df["city"] = rs.choice(["Cairo", "Giza", "Alex"], 1000)
        self.df["churn"] = np.where(y == 1, "yes", "no")
        self.df["spend"] = 100 + 30 * self.df.b + 20 * y + rs.randn(1000) * 10
        self.df["date"] = pd.date_range("2024-01-01", periods=1000, freq="D")

    def test_positive_class_slice(self):
        spec = segmentation.propose(self.df, "only the positive class", "churn", use_llm=False)
        sub, rep = segmentation.apply(self.df, spec)
        self.assertTrue((sub["churn"] == "yes").all())
        self.assertEqual(rep["rows"], (self.df["churn"] == "yes").sum())

    def test_word_boundaries_in_rule_reader(self):
        spec = segmentation.propose(self.df, "city is cairo and age > 40", "churn", use_llm=False)
        self.assertEqual({c["column"] for c in spec["conditions"]}, {"city", "age"})

    def test_last_months_uses_data_end(self):
        spec = segmentation.propose(self.df, "last 3 months", "churn", use_llm=False)
        sub, _ = segmentation.apply(self.df, spec)
        self.assertLess(len(sub), 100)
        self.assertEqual(sub["date"].max(), self.df["date"].max())

    def test_constant_target_is_explained_not_crashed(self):
        res = segmentation.analyze(self.df, segmentation.propose(self.df, "positive class", "churn", False), target="churn")
        self.assertIsNone(res["model"])
        self.assertTrue(any("single value" in n for n in res["notes"]))
        self.assertTrue(res["eda"])

    def test_secondary_target_and_imbalance_strategy(self):
        spec = segmentation.propose(self.df, "positive class", "churn", False)
        res = segmentation.analyze(self.df, spec, target="churn", secondary_target="spend")
        self.assertEqual(res["model"]["target"], "spend")
        res2 = segmentation.analyze(self.df, {"conditions": [{"column": "city", "op": "eq", "value": "Cairo"}]}, target="churn", strategy={"imbalance": "adasyn"})
        self.assertEqual(res2["model"]["strategy"]["imbalance"], "adasyn")
        self.assertNotIn("city", [r["column"] for r in res2["vs_rest"]])          # the filter column is not reported as "distinctive"

    def test_invalid_specs_rejected(self):
        for bad in ({"conditions": [{"column": "nope", "op": "eq", "value": 1}]}, {"conditions": [{"column": "city", "op": "eq", "value": "Paris"}]}, {}):
            with self.assertRaises(segmentation.SegmentError):
                segmentation.apply(self.df, bad)

    def test_registered_as_assistant_tool(self):
        from nexusml.assistant import tools
        self.assertIn("segment_analysis", tools.REGISTRY)


class NlpAdditions(unittest.TestCase):
    def test_nbsvm_binary_and_multiclass(self):
        rs = np.random.RandomState(0)
        words = {0: ["good", "great", "love"], 1: ["bad", "awful", "hate"], 2: ["ok", "fine", "average"]}
        docs, y = [], []
        for _ in range(300):
            c = rs.randint(3)
            docs.append(" ".join(list(rs.choice(words[c], 3)) + list(rs.choice(["the", "a", "movie", "film", "it"], 5))))
            y.append(c)
        y = np.array(y)
        for classes in ((0, 1), (0, 1, 2)):
            m = np.isin(y, classes)
            D = pd.Series(np.array(docs, dtype=object)[m])
            pipe = Pipeline([("t", nlp.text_transformer("word")), ("m", nlp.NBLogisticRegression())]).fit(D[:200], y[m][:200])
            self.assertGreater((pipe.predict(D[200:]) == y[m][200:]).mean(), 0.9)
            self.assertAlmostEqual(float(pipe.predict_proba(D[:3]).sum(1)[0]), 1.0, places=5)

    def test_candidates_include_new_models(self):
        names = " ".join(nlp.candidate_specs("classification", {}, 0))
        self.assertIn("NB-SVM", names)
        self.assertIn("word+char", names)


if __name__ == "__main__":
    unittest.main()
