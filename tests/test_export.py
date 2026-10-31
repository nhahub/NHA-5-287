import io
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

os.environ["NEXUSML_BACKEND"] = "builtin"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from nexusml import export, ingest, modeling
from nexusml.runtime import run_to_completion
from test_agents import FAST, SAMPLE, LLMCase


class TestPackage(LLMCase):
    def test_package_runs_standalone(self):
        v = run_to_completion(str(SAMPLE / "churn.csv"), "Why churn?", **FAST)
        data = export.build_package(v)
        tmp = tempfile.mkdtemp()
        zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
        pkg = next(Path(tmp).iterdir())
        names = {p.name for p in pkg.iterdir()}
        for want in ("model_bundle.joblib", "predict.py", "serve_app.py", "train.py", "schema.json", "input_template.csv", "requirements.txt", "Dockerfile", "README.md", "nexusml"):
            self.assertIn(want, names)
        env = {k: x for k, x in os.environ.items() if k != "PYTHONPATH"}          # must NOT rely on this project being importable
        r = subprocess.run([sys.executable, "predict.py", "input_template.csv"], cwd=pkg, capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        out = pd.read_csv(pkg / "predictions.csv")
        self.assertIn("prediction", out.columns)
        self.assertTrue(any(c.startswith("proba_") for c in out.columns))
        train_csv = SAMPLE / "churn.csv"
        r = subprocess.run([sys.executable, "train.py", str(train_csv)], cwd=pkg, capture_output=True, text=True, env=env, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn("held-out", r.stdout)
        self.assertTrue((pkg / "model_bundle_retrained.joblib").exists())
        compile((pkg / "serve_app.py").read_text(), "serve_app.py", "exec")
        schema = export.schema_for(v)
        self.assertNotIn("churn", [f["name"] for f in schema])                    # target is not an input
        self.assertNotIn("account_closed_flag", [f["name"] for f in schema])      # dropped leak is not an input
        self.assertIn("scikit-learn==", (pkg / "requirements.txt").read_text())

    def test_no_package_without_model(self):
        with self.assertRaises(ValueError):
            export.build_package({"model_key": None})


class TestModelZooAndIngest(unittest.TestCase):
    def test_new_models_and_safe_spaces(self):
        df = pd.read_csv(SAMPLE / "churn.csv").head(400)
        plan = dict(problem_type="classification", target="churn", drop_columns=["customer_id", "account_closed_flag"], datetime_columns=["signup_date"], date_column="signup_date")
        sp = modeling.make_splits(df, plan)
        out = modeling.fit_default("classification", sp, {"extra_models": True})
        names = {c["name"] for c in out["candidates"]}
        self.assertTrue({"DecisionTree", "KNN", "AdaBoost"} <= names)
        space, notes = modeling.sanitize_space("KNN", {"weights": {"type": "choice", "values": ["distance", "evil"]}, "n_neighbors": {"type": "randint", "low": 1, "high": 10 ** 6}})
        self.assertEqual(space["weights"]["values"], ["distance"])
        self.assertEqual(space["n_neighbors"]["high"], modeling.PARAM_BOUNDS["n_neighbors"][1])

    def test_excel_is_converted_to_csv(self):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            self.skipTest("openpyxl not installed")
        buf = io.BytesIO()
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buf, index=False)
        csv = ingest.to_csv_bytes("data.xlsx", buf.getvalue()).decode()
        self.assertTrue(csv.startswith("a,b"))
        self.assertEqual(ingest.to_csv_bytes("d.csv", b"a,b\n1,2\n"), b"a,b\n1,2\n")


if __name__ == "__main__":
    unittest.main()
