"""A strong-but-not-obvious leak (a post-outcome column that agrees with the target ~75% of the time) must not inflate the headline silently."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nexusml.evaluation import harness


class TestMediumLeakIsSurfaced(unittest.TestCase):
    def test_post_outcome_column_triggers_a_warning(self):
        r = np.random.RandomState(5); n = 900
        tenure, calls = r.exponential(20, n), r.poisson(2, n)
        churn = (r.uniform(size=n) < 1 / (1 + np.exp(1.0 + 0.05 * tenure - 0.35 * calls))).astype(int)
        df = pd.DataFrame({"tenure": tenure, "support_calls": calls, "plan": r.choice(["a", "b", "c"], n),
                           "refund_issued_after_cancel": churn * r.choice([1, 1, 1, 0], n), "churned": np.where(churn == 1, "Yes", "No")})
        d = Path(tempfile.mkdtemp()); p = d / "t.csv"; df.to_csv(p, index=False)
        case = {"name": "leak", "path": str(p), "truth": {"target": "churned", "problem_type": "classification"},
                "frame": {"analysis_type": "general_exploration", "problem_type": "classification", "target": "churned", "date_column": None, "segment_columns": [], "kpis": [{"name": "Records", "agg": "count"}]}}
        v = harness.run_case(case, d)["values"]
        codes = [i["code"] for i in v["audit"]["issues"]]
        self.assertIn("leakage_suspect_possible", codes)
        msg = next(i["message"] for i in v["audit"]["issues"] if i["code"] == "leakage_suspect_possible")
        self.assertIn("refund_issued_after_cancel", msg)

    def test_a_clean_dataset_gets_no_such_warning(self):
        r = np.random.RandomState(2); n = 600
        x = r.normal(size=(n, 3)); y = (x[:, 0] + 0.8 * r.normal(size=n) > 0).astype(int)
        df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "y": np.where(y == 1, "pos", "neg")})
        d = Path(tempfile.mkdtemp()); p = d / "c.csv"; df.to_csv(p, index=False)
        case = {"name": "clean", "path": str(p), "truth": {"target": "y", "problem_type": "classification"},
                "frame": {"analysis_type": "general_exploration", "problem_type": "classification", "target": "y", "date_column": None, "segment_columns": [], "kpis": [{"name": "Records", "agg": "count"}]}}
        v = harness.run_case(case, d)["values"]
        self.assertNotIn("leakage_suspect_possible", [i["code"] for i in v["audit"]["issues"]])


if __name__ == "__main__":
    unittest.main()
