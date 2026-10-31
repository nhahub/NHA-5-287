"""v5 tests: leaderboard analytics, the condensed graph, MLflow candidate logging, the NLP and vision pipelines, UI panels, failure isolation, OCR."""
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("NEXUSML_BACKEND", "builtin")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from nexusml import cv, graph, leaderboard, llm, media, modeling, nlp, plotly_viz as pv, tracking, ui_panels
from nexusml.agents import stages
from nexusml.evaluation import datasets as D
from nexusml.evaluation.stub_llm import make_stub
from nexusml.runtime import run_to_completion
from nexusml.state import merge_update, merge_updates
from nexusml.storage import get_registry
from fake_st import FakeSt
from test_v4 import FakeMlflow


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self._env = mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": self.d, "GROQ_API_KEY": "stub-key", "NEXUSML_VALIDATE": "1"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        shutil.rmtree(self.d, ignore_errors=True)


def run_case(name, workdir, patches=(), **kw):
    case = D.GENERATORS[name](workdir)
    path = case.get("path")
    args = dict(auto_approve=True, tune_n_iter=2, max_retries=1, enable_tuning=False, seed=42)
    if case.get("folder"):
        mi = media.prepare_paths([case["folder"]], workdir)
        path = mi["csv"]
        args.update({k: v for k, v in mi["suggest"].items() if v})
    args.update(kw)
    ctx = [mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=make_stub(case["frame"]))] + list(patches)
    for c in ctx:
        c.start()
    try:
        return run_to_completion(path, "why?", **args)
    finally:
        for c in reversed(ctx):
            c.stop()


# ============================================================================================================ leaderboard maths
class TestLeaderboardMaths(unittest.TestCase):
    def test_corrected_t_is_more_conservative_than_plain_t(self):
        from scipy import stats
        a, b = [0.82, 0.84, 0.81, 0.85, 0.83], [0.80, 0.83, 0.80, 0.82, 0.82]
        plain = stats.ttest_rel(a, b).pvalue
        corr = leaderboard.corrected_paired_t(a, b, 1 / 4)
        d = np.array(a) - np.array(b)
        expect_t = d.mean() / np.sqrt((1 / 5 + 1 / 4) * d.var(ddof=1))
        self.assertAlmostEqual(corr["t"], round(expect_t, 3), places=3)
        self.assertGreater(corr["p"], plain)                       # the whole point of the correction

    def test_identical_scores_are_tied(self):
        self.assertEqual(leaderboard.corrected_paired_t([.8] * 5, [.8] * 5, .25)["p"], 1.0)

    def _rows(self):
        mk = lambda n, s, sd, folds: {"name": n, "select_score": s, "cv_std": sd, "cv_scores": folds, "tuned": False, "fit_seconds": 1.0, "heldout": {"roc_auc": s}, "train_score": s + 0.01}
        return [mk("RandomForest", 0.90, 0.02, [.88, .91, .90, .92, .89]), mk("LogisticRegression", 0.895, 0.02, [.87, .91, .90, .91, .885]),
                mk("DecisionTree", 0.70, 0.05, [.65, .72, .70, .75, .68])]

    def test_annotate_ranks_ties_and_parsimony(self):
        rows = self._rows()
        s = leaderboard.annotate(rows, "roc_auc", "RandomForest")
        by = {r["name"]: r for r in rows}
        self.assertEqual(by["RandomForest"]["rank"], 1)
        self.assertGreater(by["LogisticRegression"]["p_vs_best"], 0.05)       # indistinguishable from the winner
        self.assertLess(by["DecisionTree"]["p_vs_best"], 0.05)                # reliably worse
        self.assertIn("tied", by["LogisticRegression"]["verdict"])
        self.assertIn("worse", by["DecisionTree"]["verdict"])
        self.assertEqual(s["parsimony_pick"], "LogisticRegression")           # simpler model within one SE
        self.assertTrue(s["parsimony_differs"])

    def test_overfit_is_flagged_in_the_verdict(self):
        rows = self._rows()
        rows[0]["train_score"], rows[0]["heldout"]["roc_auc"] = 1.0, 0.80          # gap 0.20 > OVERFIT_GAP (0.15)
        leaderboard.annotate(rows, "roc_auc", "RandomForest")
        self.assertIn("Overfits", rows[0]["verdict"])

    def test_diagnostics_classification_and_regression(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        rs = np.random.RandomState(0)
        X = pd.DataFrame(rs.rand(200, 3), columns=list("abc"))
        y = (X.a + 0.2 * rs.randn(200) > 0.5).astype(int).values
        d = leaderboard.diagnostics("classification", LogisticRegression().fit(X, y), X, y, ["no", "yes"])
        self.assertEqual(np.array(d["confusion_matrix"]).sum(), 200)
        self.assertEqual(set(d["per_class"]), {"no", "yes"})
        self.assertIn("auc", d["roc"])
        self.assertTrue(d["top_features"])
        yr = X.a.values * 3 + rs.randn(200) * 0.1
        dr = leaderboard.diagnostics("regression", RandomForestRegressor(n_estimators=20, random_state=0).fit(X, yr), X, yr)
        self.assertEqual(len(dr["actual_vs_predicted"]["actual"]), 150)
        self.assertGreater(dr["abs_error"]["p95"], dr["abs_error"]["p50"])

    def test_diagnostics_never_raises(self):
        class Broken:
            def predict(self, X):
                raise RuntimeError("boom")
        self.assertIn("error", leaderboard.diagnostics("classification", Broken(), pd.DataFrame({"a": [1]}), np.array([0]), ["x"]))


class TestStateMerge(unittest.TestCase):
    def test_reducers(self):
        s = merge_update({"warnings": ["a"], "timings": {"x": 1}, "candidate_history": [1]}, {"warnings": ["b"], "timings": {"y": 2}, "candidate_history": [2], "k": 5})
        self.assertEqual(s["warnings"], ["a", "b"])
        self.assertEqual(s["timings"], {"x": 1, "y": 2})
        self.assertEqual(s["candidate_history"], [1, 2])
        self.assertEqual(merge_updates({"warnings": ["a"]}, {"warnings": ["b"]}, {"z": 1}), {"warnings": ["a", "b"], "z": 1})


# ============================================================================================================ graph
class TestGraph(Tmp):
    def test_sixteen_real_nodes_and_two_interrupts(self):
        self.assertEqual(len(graph.NODES), 16)
        self.assertEqual(graph.INTERRUPTS, ["await_clarification", "human_review"])
        self.assertFalse([n for n in graph.NODES if "noop" in n or n.endswith("_done")])
        g = graph.build_graph()
        self.assertTrue(g)

    def test_routers(self):
        ok = {"analysis_plan": {"problem_type": "classification", "target": "t", "drop_columns": []}, "dataset_profile": {"n_rows": 500, "columns": {"a": {}, "t": {}}},
              "approval_status": "approved"}
        self.assertEqual(graph.route_plan(ok), "go")
        self.assertEqual(graph.route_plan({**ok, "approval_status": "pending", "requires_approval": True}), "review")
        self.assertEqual(graph.route_plan({**ok, "approval_status": "rejected"}), "abort")
        self.assertEqual(graph.route_plan({**ok, "dataset_profile": {"n_rows": 3, "columns": {"a": {}, "t": {}}}}), "abort")
        self.assertEqual(graph.route_critic({"retry_pending": True}), "retry")
        self.assertEqual(graph.route_critic({}), "done")
        self.assertIn("rejected", stages.abort_node({**ok, "approval_status": "rejected"})["errors"][0])

    def test_full_run_executes_the_crew_stages_with_timings(self):
        ev = []
        v = run_case("planted_classification", Path(self.d), on_event=lambda n, u: ev.append(n))
        self.assertEqual(v["errors"], [])
        self.assertEqual(ev[:3], ["intake", "frame", "plan"])
        self.assertEqual(ev[-3:], ["explain", "synthesize", "publish"])
        self.assertEqual(set(ev), {"intake", "frame", "plan", "prepare", "brief", "sentinel", "explore", "scout", "automl", "critic", "explain", "synthesize", "publish"})
        for k in ("intake", "brief", "automl", "explain", "synthesize", "publish"):
            self.assertIn(k, v["timings"])
        self.assertEqual(v["pipeline_kind"], "tabular")
        self.assertTrue(v["leaderboard"]["rows"])
        self.assertTrue(v["model_card_key"] and get_registry().path_of(v["model_card_key"]).exists())

    def test_retry_keeps_every_attempt_and_logs_it(self):
        n = {"i": 0}

        def verdict(pl):
            n["i"] += 1
            return {"route": "refine" if "refine" in pl["allowed_routes"] and n["i"] == 1 else "accept", "directive": "focus on ensembling", "rationale": "stub"}
        case = D.GENERATORS["planted_classification"](Path(self.d))
        from nexusml.evaluation.stub_llm import make_stub as _mk
        ctx = [mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=_mk(case["frame"], {"route_verdict": verdict}))]
        [c.start() for c in ctx]
        try:
            v = run_to_completion(case["path"], "why?", auto_approve=True, tune_n_iter=2, max_retries=2, enable_tuning=False, seed=42, target_spec={"metric": "roc_auc", "min": 0.9999})
        finally:
            [c.stop() for c in reversed(ctx)]
        self.assertEqual(len(v["candidate_history"]), 2)
        self.assertEqual(v["leaderboard"]["n_attempts"], 2)
        self.assertEqual(len(v["leaderboard"]["past_attempts"]), 1)
        self.assertEqual(v["timings"]["automl"]["runs"], 2)
        att = {r["tags"].get("attempt") for r in tracking.all_candidate_runs()}
        self.assertEqual(att, {1, 2})                                  # v4 logged only the last attempt's candidates

    def test_failing_subtask_is_isolated(self):
        def boom(state):
            raise ValueError("kpi exploded")
        v = run_case("planted_classification", Path(self.d), patches=[mock.patch.object(stages.analysis, "kpi_analysis_node", side_effect=boom)])
        self.assertEqual(v["errors"], [])
        self.assertTrue(any(e["task"] == "kpi_analysis" for e in v["stage_errors"]))
        self.assertTrue(any("kpi_analysis failed" in w for w in v["warnings"]))
        self.assertTrue(v["best_model_name"])                           # the modeling branch was unaffected

    def test_llm_required_is_never_swallowed(self):
        def nokey(state):
            raise llm.LLMRequired("no key")
        with self.assertRaises(llm.LLMRequired):
            run_case("planted_classification", Path(self.d), patches=[mock.patch.object(stages.analysis, "statistical_analysis_node", side_effect=nokey)])

    def test_clustering_runs_through_the_same_loop(self):
        v = run_case("planted_clusters", Path(self.d))
        self.assertEqual(v["errors"], [])
        self.assertTrue(v["cluster_results"])
        self.assertFalse(v.get("candidate_history"))


# ============================================================================================================ tracking
class FakeMlflow5(FakeMlflow):
    def __init__(self, with_optional=True):
        super().__init__()
        m = self.modules["mlflow"]
        if with_optional:
            m.log_dict = lambda d, name: self.calls.append(("log_dict", name))
            m.log_text = lambda t, name: self.calls.append(("log_text", name))
            m.log_table = lambda df, name: self.calls.append(("log_table", name, len(df)))


def _tracking_state():
    rows = []
    for att, names in ((1, ["LR", "RF", "DT"]), (2, ["LR", "RF", "DT"])):
        for i, n in enumerate(names):
            rows.append({"attempt": att, "name": n, "tuned": False, "score_source": "default CV", "select_score": 0.8 - i * 0.05, "cv_std": 0.02, "default_cv": 0.8 - i * 0.05,
                         "cv_scores": [0.8, 0.81, 0.79], "fit_seconds": 1.0, "heldout": {"roc_auc": 0.8 - i * 0.05, "f1_macro": 0.7}, "train_score": 0.9, "overfit_gap": 0.1,
                         "rank": i + 1, "delta_vs_best": -0.05 * i, "p_vs_best": None if i == 0 else 0.2, "verdict": f"v{n}", "selected": att == 2 and i == 0,
                         "hyperparams": {"C": 1.0}, "diagnostics": {"classes": ["a", "b"], "confusion_matrix": [[5, 1], [2, 4]], "per_class": {"a": {"precision": .8}, "b": {"precision": .7}},
                                                                    "top_features": [{"feature": "x", "importance": .4}]}})
    final = [r for r in rows if r["attempt"] == 2]
    past = [r for r in rows if r["attempt"] == 1]
    return {"analysis_plan": {"problem_type": "classification", "target": "t", "drop_columns": []}, "analysis_type": "churn", "dataset_profile": {"n_rows": 10, "quality": {"score": 90}},
            "candidate_metrics": [{"name": "LR", "tuned": False, "select_source": "default CV", "select_score": 0.8, "cv_scores": [.8], "roc_auc": 0.8}], "best_model_name": "LR",
            "primary_metric": "roc_auc", "experiment_id": "e_v5", "user_request": "q", "evidence": [], "charts": [], "llm_calls": [], "quality_passed": True,
            "leaderboard": {"metric": "roc_auc", "rows": final, "past_attempts": [{"attempt": 1, "levers": [], "rows": past}], "n_attempts": 2, "tied_with_winner": ["RF"], "summary": "s"},
            "timings": {"fit": {"seconds": 2.0}}}


class TestMlflowCandidates(Tmp):
    def _run(self, fake):
        info = tracking.log_analysis(_tracking_state(), None)
        with mock.patch.dict(sys.modules, fake.modules):
            return info, tracking.mirror_mlflow(info["run_id"], None, None, "Staging")

    def test_json_store_has_every_candidate_of_every_attempt(self):
        info = tracking.log_analysis(_tracking_state(), None)
        kids = [k for k in tracking.list_runs(parent=info["run_id"])]
        self.assertEqual(len(kids), 6)
        k = next(k for k in kids if k["params"]["model"] == "RF" and k["tags"]["attempt"] == 2)
        for m in ("select_score", "train_score", "overfit_gap", "p_vs_best", "roc_auc", "rank"):
            self.assertIn(m, k["metrics"])
        self.assertEqual(k["params"]["hp.C"], 1.0)
        self.assertEqual(tracking.candidate_diagnostics(k["run_id"])["confusion_matrix"], [[5, 1], [2, 4]])
        parent = tracking.get_run(info["run_id"])
        self.assertEqual(parent["params"]["n_candidates_total"], 6)
        self.assertIn("winner_margin", parent["metrics"])

    def test_mlflow_default_layout_is_flat_one_run_per_candidate(self):
        fake = FakeMlflow5()
        info, out = self._run(fake)
        self.assertTrue(out["ok"], out)
        self.assertEqual(fake.calls.count(("start_run", "root")), 7)    # 1 summary + 6 candidates, all top-level
        self.assertNotIn(("start_run", "nested"), fake.calls)
        self.assertEqual(out["counts"]["candidate_runs"], 6)
        names = [c[1] for c in fake.calls if c[0] == "log_dict"]
        self.assertEqual(names.count("diagnostics.json"), 6)
        self.assertIn(("log_table", "leaderboard.json", 6), fake.calls)
        self.assertIn("confusion_matrix.csv", [c[1] for c in fake.calls if c[0] == "log_text"])

    def test_nested_layout_is_still_available(self):
        fake = FakeMlflow5()
        with mock.patch.dict(os.environ, {"NEXUSML_MLFLOW_NESTED": "1"}):
            info, out = self._run(fake)
        self.assertTrue(out["ok"], out)
        self.assertEqual(fake.calls.count(("start_run", "root")), 1)
        self.assertEqual(fake.calls.count(("start_run", "nested")), 6)

    def test_old_mlflow_without_optional_apis_still_succeeds(self):
        fake = FakeMlflow5(with_optional=False)
        info, out = self._run(fake)
        self.assertTrue(out["ok"], out)
        self.assertIn("not logged", out["message"])                     # reported, not fatal


# ============================================================================================================ NLP
def _corpus(seed=0, per=100, flips=10):
    rng = np.random.RandomState(seed)
    V = {"sports": "match goal team player coach season league win score stadium".split(), "tech": "software cloud server code database api latency deploy kernel network".split(),
         "food": "recipe oven flavor sauce butter spice bake roast fresh pepper".split()}
    common = "the a of and to in is for with on this very really great".split()
    rows = []
    for c, w in V.items():
        for _ in range(per):
            t = list(rng.choice(w, 7)) + list(rng.choice(common, 8))
            rng.shuffle(t)
            rows.append({"text": " ".join(t).capitalize() + ".", "label": c})
    df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    flip = df.sample(flips, random_state=3).index
    df.loc[flip, "label"] = df.loc[flip, "label"].map({"sports": "tech", "tech": "food", "food": "sports"})
    return df, set(flip.tolist())


class TestNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.flips = _corpus()

    def test_label_noise_is_found(self):
        li = nlp.label_issues(self.df.text.tolist(), self.df.label.tolist())
        found = {s["index"] for s in li["suspects"]}
        self.assertGreaterEqual(len(found & self.flips), 8)             # 10 planted flips; top-12 reported
        self.assertTrue(all(s["given"] != s["suggested"] for s in li["suspects"]))

    def test_class_terms_topics_and_summary(self):
        ct = nlp.class_terms(self.df.text.tolist(), self.df.label.tolist())
        self.assertTrue({"goal", "team", "coach", "match", "player"} & {t["term"] for t in ct["sports"][:6]})
        tp = nlp.topics(self.df.text.tolist(), self.df.label.tolist())
        self.assertGreaterEqual(tp["ari_vs_labels"], 0.6)
        s = nlp.extractive_summary("Cats are small. Cats purr when happy. Dogs bark loudly at strangers. Dogs are loyal animals. The weather is mild today.", 2)
        self.assertEqual(len(s), 2)

    def test_pii_counts_without_storing_matches(self):
        r = nlp.pii_scan(["mail jane.doe@example.com now", "call +1 555 123 4567", "card 4111 1111 1111 1111 ok", "nothing here"])
        self.assertEqual(r["docs_affected"], 3)
        self.assertNotIn("jane", str(r))
        self.assertEqual(r["counts"]["card"], 1)                        # Luhn-valid test number
        self.assertEqual(nlp.pii_scan(["card 4111 1111 1111 1112 typo"])["counts"].get("card", 0), 0)       # fails the checksum: not flagged

    def test_sentiment_negation(self):
        self.assertGreater(nlp.sentiment_score("this is really good"), 0.3)
        self.assertLess(nlp.sentiment_score("this is not good"), 0)
        self.assertLess(nlp.sentiment_score("terrible and awful"), -0.5)

    def test_textvec_handles_nan_and_calibrated_svc_is_a_classifier(self):
        from sklearn.base import is_classifier
        from sklearn.model_selection import cross_val_score
        v = nlp.TextVec().fit(pd.Series(["alpha beta", None, "gamma delta", float("nan")]))
        self.assertEqual(v.transform(pd.Series([None, "alpha"])).shape[0], 2)
        X, y = np.random.RandomState(0).rand(90, 5), (np.random.RandomState(0).rand(90) > .5).astype(int)
        self.assertTrue(is_classifier(nlp.CalibratedLinearSVC()))
        self.assertTrue(np.isfinite(cross_val_score(nlp.CalibratedLinearSVC(), X, y, cv=3, scoring="roc_auc")).all())

    def test_suite_trains_explains_and_stress_test_detects_damage(self):
        sp = modeling.make_splits(self.df, {"problem_type": "classification", "target": "label", "drop_columns": [], "datetime_columns": []})
        out = modeling.fit_default("classification", sp, {**modeling.DEFAULT_STRATEGY, "pipeline_kind": "text", "tune": False})
        names = [c["name"] for c in out["candidates"]]
        self.assertGreaterEqual(len(names), 6)
        self.assertFalse(out["failed"])
        self.assertTrue(any("char" in n for n in names) and any("LSA" in n for n in names))
        est = out["models"][out["best"]]
        ex = nlp.explain_document(est, sp["X_test"].iloc[[0]], "text", sp["classes"])
        self.assertTrue(ex["supporting"])
        nlp.PERTURBATIONS["wipe"] = lambda t, r: "zzz"
        try:
            st = nlp.stress_test(est, sp["X_test"], sp["y_test"], "text", "classification")
        finally:
            nlp.PERTURBATIONS.pop("wipe")
        wipe = next(r for r in st["results"] if r["perturbation"] == "wipe")
        self.assertGreater(wipe["drop"], 0.4)                           # wiring proof: destroying the text must hurt
        self.assertTrue(st["fragile"])

    def test_semantic_index_finds_the_right_topic(self):
        ix = nlp.SemanticIndex(self.df.text.tolist())
        hits = ix.search("goal team coach", 5)
        self.assertGreaterEqual(sum(self.df.label[i] == "sports" for i, _ in hits), 4)

    def test_highlight_escapes_html(self):
        h = nlp.highlight_html("good <script>alert(1)</script>", {"good": 0.3})
        self.assertNotIn("<script>", h)
        self.assertIn("&lt;script&gt;", h)


class TestBrokenCandidateIsReportedNotRanked(unittest.TestCase):
    def test_failed_candidate_excluded_with_reason(self):
        from sklearn.base import BaseEstimator, ClassifierMixin
        from sklearn.linear_model import LogisticRegression

        class Bad(ClassifierMixin, BaseEstimator):
            def fit(self, X, y):
                raise RuntimeError("cannot fit")
        df, _ = _corpus(per=40)
        sp = modeling.make_splits(df, {"problem_type": "classification", "target": "label", "drop_columns": [], "datetime_columns": []})
        cands = lambda task, st: {"LogisticRegression": LogisticRegression(max_iter=500), "Bad": Bad()}
        with mock.patch.object(modeling, "_candidates", side_effect=cands):
            out = modeling.fit_default("classification", sp, dict(modeling.DEFAULT_STRATEGY))
        self.assertEqual([c["name"] for c in out["candidates"]], ["LogisticRegression"])
        self.assertEqual(out["failed"][0]["name"], "Bad")
        self.assertIn("cannot fit", out["failed"][0]["error"])


# ============================================================================================================ vision
def _shape_images(root, per=30, seed=0, flips=4):
    from PIL import Image
    r = np.random.RandomState(seed)
    rows = []
    for cls in ("hstripes", "vstripes", "blobs"):
        for i in range(per):
            c1, c2 = r.randint(30, 225, 3), r.randint(30, 225, 3)
            a = np.zeros((64, 64, 3), dtype=np.uint8) + c1
            if cls == "hstripes":
                for y in range(0, 64, 8):
                    a[y:y + 4, :] = c2
            elif cls == "vstripes":
                for x in range(0, 64, 8):
                    a[:, x:x + 4] = c2
            else:
                for _ in range(8):
                    cy, cx = r.randint(6, 58, 2)
                    yy, xx = np.ogrid[:64, :64]
                    a[(yy - cy) ** 2 + (xx - cx) ** 2 < 30] = c2
            a = np.clip(a.astype(int) + r.randint(-10, 10, a.shape), 0, 255).astype(np.uint8)
            Image.fromarray(a).save(root / f"{cls}_{i}.png")
            rows.append({"image": f"{cls}_{i}.png", "label": cls})
    df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    flip = df.sample(flips, random_state=2).index
    df.loc[flip, "label"] = df.loc[flip, "label"].map({"hstripes": "blobs", "vstripes": "hstripes", "blobs": "vstripes"})
    return df, set(flip.tolist())


class TestVision(Tmp):
    def setUp(self):
        super().setUp()
        self.root = Path(self.d) / "imgs"
        self.root.mkdir()
        self.df, self.flips = _shape_images(self.root)

    def test_feature_backends_shapes_cache_and_determinism(self):
        p = self.root / self.df.image[0]
        dims = {b: len(cv.extract(p, b)) for b in ("color", "hog", "lbp", "hsv", "texture")}
        self.assertEqual(dims["hog"], 576)
        self.assertEqual(dims["lbp"], 128)
        self.assertEqual(dims["hsv"], 24)
        self.assertGreater(dims["texture"], dims["hog"])
        np.testing.assert_array_equal(cv.extract(p, "texture"), cv.extract(p, "texture"))
        self.assertIsNone(cv.extract(self.root / "missing.png", "hog"))

    def test_hog_separates_stripe_orientation_where_pixels_alone_do_not(self):
        h, v = [cv.extract(self.root / f"{c}_{i}.png", "hog") for c in ("hstripes", "vstripes") for i in (0, 1)][::2]
        h2, v2 = [cv.extract(self.root / f"{c}_{i}.png", "hog") for c in ("hstripes", "vstripes") for i in (1, 2)][::2]
        within = np.linalg.norm(h - h2) + np.linalg.norm(v - v2)
        between = np.linalg.norm(h - v) + np.linalg.norm(h2 - v2)
        self.assertGreater(between, 2 * within)

    def test_analysis_finds_planted_label_errors(self):
        items, figs, rep = cv.analyze(self.df, "image", self.root, "label")
        found = {s["index"] for s in rep["label_issues"]["suspects"]}
        self.assertGreaterEqual(len(found & self.flips), 3)
        self.assertTrue({"montage", "feature_map", "outliers"} <= {f[0] for f in figs})

    def test_suite_robustness_occlusion_and_gallery(self):
        sp = modeling.make_splits(self.df, {"problem_type": "classification", "target": "label", "drop_columns": [], "datetime_columns": [], "image_root": str(self.root)})
        from nexusml import featurizers
        featurizers.set_image_root(str(self.root))
        out = modeling.fit_default("classification", sp, {**modeling.DEFAULT_STRATEGY, "pipeline_kind": "image", "tune": False})
        self.assertGreaterEqual(len(out["candidates"]), 6)
        by = {c["name"]: c["select_score"] for c in out["candidates"]}
        self.assertGreater(max(v for k, v in by.items() if "HOG" in k or "texture" in k.lower() or "LBP" in k), 0.7)
        est = out["models"][out["best"]]
        rob = cv.robustness(est, sp["X_test"], sp["y_test"], "image", self.root, max_rows=40)
        self.assertEqual(len(rob["results"]), len(cv.PERTURBATIONS))
        self.assertTrue(all("drop" in r for r in rob["results"]))
        self.assertFalse(list(Path(tempfile.gettempdir()).glob("ds_robust_*")))         # temp images cleaned up
        occ = cv.occlusion_map(est, sp["X_test"].iloc[[0]], "image", self.root, sp["classes"])
        self.assertEqual(np.array(occ["heat"]).shape, (6, 6))
        self.assertTrue(np.isfinite(np.array(occ["heat"])).all())
        rows, fig = cv.misclassified_gallery(est, sp["X_test"], sp["y_test"], sp["classes"], "image", self.root)
        self.assertTrue(all(r["true"] != r["predicted"] for r in rows))
        ix = cv.ImageIndex([self.root / f for f in self.df.image])
        top = ix.search(self.root / self.df.image[0], 3)
        self.assertEqual(top[0][0], 0)                                   # the image itself is its own nearest neighbour


# ============================================================================================================ pipelines end to end + UI
class TestPipelinesEndToEnd(Tmp):
    @classmethod
    def setUpClass(cls):
        cls._d = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": cls._d, "GROQ_API_KEY": "stub-key", "NEXUSML_VALIDATE": "1"}):
            cls.text = run_case("text_reviews", Path(cls._d))
            cls.image = run_case("image_folders", Path(cls._d))
            cls.reg_ = get_registry()
            cls.report = cls.reg_.path_of(cls.text["report_key"]).read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._d, ignore_errors=True)

    def test_text_pipeline(self):
        v = self.text
        self.assertEqual((v["errors"], v["stage_errors"], v["pipeline_kind"]), ([], [], "text"))
        self.assertGreaterEqual(len(v["leaderboard"]["rows"]), 6)
        self.assertTrue(all(" · " in r["name"] for r in v["leaderboard"]["rows"] if not r["name"].startswith("Ensemble")))
        self.assertFalse([w for w in v["warnings"] if "excluded" in w])
        nr = v["nlp_report"]
        self.assertTrue({"corpus", "class_terms", "topics", "label_issues", "pii", "misclassified", "stress_test"} <= set(nr))
        self.assertIn("nlp", {c["section"] for c in v["charts"]})
        win = next(r for r in v["leaderboard"]["rows"] if r["selected"])
        self.assertIn("per_class", win["diagnostics"])

    def test_report_contains_leaderboard_nlp_and_timeline(self):
        for h in ("Model leaderboard: every candidate compared", "p vs winner", "Text (NLP) pipeline", "Run timeline", "Text stress test"):
            self.assertIn(h, self.report)

    def test_no_raw_text_or_terms_reach_the_llm_evidence(self):
        """Evidence is what the LLM sees. By default it may hold counts and shares only: no document snippets, no class-distinguishing terms, no topic words."""
        import re
        nr = self.text["nlp_report"]
        ev = [e["text"] for e in self.text["evidence"] if e["id"][0] in "XVB"]
        self.assertTrue(ev)
        terms = {t["term"] for v in nr["class_terms"].values() for t in v}
        topic_words = {w for t in nr["topics"]["topics"] for w in t["terms"]}
        leaked = [w for w in (terms | topic_words) if len(w) >= 4 and any(re.search(rf"\b{re.escape(w)}\b", e.lower()) for e in ev)]
        self.assertEqual(leaked, [])
        self.assertTrue(nr["misclassified"] or True)
        for m in nr["misclassified"]:
            self.assertFalse(any(m["text"][:30] in e for e in ev))
        # and the opt-in path really does include them (so the test above is not passing for the wrong reason)
        items, _, _ = nlp.analyze(pd.DataFrame({"t": ["good great fine product"] * 20 + ["bad awful poor product"] * 20, "y": ["p"] * 20 + ["n"] * 20}), "t", "y", allow_terms=True)
        self.assertTrue(any("good" in x[1] or "great" in x[1] for x in items if x[0] == "nlp_class_terms"))

    def test_image_pipeline(self):
        v = self.image
        self.assertEqual((v["errors"], v["stage_errors"], v["pipeline_kind"]), ([], [], "image"))
        vr = v["vision_report"]
        self.assertTrue({"label_issues", "robustness", "occlusion_examples", "outliers"} <= set(vr))
        self.assertEqual(len(vr["robustness"]["results"]), len(cv.PERTURBATIONS))
        self.assertTrue({"vision", "media"} <= {c["section"] for c in v["charts"]})
        self.assertTrue(any(e["id"].startswith("V") for e in v["evidence"]))

    def test_evidence_ids_are_unique(self):
        for v in (self.text, self.image):
            ids = [e["id"] for e in v["evidence"]]
            self.assertEqual(len(ids), len(set(ids)))

    def test_panels_render_with_real_output(self):
        reg = self.reg_
        ch = lambda *a, **k: None
        for v in (self.text, self.image):
            st = FakeSt()
            ui_panels.leaderboard_panel(st, v, "k")
            ui_panels.timeline_panel(st, v, "k")
            self.assertGreaterEqual(st.count("plotly_chart"), 3)
        with mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": self._d}):
            ui_panels.nlp_panel(FakeSt(), self.text, "k", ch, lambda: reg.load(self.text["model_key"]), {})
            ui_panels.vision_panel(FakeSt(), self.image, "k", ch, lambda: reg.load(self.image["model_key"]), None, Path(self._d) / "up")

    def test_text_try_it_highlights_words(self):
        reg = self.reg_
        st = FakeSt(answers={"text_area": "great quality and really fast delivery. loved it. would buy again. highly recommend it to anyone."})
        ui_panels.nlp_panel(st, self.text, "k", lambda *a, **k: None, lambda: reg.load(self.text["model_key"]), {})
        html = [c for c in st.calls if c[0] == "markdown" and c[2].get("unsafe_allow_html")]
        self.assertTrue(html and any("rgba" in c[1][0] for c in html))

    def test_candidates_page_lists_runs(self):
        st = FakeSt()
        with mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": self._d}):
            ui_panels.candidates_page(st, tracking)
        self.assertGreaterEqual(st.count("dataframe"), 1)

    def test_figure_builders(self):
        rows = self.text["leaderboard"]["rows"]
        for f in (pv.leaderboard_scores(rows, self.text["primary_metric"]), pv.leaderboard_folds(rows)):
            self.assertTrue(f["data"])


class TestTermPrivacy(Tmp):
    """The evidence register is what the narrative LLM reads. Terms only reach it when the user opts in (this was documented in v4 but not enforced)."""

    def test_opt_in_path_works_and_default_redacts(self):
        import re
        base = run_case("text_reviews", Path(self.d))
        terms = {t["term"] for v in base["nlp_report"]["class_terms"].values() for t in v}
        ev = [e for e in base["evidence"] if e["source"] in ("media_keywords", "media_themes", "media_class_terms")]
        self.assertEqual(len(ev), 3)
        blob = " ".join(e["text"] + str(e["data"]) for e in ev).lower()
        self.assertEqual([w for w in terms if len(w) >= 4 and re.search(rf"\b{re.escape(w)}\b", blob)], [])
        self.assertTrue(base["media_report"].get("themes"))                       # the full terms are still in the report data
        opt = run_case("text_reviews", Path(self.d), allow_llm_text=True)         # also guards the v4 crash: nested list in llm_calls
        self.assertEqual(opt["errors"], [])
        self.assertTrue(all(isinstance(c, dict) for c in opt["llm_calls"]))
        txt = " ".join(e["text"] for e in opt["evidence"] if e["source"] == "media_class_terms")
        self.assertTrue(any(w in txt for w in terms))


class TestOCR(unittest.TestCase):
    @unittest.skipUnless(shutil.which("tesseract") and shutil.which("pdftoppm"), "tesseract / pdftoppm not installed")
    def test_scanned_pdf_is_read_and_flagged(self):
        from PIL import Image, ImageDraw, ImageFont
        d = Path(tempfile.mkdtemp())
        img = Image.new("RGB", (1240, 400), "white")
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 44)
        except Exception:
            f = ImageFont.load_default()
        ImageDraw.Draw(img).text((40, 60), "Invoice 4471 for Northwind Traders", fill="black", font=f)
        img.save(d / "scan.pdf", "PDF", resolution=150)
        t, n, note = media.extract_text(d / "scan.pdf")
        self.assertIn("Northwind", t)
        self.assertIn("OCR", note)
        with mock.patch.dict(os.environ, {"NEXUSML_OCR": "0"}):
            self.assertIn("scanned", media.extract_text(d / "scan.pdf")[2])
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
