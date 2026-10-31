"""v6 tests: the goal-driven AutoML engine, the LLM agent's guards, the crew (sentinel / scout / director), the blackboard and the new graph."""
import os
import shutil
import sys
import tempfile
import threading
import time
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

from nexusml import automl_agent as AA, automl_recipes as R, blackboard, director, graph, llm, modeling, sentinel, tracking, ui_panels
from nexusml.automl_core import Config, Goal, Session
from nexusml.automl_ensemble import EnsembleClassifier, caruana, finalize
from nexusml.automl_run import run as run_automl
from nexusml.evaluation.stub_llm import make_stub
from nexusml.runtime import run_to_completion
from fake_st import FakeSt


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self._env = mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": self.d, "GROQ_API_KEY": "stub-key", "NEXUSML_VALIDATE": "1"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        shutil.rmtree(self.d, ignore_errors=True)


def structured(n=1600, seed=1):
    """A hidden ratio feature plus an XOR interaction: defaults leave room that goal-directed actions can fill."""
    rs = np.random.RandomState(seed)
    df = pd.DataFrame({"a": rs.lognormal(0, .6, n), "b": rs.rand(n) + .5, "c": rs.randn(n), "d": rs.randn(n), "e": rs.randn(n), "g": rs.choice(list("xyz"), n)})
    s = 3.0 * (np.log(df.a / df.b) - 0.35) + 1.6 * np.sign(df.c) * np.sign(df.d) + rs.randn(n) * 0.5
    df["y"] = (s > 0).astype(int)
    return df


def splits_of(df, target="y"):
    return modeling.make_splits(df, {"problem_type": "classification", "target": target, "drop_columns": [], "datetime_columns": []})


# ============================================================================================================ recipes & ideas
class TestRecipes(unittest.TestCase):
    def setUp(self):
        rs = np.random.RandomState(0)
        self.X = pd.DataFrame({"a": rs.lognormal(0, 1, 300), "b": rs.rand(300) + .5, "g": rs.choice(list("xyz"), 300), "h": rs.choice([f"k{i}" for i in range(20)], 300)})
        self.X.loc[::7, "a"] = np.nan

    def test_ideas_are_validated_against_the_real_columns(self):
        ok, why = R.validate_ideas([{"op": "ratio", "cols": ["a", "b"], "name": "A over B"}, {"op": "ratio", "cols": ["a", "ghost"]}, {"op": "exec", "cols": ["a"]},
                                    {"op": "group_mean", "cols": ["g", "a"]}, {"op": "ratio", "cols": ["a", "b"]}, "not a dict", {"op": "bin_quantile", "cols": ["b"], "param": 99}], self.X)
        self.assertEqual([i["op"] for i in ok], ["ratio", "group_mean", "bin_quantile"])
        self.assertEqual(ok[2]["param"], 10)                                   # clipped into the safe range
        self.assertEqual(len(why), 4)
        self.assertTrue(ok[0]["name"].startswith("idea_"))

    def test_idea_transformer_learns_from_train_only_and_is_safe(self):
        ideas, _ = R.validate_ideas([{"op": "ratio", "cols": ["a", "b"]}, {"op": "group_mean", "cols": ["g", "a"]}, {"op": "count_encode", "cols": ["h"]}], self.X)
        t = R.IdeaTransformer(ideas).fit(self.X.iloc[:200])
        out = t.transform(self.X.iloc[200:].assign(g="NEVER_SEEN"))
        self.assertEqual(out.shape[0], 100)
        self.assertTrue(np.isfinite(out[[i["name"] for i in ideas]].drop(columns=[ideas[0]["name"]]).to_numpy()[~np.isnan(out[ideas[1]["name"]].to_numpy())]).all())
        self.assertFalse(np.isinf(out[ideas[0]["name"]].to_numpy()).any())      # division by zero never yields inf

    def test_every_recipe_builds_fits_and_predicts_on_raw_frames(self):
        from sklearn.linear_model import LogisticRegression
        y = (self.X.b > 1.0).astype(int).values
        for rec in ({"recipe": ("power",)}, {"recipe": ("missing_ind",)}, {"recipe": ("target_enc",)}, {"recipe": ("interactions",)}, {"recipe": ("select",)}):
            est, _ = modeling._build(modeling.Spec(LogisticRegression(max_iter=300), **rec), self.X, "classification", {})
            est.fit(self.X, y)
            self.assertEqual(est.predict_proba(self.X.head(4)).shape, (4, 2), rec)
        ideas, _ = R.validate_ideas([{"op": "ratio", "cols": ["a", "b"]}], self.X)
        est, _ = modeling._build(modeling.Spec(LogisticRegression(max_iter=300), ideas=ideas), self.X, "classification", {})
        self.assertEqual([n for n, _ in est.steps][:2], ["ideas", "prep"])
        est.fit(self.X, y)
        self.assertEqual(est.predict(self.X.head(3)).shape, (3,))


# ============================================================================================================ engine
class TestEngine(Tmp):
    @classmethod
    def setUpClass(cls):
        cls.df = structured()
        cls.sp = splits_of(cls.df)

    def session(self, goal=None, workers=1, seed=42):
        return Session("classification", self.sp, dict(modeling.DEFAULT_STRATEGY), goal or Goal(max_seconds=90), seed=seed, n_workers=workers)

    def test_portfolio_search_and_determinism_across_workers(self):
        a = self.session(workers=1)
        a.portfolio()
        a.round = 1
        a.run_search(4)
        b = self.session(workers=3)
        b.portfolio()
        b.round = 1
        b.run_search(4)
        sig = lambda s: [(t.cfg.key(), round(t.mean, 6)) for t in sorted(s.full(), key=lambda t: t.id)]
        self.assertEqual(sig(a)[:5], sig(b)[:5])                                # the portfolio is identical however many workers ran it
        self.assertGreater(len(a.full()), len(a.families()))                     # the search added trials beyond the defaults
        self.assertTrue(all(t.oof is not None for t in a.full()))

    def test_eci_gives_unexplored_families_a_chance_and_cheap_ones_more(self):
        s = self.session()
        s.portfolio()
        probs = {f: 1 / s.eci(f) for f in s.families() if s.space(f)}
        self.assertTrue(all(v > 0 for v in probs.values()))
        fresh = Session("classification", self.sp, dict(modeling.DEFAULT_STRATEGY), Goal(), seed=1)
        self.assertLess(fresh.eci(fresh.families()[0]), 0.1)                     # never-tried families look cheap to try (FLAML-style fairness)

    def test_recipe_effects_are_recorded_as_credit_assignment(self):
        s = self.session()
        s.portfolio()
        eff = s.variant_trials(recipe=("interactions",), top=2)
        self.assertEqual(len(eff["deltas"]), 2)
        self.assertEqual(s.effects[-1]["recipe"], ["interactions"])
        again = s.variant_trials(recipe=("interactions",), top=2)               # same config already seen: nothing re-run
        self.assertEqual(again["deltas"], [])

    def test_caruana_beats_or_matches_the_best_single_model(self):
        s = self.session()
        s.portfolio()
        pool = sorted(s.full(), key=lambda t: -t.mean)
        picks, sc = caruana([t.oof for t in pool], s.y, s.metric, 12, s.oof_mask)
        self.assertGreaterEqual(sc, pool[0].mean - 1e-3)
        self.assertTrue(picks)

    def test_finalize_confirms_on_fresh_folds_and_ensemble_is_a_drop_in_model(self):
        s = self.session()
        s.portfolio()
        s.round = 1
        s.run_search(3)
        out = finalize(s)
        self.assertEqual(out["best"], max(out["candidates"], key=lambda c: c["select_score"])["name"])
        for c in out["candidates"]:
            self.assertEqual(c["select_source"], "fresh-fold CV (re-scored after the search)")
            self.assertEqual(len(c["cv_scores"]), len(s.folds))
        est = out["models"][out["best"]]
        P = est.predict_proba(self.sp["X_test"].head(5))
        self.assertEqual(P.shape, (5, 2))
        import pickle
        self.assertTrue(pickle.loads(pickle.dumps(est)).predict(self.sp["X_test"].head(3)).shape == (3,))
        ens = [c for c in out["candidates"] if c["name"].startswith("Ensemble")]
        if ens:
            self.assertIn("mildly optimistic", ens[0]["ensemble_note"])

    def test_noise_ceiling_from_conflicting_duplicates(self):
        rs = np.random.RandomState(0)
        n = 600
        X = pd.DataFrame({"f": rs.randint(0, 5, n), "g": rs.randint(0, 3, n)})
        y = ((X.f + X.g > 3) ^ (rs.rand(n) < 0.2)).astype(int)                  # 20% label noise on duplicated feature rows
        sp = modeling.make_splits(X.assign(y=y), {"problem_type": "classification", "target": "y", "drop_columns": [], "datetime_columns": []})
        c = Session("classification", sp, dict(modeling.DEFAULT_STRATEGY), Goal()).ceiling()
        self.assertTrue(c["available"])
        self.assertLess(c["max_accuracy"], 0.9)
        self.assertGreater(c["max_accuracy"], 0.7)

    def test_learning_curve_reports_a_verdict(self):
        s = self.session()
        s.portfolio()
        lc = s.learning_curve(s.best().cfg)
        self.assertEqual(len(lc["points"]), 4)
        self.assertTrue(lc["verdict"])


# ============================================================================================================ agent guards
class TestAgentGuards(Tmp):
    @classmethod
    def setUpClass(cls):
        cls.sp = splits_of(structured())

    def setUp(self):
        super().setUp()
        self.s = Session("classification", self.sp, dict(modeling.DEFAULT_STRATEGY), Goal.from_spec({"metric": "roc_auc", "min": 0.99}, "classification", max_seconds=90))
        self.s.portfolio()
        self.s.round = 1

    def test_validate_decision_schema(self):
        good = {"diagnosis": "d", "actions": [{"type": "deepen", "params": {}, "why": "w"}], "stop": None}
        self.assertEqual(AA.validate_decision(good)["actions"][0]["type"], "deepen")
        for bad in ({}, {"diagnosis": "x", "actions": [{"type": "rm -rf"}]}, {"diagnosis": "x", "actions": []}, {"diagnosis": "x", "actions": [], "stop": {"reason": "because"}}, "text"):
            with self.assertRaises(ValueError):
                AA.validate_decision(bad)

    def test_actions_are_checked_against_the_data(self):
        acts = [{"type": "search_family", "params": {"family": "ghost"}}, {"type": "try_recipe", "params": {"recipe": "target_enc"}},      # no high-cardinality columns
                {"type": "try_recipe", "params": {"recipe": "nonsense"}}, {"type": "log_target", "params": {}}, {"type": "drop_features", "params": {"columns": ["a"]}},
                {"type": "try_ideas", "params": {"ideas": [{"op": "ratio", "cols": ["a", "ghost"]}]}}, {"type": "try_recipe", "params": {"recipe": "power"}},
                {"type": "search_family", "params": {"family": "randomforest", "n_trials": 999}}]
        ok, why = AA.validate_actions(self.s, [{**a, "why": ""} for a in acts], [], set())
        self.assertEqual([a["type"] for a in ok], ["try_recipe", "search_family"])
        self.assertEqual(ok[1]["params"]["family"], "RandomForest")
        self.assertEqual(ok[1]["params"]["n_trials"], AA.MAX_TRIALS_PER_ACTION)  # capped
        self.assertGreaterEqual(len(why), 6)

    def test_drop_features_only_for_columns_the_sentinel_flagged(self):
        a = [{"type": "drop_features", "params": {"columns": ["c", "d"]}, "why": ""}]
        self.assertEqual(AA.validate_actions(self.s, a, [], set())[0], [])
        self.assertEqual(AA.validate_actions(self.s, a, [], {"c"})[0][0]["params"]["columns"], ["c"])

    def test_at_most_four_actions_per_round(self):
        acts = [{"type": "deepen", "params": {}, "why": ""}] * 3 + [{"type": "ensemble", "params": {}, "why": ""}] * 5
        ok, why = AA.validate_actions(self.s, acts, [], set())
        self.assertLessEqual(len(ok), AA.MAX_ACTIONS)

    def test_stop_needs_evidence(self):
        self.assertFalse(AA._stop_allowed(self.s, "goal_met", [])[0])           # estimate does not meet 0.99
        self.assertFalse(AA._stop_allowed(self.s, "infeasible", [])[0])          # no ceiling, no flat rounds
        self.assertFalse(AA._stop_allowed(self.s, "plateau", [])[0])             # no flat round yet
        self.assertFalse(AA._stop_allowed(self.s, "plateau", [{"gain": 0.01}])[0])
        self.assertTrue(AA._stop_allowed(self.s, "plateau", [{"gain": 0.0}])[0])  # one flat round is evidence enough
        self.s.quarantine_ok = {"c"}
        self.assertFalse(AA._stop_allowed(self.s, "goal_met", [])[0])           # a pending leak blocks any success stop
        self.assertIn("leak", AA._stop_allowed(self.s, "goal_met", [])[1])

    def test_default_policy_always_returns_a_valid_decision(self):
        d = AA.validate_decision(AA.default_actions(self.s, [], []))
        self.assertTrue(d["actions"])
        self.s.quarantine_ok = {"c"}
        self.assertEqual(AA.default_actions(self.s, [], [])["actions"][0]["type"], "drop_features")


# ============================================================================================================ round loop (scripted LLM)
class TestRoundLoop(Tmp):
    def test_actions_run_before_an_accepted_stop_and_goal_met_is_blocked_by_a_leak(self):
        sp = splits_of(structured())
        goal = Goal.from_spec({"metric": "roc_auc", "min": 0.90}, "classification", max_seconds=120, max_rounds=4)
        board = blackboard.Blackboard()
        board.post("sentinel", "quarantine_proposal", {"columns": ["e"]}, severity="high", text="drop e")
        board.mark_done("sentinel")
        script = [{"diagnosis": "d", "actions": [{"type": "drop_features", "params": {"columns": ["e"]}, "why": "leak"}, {"type": "try_recipe", "params": {"recipe": "power"}, "why": "skew"}],
                   "stop": {"reason": "goal_met", "why": "looks fine"}}]
        asked = []

        def ask(purpose, system, payload, validate, mt, default):
            asked.append(payload)
            return validate(script[0]), "scripted", [], []
        out = run_automl("classification", sp, dict(modeling.DEFAULT_STRATEGY), goal, board, ask, wait_for_peers=0.1)
        sess = out["sess"]
        self.assertEqual(sess.dropped, ["e"])                                    # the leak was quarantined although the same answer asked to stop
        self.assertTrue(any(a["type"] == "try_recipe" for r in sess.rounds for a in r["actions"]))
        self.assertTrue(any("possible leak" in x for x in sess.rounds[0]["rejected"]))      # the same answer's 'goal_met' stop was refused while the leak was pending
        self.assertIn("messages_from_other_agents", asked[0])
        self.assertEqual(asked[0]["messages_from_other_agents"][0]["columns"], ["e"])

    def test_goal_verdict_uses_three_looks_and_ci_nuance(self):
        sp = splits_of(structured())
        goal = Goal.from_spec({"metric": "roc_auc", "min": 0.999}, "classification", max_seconds=60, max_rounds=1)
        out = run_automl("classification", sp, dict(modeling.DEFAULT_STRATEGY), goal, blackboard.Blackboard(), lambda p, s, pl, v, m, d: (v({"diagnosis": "d", "actions": [], "stop": {"reason": "plateau", "why": "x"}}), "scripted", [], []),
                         wait_for_peers=0.1)
        gr = out["goal_report"]
        self.assertEqual(gr["status"], "not_met")
        for k in ("confirmed_on_fresh_folds", "heldout", "search_estimate", "protocol", "ceiling"):
            self.assertIn(k, gr)
        self.assertIn("scored once", gr["protocol"])

    def test_best_effort_run_has_no_goal(self):
        sp = splits_of(structured(800))
        out = run_automl("classification", sp, dict(modeling.DEFAULT_STRATEGY), Goal(max_seconds=60, max_rounds=1), blackboard.Blackboard(),
                         lambda p, s, pl, v, m, d: (v({"diagnosis": "d", "actions": [], "stop": {"reason": "satisfied", "why": "x"}}), "scripted", [], []), wait_for_peers=0.1)
        self.assertEqual(out["goal_report"]["status"], "no_goal")


# ============================================================================================================ blackboard & director
class TestBlackboard(unittest.TestCase):
    def test_messages_are_addressed_and_never_echoed_to_the_sender(self):
        b = blackboard.Blackboard()
        b.post("scout", "idea", text="x")
        b.post("director", "note", text="for automl only", to="automl")
        msgs, cur = b.read("automl")
        self.assertEqual([m["kind"] for m in msgs], ["idea", "note"])
        self.assertEqual(b.read("automl", cur)[0], [])
        self.assertEqual(b.read("scout")[0], [])                                 # the scout sees neither its own post nor mail addressed to the AutoML agent

    def test_wait_for_blocks_until_peer_is_done_or_times_out(self):
        b = blackboard.Blackboard()
        threading.Timer(0.2, lambda: b.mark_done("sentinel")).start()
        t = time.time()
        self.assertTrue(b.wait_for(["sentinel"], 3))
        self.assertLess(time.time() - t, 2)
        self.assertFalse(b.wait_for(["never"], 0.2))

    def test_thread_safety(self):
        b = blackboard.Blackboard()
        ts = [threading.Thread(target=lambda i=i: [b.post(f"a{i}", "k", text=str(j)) for j in range(50)]) for i in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(b.count(), 400)
        self.assertEqual(len({m["id"] for m in b.export()}), 400)


class TestDirector(Tmp):
    def test_llm_answer_used_and_default_when_llm_fails(self):
        ok = lambda s, p, purpose="", max_tokens=0, temperature=0: ({"v": 1}, None, {"purpose": purpose, "ok": True, "tokens": 5})
        bad = lambda *a, **k: (None, "HTTP 500", {"ok": False})
        with mock.patch.object(llm, "available", return_value=True):
            with mock.patch.object(llm, "ask_json", side_effect=ok):
                d, src, calls, warns = director.decide({}, "j", "p", {}, lambda x: x, lambda: {"v": 0})
            self.assertEqual((d["v"], src), (1, "llm"))
            with mock.patch.object(llm, "ask_json", side_effect=bad):
                d, src, calls, warns = director.decide({}, "j", "p", {}, lambda x: x, lambda: {"v": 0})
                self.assertEqual((d["v"], src), (0, "default"))
                self.assertTrue(warns)
                with mock.patch.dict(os.environ, {"NEXUSML_STRICT_LLM": "1"}):
                    with self.assertRaises(llm.LLMRequired):
                        director.decide({}, "j", "p", {}, lambda x: x, lambda: {"v": 0})


# ============================================================================================================ sentinel
class TestSentinel(unittest.TestCase):
    def test_review_zone_leak_and_legitimate_strong_predictor(self):
        rs = np.random.RandomState(0)
        n = 1200
        df = pd.DataFrame({"tenure": rs.rand(n) * 60, "x": rs.randn(n)})
        y = (rs.rand(n) < 1 / (1 + np.exp(-(0.1 * (30 - df.tenure) + df.x)))).astype(int)
        df["after_the_fact"] = y ^ (rs.rand(n) < 0.06)                          # AUC ~0.94: below the profiler's automatic 0.97 cut-off
        df["y"] = y
        sp = splits_of(df)
        rep = sentinel.scan(sp, "classification", "y")
        flagged = {s["column"]: s["severity"] for s in rep["leakage_suspects"]}
        self.assertEqual(flagged.get("after_the_fact"), "high")
        self.assertNotIn("x", [c for c, v in flagged.items() if v == "high"])
        self.assertFalse(rep["drift"]["shifted"])                                # a random split must not look like drift

    def test_drift_detected_when_test_differs(self):
        rs = np.random.RandomState(0)
        a = pd.DataFrame({"f": rs.randn(600)})
        b = pd.DataFrame({"f": rs.randn(600) + 3})
        self.assertTrue(sentinel.drift(a, b)["shifted"])


class TestEnsembleIsLeakFreeUnderCloneAndFit(unittest.TestCase):
    """Utilities such as cross_val_score / learning_curve / out-of-fold thresholds do clone(model).fit(train).predict(validation). The ensemble's members arrive pre-fitted on ALL rows,
    so a no-op fit would score them on rows they already saw. On pure-noise labels an honest model must score ~chance."""

    def test_cross_validation_of_an_ensemble_scores_chance_on_noise(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from nexusml.automl_models import EnsembleClassifier, StackClassifier
        rs = np.random.RandomState(0)
        X = pd.DataFrame(rs.randn(400, 6))
        y = rs.randint(0, 2, 400)
        members = [RandomForestClassifier(n_estimators=60, random_state=0).fit(X, y), RandomForestClassifier(n_estimators=40, max_depth=None, random_state=1).fit(X, y)]
        self.assertGreater(roc_auc(members[0], X, y), 0.99)                      # the pre-fitted forest has memorised the noise
        for ens in (EnsembleClassifier(members=members, weights=[.5, .5], classes=[0, 1]),
                    StackClassifier(members=members, weights=None, classes=[0, 1], meta=LogisticRegression())):
            auc = cross_val_score(ens, X, y, cv=3, scoring="roc_auc").mean()
            self.assertLess(auc, 0.62, type(ens).__name__)                       # an in-sample score would be ~1.0
        ens = EnsembleClassifier(members=members, weights=[.5, .5], classes=[0, 1])
        self.assertGreater(roc_auc(ens, X, y), 0.99)                             # the deployed (not refitted) model still predicts with its pre-fitted members


def roc_auc(m, X, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, m.predict_proba(X)[:, 1])


class TestPandasStringDtype(Tmp):
    """pandas 3 gives text columns StringDtype, which np.issubdtype cannot interpret (this crashed the scout and the AutoML digest once)."""

    def test_string_columns_through_facts_digest_sentinel_and_ideas(self):
        from nexusml.agents import crew
        rs = np.random.RandomState(0)
        n = 500
        df = pd.DataFrame({"num": rs.randn(n), "cat": pd.Series(rs.choice(list("abc"), n), dtype="string"), "hi": pd.Series(rs.choice([f"k{i}" for i in range(30)], n), dtype="string")})
        df["y"] = (df.num + (df.cat == "a") + rs.randn(n) * .5 > 0.5).astype(int)
        self.assertEqual({f["name"]: f["kind"] for f in crew._column_facts(df.drop(columns="y"))}, {"num": "numeric", "cat": "categorical", "hi": "categorical"})
        sp = splits_of(df)
        s = Session("classification", sp, dict(modeling.DEFAULT_STRATEGY), Goal(max_seconds=30))
        facts = AA._data_facts(s)
        self.assertEqual((facts["n_numeric"], facts["n_categorical"]), (1, 2))
        self.assertEqual(facts["high_cardinality_categoricals"], ["hi"])
        self.assertIn("target_enc", AA.applicable_recipes(s))
        self.assertTrue(sentinel.scan(sp, "classification", "y")["n_features"] == 3)
        ok, _ = R.validate_ideas([{"op": "group_mean", "cols": ["cat", "num"]}, {"op": "count_encode", "cols": ["hi"]}], sp["X_train"])
        self.assertEqual(len(ok), 2)
        s.portfolio()
        self.assertGreater(len(s.full()), 0)
        self.assertGreater(len(s.variant_trials(recipe=("target_enc",))["deltas"]), 0)


# ============================================================================================================ the new graph, end to end
def run_csv(path, frame, workdir, overrides=None, patches=(), **kw):
    ctx = [mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=make_stub(frame, overrides))] + list(patches)
    for c in ctx:
        c.start()
    try:
        return run_to_completion(str(path), "why?", **{"auto_approve": True, "tune_n_iter": 2, "max_retries": 1, "enable_tuning": False, "seed": 42, **kw})
    finally:
        for c in reversed(ctx):
            c.stop()


def churn_with_leak(path, noise=0.06, n=1400):
    rs = np.random.RandomState(3)
    df = pd.DataFrame({"tenure": rs.rand(n) * 60, "spend": rs.lognormal(3, .5, n), "visits": rs.poisson(4, n), "region": rs.choice(list("ABCD"), n), "score": rs.randn(n)})
    p = 1 / (1 + np.exp(-(0.04 * (30 - df.tenure) + 0.5 * df.score - 0.1 * df.visits + rs.randn(n) * .8)))
    df["churned"] = (rs.rand(n) < p).astype(int)
    df["refund_issued"] = df.churned ^ (rs.rand(n) < noise)
    df.to_csv(path, index=False)
    return {"analysis_type": "churn", "problem_type": "classification", "target": "churned", "date_column": None, "segment_columns": ["region"], "kpis": [{"name": "Records", "agg": "count"}]}


class TestCrewEndToEnd(Tmp):
    def test_graph_shape(self):
        self.assertEqual(len(graph.NODES), 16)
        self.assertEqual(graph.INTERRUPTS, ["await_clarification", "human_review"])
        for n in ("brief", "sentinel", "scout", "automl", "critic", "explain"):
            self.assertIn(n, graph.NODES)
        self.assertTrue(graph.build_graph())

    def test_leak_is_caught_by_the_crew_and_the_score_becomes_honest(self):
        """The regression that motivated the guards: a leaky column made 'goal met' (AUC 0.97) a lie; the honest answer is ~0.68 and NOT met."""
        p = Path(self.d) / "leak.csv"
        frame = churn_with_leak(p)
        v = run_csv(p, frame, self.d, target_spec={"metric": "roc_auc", "min": 0.80})
        self.assertEqual(v["errors"], [])
        self.assertEqual(v["sentinel_report"]["quarantine_proposed"], ["refund_issued"])
        self.assertEqual(v["quarantined"], ["refund_issued"])
        self.assertEqual(v["automl_report"]["goal"]["status"], "not_met")
        best = next(c for c in v["candidate_metrics"] if c["name"] == v["best_model_name"])
        self.assertLess(best["roc_auc"], 0.85)
        kinds = {(m["sender"], m["kind"]) for m in v["board"]}
        self.assertTrue({("sentinel", "quarantine_proposal"), ("director", "crew")} <= kinds)
        first = v["automl_report"]["rounds"][0]
        self.assertIn("drop_features", [a["type"] for a in first["actions"]])
        self.assertTrue(any("possible leak" in x for x in first["rejected"]))     # the premature 'goal_met' stop was refused

    def test_branches_really_run_concurrently(self):
        p = Path(self.d) / "x.csv"
        frame = churn_with_leak(p)
        v = run_csv(p, frame, self.d)
        tm = v["timings"]
        span = {k: (tm[k]["first_start"], tm[k]["last_end"]) for k in ("sentinel", "explore", "scout", "automl")}
        overlap = [(a, b) for a in span for b in span if a < b and span[a][0] < span[b][1] and span[b][0] < span[a][1]]
        self.assertGreaterEqual(len(overlap), 4)
        self.assertTrue(any("automl" in pr for pr in overlap))

    def test_decisions_are_recorded_with_source_and_guard(self):
        p = Path(self.d) / "x.csv"
        v = run_csv(p, churn_with_leak(p), self.d)
        by = {d["junction"] for d in v["decisions"]}
        self.assertTrue({"brief_crew", "automl_round", "route_verdict"} <= by)
        self.assertTrue(all(d["source"] in ("llm", "default") or d.get("accepted") for d in v["decisions"]))
        verdict = next(d for d in v["decisions"] if d["junction"] == "route_verdict")
        self.assertIn("allowed", verdict["guard"])

    def test_refine_is_an_llm_choice_bounded_by_guards(self):
        p = Path(self.d) / "x.csv"
        frame = churn_with_leak(p)
        n = {"i": 0}

        def verdict(pl):
            n["i"] += 1
            return {"route": "refine" if "refine" in pl["allowed_routes"] else "accept", "directive": "Focus on ensembling.", "rationale": f"stub {n['i']}"}
        v = run_csv(p, frame, self.d, overrides={"route_verdict": verdict}, target_spec={"metric": "roc_auc", "min": 0.99}, max_retries=2)
        routes = [d["decision"] for d in v["decisions"] if d["junction"] == "route_verdict"]
        self.assertEqual(routes[-1], "accept")
        self.assertIn("refine", routes)
        self.assertLessEqual(routes.count("refine"), 2)                          # never more than the guard allows
        self.assertEqual(v["automl_report"]["attempt"], len(routes))
        self.assertEqual([h["attempt"] for h in v["candidate_history"]], list(range(1, len(routes) + 1)))
        self.assertGreaterEqual(len(v["automl_report"]["trials"]), 10)           # a refine pass keeps every earlier trial

    def test_invalid_llm_route_falls_back_to_a_guarded_default(self):
        p = Path(self.d) / "x.csv"
        v = run_csv(p, churn_with_leak(p), self.d, overrides={"route_verdict": lambda pl: {"route": "explode", "rationale": "x"}})
        self.assertEqual(v["errors"], [])
        d = next(d for d in v["decisions"] if d["junction"] == "route_verdict")
        self.assertEqual(d["decision"], "accept")

    def test_llm_outage_at_decision_points_degrades_not_dies(self):
        p = Path(self.d) / "x.csv"
        frame = churn_with_leak(p)
        real = make_stub(frame)

        def flaky(system, prompt, purpose="", max_tokens=0, temperature=0):
            if purpose in ("brief_crew", "automl_round", "scout_ideas", "sentinel_review", "route_verdict"):
                return None, "HTTP 503", {"ok": False, "purpose": purpose}
            return real(system, prompt, purpose, max_tokens, temperature)
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=flaky):
            v = run_to_completion(str(p), "why?", auto_approve=True, tune_n_iter=2, max_retries=1, enable_tuning=False, seed=42)
        self.assertEqual(v["errors"], [])
        self.assertTrue(v["best_model_name"])
        self.assertTrue(all(d["source"] == "default" for d in v["decisions"] if d["junction"] in ("brief_crew", "route_verdict")))
        self.assertTrue(any("deterministic default" in w for w in v["warnings"]))

    def test_goal_in_plain_words_is_extracted_and_validated(self):
        p = Path(self.d) / "x.csv"
        frame = churn_with_leak(p)
        brief = lambda pl: {"crew": {"sentinel": True, "scout": False, "explore": True}, "automl": {"max_seconds": None, "max_rounds": 2, "workers": 99, "style": "quick"},
                            "goal": {"metric": "roc_auc", "bound": "min", "value": 0.7}, "rationale": "stub"}
        v = run_csv(p, frame, self.d, overrides={"brief_crew": brief})
        self.assertEqual(v["target_spec"], {"metric": "roc_auc", "min": 0.7})
        self.assertFalse(v["crew"]["scout"])
        self.assertEqual(v["scout_report"], {"skipped": True})
        self.assertLessEqual(v["automl_plan"]["workers"], int(os.environ.get("NEXUSML_AUTOML_WORKERS", "4")))      # the guard clips the LLM's 99 workers
        bad = lambda pl: {**brief(pl), "goal": {"metric": "vibes", "bound": "min", "value": 3}}
        v2 = run_csv(p, frame, self.d, overrides={"brief_crew": bad})
        self.assertFalse(v2.get("target_spec"))
        self.assertTrue(any("ignored" in w for w in v2["warnings"]))

    def test_goal_with_a_secondary_constraint_in_plain_words(self):
        p = Path(self.d) / "x.csv"
        frame = churn_with_leak(p)
        brief = lambda pl: {"crew": {"sentinel": True, "scout": True, "explore": True}, "automl": {"max_seconds": None, "max_rounds": 1, "workers": 1, "style": "quick"},
                            "goal": {"metric": "Recall", "bound": "max", "value": 0.6, "subject_to": {"metric": "precision", "value": 0.4}}, "rationale": "stub"}
        v = run_csv(p, frame, self.d, overrides={"brief_crew": brief})
        self.assertEqual(v["target_spec"], {"metric": "recall", "min": 0.6, "subject_to": {"metric": "precision", "min": 0.4}})   # direction comes from the metric, not the LLM
        g = v["automl_report"]["goal"]
        self.assertEqual(g["text"], "recall >= 0.6 with precision >= 0.4")
        self.assertIn(g["status"], ("met", "met_on_test_only", "not_met"))
        self.assertIsNotNone(g["threshold"])                                   # a decision cut-off was chosen on out-of-fold predictions

    def test_report_tracking_and_panels_cover_automl(self):
        from nexusml.storage import get_registry
        p = Path(self.d) / "x.csv"
        v = run_csv(p, churn_with_leak(p), self.d, target_spec={"metric": "roc_auc", "min": 0.9})
        text = get_registry().path_of(v["report_key"]).read_text(encoding="utf-8")
        for h in ("Goal-driven AutoML", "round by round", "The crew: decisions and conversation", "Blackboard"):
            self.assertIn(h, text)
        run = tracking.get_run(v["tracking"]["run_id"])
        self.assertEqual(run["params"]["search_engine"], "automl-agent")
        self.assertIn("goal_target", run["metrics"])
        self.assertTrue({"automl_trials", "decisions", "blackboard"} <= set(run["artifacts"]))
        st = FakeSt()
        ui_panels.automl_panel(st, v, "k")
        ui_panels.crew_panel(st, v, "k")
        self.assertGreaterEqual(st.count("plotly_chart"), 3)


class TestExportOfNewModels(Tmp):
    """A pickled model references the module that defines its classes; the exported package must ship every such module (v5's text/image models had never been exported)."""

    def _predict_in_clean_process(self, v):
        import io
        import subprocess
        import zipfile
        from nexusml import export
        data = export.build_package(v)
        tmp = tempfile.mkdtemp()
        zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
        pkg = next(Path(tmp).iterdir())
        env = {k: x for k, x in os.environ.items() if k != "PYTHONPATH"}
        r = subprocess.run([sys.executable, "predict.py", "input_template.csv"], cwd=pkg, capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-900:])
        out = pd.read_csv(pkg / "predictions.csv")
        self.assertIn("prediction", out.columns)
        self.assertTrue(any(c.startswith("proba_") for c in out.columns))
        return out

    def test_ensemble_winner_deploys(self):
        p = Path(self.d) / "x.csv"
        frame = churn_with_leak(p)
        pick = lambda rs: next(r["name"] for r in rs if r["name"].startswith("Ensemble"))
        v = run_csv(p, frame, self.d, patches=[mock.patch.object(modeling, "pick_best", side_effect=pick)])
        self.assertTrue(v["best_model_name"].startswith("Ensemble"))
        out = self._predict_in_clean_process(v)
        self.assertGreater(len(out), 0)

    def test_text_pipeline_deploys(self):
        from test_v5 import run_case
        v = run_case("text_reviews", Path(self.d))
        self.assertEqual(v["pipeline_kind"], "text")
        self._predict_in_clean_process(v)


if __name__ == "__main__":
    unittest.main()
