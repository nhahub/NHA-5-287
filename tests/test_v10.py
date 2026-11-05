"""v10 tests: building on an earlier analysis, managing analyses (delete / retry), the assistant's view of the whole workspace, and the redesigned Analyses page."""
import json, os, shutil, sys, tempfile, time, unittest, warnings
from pathlib import Path
from unittest import mock
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("NEXUSML_BACKEND", "builtin"); os.environ.setdefault("NEXUSML_VALIDATE", "1")
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from nexusml import jobs, lineage, llm, profiling, resume, runstore, runtime, workspace as W
from nexusml.assistant import agent as A, tools as T, workspace_ctx
from nexusml.evaluation.stub_llm import make_stub
from nexusml.storage import get_registry
from nexusml.ui import _st, app, context, dock, history, nav
from nexusml.ui.pages import analyze, assistant, data, home, models, runs, settings
from fake_st import FakeSt, run_page

KPIS = [{"name": "Records", "agg": "count"}, {"name": "Avg spend", "agg": "mean", "column": "spend"}]
F_DESC = {"analysis_type": "churn", "problem_type": "descriptive", "target": "churned", "date_column": None, "segment_columns": ["region"], "kpis": KPIS}
F_CLS = {**F_DESC, "problem_type": "classification"}
FOLLOWUP_MODELS = [True]                         # does a follow-up in these tests ask for a model (slow) or stay descriptive (fast)?
SEEN: list = []


def frame(pl):
    SEEN.append("previous_analysis" in pl)
    return {**(F_CLS if "previous_analysis" in pl and FOLLOWUP_MODELS[0] else F_DESC), "business_question": "Why do customers leave?", "hypotheses": ["Tenure matters"]}


def churn(n=700):
    rs = np.random.RandomState(3)
    df = pd.DataFrame({"tenure": rs.rand(n) * 60, "spend": rs.lognormal(3, .5, n), "region": rs.choice(list("ABCD"), n), "score": rs.randn(n)})
    df["churned"] = ((0.04 * (30 - df.tenure) + 0.5 * df.score + rs.randn(n) * .8) > 0).astype(int)
    return df


def wait_attempt(jid, status="failed"):
    """A job's status flips before its record is stored, so wait for the record."""
    for _ in range(200):
        a = runstore.get_attempt(jid)
        if a and a["status"] == status:
            return a
        time.sleep(.05)
    return None


class Env(unittest.TestCase):
    """A scratch workspace with the language model replaced by the offline test double."""
    @classmethod
    def start_env(cls):
        cls.d = tempfile.mkdtemp()
        cls.env = mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": cls.d, "GROQ_API_KEY": "sk-secret-value-xyz"}); cls.env.start()
        cls.patches = [mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "preflight", return_value=None),
                       mock.patch.object(llm, "ask_json", side_effect=make_stub(F_DESC, {"frame_business_problem": frame}))]
        [p.start() for p in cls.patches]

    @classmethod
    def stop_env(cls):
        [p.stop() for p in cls.patches]; cls.env.stop(); _st.use(None); shutil.rmtree(cls.d, ignore_errors=True)

    @staticmethod
    def analyse(path, request, **kw):
        return runtime.run_to_completion(str(path), request, auto_approve=True, enable_tuning=False, seed=1, **kw)


class TestResume(Env):
    @classmethod
    def setUpClass(cls):
        cls.start_env()
        FOLLOWUP_MODELS[0] = True
        cls.meta = W.add_dataset("churn.csv", churn().to_csv(index=False).encode())
        cls.path = W.dataset_path(cls.meta["id"])
        cls.profiled = 0
        orig = profiling.profile_dataset

        def counting(*a, **k):
            cls.profiled += 1
            return orig(*a, **k)
        with mock.patch.object(profiling, "profile_dataset", side_effect=counting):
            cls.v1 = cls.analyse(cls.path, "just describe the data and the KPIs, no model")
            cls.after_first = cls.profiled
            del SEEN[:]
            cls.v2 = cls.analyse(cls.path, "Now build a predictive model", resume_from=cls.v1["experiment_id"])
        cls.profiled_second = cls.profiled - cls.after_first
        cls.seen_second = list(SEEN)                      # what framing was sent during the follow-up only

    @classmethod
    def tearDownClass(cls):
        cls.stop_env()

    def test_first_run_is_descriptive_and_builds_no_model(self):
        self.assertEqual(self.v1["analysis_plan"]["problem_type"], "descriptive"); self.assertIsNone(self.v1.get("best_model_name")); self.assertEqual(self.v1["errors"], [])

    def test_followup_builds_a_model_on_top_of_the_earlier_analysis(self):
        v = self.v2
        self.assertEqual(v["errors"], []); self.assertEqual(v["analysis_plan"]["problem_type"], "classification"); self.assertTrue(v["best_model_name"]); self.assertTrue(v["report_markdown"])
        self.assertEqual(v["parent_experiment_id"], self.v1["experiment_id"]); self.assertNotEqual(v["experiment_id"], self.v1["experiment_id"])
        self.assertEqual(runstore.parent_of(v["experiment_id"]), self.v1["experiment_id"]); self.assertEqual(lineage.family(self.v1["experiment_id"])["children"], [v["experiment_id"]])

    def test_the_dataset_is_not_profiled_again(self):
        self.assertEqual(self.after_first, 1); self.assertEqual(self.profiled_second, 0)
        self.assertEqual(self.v2["dataset_profile"], self.v1["dataset_profile"])

    def test_framing_is_told_what_was_found_before(self):
        self.assertTrue(self.seen_second and all(self.seen_second))
        ctx = self.v2["resume"]["context"]
        self.assertEqual(ctx["request"], "just describe the data and the KPIs, no model"); self.assertFalse(ctx["modeling_done"]); self.assertIn("tenure", ctx["columns"])

    def test_kpis_and_trends_are_reused_and_statistics_redone_because_the_question_changed(self):
        who = lambda v: {e["agent"] for e in v["agent_log"]}
        self.assertIn("KPI Analyst", who(self.v1)); self.assertNotIn("KPI Analyst", who(self.v2)); self.assertNotIn("Trend Analyst", who(self.v2)); self.assertIn("Statistician", who(self.v2))
        self.assertEqual(self.v2["kpi_results"], self.v1["kpi_results"]); self.assertEqual(self.v2["trend_results"], self.v1["trend_results"])
        self.assertTrue(any("recomputed the statistical tests" in e["message"] for e in self.v2["agent_log"]))

    def test_evidence_ids_never_collide_and_reused_evidence_keeps_its_ids(self):
        ids = [e["id"] for e in self.v2["evidence"]]
        self.assertEqual(len(ids), len(set(ids)))
        by1 = {e["id"]: e["text"] for e in self.v1["evidence"]}
        for k in ("K01", "Q01", "T01"):
            self.assertEqual({e["id"]: e["text"] for e in self.v2["evidence"]}[k], by1[k])
        self.assertTrue({"B", "M"} & {i[0] for i in ids})                                      # the new work added its own evidence

    def test_saved_run_does_not_carry_the_reuse_payload(self):
        saved = lineage.load_state(self.v2["experiment_id"])
        self.assertNotIn("carry", saved["resume"]); self.assertEqual(saved["resume"]["parent"], self.v1["experiment_id"]); self.assertTrue(saved["resume"]["profile_reused"])
        self.assertLess(len(json.dumps(saved["resume"])), 6000)

    def test_a_follow_up_must_use_the_same_dataset(self):
        other = W.add_dataset("other.csv", churn(120).to_csv(index=False).encode())
        with self.assertRaises(resume.ResumeError):
            resume.seed(self.v1["experiment_id"], str(W.dataset_path(other["id"])))
        with self.assertRaises(resume.ResumeError):
            resume.seed("EXP_missing", str(self.path))

    def test_reuse_can_be_restricted(self):
        none = resume.seed(self.v1["experiment_id"], str(self.path), reuse=[])
        self.assertEqual(none["resume"]["carry"], {}); self.assertFalse(none["resume"]["profile_reused"]); self.assertNotIn("dataset_profile", none["state"])
        prof = resume.seed(self.v1["experiment_id"], str(self.path), reuse=["profile"])
        self.assertTrue(prof["resume"]["profile_reused"]); self.assertEqual(prof["resume"]["carry"], {})
        self.assertEqual({e["id"][0] for e in prof["state"]["evidence"]}, {"Q"})

    def test_which_phases_apply_depends_on_the_new_plan(self):
        sd = resume.seed(self.v1["experiment_id"], str(self.path))
        same = {"resume": sd["resume"], "analysis_plan": {"target": "churned", "problem_type": "descriptive", "date_column": None}}
        self.assertEqual(resume.applicable(same), ["statistics", "kpis", "trends"])
        self.assertEqual(resume.applicable({**same, "analysis_plan": {"target": "churned", "problem_type": "classification", "date_column": None}}), ["kpis", "trends"])
        self.assertEqual(resume.applicable({**same, "analysis_plan": {"target": "spend", "problem_type": "descriptive", "date_column": None}}), ["kpis", "trends"])
        self.assertEqual(resume.applicable({**same, "analysis_plan": {"target": "churned", "problem_type": "descriptive", "date_column": "when"}}), ["statistics", "kpis"])

    def test_a_changed_dataset_reuses_nothing(self):
        meta = W.add_dataset("ch.csv", churn(300).to_csv(index=False).encode()); p = W.dataset_path(meta["id"])
        FOLLOWUP_MODELS[0] = False
        v = self.analyse(p, "describe it")
        t = time.time() + 120; os.utime(p, (t, t))                                              # the file is modified after the analysis finished
        sd = resume.seed(v["experiment_id"], str(p))
        self.assertEqual(sd["resume"]["requested"], []); self.assertFalse(sd["resume"]["profile_reused"]); self.assertTrue(sd["resume"]["notes"])
        FOLLOWUP_MODELS[0] = True

    def test_an_invalid_follow_up_is_refused_when_submitted_not_inside_the_background_run(self):
        with self.assertRaises(resume.ResumeError):
            jobs.submit(str(self.path), "x", None, "churn.csv", resume_from="EXP_nope")

    def test_reproducibility_kit_runs_a_follow_up_from_scratch_and_says_so(self):
        from nexusml import export
        blob = export.build_reproducibility_kit(lineage.load_state(self.v2["experiment_id"]), get_registry())
        import io, zipfile
        z = zipfile.ZipFile(io.BytesIO(blob))
        self.assertNotIn("resume_from", z.read("run_args.json").decode()); self.assertIn("built on an earlier one", z.read("README.txt").decode())


class TestLifecycle(Env):
    """Deleting, unfinished attempts and retry data. Uses fast descriptive analyses."""
    def setUp(self):
        self.start_env(); FOLLOWUP_MODELS[0] = False
        self.meta = W.add_dataset("churn.csv", churn(400).to_csv(index=False).encode()); self.path = W.dataset_path(self.meta["id"])

    def tearDown(self):
        self.stop_env()

    def files(self):
        root = Path(self.d)
        return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and "sqlite" not in p.name and p.parent != root and "workspace" not in p.parts}

    def test_deleting_the_parent_leaves_the_follow_up_intact_and_deleting_both_leaves_nothing(self):
        a = self.analyse(self.path, "describe"); b = self.analyse(self.path, "go on", resume_from=a["experiment_id"])
        out = lineage.delete_run(a["experiment_id"])
        self.assertGreater(out["removed_files"], 5); self.assertIsNone(runstore.get_run(a["experiment_id"])); self.assertIsNone(lineage.load_state(a["experiment_id"]))
        kept = lineage.load_state(b["experiment_id"]); self.assertTrue(kept and kept["report_markdown"])
        self.assertTrue(all(get_registry().path_of(k).exists() for c in kept["charts"] for k in (c.get("key"), c.get("spec_key")) if k))
        self.assertTrue(get_registry().path_of(kept["report_key"]).exists())
        fam = lineage.family(b["experiment_id"]); self.assertTrue(fam["parent_deleted"]); self.assertIsNone(fam["parent"])
        lineage.delete_run(b["experiment_id"])
        left = {f for f in self.files() if not f.endswith("experiments_index.json") and "registry.json" not in f}
        self.assertEqual(left, set(), left); self.assertEqual(runstore.list_runs(10), []); self.assertEqual(get_registry().list_runs(), [])

    def test_a_file_another_analysis_still_points_at_is_never_deleted(self):
        a = self.analyse(self.path, "describe"); b = self.analyse(self.path, "other")
        shared = next(c["key"] for c in lineage.load_state(a["experiment_id"])["charts"] if c.get("key"))
        sb = lineage.load_state(b["experiment_id"]); sb["charts"].append({"id": "x", "title": "x", "caption": "", "key": shared, "spec_key": None, "section": "x"})
        get_registry().path_of(f"runs/{b['experiment_id']}/state.json").write_text(json.dumps(sb))
        out = lineage.delete_run(a["experiment_id"])
        self.assertTrue(get_registry().path_of(shared).exists()); self.assertTrue(any(shared.startswith(k) or k == shared or shared.startswith(k + "/") for k in out["kept_shared"]), out["kept_shared"])

    def test_deleting_removes_registered_model_versions_that_pointed_at_it(self):
        FOLLOWUP_MODELS[0] = True
        a = self.analyse(self.path, "describe"); b = self.analyse(self.path, "model please", resume_from=a["experiment_id"])
        from nexusml import tracking
        self.assertTrue(tracking.registry()["models"]); self.assertTrue(get_registry().path_of(b["model_key"]).exists())
        lineage.delete_run(b["experiment_id"])
        self.assertEqual(tracking.registry()["models"], {}); self.assertFalse(get_registry().path_of(b["model_key"]).exists())
        self.assertEqual([r for r in tracking.list_runs() if (r.get("tags") or {}).get("experiment_id") == b["experiment_id"]], [])
        self.assertTrue([r for r in tracking.list_runs() if (r.get("tags") or {}).get("experiment_id") == a["experiment_id"]])        # the parent's own records are untouched

    def test_a_failed_analysis_is_kept_so_it_can_be_retried_or_deleted_after_a_restart(self):
        with mock.patch.object(runtime, "run_to_completion", side_effect=RuntimeError("boom")):
            jid = jobs.submit(str(self.path), "predict churn", self.meta["id"], "churn.csv", options={"minutes": 7}, enable_tuning=False, target_column="churned", seed=5)
            wait_attempt(jid)
        jobs.forget(jid)                                                                          # as if the app had restarted: only the stored record is left
        att = runstore.get_attempt(jid)
        self.assertEqual(att["status"], "failed"); self.assertIn("boom", att["error"]); self.assertEqual(att["request"], "predict churn")
        spec = lineage.retry_spec(attempt_id=jid)
        self.assertEqual(spec["request"], "predict churn"); self.assertEqual(spec["options"], {"minutes": 7})
        self.assertEqual(spec["kwargs"]["target_column"], "churned"); self.assertEqual(spec["kwargs"]["seed"], 5); self.assertFalse(spec["kwargs"]["enable_tuning"])
        for k in ("dataset_path", "request", "options", "resume_from"):
            self.assertNotIn(k, spec["kwargs"])
        lineage.delete_attempt(jid); self.assertIsNone(runstore.get_attempt(jid))

    def test_retry_data_of_a_finished_analysis(self):
        a = self.analyse(self.path, "describe the churn", target_column="churned")
        spec = lineage.retry_spec(exp=a["experiment_id"])
        self.assertEqual(spec["request"], "describe the churn"); self.assertEqual(spec["kwargs"]["target_column"], "churned"); self.assertEqual(spec["source"], "run")
        self.assertTrue(Path(spec["dataset_path"]).exists()); self.assertNotIn("resume_from", spec["kwargs"])

    def test_notes_belong_to_their_analysis_and_go_with_it(self):
        a = self.analyse(self.path, "describe")
        runstore.add_note(a["experiment_id"], "the spend column looks odd")
        self.assertEqual([n["text"] for n in runstore.list_notes(a["experiment_id"])], ["the spend column looks odd"])
        lineage.delete_run(a["experiment_id"]); self.assertEqual(runstore.list_notes(a["experiment_id"]), [])


class TestAssistantWorkspace(Env):
    @classmethod
    def setUpClass(cls):
        cls.start_env(); FOLLOWUP_MODELS[0] = True
        cls.meta = W.add_dataset("churn.csv", churn().to_csv(index=False).encode()); cls.path = W.dataset_path(cls.meta["id"])
        cls.v1 = cls.analyse(cls.path, "describe the data, no model")
        cls.v2 = cls.analyse(cls.path, "Now build a predictive model", resume_from=cls.v1["experiment_id"])
        cls.id1, cls.id2 = cls.v1["experiment_id"], cls.v2["experiment_id"]

    @classmethod
    def tearDownClass(cls):
        cls.stop_env()

    def tool(self, name, view=None, run_state=None, **a):
        return T.run_tool(T.ToolContext(pd.DataFrame(), "", run_state=run_state, view=view or {}), name, a)

    def test_every_part_of_an_analysis_can_be_read_without_a_dataset(self):
        from nexusml.assistant import runfacts
        for part in runfacts.PARTS:
            r = self.tool("analysis", run=self.id2, part=part, query="churn" if part in ("evidence", "charts", "report") else "")
            self.assertEqual(r.summary["analysis"], self.id2, part)
        m = self.tool("analysis", run="latest", part="metrics").summary
        self.assertTrue(m["modelled"]); self.assertEqual(m["primary_metric"], self.v2["primary_metric"]); self.assertAlmostEqual(m["held_out"]["value"], self.v2["metric_ci"]["value"], 4)
        self.assertFalse(self.tool("analysis", run=self.id1, part="metrics").summary["modelled"])

    def test_an_analysis_can_be_found_by_id_suffix_words_or_position(self):
        for ref in ("latest", self.id2, self.id2[-6:], "predictive model"):
            self.assertEqual(self.tool("analysis", run=ref, part="overview").summary["analysis"], self.id2, ref)
        self.assertEqual(self.tool("analysis", run="first", part="overview").summary["analysis"], self.id1)
        self.assertEqual(self.tool("analysis", run="previous", part="overview").summary["analysis"], self.id1)
        with self.assertRaises(T.ToolError):
            self.tool("analysis", run="no such analysis", part="overview")
        with self.assertRaises(T.ToolError):
            self.tool("analysis", run="EXP_2026", part="overview")                            # ambiguous prefix is refused, not guessed

    def test_the_open_analysis_is_the_default_and_the_page_view_selects_one(self):
        self.assertEqual(self.tool("analysis", run_state=self.v1, part="overview").summary["analysis"], self.id1)
        self.assertEqual(self.tool("analysis", view={"run": self.id1}, part="overview").summary["analysis"], self.id1)

    def test_listing_and_comparing_analyses(self):
        r = self.tool("analyses"); self.assertEqual(r.summary["finished_analyses"], 2); self.assertEqual(r.table.iloc[0]["id"], self.id2); self.assertEqual(r.table.iloc[0]["built_on"], self.id1)
        self.assertEqual(self.tool("analyses", query="predictive").summary["finished_analyses"], 1)
        cmp = self.tool("analyses", compare=f"{self.id1},{self.id2}"); self.assertEqual(len(cmp.table), 2)
        with self.assertRaises(T.ToolError):
            self.tool("analyses", compare=self.id1)

    def test_a_follow_up_reports_what_it_reused(self):
        o = self.tool("analysis", run=self.id2, part="overview").summary
        self.assertEqual(o["built_on"], self.id1); self.assertIn("data profile", o["reused_from_parent"]); self.assertIn("kpis", o["reused_from_parent"])

    def test_charts_are_listed_and_shown(self):
        lst = self.tool("analysis_chart", run=self.id2); self.assertGreater(lst.summary["charts"], 5)
        shown = self.tool("analysis_chart", run=self.id2, chart="Model comparison")
        self.assertTrue(shown.figure or shown.image); self.assertEqual(shown.summary["chart"], "Model comparison")
        self.assertNotIn("data", json.dumps(shown.summary).lower().replace("what_it_shows", ""))
        with self.assertRaises(T.ToolError):
            self.tool("analysis_chart", run=self.id2, chart="not a chart")

    def test_report_sections_can_be_found_and_are_bounded(self):
        secs = self.tool("analysis", run=self.id2, part="report").summary["sections"]; self.assertTrue(any("Quality gate" in s for s in secs))
        txt = self.tool("analysis", run=self.id2, part="report", query="quality gate").summary
        self.assertIn("PASSED", txt["text"]); self.assertLessEqual(len(txt["text"]), 3500)

    def test_nothing_from_a_run_exposes_rows_paths_or_keys(self):
        blob = ""
        for part in ("overview", "metrics", "explanation", "statistics", "kpis", "audit", "findings", "evidence", "decisions"):
            blob += json.dumps(self.tool("analysis", run=self.id2, part=part).summary, default=str)
        blob += json.dumps(workspace_ctx.digest({"page": "x"}, self.id2), default=str)
        for secret in (self.d, "GROQ", "sk-secret-value-xyz"):
            self.assertNotIn(secret, blob)

    def test_a_concern_is_saved_with_the_analysis_and_visible_afterwards(self):
        r = self.tool("note_concern", run=self.id2, concern="The AUC seems too good for churn data")
        self.assertTrue(r.summary["saved"]); notes = self.tool("analysis", run=self.id2, part="notes").summary
        self.assertEqual(notes["count"], 1); self.assertIn("too good", notes["notes"][0]["text"])

    def test_a_follow_up_proposal_names_what_it_reuses_and_checks_the_target_against_the_analysis(self):
        r = self.tool("propose_extension", run=self.id1, request="Now build a predictive model", target="churned")
        self.assertEqual(r.proposal["kind"], "extend"); self.assertEqual(r.proposal["parent"], self.id1); self.assertIn("KPIs", r.proposal["reuse"]); self.assertEqual(r.proposal["target"], "churned")
        with self.assertRaises(T.ToolError) as cm:
            self.tool("propose_extension", run=self.id1, request="x", target="churnd_typo_col")
        self.assertIn("tenure", str(cm.exception))

    def test_installed_packages_settings_and_models_are_visible(self):
        p = self.tool("packages").summary; self.assertIn("pandas", p["installed"]); self.assertIn("installed_total", p)
        q = self.tool("packages", query="scikit").summary; self.assertTrue(any("scikit" in k for k in q["found"]))
        s = self.tool("workspace_info", part="settings").summary; self.assertIn("provider", s["language_model"]); self.assertNotIn("key", json.dumps(s).replace("Keys are never shown", "").lower())
        self.assertGreaterEqual(self.tool("workspace_info", part="models").summary["models"], 1)
        self.assertEqual(self.tool("workspace_info", part="datasets").summary["count"], 1)
        self.assertEqual(self.tool("workspace_info", part="page", view={"page": "data"}).summary["view"], {"page": "data"})

    def test_what_is_running_and_what_did_not_finish(self):
        with mock.patch.object(runtime, "run_to_completion", side_effect=RuntimeError("it broke")):
            jid = jobs.submit(str(self.path), "predict", self.meta["id"], "churn.csv")
            wait_attempt(jid)
        try:
            r = self.tool("jobs_status").summary
            self.assertEqual(r["in_this_session"][0]["status"], "failed"); self.assertIn("it broke", r["in_this_session"][0]["problem"])
            self.assertEqual(workspace_ctx.digest({})["unfinished"], 1)
        finally:
            lineage.delete_attempt(jid)

    def test_workspace_digest_says_where_the_person_is_and_what_exists(self):
        d = workspace_ctx.digest({"page": "analysis", "run": self.id2}, self.id2)
        self.assertEqual(d["page"], {"page": "analysis", "run": self.id2}); self.assertEqual(d["analyses"]["total"], 2); self.assertEqual(d["analyses"]["open"], self.id2)
        self.assertEqual(d["analyses"]["recent"][0]["built_on"], self.id1); self.assertEqual(d["datasets"][0]["name"], "churn.csv"); self.assertLess(len(json.dumps(d)), 2500)

    def say(self, message, steps, ctx_view=None, history=None):
        """Drive the assistant loop with scripted model replies; returns (result, payloads the model was sent)."""
        sent, script = [], list(steps)

        def step(pl):
            sent.append(pl)
            return script.pop(0) if script else {"reply": "done"}
        ctx = T.ToolContext(pd.DataFrame(), "", view=ctx_view or {})
        with mock.patch.object(llm, "ask_json", side_effect=make_stub(F_DESC, {"assistant_step": step})):
            r = A.respond(message, history or [], ctx, workspace=workspace_ctx.digest(ctx_view or {}, None))
        return r, sent

    def test_the_assistant_is_sent_the_workspace_on_every_step_and_can_answer_from_it(self):
        r, sent = self.say("what have I analysed?", [{"action": {"tool": "analyses", "args": {}}}, {"reply": "You have 2 analyses; the latest is the predictive model."}], {"page": "home"})
        self.assertTrue(all("workspace" in p for p in sent)); self.assertEqual(sent[0]["workspace"]["analyses"]["total"], 2)
        self.assertEqual([s["tool"] for s in r["steps"]], ["analyses"]); self.assertFalse(r["warnings"]); self.assertIn("2 analyses", r["text"])

    def test_a_number_the_assistant_invents_is_still_caught_but_counts_from_the_workspace_are_not(self):
        r, _ = self.say("what have I analysed?", [{"reply": "You have 2 analyses, and 87.3% of them passed."}], {"page": "home"})
        self.assertNotIn("87.3", r["text"])

    def test_reading_an_analysis_then_answering_with_its_numbers(self):
        v = self.v2["metric_ci"]["value"]
        r, sent = self.say("how good is the model?", [{"action": {"tool": "analysis", "args": {"run": "latest", "part": "metrics"}}}, {"reply": f"The held-out score is {v:.3f}."}], {"page": "analysis", "run": self.id2})
        self.assertEqual(r["steps"][0]["tool"], "analysis"); self.assertIn(f"{v:.3f}", r["text"]); self.assertFalse(r["warnings"])
        self.assertTrue(r["artifacts"] and r["artifacts"][0]["table"])

    def test_a_concern_leads_to_a_follow_up_proposal_the_person_confirms(self):
        r, _ = self.say("I think the model cheats", [{"action": {"tool": "analysis", "args": {"run": "latest", "part": "audit"}}},
                                                        {"action": {"tool": "propose_extension", "args": {"request": "Check the model without the strongest feature", "run": "latest"}}},
                                                        {"reply": "The review found no leakage; I prepared a follow-up that tests it."}])
        self.assertEqual(r["proposal"]["kind"], "extend"); self.assertEqual(r["proposal"]["parent"], self.id2)

    def test_the_system_rules_tell_the_assistant_how_to_treat_concerns_and_follow_ups(self):
        for phrase in ("workspace", "concern", "propose_extension", "never start an analysis"):
            self.assertIn(phrase.lower(), A.SYSTEM.lower(), phrase)


class TestAnalysesPage(Env):
    @classmethod
    def setUpClass(cls):
        cls.start_env(); FOLLOWUP_MODELS[0] = False
        cls.meta = W.add_dataset("churn.csv", churn(400).to_csv(index=False).encode()); cls.path = W.dataset_path(cls.meta["id"])
        cls.v1 = cls.analyse(cls.path, "describe the churn data"); cls.id1 = cls.v1["experiment_id"]

    @classmethod
    def tearDownClass(cls):
        cls.stop_env()

    def setUp(self):
        nav.register("analyze", "PAGE-ANALYSES"); nav.register("assistant", "PAGE-ASSISTANT"); nav.register("data", "PAGE-DATA"); nav.register("history", "PAGE-ANALYSES")
        for a in runstore.list_attempts(50):
            lineage.delete_attempt(a["id"])

    def page(self, fn, **kw):
        f = FakeSt(**kw); _st.use(f); run_page(fn); return f

    def failed_attempt(self, req="predict churn", **kw):
        aid = "att" + str(int(time.time() * 1000))[-7:]
        runstore.record_attempt({"id": aid, "created": time.time(), "dataset_id": self.meta["id"], "dataset_name": "churn.csv", "request": req, "status": "failed", "error": "boom", "kwargs": {"dataset_path": str(self.path), "seed": 9, **kw, "options": {"minutes": 3}}})
        return aid

    # layout
    def test_new_analysis_is_the_large_focal_area_and_history_a_narrow_rail(self):
        f = self.page(analyze.render)
        cols = [(a, k) for n, a, k in f.calls if n == "columns"][0]
        self.assertEqual(cols[0][0], [7, 3]); self.assertGreater(cols[0][0][0], 2 * cols[0][0][1])
        self.assertTrue(any("## New analysis" in t for t in f.texts())); self.assertTrue(any("History" in str(a[0]) for n, a, k in f.calls if n == "markdown" and a))
        area = next(k for n, a, k in f.calls if n == "text_area" and k.get("key") == "an-req"); self.assertGreaterEqual(area["height"], 150)
        sub = next(k for n, a, k in f.calls if n == "form_submit_button"); self.assertTrue(sub.get("use_container_width")); self.assertEqual(sub.get("type"), "primary")
        box = next(k for n, a, k in f.calls if n == "container" and k.get("height")); self.assertGreaterEqual(box["height"], 400)   # the rail scrolls on its own

    def test_there_is_one_analyses_page_and_both_old_names_lead_to_it(self):
        f = self.page(app.main); pages = next(a[0] for n, a, k in f.calls if n == "navigation")
        self.assertEqual([p["title"] for p in pages["Analyses"]], ["Analyses"]); self.assertIs(nav.PAGES["analyze"], nav.PAGES["history"])

    def test_the_rail_lists_finished_unfinished_and_running_analyses_newest_first(self):
        aid = self.failed_attempt()
        es = history.entries()
        self.assertEqual({(e["kind"], e["status"]) for e in es}, {("run", "done"), ("attempt", "failed")}); self.assertEqual(es[0]["id"], aid)
        self.assertEqual(history.entries("describe")[0]["id"], self.id1); self.assertEqual(history.entries("nothing like this"), [])

    def test_the_rail_works_with_no_analyses_and_with_no_dataset(self):
        shutil.rmtree(Path(self.d) / "runs", ignore_errors=True)
        with mock.patch.object(runstore, "list_runs", return_value=[]), mock.patch.object(W, "list_datasets", return_value=[]):
            f = self.page(analyze.render)
        self.assertEqual(f.count("exception"), 0); self.assertTrue(any("Add a dataset first" in str(a[0]) for n, a, k in f.calls if n == "markdown" and a))

    # delete
    def test_deleting_from_the_rail_removes_the_analysis_and_its_files(self):
        a = self.analyse(self.path, "to be removed"); eid = a["experiment_id"]
        self.assertTrue(get_registry().path_of(a["report_key"]).exists())
        f = self.page(analyze.render, clicks=[f"h-del-{eid}"])
        self.assertIsNone(runstore.get_run(eid)); self.assertFalse(get_registry().path_of(a["report_key"]).exists()); self.assertIsNotNone(runstore.get_run(self.id1))
        self.assertTrue(any(n == "toast" for n, _, _ in f.calls))

    def test_deleting_an_unfinished_analysis_and_clearing_them_all(self):
        a1, a2 = self.failed_attempt("one"), self.failed_attempt("two"); time.sleep(.002)
        self.page(analyze.render, clicks=[f"h-del-{a1}"]); self.assertIsNone(runstore.get_attempt(a1)); self.assertIsNotNone(runstore.get_attempt(a2))
        self.page(analyze.render, clicks=["hist-clear"]); self.assertEqual(runstore.list_attempts(10), []); self.assertIsNotNone(runstore.get_run(self.id1))

    def test_deleting_the_open_analysis_closes_it(self):
        a = self.analyse(self.path, "open then delete"); eid = a["experiment_id"]
        f = FakeSt(clicks=[f"run-del-{eid}"], session={"open_run": eid}); f.query_params["run"] = eid; _st.use(f); run_page(runs.render)
        self.assertIsNone(runstore.get_run(eid)); self.assertIsNone(f.session_state.get("open_run")); self.assertNotIn("run", f.query_params)

    # retry
    def test_retry_reopens_the_question_for_editing_and_replaces_the_failed_attempt(self):
        aid = self.failed_attempt("predict churn", target_column="churned", enable_tuning=False)
        sent = {}
        with mock.patch.object(jobs, "submit", side_effect=lambda path, req, ds, name, **k: sent.update(path=path, req=req, ds=ds, name=name, **k) or "job-new"):
            f = self.page(analyze.render, clicks=[f"h-retry-{aid}", f"rt-go-{aid}"], values={f"rt-req-{aid}": "predict churn for the next quarter only", f"rt-target-{aid}": "Keep as before"})
        self.assertEqual(sent["req"], "predict churn for the next quarter only"); self.assertEqual(sent["target_column"], "churned"); self.assertEqual(sent["seed"], 9)
        self.assertFalse(sent["enable_tuning"]); self.assertEqual(sent["options"], {"minutes": 3}); self.assertEqual(sent["name"], "churn.csv"); self.assertNotIn("resume_from", sent)
        self.assertIsNone(runstore.get_attempt(aid)); self.assertEqual(f.session_state.job_id, "job-new")

    def test_retry_can_change_the_target_and_kind(self):
        aid = self.failed_attempt("predict", target_column="churned"); sent = {}
        with mock.patch.object(jobs, "submit", side_effect=lambda path, req, ds, name, **k: sent.update(k) or "j"):
            self.page(analyze.render, clicks=[f"h-retry-{aid}", f"rt-go-{aid}"], values={f"rt-target-{aid}": "spend", f"rt-kind-{aid}": "Predict a number (regression)"})
        self.assertEqual(sent["target_column"], "spend"); self.assertEqual(sent["problem_type"], "regression")

    def test_retrying_a_finished_analysis_keeps_it_and_can_reuse_its_results(self):
        sent = {}
        with mock.patch.object(jobs, "submit", side_effect=lambda path, req, ds, name, **k: sent.update(k, req=req) or "j2"):
            self.page(analyze.render, clicks=[f"h-retry-{self.id1}", f"rt-go-{self.id1}"], values={f"rt-req-{self.id1}": "describe it by region", f"rt-reuse-{self.id1}": True})
        self.assertEqual(sent["resume_from"], self.id1); self.assertEqual(sent["req"], "describe it by region"); self.assertIsNotNone(runstore.get_run(self.id1))

    def test_retry_without_the_dataset_explains_instead_of_failing(self):
        aid = self.failed_attempt(); os.rename(self.path, str(self.path) + ".gone")
        try:
            f = self.page(analyze.render, clicks=[f"h-retry-{aid}"])
        finally:
            os.rename(str(self.path) + ".gone", self.path)
        self.assertTrue(any("no longer in the workspace" in t for t in f.texts()) or any(n == "error" for n, _, _ in f.calls)); self.assertIsNotNone(runstore.get_attempt(aid))

    # follow-up
    def test_build_on_this_opens_the_composer_in_follow_up_mode_for_that_dataset(self):
        f = self.page(analyze.render, clicks=[f"h-ext-{self.id1}"])
        self.assertEqual(f.session_state.extend_from, self.id1)
        f2 = self.page(analyze.render, session={"extend_from": self.id1})
        self.assertTrue(any("## Follow-up" in t for t in f2.texts())); self.assertTrue(any("Building on an earlier analysis" in str(a[0]) for n, a, k in f2.calls if n == "markdown" and a))
        self.assertTrue(any("Reused, not recomputed" in t for t in f2.texts()))
        self.assertTrue(next(k for n, a, k in f2.calls if n == "selectbox" and k.get("key") == "an-ds")["disabled"])

    def test_a_follow_up_is_submitted_with_the_analysis_it_builds_on(self):
        sent = {}
        with mock.patch.object(jobs, "submit", side_effect=lambda path, req, ds, name, **k: sent.update(k, req=req, ds=ds) or "jf"):
            f = self.page(analyze.render, session={"extend_from": self.id1}, values={"an-req": "Now build a predictive model"}, clicks=["Start follow-up"])
        self.assertEqual(sent["resume_from"], self.id1); self.assertEqual(sent["req"], "Now build a predictive model"); self.assertEqual(sent["ds"], self.meta["id"]); self.assertNotIn("resume_reuse", sent)
        self.assertEqual(f.session_state.job_id, "jf"); self.assertIsNone(f.session_state.extend_from)
        sent.clear()
        with mock.patch.object(jobs, "submit", side_effect=lambda path, req, ds, name, **k: sent.update(k) or "jf"):
            self.page(analyze.render, session={"extend_from": self.id1}, values={"an-req": "again", "an-redo": True}, clicks=["Start follow-up"])
        self.assertEqual(sent["resume_reuse"], ["profile"])

    def test_a_follow_up_to_a_deleted_analysis_falls_back_to_a_plain_new_analysis(self):
        f = self.page(analyze.render, session={"extend_from": "EXP_gone"})
        self.assertTrue(any("## New analysis" in t for t in f.texts())); self.assertIsNone(f.session_state.extend_from)

    def test_a_follow_up_that_can_no_longer_be_run_says_why(self):
        with mock.patch.object(jobs, "submit", side_effect=resume.ResumeError("The dataset was changed.")):
            f = self.page(analyze.render, session={"extend_from": self.id1}, values={"an-req": "x"}, clicks=["Start follow-up"])
        self.assertTrue(any("was changed" in str(a[0]) for n, a, k in f.calls if n == "error"))

    # detail
    def test_the_open_analysis_shows_its_actions_lineage_and_notes(self):
        child = self.analyse(self.path, "and now by region", resume_from=self.id1); cid = child["experiment_id"]
        runstore.add_note(cid, "check the tenure column")
        f = FakeSt(); f.query_params["run"] = cid; _st.use(f); run_page(runs.render)
        labels = [a[0] for n, a, k in f.calls if n in ("button", "download_button") and a]
        for want in ("Build on this", "Retry with changes", "Ask the assistant", "Back"):
            self.assertIn(want, labels)
        self.assertTrue(any("Built on:" in t for t in f.texts())); self.assertTrue(any("tenure column" in str(a[0]) for n, a, k in f.calls if n == "markdown" and a))
        self.assertEqual(context.get_view()["page"], "analysis"); self.assertEqual(context.get_view()["run"], cid)
        f1 = FakeSt(); f1.query_params["run"] = self.id1; _st.use(f1); run_page(runs.render)
        self.assertTrue(any("Follow-up:" in t for t in f1.texts()))

    def test_a_note_can_be_added_from_the_page(self):
        f = FakeSt(values={f"note-text-{self.id1}": "why is spend skewed?"}, clicks=["Save note"]); f.query_params["run"] = self.id1; _st.use(f); run_page(runs.render)
        self.assertIn("why is spend skewed?", [n["text"] for n in runstore.list_notes(self.id1)])

    def test_a_job_that_needs_an_answer_offers_to_retry_with_a_column(self):
        snap = {"id": "jn", "request": "predict it", "dataset_name": "churn.csv", "elapsed": 3.0, "status": "needs_input", "progress": .1, "stage_label": "Plan", "log": [], "message": "Which column should be predicted?", "error": None}
        with mock.patch.object(jobs, "get", return_value=snap), mock.patch.object(jobs, "active", return_value=[]):
            f = self.page(analyze.render, session={"job_id": "jn"})
        self.assertIn("Choose a column and retry", [a[0] for n, a, k in f.calls if n == "button"])


class TestAssistantEverywhere(Env):
    @classmethod
    def setUpClass(cls):
        cls.start_env(); FOLLOWUP_MODELS[0] = False
        cls.meta = W.add_dataset("churn.csv", churn(300).to_csv(index=False).encode()); cls.path = W.dataset_path(cls.meta["id"])
        cls.v1 = cls.analyse(cls.path, "describe the churn data"); cls.id1 = cls.v1["experiment_id"]

    @classmethod
    def tearDownClass(cls):
        cls.stop_env()

    def setUp(self):
        nav.register("analyze", "PAGE-ANALYSES"); nav.register("assistant", "PAGE-ASSISTANT")

    def run_dock(self, **kw):
        f = FakeSt(**kw); _st.use(f); run_page(dock.render); return f

    def test_every_page_reports_where_the_person_is(self):
        for fn, page in ((home.render, "home"), (data.render, "data"), (models.render, "models"), (settings.render, "settings"), (assistant.render, "assistant"), (analyze.render, "analyses")):
            f = FakeSt(); _st.use(f); run_page(fn)
            self.assertEqual(context.get_view()["page"], page, fn.__module__)
            self.assertEqual(f.count("exception"), 0)
        f = FakeSt(session={"ds_id": self.meta["id"]}); _st.use(f); run_page(data.render); self.assertEqual(context.get_view()["dataset"], "churn.csv")

    def test_the_assistant_is_offered_on_every_page_except_its_own(self):
        for page in ("home", "data", "analyses", "analysis", "models", "settings"):
            f = FakeSt(session={"view_ctx": {"page": page}}); _st.use(f); run_page(dock.render)
            self.assertIn("Ask the assistant", [a[0] for n, a, k in f.calls if n == "button"], page)
        f = FakeSt(session={"view_ctx": {"page": "assistant"}}); _st.use(f); run_page(dock.render); self.assertEqual(f.count("button"), 0)

    def test_the_app_shell_draws_the_dock_after_the_page(self):
        f = FakeSt(); _st.use(f); run_page(app.main); names = [(n, a[0] if a else None) for n, a, k in f.calls]
        self.assertLess([n for n, _ in names].index("navigation"), names.index(("button", "Ask the assistant")))

    def test_the_dock_opens_the_shared_conversation_with_page_specific_suggestions(self):
        f = self.run_dock(session={"view_ctx": {"page": "analysis", "run": self.id1, "request": "describe the churn data"}}, clicks=["dock-open"])
        self.assertEqual(f.count("form"), 1); self.assertTrue(any("describe the churn data" in t for t in f.texts()))
        self.assertIn("Is there anything in this analysis I should be concerned about?", [a[0] for n, a, k in f.calls if n == "button"])
        self.assertEqual(next(k for n, a, k in f.calls if n == "form_submit_button")["type"], "primary")

    def test_a_question_asked_in_the_dock_is_answered_with_the_page_and_workspace_in_mind(self):
        sent = []

        def step(pl):
            sent.append(pl)
            return {"action": {"tool": "analysis", "args": {"part": "overview"}}} if len(sent) == 1 else {"reply": "It is a descriptive analysis of the churn data."}
        with mock.patch.object(llm, "ask_json", side_effect=make_stub(F_DESC, {"assistant_step": step})):
            f = self.run_dock(session={"view_ctx": {"page": "analysis", "run": self.id1, "request": "describe the churn data"}, "open_run": self.id1}, clicks=["dock-open", "Send"], values={"dock-text": "what does this analysis show?"})
        roles = [m["role"] for m in f.session_state.messages]; self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual(sent[0]["workspace"]["page"]["run"], self.id1); self.assertEqual(sent[0]["workspace"]["analyses"]["open"], self.id1); self.assertTrue(sent[0]["a_finished_analysis_is_open"])
        self.assertEqual(f.session_state.messages[1]["steps"][0]["tool"], "analysis"); self.assertTrue(W.list_conversations())             # saved like any other conversation

    def test_the_dock_and_the_assistant_page_share_one_conversation(self):
        msgs = [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "Hi, how can I help?", "steps": [], "artifacts": [], "proposal": None, "followups": [], "warnings": []}]
        f = self.run_dock(session={"view_ctx": {"page": "home"}, "messages": msgs, "uid": "u1"}, clicks=["dock-open"]); self.assertTrue(any("Hi, how can I help?" in t for t in f.texts()))

    def test_a_follow_up_proposal_in_the_chat_leads_to_the_follow_up_composer(self):
        prop = {"kind": "extend", "parent": self.id1, "objective": "Now build a predictive model", "target": "churned", "reuse": ["data profile", "KPIs"]}
        msgs = [{"role": "assistant", "text": "I prepared a follow-up.", "steps": [], "artifacts": [], "proposal": prop, "followups": [], "warnings": []}]
        f = self.run_dock(session={"view_ctx": {"page": "home"}, "messages": msgs, "uid": "u1", "ds_id": self.meta["id"]}, clicks=["dock-open", "prop-dock-u1-0"])
        self.assertEqual(f.session_state.extend_from, self.id1); self.assertEqual(f.session_state.prefill["objective"], "Now build a predictive model"); self.assertEqual(f.switched, ["PAGE-ANALYSES"])
        self.assertTrue(any("Proposed follow-up" in str(a[0]) for n, a, k in f.calls if n == "markdown" and a)); self.assertTrue(any("Reuses: data profile, KPIs" in t for t in f.texts()))

    def test_a_chart_the_assistant_shows_from_a_finished_analysis_is_drawn(self):
        r = T.run_tool(T.ToolContext(pd.DataFrame(), ""), "analysis_chart", {"run": self.id1, "chart": self.v1["charts"][0]["title"]})
        art = {"title": r.title, "figure": r.figure, "image_key": r.image, "table": None, "notes": []}
        f = FakeSt(); _st.use(f)
        from nexusml.ui import components as c
        c.render_artifact(art, "k"); self.assertGreaterEqual(f.count("plotly_chart") + f.count("image"), 1)

    def test_without_a_language_model_the_dock_explains_and_disables_sending(self):
        with mock.patch.object(llm, "available", return_value=False):
            f = self.run_dock(session={"view_ctx": {"page": "home"}}, clicks=["dock-open"])
        self.assertTrue(any("not connected" in t for t in f.texts())); self.assertTrue(next(k for n, a, k in f.calls if n == "form_submit_button")["disabled"])


if __name__ == "__main__":
    unittest.main()
