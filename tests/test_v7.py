"""v7 tests: the assistant (tools, loop, honesty), workspace, background jobs, and every page of the new application."""
import os, re, zipfile, shutil, sys, tempfile, threading, time, tomllib, unittest, warnings
from pathlib import Path
from unittest import mock
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("NEXUSML_BACKEND", "builtin")
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from nexusml import graph, jobs, llm, palette, runstore, runtime, workspace as W
from nexusml.assistant import agent as A, tools as T
from nexusml.evaluation.stub_llm import make_stub
from nexusml.ui import _st, app, nav, theme
from nexusml.ui.pages import analyze, assistant, data, home, models, runs, settings
from fake_st import FakeSt, run_page

FRAME = {"analysis_type": "churn", "problem_type": "classification", "target": "churned", "date_column": None, "segment_columns": ["region"], "kpis": [{"name": "Records", "agg": "count"}]}


def sales(n=1200):
    rs = np.random.RandomState(0)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="8h"), "region": rs.choice(["West", "East"], n), "price": rs.lognormal(3, .4, n), "qty": rs.poisson(3, n) + 1})
    df.loc[[10, 200, 777], "price"] = [9000, 12000, 15000]
    return df


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": self.d, "GROQ_API_KEY": "k"}); self.env.start()
        self.p = [mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=make_stub(FRAME))]
        [p.start() for p in self.p]
        self.df = sales()

    def tearDown(self):
        [p.stop() for p in self.p]; self.env.stop(); _st.use(None); shutil.rmtree(self.d, ignore_errors=True)

    def add(self, df=None):
        return W.add_dataset("sales.csv", (df if df is not None else self.df).to_csv(index=False).encode())


class TestTools(Base):
    def run_t(self, name, **kw):
        return T.run_tool(T.ToolContext(self.df, "sales"), name, kw)

    def test_outliers_find_planted_rows_and_summary_has_no_rows(self):
        r = self.run_t("outliers")
        self.assertTrue({10, 200, 777} <= set(r.table["row"]))
        self.assertIsNotNone(r.figure)
        blob = str(r.summary)
        self.assertNotIn("12000", blob)                                              # the summary holds fences and counts, not the flagged rows' values
        self.assertIn("by_column", r.summary)

    def test_every_tool_runs(self):
        for name, kw in [("overview", {}), ("distribution", {"column": "price"}), ("value_counts", {"column": "region"}), ("correlations", {}), ("missing", {}),
                         ("group_summary", {"by": "region", "value": "price"}), ("relationship", {"a": "region", "b": "price"}), ("relationship", {"a": "price", "b": "qty"}),
                         ("plot", {"kind": "scatter", "x": "price", "y": "qty"}), ("filter_rows", {"conditions": [{"column": "region", "op": "==", "value": "West"}]}),
                         ("time_trend", {"date": "date", "value": "price"}), ("duplicates", {}), ("drivers", {"target": "price"}), ("propose_analysis", {"objective": "x"})]:
            r = self.run_t(name, **kw)
            self.assertTrue(r.title and isinstance(r.summary, dict), name)

    def test_hostile_and_malformed_requests_are_rejected_clearly(self):
        for name, kw in [("outliers", {"columns": ["nope"]}), ("distribution", {}), ("plot", {"kind": "pie", "x": "price"}), ("distribution", {"column": "price; import os"}),
                         ("filter_rows", {"conditions": [{"column": "price", "op": "drop table", "value": 1}]}), ("nosuch", {}), ("outliers", {"bogus": 1}), ("time_trend", {"date": "region"})]:
            with self.assertRaises(T.ToolError, msg=f"{name}{kw}"):
                self.run_t(name, **kw)

    def test_no_data_is_a_clear_error(self):
        with self.assertRaises(T.ToolError):
            T.run_tool(T.ToolContext(pd.DataFrame()), "outliers", {})


class TestAgent(Base):
    def say(self, msg, overrides=None, ctx="data", history=None):
        c = T.ToolContext(self.df, "sales") if ctx == "data" else None
        with mock.patch.object(llm, "ask_json", side_effect=make_stub(FRAME, overrides)), mock.patch.object(runtime, "run_to_completion", side_effect=AssertionError("pipeline")), \
                mock.patch.object(graph, "build_graph", side_effect=AssertionError("pipeline")):
            return A.respond(msg, history, c)

    def test_single_question_uses_one_tool_and_never_starts_the_pipeline(self):
        t = time.time(); r = self.say("Find me outliers in this data")
        self.assertEqual([s["tool"] for s in r["steps"]], ["outliers"]); self.assertEqual(r["calls"], 2)
        self.assertTrue(r["artifacts"][0]["table"] and r["artifacts"][0]["figure"]); self.assertLess(time.time() - t, 10)

    def test_general_question_needs_no_tool_and_no_data(self):
        r = self.say("hello", ctx=None); self.assertEqual(r["steps"], [])

    def test_missing_dataset_is_explained(self):
        self.assertIn("dataset", self.say("find outliers", ctx=None)["text"].lower())

    def test_invented_figures_are_held_back_then_rewritten(self):
        bad = lambda pl: {"reply": "About 93.7% of rows are fine."} if pl.get("observations") and not pl.get("problem_with_your_last_reply") else (
            {"reply": "See the table."} if pl.get("problem_with_your_last_reply") else {"action": {"tool": "outliers", "args": {}}})
        r = self.say("find outliers", {"assistant_step": bad})
        self.assertIn("check", [s["tool"] for s in r["steps"]]); self.assertNotIn("93.7", r["text"])

    def test_bad_tool_call_is_fed_back_and_loop_is_bounded(self):
        r = self.say("x", {"assistant_step": lambda pl: {"action": {"tool": "outliers", "args": {"columns": ["ghost"]}}}})
        self.assertLessEqual(len(r["steps"]), A.MAX_STEPS); self.assertIn("step limit", " ".join(r["warnings"]))

    def test_full_analysis_is_only_a_proposal(self):
        r = self.say("do a full analysis and build a model"); self.assertTrue(r["proposal"])

    def test_llm_outage_is_a_plain_message(self):
        with mock.patch.object(llm, "ask_json", side_effect=lambda *a, **k: (None, "HTTP 503", {"ok": False})):
            r = A.respond("find outliers", None, T.ToolContext(self.df, "s"))
        self.assertIn("language model", r["text"])

    def test_digest_has_no_rows(self):
        dg = A.digest(self.df, "s"); self.assertNotIn("rows_data", dg); self.assertTrue(all("name" in c for c in dg["column_list"]))
        self.assertNotIn("top_values", str(A.digest(self.df, "s", share_labels=False)))


class TestWorkspace(Base):
    def test_dataset_lifecycle_and_errors(self):
        m = self.add(); self.assertEqual(self.add()["id"], m["id"]); self.assertEqual(len(W.list_datasets()), 1)
        for bad in (b"", b"\x00\x01garbage\xff"):
            with self.assertRaises(ValueError): W.add_dataset("x.csv", bad)
        W.delete_dataset(m["id"]); self.assertIsNone(W.get_dataset(m["id"]))

    def test_conversations_roundtrip_pin_rename_and_isolation(self):
        cid = W.new_conversation_id(); msgs = [{"role": "user", "text": "find outliers"}, {"role": "assistant", "text": "ok", "artifacts": [{"title": "t", "table": {"columns": ["a"], "rows": [[1]], "total_rows": 1}}]}]
        W.save_conversation(cid, msgs); W.pin_conversation(cid, True); W.rename_conversation(cid, "Outlier check")
        c = W.load_conversation(cid); self.assertEqual((c["title"], c["pinned"], c["messages"][1]["artifacts"][0]["table"]["rows"]), ("Outlier check", True, [[1]]))
        W.save_conversation("empty", [{"role": "assistant", "text": "hi"}]); self.assertIsNone(W.load_conversation("empty"))
        with mock.patch.object(runstore, "current_user", return_value="other"):
            self.assertEqual((W.list_datasets(), W.list_conversations()), ([], []))

    def test_user_identity_is_per_session_not_process_wide(self):
        seen = {}
        def s(n): runstore.set_user(n); time.sleep(.05); seen[n] = runstore.current_user()
        ts = [threading.Thread(target=s, args=(n,)) for n in "abc"]; [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(seen, {"a": "a", "b": "b", "c": "c"})


class TestJobs(Base):
    def test_background_analysis_reports_progress_and_can_be_cancelled(self):
        rs = np.random.RandomState(3); n = 700
        df = pd.DataFrame({"tenure": rs.rand(n) * 60, "spend": rs.lognormal(3, .5, n), "region": rs.choice(list("ABCD"), n), "score": rs.randn(n)})
        df["churned"] = ((0.04 * (30 - df.tenure) + 0.5 * df.score + rs.randn(n) * .8) > 0).astype(int)
        m = self.add(df); jid = jobs.submit(W.dataset_path(m["id"]), "why churn?", m["id"], m["name"], enable_tuning=False, seed=1)
        t = time.time()
        while time.time() - t < 240 and jobs.get(jid)["status"] in ("queued", "running"): time.sleep(.3)
        s = jobs.get(jid); self.assertEqual(s["status"], "done"); self.assertTrue(s["log"]); self.assertTrue(any(r["run_id"] == s["experiment_id"] for r in runstore.list_runs()))
        j2 = jobs.submit(W.dataset_path(m["id"]), "again", m["id"], m["name"], enable_tuning=False); time.sleep(1.5); jobs.cancel(j2)
        t = time.time()
        while time.time() - t < 120 and jobs.get(j2)["status"] in ("queued", "running"): time.sleep(.3)
        self.assertEqual(jobs.get(j2)["status"], "cancelled")


class Up:
    def __init__(s, name, b): s.name, s._b = name, b
    def getvalue(s): return s._b


class TestPages(Base):
    def page(self, fn, **kw):
        f = FakeSt(**kw); _st.use(f); run_page(fn); return f

    def test_shell_and_theme(self):
        f = self.page(app.main); names = [c[0] for c in f.calls]
        self.assertEqual(names[0], "set_page_config"); self.assertIn("navigation", names)
        for mode in ("light", "dark"):
            d = tomllib.loads(theme.config_toml(mode)); self.assertEqual(d["theme"]["base"], mode); self.assertTrue(d["browser"]["gatherUsageStats"] is False)
        for w in ("gradient", "glow", "blur"): self.assertNotIn(w, theme.css())

    def test_first_run_then_upload_then_every_page_renders(self):
        self.assertTrue(any("first dataset" in t for t in self.page(home.render).texts()))
        self.page(home.render, values={"home-upload": [Up("sales.csv", self.df.to_csv(index=False).encode())]}, clicks=["home-add"]); self.assertEqual(len(W.list_datasets()), 1)
        for fn in (home.render, data.render, assistant.render, analyze.render, runs.render, models.render, settings.render):
            f = self.page(fn); self.assertEqual(f.count("exception"), 0, fn.__module__)

    def test_assistant_page_answers_a_question_and_saves_the_conversation(self):
        self.add()
        f = self.page(assistant.render, chat=["Find me outliers in this data"])
        ms = f.session_state.messages; self.assertEqual([m["role"] for m in ms], ["user", "assistant"])
        self.assertTrue(ms[1]["artifacts"] and f.count("plotly_chart") == 1 and f.count("dataframe") >= 1)
        self.assertEqual([c["title"] for c in W.list_conversations()], ["Find me outliers in this data"])
        chip = "fu-%s-1-0" % f.session_state.uid
        f2 = self.page(assistant.render, session=dict(f.session_state), clicks=[chip]); self.assertEqual(sum(m["role"] == "user" for m in f2.session_state.messages), 2)

    def test_assistant_page_without_key_explains_and_disables_input(self):
        with mock.patch.object(llm, "available", return_value=False):
            f = self.page(assistant.render)
        self.assertTrue(any("not connected" in t for t in f.texts())); self.assertTrue(all(k.get("disabled") for n, a, k in f.calls if n == "chat_input"))

    def test_every_plotly_chart_opts_out_of_the_framework_theme(self):
        self.add(); f = self.page(data.render)
        self.assertGreater(f.count("plotly_chart"), 0); self.assertTrue(all(k.get("theme") is None and "theme" in k for n, a, k in f.calls if n == "plotly_chart"))

    def test_error_boundary_shows_a_calm_message(self):
        f = FakeSt(); _st.use(f)
        with app.error_boundary("t"):
            raise RuntimeError("boom")
        self.assertTrue(any("safe" in t for t in f.texts()))

    def test_sign_in_gate(self):
        with mock.patch.dict(os.environ, {"NEXUSML_APP_PASSWORD": "pw"}):
            f = self.page(app.main, values={"gate-name": "Ana", "gate-pw": "wrong"}, clicks=["gate-go"])
            self.assertFalse(f.session_state.get("authed")); self.assertTrue(any("not correct" in t for t in f.texts()))
            f = self.page(app.main, values={"gate-name": "Ana", "gate-pw": "pw"}, clicks=["gate-go"]); self.assertTrue(f.session_state.get("authed"))

    def test_analysis_form_builds_a_valid_goal(self):
        self.assertEqual(analyze.build_spec("roc_auc", .9, "recall", .7), {"metric": "roc_auc", "min": 0.9, "subject_to": {"metric": "recall", "min": 0.7}})
        self.assertEqual(analyze.build_spec("rmse", 5, None, 0), {"metric": "rmse", "max": 5.0}); self.assertIsNone(analyze.build_spec(None, 1, None, 0))

    def test_no_emoji_or_machine_wording_in_interface_text(self):
        emoji = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2705\u274C\u26A0\u2728]"); bad = re.compile(r"\b(AI|LLM|Claude|stub|magic)\b")
        for f in list(Path("nexusml/ui").rglob("*.py")) + list(Path("nexusml/assistant").glob("*.py")) + [Path("nexusml/ui_panels.py")]:
            for i, l in enumerate(f.read_text().splitlines(), 1):
                if l.strip().startswith(("#", '"""')): continue
                for m in re.finditer(r'f?"[^"\n]*"', l):
                    self.assertIsNone(emoji.search(m.group(0)), f"{f}:{i}"); self.assertIsNone(bad.search(m.group(0)), f"{f}:{i} {m.group(0)}")


if __name__ == "__main__":
    unittest.main()


class TestCombineTables(unittest.TestCase):
    """Several workspace datasets -> one combined dataset, through the same calls the Data page makes."""

    def test_combine_three_tables_into_one_dataset(self):
        import tempfile
        from pathlib import Path
        from nexusml import config, multitable, workspace
        from nexusml.evaluation import datasets as ev
        from nexusml.ui.pages import data as data_page

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["NEXUSML_ARTIFACTS"] = tmp
            case = ev.multitable_churn(Path(tmp))
            ids = [workspace.add_dataset(Path(p).name, Path(p).read_bytes())["id"] for p in case["paths"]]
            names = {d["id"]: d["name"] for d in workspace.list_datasets()}
            frames = data_page._frames(ids, names)
            self.assertEqual(sorted(frames), ["customers", "orders", "products"])
            plan = multitable.propose(frames)
            self.assertEqual(plan["base"], "customers")
            merged = multitable.apply(frames, {**plan, "approved": True})
            self.assertEqual(len(merged), len(frames["customers"]))
            meta = workspace.add_dataset("combined_customers.csv", merged.to_csv(index=False).encode())
            self.assertGreater(workspace.load_df(meta["id"]).shape[1], frames["customers"].shape[1])
            with self.assertRaises(multitable.ApprovalRequired):
                multitable.apply(frames, plan)


class TestImageUploads(unittest.TestCase):
    """Uploaded image archives must stay usable after they are stored in the workspace (the table points at the files)."""

    def _zip(self):
        import io, zipfile
        import numpy as np
        from PIL import Image
        r = np.random.RandomState(0)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for cls in ("dark", "light"):
                for i in range(8):
                    b = io.BytesIO()
                    Image.fromarray((r.rand(24, 24, 3) * 60 + (0 if cls == "dark" else 190)).astype("uint8")).save(b, "PNG")
                    z.writestr(f"{cls}/{cls}_{i}.png", b.getvalue())
        return buf.getvalue()

    def test_stored_dataset_points_at_real_files_and_old_ones_are_repaired(self):
        import pandas as pd
        from pathlib import Path
        from nexusml import featurizers, media, workspace
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["NEXUSML_ARTIFACTS"] = tmp
            meta = workspace.add_upload("pics.zip", self._zip())
            path = Path(workspace.dataset_path(meta["id"]))
            df = pd.read_csv(path)
            self.assertTrue(all(Path(p).is_absolute() and Path(p).is_file() for p in df["image"]))
            featurizers.check_images(df["image"], None)                       # all found: no error
            rel = [str(Path(p).relative_to(Path(p).parents[1])) for p in df["image"]]
            self.assertEqual(media.locate_root(rel, str(path)), Path(df["image"].iloc[0]).parents[1])   # old relative paths are found again
            with self.assertRaises(ValueError):
                featurizers.check_images(pd.Series(rel), str(path.parent))    # not found from the dataset folder: explicit error

    def test_odd_image_modes_and_rotation(self):
        import numpy as np
        from pathlib import Path
        from PIL import Image
        from nexusml import featurizers
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            Image.new("RGBA", (40, 30), (255, 0, 0, 0)).save(d / "a.png")
            Image.fromarray((np.random.rand(30, 40) * 60000).astype("uint16")).save(d / "b.png")
            im = Image.new("RGB", (60, 20), (10, 200, 10))
            ex = im.getexif()
            ex[274] = 6
            im.save(d / "c.jpg", exif=ex)
            sizes = {f.name: featurizers.load_rgb(f).size for f in d.iterdir()}
            self.assertEqual(sizes["c.jpg"], (20, 60))                        # EXIF rotation applied
            self.assertTrue(all(featurizers.handcrafted_image_features(f) is not None for f in d.iterdir()))


class TestImageOptions(Base):
    """Preprocessing recipe and deep-network choice on the New analysis page, and the recipe's effect on features."""

    def test_page_offers_image_options_and_passes_them_to_the_run(self):
        import io
        import numpy as np
        from PIL import Image
        r = np.random.RandomState(0)
        files = []
        for cls in ("a", "b"):
            for i in range(6):
                b = io.BytesIO()
                Image.fromarray((r.rand(20, 20, 3) * 255).astype("uint8")).save(b, "PNG")
                files.append((f"{cls}/{cls}{i}.png", b.getvalue()))
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, "w") as z:
            for n, d in files:
                z.writestr(n, d)
        meta = W.add_upload("pics.zip", zb.getvalue())
        sent = {}
        _st.use(FakeSt(values={"an-img-steps": ["blur", "edges"], "an-img-deep": "Off (hand-crafted features only)", "an-req": "classify the pictures"},
                       clicks=["Start analysis"], session={"ds_id": meta["id"]}))
        with mock.patch.object(analyze.jobs, "submit", side_effect=lambda *a, **k: sent.update(k) or "job1"):
            run_page(analyze.render)
        self.assertEqual([s["op"] for s in sent["image_prep"]], ["blur", "edges"])
        self.assertEqual(sent["deep_mode"], "off")

    def test_recipe_changes_features_and_is_validated(self):
        import numpy as np
        from PIL import Image
        from nexusml import cv, featurizers, imageprep
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.png"
            Image.fromarray((np.random.RandomState(1).rand(40, 40, 3) * 255).astype("uint8")).save(p)
            featurizers.set_image_prep(None)
            a = cv.extract(p, "hog")
            featurizers.set_image_prep("blur:3")
            b = cv.extract(p, "hog")
            featurizers.set_image_prep(None)
            self.assertFalse(np.allclose(a, b))
        with self.assertRaises(ValueError):
            imageprep.parse("blur:0")
        with self.assertRaises(ValueError):
            imageprep.parse([{"op": "resize", "mode": "zig"}])

    def test_deep_backend_batches_and_is_optional(self):
        import numpy as np
        from PIL import Image
        from nexusml import cv, featurizers
        calls = []
        featurizers.EMBEDDER_OVERRIDE["resnet"] = lambda ims: calls.append(len(ims)) or np.ones((len(ims), 5))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ps = []
                for i in range(40):
                    q = Path(tmp) / f"{i}.png"
                    Image.new("RGB", (16, 16), (i, 0, 0)).save(q)
                    ps.append(q)
                X, ok = cv.feature_matrix(ps, "resnet")
            self.assertEqual((X.shape, calls), ((40, 5), [32, 8]))
            self.assertIn("SVC · ResNet-18", cv.candidate_specs("classification", {}, 0))
            featurizers.set_deep_mode("off")
            self.assertNotIn("SVC · ResNet-18", cv.candidate_specs("classification", {}, 0))
        finally:
            featurizers.EMBEDDER_OVERRIDE.clear()
            featurizers.set_deep_mode("auto")


class TestLLMProvider(unittest.TestCase):
    """Gemini is selected by its key, requests go to Google's OpenAI-compatible endpoint, and the key never appears in errors."""

    def run_py(self, env, code):
        import subprocess
        e = {k: v for k, v in os.environ.items() if not k.startswith(("GEMINI", "GOOGLE_API", "GROQ", "NEXUSML_LLM"))}
        e.update(env, PYTHONPATH=str(Path(__file__).resolve().parent.parent), NEXUSML_ARTIFACTS=tempfile.mkdtemp())
        return subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=e, capture_output=True, text=True, timeout=60, cwd=tempfile.mkdtemp())

    def test_provider_selection(self):
        code = "from nexusml import config; print(config.LLM_PROVIDER, config.LLM_KEY_ENV, config.LLM_MODEL, config.LLM_BASE_URL)"
        self.assertEqual(self.run_py({}, code).stdout.split()[:3], ["groq", "GROQ_API_KEY", "openai/gpt-oss-120b"])
        g = self.run_py({"GEMINI_API_KEY": "k"}, code).stdout.split()
        self.assertEqual(g[:3], ["gemini", "GEMINI_API_KEY", "gemini-3.1-flash-lite"])
        self.assertTrue(g[3].startswith("https://generativelanguage.googleapis.com/"))
        self.assertEqual(self.run_py({"GEMINI_API_KEY": "k", "NEXUSML_LLM_PROVIDER": "groq", "GROQ_API_KEY": "x"}, code).stdout.split()[0], "groq")

    def test_gemini_request_shape_and_error_hides_key(self):
        code = r'''
import json, io, urllib.request, urllib.error
from unittest import mock
from nexusml import llm
seen = {}
class R(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): pass
def ok(req, timeout=0):
    seen.update(url=req.full_url, auth=req.get_header("Authorization"), body=json.loads(req.data))
    return R(json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}], "usage": {"total_tokens": 7}}).encode())
with mock.patch("urllib.request.urlopen", ok):
    d, err, rec = llm.ask_json("sys", "hi", purpose="t", max_tokens=20)
print(d, err, seen["url"], seen["auth"], seen["body"]["model"])
def bad(req, timeout=0):
    raise urllib.error.HTTPError(req.full_url, 401, "x", {}, io.BytesIO(b'{"error":{"message":"bad key"}}'))
with mock.patch("urllib.request.urlopen", bad):
    d, err, rec = llm.ask_json("sys", "hi", purpose="t", max_tokens=20)
print(err)
'''
        out = self.run_py({"GEMINI_API_KEY": "SECRETKEY123"}, code)
        self.assertIn("{'ok': True} None https://generativelanguage.googleapis.com/v1beta/openai/chat/completions Bearer SECRETKEY123 gemini-3.1-flash-lite", out.stdout, out.stderr[-400:])
        self.assertNotIn("SECRETKEY123", out.stdout.split("\n")[1])
