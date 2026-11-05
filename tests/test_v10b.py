"""Assistant without a dataset, pictures and documents in the conversation, summarising an analysis, and clear feedback after adding data."""
import io, json, os, shutil, sys, tempfile, unittest, warnings
from pathlib import Path
from unittest import mock
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("NEXUSML_BACKEND", "builtin")
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from nexusml import linkfetch, llm, runtime, workspace as W
from nexusml.assistant import agent as A, attachments as AT, tools as T
from nexusml.evaluation.stub_llm import make_stub
from nexusml.ui import _st, chat, nav, state
from nexusml.ui.pages import assistant, data, home
from fake_st import FakeSt, run_page

F = {"analysis_type": "churn", "problem_type": "descriptive", "target": None, "date_column": None, "segment_columns": [], "kpis": [{"name": "Records", "agg": "count"}]}


class Up:
    def __init__(s, name, b): s.name, s._b = name, b
    def getvalue(s): return s._b


def csv_bytes(n=60):
    rs = np.random.RandomState(1)
    return pd.DataFrame({"a": rs.rand(n), "b": rs.choice(list("xyz"), n)}).to_csv(index=False).encode()


def png_bytes(w=64, h=48, color=(200, 30, 30)):
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (w, h), color).save(buf, "PNG"); return buf.getvalue()


def docx_bytes():
    import docx
    d = docx.Document()
    d.add_heading("Supplier agreement", 1)
    for t in ("This agreement between Acme Ltd and Borealis Co. starts on 2026-03-01 and runs for twelve months.",
              "Acme will deliver 400 units per month at a price of $1,250 per unit, invoiced on the first day of each month.",
              "Either side may end the agreement with 60 days notice. Late payments carry a 2% monthly fee.",
              "Borealis must keep the delivered goods insured and report any damage within five working days.",
              "The agreement renews automatically unless cancelled in writing before the end of the term."):
        d.add_paragraph(t)
    buf = io.BytesIO(); d.save(buf); return buf.getvalue()


def pdf_bytes():
    from reportlab.pdfgen import canvas
    buf = io.BytesIO(); c = canvas.Canvas(buf)
    y = 800
    for line in ("Quarterly report", "Revenue grew steadily across all regions during the quarter and the team met its main goals.",
                 "The largest increase came from the northern region where new customers joined in March 2026.",
                 "Costs stayed flat, with a total spend of $48,200 on marketing and support combined.",
                 "Management plans to hire two analysts and review pricing before the next quarter begins."):
        c.drawString(40, y, line); y -= 24
    c.showPage(); c.save(); return buf.getvalue()


class Env(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"NEXUSML_ARTIFACTS": self.d, "GROQ_API_KEY": "sk-secret-value-xyz"}); self.env.start()
        self.p = [mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "preflight", return_value=None)]
        [p.start() for p in self.p]

    def tearDown(self):
        [p.stop() for p in self.p]; self.env.stop(); _st.use(None); shutil.rmtree(self.d, ignore_errors=True)

    def say(self, message, steps, ctx=None, **kw):
        sent, script = [], list(steps)

        def step(pl):
            sent.append(pl)
            return script.pop(0) if script else {"reply": "done", "about": "general"}
        with mock.patch.object(llm, "ask_json", side_effect=make_stub(F, {"assistant_step": step})):
            r = A.respond(message, [], ctx, **kw)
        return r, sent


class TestAssistantWithoutData(Env):
    def test_a_general_question_is_answered_normally_with_no_dataset_and_no_warnings(self):
        r, sent = self.say("What is a p-value, and what cut-off do people use?", [{"reply": "A p-value is the chance of seeing a result at least this extreme if nothing is going on. Most fields use a cut-off of 0.05, which means about 1 in 20.", "about": "general"}])
        self.assertIn("cut-off of 0.05", r["text"]); self.assertEqual(r["warnings"], []); self.assertEqual(r["steps"], []); self.assertEqual(len(sent), 1)
        self.assertIsNone(sent[0]["dataset"])

    def test_without_a_dataset_the_model_is_not_offered_tools_that_need_one(self):
        _, sent = self.say("hello", [{"reply": "Hello! How can I help?", "about": "general"}])
        names = " ".join(sent[0]["tools"])
        self.assertNotIn("distribution(", names); self.assertNotIn("filter_rows(", names); self.assertIn("analyses(", names); self.assertIn("packages(", names)
        _, sent2 = self.say("hello", [{"reply": "Hi", "about": "general"}], T.ToolContext(pd.DataFrame({"a": [1, 2, 3]}), "x"))
        self.assertIn("distribution(", " ".join(sent2[0]["tools"]))

    def test_the_prompt_says_to_answer_general_questions_and_never_to_refuse_for_lack_of_data(self):
        low = A.SYSTEM.lower()
        for phrase in ("any good assistant", "never refuse", "general question", "attachments", "part=summary", '"about"'):
            self.assertIn(phrase, low, phrase)

    def test_a_general_label_only_excuses_a_reply_that_used_no_tools(self):
        invented = "You have 87.3% churn."
        r, _ = self.say("how is my churn?", [{"reply": invented, "about": "workspace"}, {"reply": "I can't state a churn figure without checking.", "about": "workspace"}])
        self.assertNotIn("87.3", r["text"])
        r2, _ = self.say("how many analyses?", [{"action": {"tool": "analyses", "args": {}}}, {"reply": "You have 87.3 analyses.", "about": "general"}, {"reply": "None yet.", "about": "general"}])
        self.assertNotIn("87.3", r2["text"])                                                    # a tool ran: figures are checked even if the model calls it general

    def test_data_figures_are_still_checked_when_a_dataset_is_loaded(self):
        r, _ = self.say("describe it", [{"reply": "The mean is 12345.678.", "about": "workspace"}, {"reply": "I would need to check.", "about": "workspace"}], T.ToolContext(pd.DataFrame({"a": [1, 2, 3]}), "x"))
        self.assertNotIn("12345.678", r["text"])


class TestSummariseAnAnalysis(Env):
    def test_one_call_gives_everything_needed_to_summarise_an_analysis(self):
        with mock.patch.object(llm, "ask_json", side_effect=make_stub(F)):
            meta = W.add_dataset("c.csv", csv_bytes(300))
            v = runtime.run_to_completion(str(W.dataset_path(meta["id"])), "describe the data", auto_approve=True, enable_tuning=False, seed=1)
        r = T.run_tool(T.ToolContext(pd.DataFrame(), ""), "analysis", {"run": "latest", "part": "summary"})
        for k in ("asked", "data", "approach", "headline", "key_findings", "recommendations", "caveats", "result"):
            self.assertIn(k, r.summary, k)
        self.assertEqual(r.summary["analysis"], v["experiment_id"]); self.assertIn("No predictive model", r.summary["result"]["note"]); self.assertLess(len(json.dumps(r.summary, default=str)), 3000)
        reply, sent = self.say("summarise my last analysis", [{"action": {"tool": "analysis", "args": {"run": "latest", "part": "summary"}}}, {"reply": "You asked to describe the data; no model was built.", "about": "workspace"}])
        self.assertEqual([s["tool"] for s in reply["steps"]], ["analysis"]); self.assertEqual(reply["warnings"], []); self.assertIn("no model was built", reply["text"])


class TestPicturesAndDocuments(Env):
    def test_what_kind_of_file_is_it(self):
        self.assertEqual([AT.kind_of(n) for n in ("a.PNG", "b.jpeg", "c.pdf", "d.docx", "e.md", "f.csv", "g.zip", "h.xlsx")], ["image", "image", "document", "document", "document", None, None, None])

    def vision(self, **over):
        d = {"description": "A solid red rectangle on a plain background.", "type": "drawing", "objects": ["red rectangle"], "visible_text": "", "details": ["No text is visible."], **over}
        return mock.patch.object(llm, "ask_vision", return_value=(d, None, {}))

    def test_a_picture_is_described(self):
        with self.vision() as v:
            att = AT.read("red.png", png_bytes())
        self.assertTrue(att["ok"]); self.assertEqual(att["size"], "64 x 48"); self.assertEqual(v.call_args[0][4], 700); self.assertEqual(v.call_args[0][3], "image/png")
        txt = AT.message_text(att)
        self.assertIn("solid red rectangle", txt); self.assertIn("red.png", txt); self.assertIn("was sent to", txt)
        ctx = AT.context(att); self.assertEqual(ctx["kind"], "image"); self.assertIn("red rectangle", ctx["what_it_shows"])

    def test_text_in_a_picture_is_read_out(self):
        with self.vision(type="document", visible_text="Invoice 1042 total 318.50"):
            att = AT.read("inv.jpg", png_bytes())
        self.assertIn("Invoice 1042 total 318.50", AT.message_text(att))

    def test_a_picture_that_cannot_be_looked_at_says_why_instead_of_failing(self):
        with mock.patch.object(llm, "ask_vision", return_value=(None, "HTTP 400 (model not available)", {})):
            att = AT.read("a.png", png_bytes())
        self.assertFalse(att["ok"]); self.assertIn("model not available", AT.message_text(att)); self.assertEqual(AT.followups(att), [])
        with mock.patch.object(llm, "available", return_value=False), mock.patch.object(llm, "ask_vision") as v:
            att2 = AT.read("a.png", png_bytes())
        self.assertIn("not connected", att2["error"]); v.assert_not_called()

    def test_a_very_large_picture_is_made_smaller_or_refused_without_calling_the_model(self):
        big = b"\x89PNG" + b"0" * (AT.MAX_IMAGE_BYTES + 10)
        with mock.patch.object(llm, "ask_vision") as v:
            att = AT.read("big.png", big)
        self.assertFalse(att["ok"]); self.assertIn("too large", att["error"]); v.assert_not_called()
        with mock.patch.object(AT, "_shrink", return_value=png_bytes()), self.vision() as v2:
            att2 = AT.read("big.png", big)
        self.assertTrue(att2["ok"]); self.assertEqual(v2.call_args[0][3], "image/jpeg")

    def test_a_word_document_is_summarised(self):
        with mock.patch.object(llm, "ask_json", side_effect=make_stub(F, {"summarize_document": lambda pl: {"summary": "A twelve month supply agreement between Acme and Borealis.", "key_points": ["400 units a month at $1,250", "60 days notice to end"], "action_items": ["Report damage within five days"], "topics": ["supply", "payment"], "questions_to_ask": ["What happens on late payment?"]}})):
            att = AT.read("agreement.docx", docx_bytes())
        self.assertTrue(att["ok"]); self.assertEqual(att["mode"], "model")
        txt = AT.message_text(att)
        for want in ("twelve month supply agreement", "400 units a month", "Report damage", "was sent to"): self.assertIn(want, txt)
        ctx = AT.context(att); self.assertIn("$1,250", ctx["amounts"]); self.assertIn("2026-03-01", ctx["dates"]); self.assertEqual(AT.followups(att), ["What happens on late payment?"])

    def test_a_pdf_is_summarised(self):
        with mock.patch.object(llm, "ask_json", side_effect=make_stub(F, {"summarize_document": lambda pl: {"summary": "Revenue grew in every region; the north grew most.", "key_points": ["Costs flat at $48,200"], "action_items": [], "topics": ["revenue"], "questions_to_ask": []}})):
            att = AT.read("q1.pdf", pdf_bytes())
        self.assertTrue(att["ok"], att.get("error")); self.assertEqual(att["pages"], 1); self.assertIn("Revenue grew", AT.message_text(att)); self.assertIn("$48,200", AT.context(att)["amounts"])

    def test_without_the_language_model_a_document_still_gets_a_local_summary(self):
        with mock.patch.object(llm, "available", return_value=False):
            att = AT.read("agreement.docx", docx_bytes())
        self.assertTrue(att["ok"]); self.assertEqual(att["mode"], "local"); txt = AT.message_text(att)
        self.assertIn("not connected", txt); self.assertIn("Acme", txt); self.assertNotIn("was sent to", txt)

    def test_if_the_model_fails_the_local_summary_is_used(self):
        with mock.patch.object(llm, "ask_json", side_effect=llm.LLMRequired("down")):
            att = AT.read("agreement.docx", docx_bytes())
        self.assertTrue(att["ok"]); self.assertTrue(att["model_failed"]); self.assertIn("did not answer", AT.message_text(att))

    def test_an_unreadable_document_gives_a_message_not_a_crash(self):
        att = AT.read("broken.docx", b"this is not a word file")
        self.assertFalse(att["ok"]); self.assertTrue(AT.message_text(att).startswith("**broken.docx**"))

    def test_the_word_AI_never_appears_in_what_the_person_reads(self):
        with self.vision():
            texts = [AT.message_text(AT.read("a.png", png_bytes()))]
        texts.append(AT.message_text(AT.read("a.docx", docx_bytes())))
        import re
        self.assertFalse(any(re.search(r"\bAI\b|\bLLM\b", t) for t in texts))


class TestFilesInTheConversation(Env):
    def page_ingest(self, files, **kw):
        f = FakeSt(**kw); _st.use(f); state.init(); chat.ingest(files); return f

    def test_a_dropped_picture_is_described_in_the_chat_and_not_added_as_a_dataset(self):
        with mock.patch.object(llm, "ask_vision", return_value=({"description": "A red square.", "type": "drawing", "objects": ["square"], "visible_text": "", "details": []}, None, {})):
            f = self.page_ingest([Up("red.png", png_bytes())])
        roles = [m["role"] for m in f.session_state.messages]
        self.assertEqual(roles, ["user", "assistant"]); self.assertIn("red.png", f.session_state.messages[0]["text"]); self.assertIn("A red square.", f.session_state.messages[1]["text"])
        self.assertEqual(f.session_state.attachments[0]["name"], "red.png"); self.assertEqual(W.list_datasets(), [])

    def test_a_dropped_document_is_summarised_in_the_chat(self):
        with mock.patch.object(llm, "ask_json", side_effect=make_stub(F)):
            f = self.page_ingest([Up("agreement.docx", docx_bytes())])
        self.assertIn("Stub summary.", f.session_state.messages[1]["text"]); self.assertEqual(f.session_state.attachments[0]["kind"], "document"); self.assertEqual(W.list_datasets(), [])

    def test_a_dropped_pdf_that_holds_a_table_is_also_added_as_a_dataset(self):
        tbl = {"name": "t.csv", "id": "dsx", "rows": 12, "cols": 3}
        with mock.patch.object(llm, "ask_json", side_effect=make_stub(F)), mock.patch.object(W, "_add_pdf_tables", return_value=tbl):
            f = self.page_ingest([Up("q1.pdf", pdf_bytes())])
        self.assertEqual(f.session_state.ds_id, "dsx"); self.assertTrue(any("also contains a table" in m["text"] for m in f.session_state.messages))

    def test_tables_still_become_datasets_as_before(self):
        f = self.page_ingest([Up("sales.csv", csv_bytes())])
        self.assertEqual(len(W.list_datasets()), 1); self.assertEqual(f.session_state.ds_id, W.list_datasets()[0]["id"]); self.assertEqual(f.session_state.messages[0]["role"], "note"); self.assertFalse(f.session_state.get("attachments"))

    def test_what_was_learned_from_a_file_reaches_the_assistant_and_its_figures_can_be_quoted(self):
        att = {"name": "agreement.docx", "kind": "document", "summary": "A supply agreement.", "amounts": ["$1,250"], "dates": ["2026-03-01"], "excerpt": "Acme will deliver 400 units"}
        r, sent = self.say("what is the unit price?", [{"reply": "The price is $1,250 per unit, from 2026-03-01.", "about": "workspace"}], attachments=[att])
        self.assertEqual(sent[0]["attachments"], [att]); self.assertEqual(r["warnings"], []); self.assertIn("$1,250", r["text"])
        r2, sent2 = self.say("what is the unit price?", [{"reply": "The price is $9,999 per unit.", "about": "workspace"}, {"reply": "I can't confirm a price.", "about": "workspace"}], attachments=[att])
        self.assertNotIn("9,999", r2["text"])
        _, sent3 = self.say("hi", [{"reply": "hi", "about": "general"}]); self.assertNotIn("attachments", sent3[0])

    def test_a_question_about_the_file_goes_through_the_chat_with_the_file_in_context(self):
        att = {"name": "red.png", "kind": "image", "what_it_shows": "A red square."}
        seen = []

        def step(pl):
            seen.append(pl); return {"reply": "It shows a red square.", "about": "workspace"}
        with mock.patch.object(llm, "ask_json", side_effect=make_stub(F, {"assistant_step": step})):
            f = FakeSt(session={"attachments": [att]}); _st.use(f); state.init(); run_page(lambda: chat.answer("what is in the picture?", None, None))
        self.assertEqual(seen[0]["attachments"][0]["what_it_shows"], "A red square."); self.assertEqual(f.session_state.messages[-1]["text"], "It shows a red square.")

    def test_a_new_conversation_forgets_the_files_of_the_old_one(self):
        f = FakeSt(session={"attachments": [{"name": "x"}]}); _st.use(f); state.init(); state.new_chat(); self.assertEqual(f.session_state.attachments, [])

    def test_the_assistant_page_invites_questions_and_files_without_a_dataset(self):
        f = FakeSt(); _st.use(f); run_page(assistant.render)
        ci = next(k for n, a, k in f.calls if n == "chat_input"); self.assertIn("Ask anything", f.calls[[n for n, _, _ in f.calls].index("chat_input")][1][0])
        self.assertTrue({"png", "jpg", "pdf", "docx"} <= set(ci["file_type"])); self.assertTrue(any("No dataset needed" in t for t in f.texts()))
        self.assertFalse(ci.get("disabled"))


class TestUploadFeedback(Env):
    def setUp(self):
        super().setUp(); nav.register("data", "PAGE-DATA")

    def page(self, fn, **kw):
        f = FakeSt(**kw); _st.use(f); run_page(fn); return f

    def success_texts(self, f):
        return [str(a[0]) for n, a, k in f.calls if n == "success"]

    def test_after_adding_a_dataset_the_data_page_says_so_and_shows_it_selected(self):
        old = W.add_dataset("old.csv", csv_bytes(30))
        f = self.page(data.render, session={"ds_id": old["id"], "data-pick": old["id"], "_seen:data-pick": old["id"]}, values={"data-upload": [Up("sales.csv", csv_bytes(80))]}, clicks=["data-add"])
        new = next(d for d in W.list_datasets() if d["name"] == "sales.csv")
        s = self.success_texts(f)
        self.assertEqual(len(s), 1); self.assertIn("sales.csv", s[0]); self.assertIn("80 rows and 2 columns", s[0]); self.assertIn("now the selected dataset", s[0])
        self.assertEqual(f.session_state.ds_id, new["id"]); self.assertEqual(f.session_state["data-pick"], new["id"])                 # the picker follows: it is not left on the old dataset
        pick = next(k for n, a, k in f.calls if n == "selectbox" and k.get("key") == "data-pick"); self.assertEqual(pick["index"], list(d["id"] for d in W.list_datasets()).index(new["id"]))

    def test_the_message_is_shown_once_and_the_file_box_is_emptied(self):
        f = self.page(data.render, values={"data-upload": [Up("sales.csv", csv_bytes())]}, clicks=["data-add"])
        self.assertEqual(f.session_state.get("_up:data"), 1)
        keys = [k["key"] for n, a, k in f.calls if n == "file_uploader"]; self.assertEqual(keys, ["data-upload", "data-upload-1"])
        f2 = FakeSt(session=dict(f.session_state)); _st.use(f2); run_page(data.render)
        self.assertEqual(self.success_texts(f2), [])

    def test_a_file_that_cannot_be_read_is_explained_even_though_the_page_reloads(self):
        f = self.page(data.render, values={"data-upload": [Up("tool.exe", b"MZ"), Up("sales.csv", csv_bytes())]}, clicks=["data-add"])
        errs = [str(a[0]) for n, a, k in f.calls if n == "error"]
        self.assertTrue(any("tool.exe" in e for e in errs)); self.assertEqual(len(self.success_texts(f)), 1)

    def test_several_files_are_each_listed_and_only_the_last_is_called_selected(self):
        f = self.page(data.render, values={"data-upload": [Up("a.csv", csv_bytes(10)), Up("b.csv", csv_bytes(20))]}, clicks=["data-add"])
        s = self.success_texts(f); self.assertEqual(len(s), 2); self.assertNotIn("selected", s[0]); self.assertIn("selected", s[1]); self.assertIn("b.csv", s[1])

    def test_nothing_is_announced_when_nothing_was_added(self):
        f = self.page(data.render, values={"data-upload": [Up("tool.exe", b"MZ")]}, clicks=["data-add"])
        self.assertEqual(self.success_texts(f), []); self.assertEqual(W.list_datasets(), [])
        self.assertEqual(f.session_state.get("_up:data", 0), 0)

    def test_the_home_page_says_so_too(self):
        f = self.page(home.render, values={"home-upload": [Up("sales.csv", csv_bytes(40))]}, clicks=["home-add"])
        self.assertTrue(any("sales.csv" in s and "40 rows" in s for s in self.success_texts(f)))

    def test_a_link_reports_what_it_added(self):
        meta = W.add_dataset("kag.csv", csv_bytes(25))
        res = {"primary": meta, "datasets": [meta], "notes": [], "source": "Kaggle"}
        with mock.patch.object(linkfetch, "add_from_link", return_value=res):
            f = self.page(data.render, values={"data-link": "https://www.kaggle.com/datasets/a/b"}, clicks=["data-fetch"])
        self.assertTrue(any("kag.csv" in s and "from Kaggle" in s for s in self.success_texts(f)))

    def test_the_picker_follows_a_new_selection_but_never_overrides_the_persons_own_choice(self):
        f = FakeSt(session={"ds_id": "B", "data-pick": "A", "_seen:data-pick": "A"}); _st.use(f)
        state.sync_pick("data-pick", {"A": 1, "B": 2}); self.assertEqual(f.session_state["data-pick"], "B")             # selection changed elsewhere (an upload)
        f.session_state["data-pick"] = "A"                                                                                # the person picks A in the box: ds_id has not moved yet
        state.sync_pick("data-pick", {"A": 1, "B": 2}); self.assertEqual(f.session_state["data-pick"], "A")
        f2 = FakeSt(session={"ds_id": None, "asst-ds": "B", "_seen:asst-ds": "B"}); _st.use(f2)
        state.sync_pick("asst-ds", {"B": 1}, "No dataset"); self.assertEqual(f2.session_state["asst-ds"], "No dataset")


if __name__ == "__main__":
    unittest.main()
