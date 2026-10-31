"""Benchmark runner.  python -m nexusml.evaluation.harness [--suite quick|full] [--out results.json] [--markdown EVAL.md] [--check] [--live]

Default mode uses the deterministic stub LLM (offline, reproducible, no secrets): it measures everything that CODE controls
(framing plumbing, cleaning, driver recovery, honesty of the model numbers, claim validation, determinism, cost). `--live` uses the
real LLM from your environment (slower; run it on a schedule). External tools (ChatGPT/Gemini/...) can be added to the table with
--external file.json (see docs/EVALUATION.md for the protocol)."""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from .. import llm, multitable, runlog
from . import datasets as D
from .metrics import score_case
from .stub_llm import make_stub

THRESHOLDS = Path(__file__).with_name("thresholds.json")


def run_case(case: Dict[str, Any], workdir: Path, live: bool = False, parallel: bool = True, seed: int = 42, **state_kw) -> Dict[str, Any]:
    from ..runtime import run_to_completion
    art = tempfile.mkdtemp(dir=workdir)
    env = {"NEXUSML_ARTIFACTS": art, "NEXUSML_PARALLEL": "1" if parallel else "0"}
    if not live:
        env["GROQ_API_KEY"] = "stub-key"
    with mock.patch.dict(os.environ, env):
        ctx = [] if live else [mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "ask_json", side_effect=make_stub(case["frame"]))]
        for c in ctx:
            c.start()
        try:
            t0 = time.time()
            path = case.get("path")
            plan = None
            media_info = None
            if case.get("folder"):
                from .. import media
                media_info = media.prepare_paths([case["folder"]], workdir)
                path = media_info["csv"]
                state_kw = {**{k: v for k, v in media_info["suggest"].items() if v}, **state_kw}
            if case.get("paths"):
                path, plan = multitable.prepare(case["paths"], approve=lambda p: True)
            kw = dict(auto_approve=True, tune_n_iter=2, max_retries=1, enable_tuning=False, seed=seed)
            kw.update(state_kw)
            values = run_to_completion(path, case.get("request", "Why does this happen and what should we do?"), **kw)
            secs = time.time() - t0
            summ = runlog.summary(values["run_id"]) if values.get("run_id") else {}
            scores = score_case({**case, "path": path}, values, seed)
            if media_info:
                scores["media"] = {"n": media_info["n"], "classes": len(media_info["classes"] or {}), "unreadable": len(media_info["unreadable"]),
                                   "kind": media_info["kind"], "task": media_info["suggest"].get("problem_type"), "xs": [e["text"] for e in values["evidence"] if e["id"].startswith("X")]}
            if plan:
                scores["multitable"] = {"rows_ok": plan.get("rows_after") == case["truth"]["rows"], "joins": len(plan["joins"]), "rejected": len(plan["rejected"])}
        finally:
            for c in ctx:
                c.stop()
    shutil.rmtree(art, ignore_errors=True)
    return {"values": values, "scores": scores, "seconds": round(secs, 1), "llm_calls": summ.get("llm_calls"), "tokens": summ.get("tokens")}


def determinism(case, workdir) -> Dict[str, Any]:
    """Same seed twice, and parallel vs sequential: identical evidence?"""
    a = run_case(case, workdir, parallel=True)["values"]
    b = run_case(case, workdir, parallel=False)["values"]
    c = run_case(case, workdir, parallel=True)["values"]
    texts = lambda v: [(e["id"], e["text"]) for e in v["evidence"]]
    return {"parallel_equals_sequential": texts(a) == texts(b), "repeat_equals_first": texts(a) == texts(c)}


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fr = [r["scores"]["framing"] for r in rows]
    drv = [r["scores"]["drivers"]["recall"] for r in rows if r["scores"]["drivers"]]
    gaps = [r["scores"]["model"]["gap"] for r in rows if r["scores"]["model"]]
    unv = [r["scores"]["claims"]["unverified_share"] for r in rows]
    clean = [r["scores"]["cleaning"]["dropped_all"] for r in rows if r["scores"]["cleaning"]]
    ari = [r["scores"]["clustering"]["ari"] for r in rows if r["scores"]["clustering"].get("ari") is not None]
    inter = [r["scores"]["interaction"]["found"] for r in rows if r["scores"]["interaction"]]
    mt = [r["scores"]["multitable"]["rows_ok"] for r in rows if r["scores"].get("multitable")]
    md = [r for r in rows if r["scores"].get("media")]
    dups = [any(("byte-identical" in t or "identical text" in t) for t in r["scores"]["media"]["xs"]) for r in md if r["name"] in ("image_folders", "document_folders")]
    unread = [r["scores"]["media"]["unreadable"] == 1 for r in md if r["name"] == "image_folders"]
    return {"cases": len(rows),
            "framing_accuracy": round(sum(f["target_ok"] and f["problem_type_ok"] for f in fr) / len(fr), 3) if fr else None,
            "driver_recall_mean": round(sum(drv) / len(drv), 3) if drv else None,
            "model_gap_min": round(min(gaps), 4) if gaps else None, "model_gap_mean": round(sum(gaps) / len(gaps), 4) if gaps else None,
            "unverified_claim_share_max": max(unv) if unv else None,
            "cleaning_all_ok": all(clean) if clean else None, "ari_min": min(ari) if ari else None,
            "interaction_found_all": all(inter) if inter else None, "media_duplicates_found": all(dups) if dups else None, "media_unreadable_reported": all(unread) if unread else None, "multitable_rows_ok": all(mt) if mt else None,
            "errors": sum(len(r["scores"]["errors"]) for r in rows),
            "seconds_total": round(sum(r["seconds"] for r in rows), 1)}


def check(summary: Dict[str, Any], det: Dict[str, Any]) -> List[str]:
    th = json.loads(THRESHOLDS.read_text())
    bad = []
    if summary["errors"] > th["max_errors"]:
        bad.append(f"{summary['errors']} run error(s)")
    if summary["framing_accuracy"] is not None and summary["framing_accuracy"] < th["framing_accuracy_min"]:
        bad.append(f"framing accuracy {summary['framing_accuracy']} < {th['framing_accuracy_min']}")
    if summary["driver_recall_mean"] is not None and summary["driver_recall_mean"] < th["driver_recall_mean_min"]:
        bad.append(f"driver recall {summary['driver_recall_mean']} < {th['driver_recall_mean_min']}")
    if summary["model_gap_min"] is not None and summary["model_gap_min"] < th["model_gap_min"]:
        bad.append(f"model gap vs plain baseline {summary['model_gap_min']} < {th['model_gap_min']}")
    if summary["unverified_claim_share_max"] is not None and summary["unverified_claim_share_max"] > th["unverified_claim_share_max"]:
        bad.append(f"unverified claim share {summary['unverified_claim_share_max']} > {th['unverified_claim_share_max']}")
    if summary["cleaning_all_ok"] is False:
        bad.append("a must-drop column was kept")
    if summary.get("interaction_found_all") is False:
        bad.append("the planted two-factor interaction was not found")
    if summary.get("media_duplicates_found") is False:
        bad.append("planted duplicate images / documents were not found")
    if summary.get("media_unreadable_reported") is False:
        bad.append("the unreadable image was not reported")
    if summary.get("multitable_rows_ok") is False:
        bad.append("a multi-table join changed the row count")
    if summary["ari_min"] is not None and summary["ari_min"] < th["ari_min"]:
        bad.append(f"clustering ARI {summary['ari_min']} < {th['ari_min']}")
    if det and not (det["parallel_equals_sequential"] and det["repeat_equals_first"]):
        bad.append(f"non-deterministic: {det}")
    return bad


def to_markdown(rows, summary, det, external=None) -> str:
    L = ["# Evaluation results", "", f"Mode: stub LLM (deterministic). Cases: {summary['cases']}. Total time: {summary['seconds_total']} s.", "",
         "| case | framing | driver recall | model vs plain baseline (gap) | cleaning ok | unverified claims | ARI | seconds | LLM calls |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["scores"]
        L.append(f"| {r['name']} | {'ok' if s['framing']['target_ok'] and s['framing']['problem_type_ok'] else 'WRONG'} | "
                 f"{s['drivers'].get('recall', '-')} | {s['model'].get('gap', '-')} ({s['model'].get('system', '-')} vs {s['model'].get('baseline', '-')}) | "
                 f"{s['cleaning'].get('dropped_all', '-')} | {s['claims']['unverified_share']} | {s['clustering'].get('ari', '-')} | {r['seconds']} | {r['llm_calls']} |")
    L += ["", f"Determinism: {det or 'not run'}", "",
          "Note: with the stub LLM, claim-faithfulness is trivially clean (stub text contains no numbers) and framing only checks the plumbing; "
          "LLM-dependent quality is measured with `--live`.", "", f"Summary: `{json.dumps(summary)}`"]
    if external:
        L += ["", "## External tools (supplied by you)", "", "| tool | case | metric | value |", "|---|---|---|---|"]
        L += [f"| {e['tool']} | {e['case']} | {e['metric']} | {e['value']} |" for e in external]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluation harness")
    ap.add_argument("--suite", choices=["quick", "full"], default="quick")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--markdown")
    ap.add_argument("--check", action="store_true", help="exit 1 if a threshold in thresholds.json is violated")
    ap.add_argument("--live", action="store_true", help="use the real LLM from the environment")
    ap.add_argument("--external", help="JSON list of {tool, case, metric, value} rows to include in the table")
    a = ap.parse_args(argv)
    names = D.QUICK if a.suite == "quick" else list(D.GENERATORS)
    work = Path(tempfile.mkdtemp())
    rows = []
    for n in names:
        case = D.GENERATORS[n](work)
        res = run_case(case, work, live=a.live)
        res["name"] = n
        rows.append({k: v for k, v in res.items() if k != "values"})
        print(f"[{n}] {res['seconds']}s framing={res['scores']['framing']} drivers={res['scores']['drivers'].get('recall')} model={res['scores']['model']}")
    det = determinism(D.planted_classification(work), work)
    summary = aggregate(rows)
    ext = json.loads(Path(a.external).read_text()) if a.external else None
    Path(a.out).write_text(json.dumps({"summary": summary, "determinism": det, "cases": rows}, indent=2, default=str))
    md = to_markdown(rows, summary, det, ext)
    if a.markdown:
        Path(a.markdown).write_text(md)
    print(md)
    shutil.rmtree(work, ignore_errors=True)
    if a.check:
        bad = check(summary, det)
        for b in bad:
            print("REGRESSION:", b)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
