"""CLI: python -m nexusml.cli data.csv "Why is churn high and what should we do?" [options]
      python -m nexusml.cli customers.csv,orders.csv,products.csv "Why do customers churn?" [options]     (several tables)
      python -m nexusml.cli photos.zip "What separates these image classes?"                              (images / documents: a ZIP of
      python -m nexusml.cli ./reports_folder "What themes are in these documents?"                        folders, a folder, or files like a.pdf,b.docx)
      python -m nexusml.cli monitor reference.csv new.csv [--every 3600]                                  (drift monitoring)
      python -m nexusml.cli runs                                                                          (list past runs)
      python -m nexusml.cli doctor                                                                        (which file types work here)
      python -m nexusml.cli ask "What is the difference between bagging and boosting?"                    (no dataset needed)
      python -m nexusml.cli describe photo.jpg [--ai] [--question "what breed?"]                          (one image / one document)
      python -m nexusml.cli describe report.pdf [--ai]

The LLM (Groq) is mandatory: set GROQ_API_KEY in the environment or a .env file. No rule-based fallback."""
import argparse
import os
import sys
import time

from . import catalog, concierge, config, docchat, llm, media, multitable, runstore, vision
from .graph import BACKEND
from .runtime import run_to_completion
from .storage import get_registry


def _monitor(argv) -> int:
    from . import monitoring
    ap = argparse.ArgumentParser(prog="monitor", description="Compare new data with a reference dataset")
    ap.add_argument("reference")
    ap.add_argument("new")
    ap.add_argument("--every", type=int, help="repeat every N seconds (otherwise run once)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    while True:
        rep = monitoring.monitor(a.reference, a.new, a.out)
        print(f"[{rep['ts']}] drift: {'ALERT' if rep['alert'] else 'ok'}; major={rep['major']} moderate={rep['moderate']} missing={rep['missing_columns']}")
        if not a.every:
            return 3 if rep["alert"] else 0
        time.sleep(a.every)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "monitor":
        return _monitor(argv[1:])
    if argv and argv[0] == "doctor":
        print("File support on this machine:")
        bad = 0
        for k, v in media.dependency_status().items():
            print(f"  [{'ok' if v['ok'] else 'MISSING'}] {k:18s} {v['how']}" + (f"   ->  {v['fix']}" if v["fix"] else ""))
            bad += not v["ok"]
        print(f"LLM key: {'set' if llm.available() else f'NOT set ({config.LLM_KEY_ENV})'}")
        return 1 if bad else 0
    if argv and argv[0] == "ask":
        q = " ".join(argv[1:]).strip()
        if not q:
            print('usage: ask "your question"')
            return 2
        try:
            r = concierge.answer({}, q, [], concierge.route(q, False))
        except llm.LLMRequired as e:
            print(e)
            return 2
        print(r["answer"])
        if r["needs_data"]:
            print("\n(If you want this on your own data, give me a file: python -m nexusml.cli data.csv \"your question\")")
        return 0
    if argv and argv[0] == "describe":
        from pathlib import Path
        ap2 = argparse.ArgumentParser(prog="describe")
        ap2.add_argument("file")
        ap2.add_argument("--ai", action="store_true", help="send the image / document text to the AI provider to describe / summarise it")
        ap2.add_argument("--question")
        a2 = ap2.parse_args(argv[1:])
        p = Path(a2.file)
        try:
            if media.kind_of(p.name) == "image":
                print(vision.format_answer(vision.describe(p, a2.question, send_to_ai=a2.ai)))
            elif media.kind_of(p.name) == "document":
                an = docchat.analyze(p)
                print(docchat.format_local(an))
                if a2.ai:
                    print("\n" + docchat.format_ai(docchat.summarize_with_ai({}, an, a2.question)))
                else:
                    print("\n(Add --ai for an AI-written summary; the document text is then sent to the LLM provider.)")
            else:
                print("describe works on one image or one document (png/jpg/... or pdf/docx/txt/html).")
                return 2
        except (media.MediaError, llm.LLMRequired) as e:
            print(f"Stopped: {e}")
            return 2
        return 0
    if argv and argv[0] in ("leaderboard", "card"):
        import json as _json
        from . import leaderboard as _lb
        from .storage import get_registry
        rows = runstore.list_runs(50)
        want = argv[1] if len(argv) > 1 else None
        meta = next((r for r in rows if (want is None or r["run_id"] == want) and r.get("state_key")), None) if rows else None
        if not meta:
            print("No finished run found" + (f" for {want}" if want else "") + ". Run an analysis first (python -m nexusml.cli data.csv \"question\").")
            return 2
        state = _json.loads(get_registry().path_of(meta["state_key"]).read_text(encoding="utf-8"))
        if argv[0] == "card":
            key = state.get("model_card_key")
            print(get_registry().path_of(key).read_text(encoding="utf-8") if key else _lb.model_card(state))
            return 0
        lb = state.get("leaderboard")
        if not lb:
            print("This run has no leaderboard (descriptive or clustering analysis).")
            return 2
        import pandas as _pd
        with _pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60):
            print(lb["summary"] + "\n")
            print(_lb.to_dataframe(lb, include_past=True).drop(columns=["verdict"]).to_string(index=False))
            print("\nVerdicts:")
            for r in lb["rows"]:
                print(f"  #{r['rank']} {r['name']}: {r['verdict']}")
        return 0
    if argv and argv[0] == "runs":
        for r in runstore.list_runs(20):
            print(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['created']))}  {r['run_id']}  {r['analysis_type']}/{r['target']}  best={r['best_model']}  "
                  f"heldout={r['heldout']}  llm_calls={r['llm_calls']}")
        return 0
    ap = argparse.ArgumentParser(description="Multi-agent Business Analyst")
    ap.add_argument("dataset")
    ap.add_argument("request", nargs="?", default="")
    ap.add_argument("--provider", choices=sorted(config.PROVIDERS), help="LLM provider (its API key is read from its environment variable, e.g. GROQ_API_KEY / GEMINI_API_KEY)")
    ap.add_argument("--model", help="model id for the chosen provider")
    ap.add_argument("--target")
    ap.add_argument("--problem-type", "--task", dest="problem_type", choices=catalog.PROBLEM_TYPES + ["forecasting"],
                    help="analytical problem (classification|regression|time_series|clustering|descriptive)")
    ap.add_argument("--analysis-type", choices=list(catalog.ANALYSIS_TYPES), help="business problem, e.g. sales_performance, churn")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--auto-approve", action="store_true", help="skip human review of column drops")
    ap.add_argument("--no-tune", action="store_true", help="skip hyper-parameter tuning (faster)")
    ap.add_argument("--tune-iter", type=int, help="random-search iterations per tuned model")
    ap.add_argument("--powerbi", action="store_true", help="write a Power BI package (and push if POWERBI_* env vars are set)")
    ap.add_argument("--seed", type=int, help="random seed for this run (recorded in the run log)")
    ap.add_argument("--cost-fn", type=float, help="business cost of a MISSED positive (used to choose the decision cutoff)")
    ap.add_argument("--cost-fp", type=float, help="business cost of a FALSE ALARM")
    ap.add_argument("--target-metric", help="metric to reach, e.g. roc_auc, recall, precision, f1, r2, rmse")
    ap.add_argument("--target-value", type=float, help="value the target metric must reach (<= for rmse/mae, >= otherwise)")
    ap.add_argument("--target-subject-to", help="secondary goal as metric=value, e.g. precision=0.5")
    ap.add_argument("--time-budget", type=float, help="seconds the AutoML agent may spend searching (default: sized from the data)")
    ap.add_argument("--workers", type=int, help="parallel AutoML workers (default: chosen by the Director, at most your CPU count)")
    ap.add_argument("--exclude-sensitive", action="store_true", help="drop detected protected attributes (gender, age, ...) from the model features")
    ap.add_argument("--no-flexible", action="store_true", help="always run the full pipeline (disable the bounded supervisor)")
    ap.add_argument("--explore", action="store_true", help="seeded exploration of retry strategies")
    ap.add_argument("--sequential", action="store_true", help="run graph branches one after another (same results, easier debugging)")
    ap.add_argument("--llm-theme-names", action="store_true", help="let the LLM see the top KEYWORDS of document themes (never raw text) to name them")
    ap.add_argument("--approve-joins", action="store_true", help="approve the proposed multi-table join plan without asking")
    a = ap.parse_args(argv)
    if a.sequential:
        import os
        os.environ["NEXUSML_PARALLEL"] = "0"
    target_spec = None
    if a.target_metric and a.target_value is not None:
        target_spec = {"metric": a.target_metric, ("max" if a.target_metric in ("rmse", "mae") else "min"): a.target_value}
        if a.target_subject_to:
            m, _, v = a.target_subject_to.partition("=")
            target_spec["subject_to"] = {"metric": m, ("max" if m in ("rmse", "mae") else "min"): float(v)}
    if a.dataset.lower().startswith(("http://", "https://")):          # a link: download it first, then continue exactly as for a local path
        from . import linkfetch
        try:
            a.dataset = linkfetch.materialize(a.dataset, str(__import__("pathlib").Path(config.artifacts_dir()) / "downloads" / str(int(time.time()))))
        except linkfetch.LinkError as e:
            print(f"Could not use the link: {e}", file=sys.stderr)
            return 2
        print(f"(downloaded: {a.dataset})")
    table_plan = None
    dataset = a.dataset
    media_info = None
    _ds = [p for p in a.dataset.split(",") if p]
    _is_media = any(__import__("pathlib").Path(p).is_dir() and not any(f.suffix.lower() in media.TABULAR_EXT for f in __import__("pathlib").Path(p).iterdir() if f.is_file())
                    or media.kind_of(p) in ("image", "document", "zip") for p in _ds)
    if _is_media:                                                     # images / documents / folders / ZIPs
        try:
            media_info = media.prepare_paths(_ds, __import__("pathlib").Path(config.artifacts_dir()) / "media_uploads")
        except media.MediaError as e:
            print(f"Stopped: {e}")
            return 2
        dataset = media_info["csv"]
        sg = media_info["suggest"]
        a.target = a.target or sg.get("target_column")
        a.problem_type = a.problem_type or sg.get("problem_type")
        a.analysis_type = a.analysis_type or sg.get("analysis_type")
        print(f"Read {media_info['n']} {media_info['kind']}" + (f" in classes {media_info['classes']}" if media_info["classes"] else " (no class folders: grouping by similarity)")
              + (f"; unreadable: {len(media_info['unreadable'])}" if media_info["unreadable"] else ""))
        for w in media_info["warnings"]:
            print("  note:", w)
    if not _is_media and ("," in a.dataset or (a.dataset and not a.dataset.lower().endswith(tuple(media.TABULAR_EXT)) and __import__("pathlib").Path(a.dataset).is_dir())):
        paths = [p for p in a.dataset.split(",") if p]
        def approve(plan):
            print("\nProposed table joins:\n" + multitable.describe(plan))
            return a.approve_joins or input("Approve these joins? [Y/n] ").strip().lower() != "n"
        try:
            dataset, table_plan = multitable.prepare(paths, approve=approve, target=a.target)      # the table holding the target column becomes the main table
            runstore.audit("join_plan_approved", None, {"base": table_plan["base"], "joins": [j["table"] for j in table_plan["joins"]]})
        except multitable.ApprovalRequired as e:
            print(f"Stopped: {e}")
            return 2

    def on_event(node, upd):
        for e in upd.get("agent_log", []):
            print(f"[{e['ts']}] {e['agent']:<18} {e['message']}")

    def on_clarify(values):
        print("\n? " + values.get("clarification_question", ""))
        t = input("Target column (blank for none): ").strip()
        p = input("Problem type [classification/regression/time_series/clustering/descriptive/auto]: ").strip()
        return {"target_column": t or None, "problem_type": None if p in ("", "auto") else p}

    def on_approval(values):
        print("\nProposed destructive actions:")
        for x in [x for x in values["analysis_plan"]["actions"] if x["destructive"]]:
            print(f"  - drop {x['column']!r} ({x['issue']}): {x['reason']}" + (f"  [LLM: {x['llm_opinion']}]" if x.get("llm_opinion") else ""))
        return {"approved": input("Approve all? [Y/n] ").strip().lower() != "n"}

    if getattr(a, "provider", None) or getattr(a, "model", None):
        config.set_provider(a.provider or config.LLM_PROVIDER, a.model or "")
    print(f"(backend: {BACKEND}; LLM: {config.LLM_PROVIDER_NAME} {config.LLM_MODEL})")
    try:
        v = run_to_completion(dataset, a.request, on_event=on_event, on_clarify=on_clarify, on_approval=on_approval,
                              seed=a.seed, cost_fn=a.cost_fn, cost_fp=a.cost_fp, target_spec=target_spec, exclude_sensitive=a.exclude_sensitive,
                              flexible=not a.no_flexible, explore=a.explore, table_plan=table_plan, allow_llm_text=a.llm_theme_names,
                              target_column=a.target, problem_type=a.problem_type, analysis_type=a.analysis_type,
                              max_retries=a.max_retries, auto_approve=a.auto_approve,
                              enable_tuning=not a.no_tune, time_budget=a.time_budget, workers=a.workers, tune_n_iter=a.tune_iter,
                              powerbi_enabled=a.powerbi)
    except llm.LLMRequired as e:
        print(f"\nStopped: {e}")
        return 2
    except (EOFError, KeyboardInterrupt):
        return 130
    if v.get("errors"):
        print("\nFailed:", *v["errors"], sep="\n  ")
        return 1
    reg = get_registry()
    print(f"\nHeadline: {v['insights'].get('headline', '')}\n")
    print(f"Report (Markdown): {reg.path_of(v['report_key'])}\nReport (HTML, charts embedded): {reg.path_of(v['report_html_key'])}")
    if v.get("exec_summary_key"):
        print(f"Executive brief (1 page): {reg.path_of(v['exec_summary_key'])}")
    if v.get("repro_key"):
        print(f"Reproducibility kit: {reg.path_of(v['repro_key'])}")
    ci = v.get("metric_ci") or {}
    if ci.get("value") is not None:
        print(f"Held-out {ci['metric']}: {ci['value']}" + (f" (95% CI {ci['lo']}-{ci['hi']})" if ci.get("lo") is not None else ""))
    if (v.get("target_report") or {}).get("spec"):
        tr = v["target_report"]
        print(f"Target {'MET' if tr.get('met_heldout') else 'NOT met'} on held-out data (search stopped: {tr.get('stopped_because')})")
    if v.get("model_key"):
        print(f"Model bundle: {reg.path_of(v['model_key'])}")
    if v.get("powerbi_result"):
        print(f"Power BI package: {v['powerbi_result']['package_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
