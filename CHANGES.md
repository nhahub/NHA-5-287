# Revision 3.1: the assistant without a dataset, pictures and documents, summaries, and upload feedback

- **General questions work with no dataset.** The reply check that every figure must come from the data no longer blocks plain answers: the model labels each reply `about: general | workspace`, and a
  general reply that used no tool is not checked (anything else still is). Tools that need a dataset are not offered when none is loaded. The prompt and the Assistant page no longer read as data-only.
- **Pictures and documents in the assistant only** (assistant/attachments.py, ui/chat.py): a dropped picture (png, jpg, webp, gif) is described, including any text in it; a dropped PDF, Word, ODT, HTML or
  Markdown file is summarised (key points, actions, dates, amounts). They are NOT added to the workspace's datasets and the analysis pipeline is untouched (a PDF that contains a table is still also added as a dataset).
  The reply says the file was sent to the model provider. Without a connection a document gets a local summary; a picture gets a clear message. What was learned is kept for follow-up questions in that conversation
  (numbers quoted from a file pass the figure check).
- **Summarise an analysis:** `analysis(part="summary")` returns what was asked, what was found, how reliable it is, caveats and next steps in one call.
- **Adding data is now visible.** After an upload or a pasted link the Data and Home pages say what was added (name, rows, columns), mark it selected, list any file that could not be read (these used to vanish on reload), and
  empty the file box. Root cause of "I have to look for it": the dataset picker kept its old value, so the page quietly switched back to the previously selected dataset. Both pickers (Data, Assistant) now follow the selection.
- Tests: tests/test_v10b.py (33).
- Limits: pictures/documents are per conversation (not restored when an old conversation is reopened); a picture is only as well described as the provider's vision model allows; the dock accepts text only (drop files on the Assistant page).

# Revision 3: building on analyses, managing them, and an assistant that sees the whole workspace

## 1. The assistant can read everything (assistant/runfacts.py, assistant/workspace_ctx.py, assistant/tools.py)
- New tools that need no dataset: `analyses` (list / compare), `analysis` (one analysis: overview, metrics, explanation, trials, statistics, kpis, trends, findings, evidence, audit, decisions, charts,
  report sections, notes), `analysis_chart` (list or show a chart), `jobs_status`, `workspace_info` (datasets, registered models, settings, current page), `packages` (installed packages),
  `note_concern`, `propose_extension`. Everything is bounded and aggregate-only; no rows, paths or keys.
- Every turn carries a compact `workspace` block (page, open analysis, datasets, recent analyses, what is running, language model name). Counts in it can be quoted without tripping the invented-figure check.
- New rules: look at the evidence before answering a concern and say whether it is supported; record it with `note_concern` if asked; use `propose_extension` to continue an analysis.
- Concerns are stored per analysis (`runstore.notes`) and shown on the analysis page.

## 2. Build on an analysis instead of starting over (resume.py, state.py, agents/stages.py)
- `run_to_completion(..., resume_from=<experiment id>, resume_reuse=None|[...])`. The follow-up is a NEW analysis with its own report that remembers its parent (`parent_experiment_id`, `lineage` table).
- Reused when still true: data profile (dataset file unchanged), KPIs (definitions pinned to the parent), trends/outliers (same date column), statistics (only if target and predictive-or-not are unchanged).
  Planning, split, model search, explanation, charts and the report are always computed fresh. Reused evidence keeps its ids and only reused phases' evidence is carried, so ids cannot collide.
- The framing step receives `previous_analysis` (what was asked and found), so "now build a model" refers to the earlier question and target.
- A follow-up is validated when submitted (`jobs.submit`), so a deleted parent or a different dataset is explained at once.
- Measured with the offline test double: run 1 (describe only) 2 s; run 2 (model) reused profile, KPIs, trends; 0 profile recomputations; only the Statistician reran.

## 3. History management (lineage.py, runstore.py, jobs.py, ui/history.py)
- **Delete** a finished analysis: removes its state, reports, charts, model bundle, run log, graph checkpoint, tracking records, registered model versions and index entry. A file another analysis still
  references is kept (checked by scanning the other saved states). Follow-ups keep their own results. The audit log records the deletion.
- Failed, cancelled and "needs input" analyses are now stored (`attempts`), so they survive a restart and can be retried or removed. "Clear unfinished" removes them all.
- **Retry with changes**: a dialog with the question, target and kind editable, using the original options. For a finished analysis an option reuses its still-valid results; an unfinished attempt is replaced.
- The reproducibility kit of a follow-up omits `resume_from` and says it runs from scratch.

## 4. Assistant everywhere, and the Analyses page (ui/dock.py, ui/context.py, ui/chat.py, ui/pages/analyze.py, runs.py)
- Every page reports its view (`context.set_view`). A sidebar button on every page except the Assistant page opens the same conversation in a dialog; the assistant is told the page and the open analysis.
- **One Analyses page**: a large "New analysis" composer (or "Follow-up" when building on one) takes the main area; history is a compact scrolling rail with Open / Build on this / Retry / Delete per entry.
  "New analysis" and the old history page are no longer separate navigation entries (both names still route here).
- Analysis page: Build on this, Retry with changes, Delete, lineage links (built on / follow-ups), notes.

## Tests
- New: tests/test_v10.py (64): resume, lifecycle, assistant workspace tools and loop, the Analyses page, the dock. Existing suites unchanged.

## Known limits
- A follow-up searches for a new model; it does not warm-start from the parent's models.
- Media analysis (text/images) is recomputed in a follow-up; only profile, statistics, KPIs and trends are reused.
- Runs are listed for everyone using the workspace (as before); delete is not restricted per user.
- The deletion of MLflow-mirrored runs is not attempted; only the local store is cleaned.
- The UI was verified with the test harness (FakeSt), not in a browser: layout proportions are asserted, not seen.

# Changes in this revision

## Bugs fixed
- **False near-duplicate alarms (featurizers.py, mediamining.py).** The image hash was greyscale-only, so differently coloured images with the same layout collided
  (36% of a 3-class shapes dataset flagged, with a false "scores are inflated" warning). The hash now includes a colour signature and a hash match is confirmed on a 16x16 thumbnail.
- **Silent medium-severity leakage (agents/critic.py).** The Sentinel found a post-outcome column (score 0.87, "medium") but nothing reached the user while the headline AUC was inflated (0.95 vs ~0.65 without it).
  The Critic now adds a `leakage_suspect_possible` warning. Columns are not auto-dropped.

## Tool-calling efficiency (assistant/agent.py, assistant/tools.py)
- Independent tools can be requested together: `{"actions": [...]}`, max `MAX_BATCH` (4) per step. The single `{"action": ...}` form still works.
- A repeated successful call is not re-run; a repeated failing call is not retried, and after two the model must answer.
- The last step is reply-only (`must_reply_now`), so no tool call is run and then discarded.
- Near-miss column names (punctuation, spacing, typos in long names) are repaired and reported; short, ambiguous or hostile names are still errors.
- Compact tool catalog (-44%), timing noise removed, older observations shortened.
- `respond()` returns a new `stats` dict (tool_runs, tool_batches, skipped_repeats, skipped_failures, name_corrections, payload_chars).
- Offline benchmark on 6 scripted scenarios (scripted model, NOT a live LLM): LLM calls 27 -> 16, tool runs 23 -> 17, prompt chars -200k -> 92k, answered 4/6 -> 6/6. See `benchmarks/bench_agent.py`.

## Tests
- New: tests/test_agent_efficiency.py (17), tests/test_leak_warning.py (2).
- Changed: tests/test_v7.py `test_bad_tool_call_is_fed_back_and_loop_is_bounded` now asserts the call count is below MAX_STEPS (old behaviour used all 6).

# Revision 2: multi-table input (multitable.py), found by stress-testing with variants of a 3-CSV star schema

## Bugs fixed
- **Tables that each have a primary key called `id` were joined to each other** (`customers.id = products.id`, both small ints, "100% match"), gluing product columns onto the wrong customers and skipping the real join. Generic `id`~`id` is no longer evidence of a relationship.
- **Keys that differ only in formatting never matched** (`C0001` vs ` c0001 `, `3` vs `3.0`, `0007` vs `7`). The customers table was dropped silently and the analysis ran on the 20-row products table. Keys are now normalised on both sides.
- **Wrong main table**: with no target known the table could be anything, and the target column in a child table was lost. `propose/prepare(target=...)` now makes the table holding the target the main table (CLI passes `--target`). When several tables tie, the alternatives are listed.
- **Records with no child rows got NaN totals**, which the model later imputed with the median (hiding "never ordered"). Counts, sums, distinct counts and shares are now 0.
- **Orphan child rows** (e.g. orders for unknown customers) were dropped silently; they are now reported with their share.
- **Differently named keys** (`cust_id` ~ `customer_id`) are linked only if the values also agree (>= 90%), and flagged for confirmation.
- **Dangerous plans warn loudly**: an unconnected table larger than the main table, or nothing linkable at all.
- **File reading**: BOM, `;`/tab/`|` delimiters, and cp1256 (Arabic) vs latin-1 are told apart by plausibility. cp1256 accepts any byte and used to turn "Málaga" into "Mلlaga".
- **media.py text documents**: `utf-16` without a BOM "decoded" Latin-1 files into CJK gibberish; UTF-16 is now used only when a BOM says so.

## Refinements
- Categorical child columns (<= 6 levels) become per-level **shares** (the mix) instead of an uninformative distinct count.
- Date child columns give **recency, tenure and span in days**.
- Verified at 1M order rows: propose + apply in about 6 s.

## Evidence
- tests/test_multitable_robust.py (14 tests). benchmarks/mt_variants.py reproduces the 6 stress variants.
- On the supplied (clean) data the result is unchanged within noise (mean held-out AUC over 3 seeds 0.793 original vs 0.784 modified; both pick order-based drivers).

# Revision 4: NLP robustness on top of v3 (nlp.py, evaluation/metrics.py, agents/critic.py)

## Bugs fixed
- **Tokenizer cut words at combining marks** (Python `\w` ignores them): Hindi भारत -> `रत`, जहाँ -> `जह`; vowelled Arabic fragmented; Thai split mid-word. The token pattern now keeps Arabic/Hebrew points, Brahmic vowel signs, Thai/Lao/Tibetan/Myanmar/Khmer marks and ZWNJ/ZWJ inside words. Used by TextVec and corpus_stats.
- **Arabic optional vowel marks and tatweel** are removed in clean_text, so 'الْقَاهِرَةُ' and 'القاهرة' are one word.
- **Languages without spaces** (Chinese, Japanese, Thai, Lao, Khmer, Myanmar, Tibetan) were one giant "word" per sentence for every word-based model; the vectoriser now switches to character n-grams when more than 30% of letters are in such scripts.
- **HTML markup** (tags, entities) was left in the text and became features (`br`, `nbsp`, `amp`). A whitelist of real HTML tags is stripped, entities decoded; text like `<class 'int'>` is kept.
- **Personal-data scanner false alarms**: the phone pattern matched any 10+ character run of digits/spaces/dots/dashes (126 false phones and 2 false cards in 2,000 docstrings with no personal data; timestamps, decimals, arrays). Replaced by structured patterns (+CC, US, national, Egyptian mobile; grouped or prefix-checked cards; valid IPv4 octets; decimals excluded). Real formats are still detected.
- **Evaluation scorer** crashed on tables whose only feature is text, and for multiclass tasks compared the system's macro-F1 with a one-class AUC baseline (the "gap" was meaningless). The baseline now reads text (TF-IDF) and uses the same metric as the system.
- **Language labels** for Japanese kana, Indic and South-East-Asian scripts.
- **Audit message** said "Minority class is only 0%" for a 0.25% class (now one decimal).
- **Label-issue headline** claimed N documents "look mislabelled" (194 flagged for 105 real errors). It now says "up to N may be mislabelled or ambiguous" and that the ranking, not the total, is reliable.

## Evidence (3 splits each, same held-out rows as my own models; macro-F1)
- Messy text (HTML, junk, 5% duplicates, 5% wrong labels): v3 0.689 -> 0.703 (ahead on 3 of 3 splits); my own models 0.700.
- Clean `licenses`: identical to v3 (0.867). Clean `libdocs`: 0.774 vs 0.779 (one split -0.017, others ~0; within noise).
- An extra char-level LinearSVC candidate was tried and removed: it cost ~0.007 on clean data and ~30% runtime and v3's word+char / NB-SVM already cover it.
- Label-issue ranking on 105 planted errors: precision 1.00 @10, 0.92 @12, 0.72 @25, 0.29 overall; recall 0.52-0.59.
- New tests: tests/test_nlp_robust.py (13). benchmarks/nlp_compare.py, nlp_build_datasets.py, nlp_label_issues.py reproduce the comparison on real local text (numpy/pandas/scipy/sklearn/matplotlib docstrings, OS license texts).

## Not covered
No real labelled sentiment or multilingual benchmark (the sandbox had no network); multilingual handling is checked with tokenisation probes only.
