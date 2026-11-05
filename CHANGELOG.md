# Changelog

## v5
- Changed (modeling): the AutoML search no longer runs one fixed zoo on every dataset. `advisor.py` measures the data first (season length and strength, trend, log scale, random-walk-ness for time series; size, feature mix, missing values, imbalance and a linear-vs-boosted probe for tables) and builds a shortlist with a stated reason for every inclusion and exclusion. A verification step then fits each excluded family once and promotes any that beats the incumbent. The AI agent can still request any family. The plan and the audit appear in the decision log.
- Added (time series): seasonal lags and position / phase features chosen from the data, measured on the training part only; new families `TrendSeasonal` (trend plus season regression), seasonal-change wrappers that let Ridge and boosted trees forecast trending seasonal series, and `Seasonal naive`; the baseline every model must beat now includes "same period last season". The ensemble is only used on a time series when it clearly beats the best single model (its weights were previously tuned and scored on the same windows).
- Added (neural nets): `MLP` family (scaled target, early stopping, searchable width / depth / regularisation), shortlisted only when the data is large enough, mostly numeric and non-linear.
- Added (links): paste a Kaggle dataset / competition, GitHub or direct file link on Home / Data, in the assistant chat, or pass it to the CLI; it is downloaded, size-capped, unzipped safely and added. Kaggle keys come only from environment variables. Internal addresses are refused.
- Benchmarks: `benchmarks/real_data_bench.py`, `make_local_datasets.py`, `compare_results.py` and stored results for v4, v5 and a hand-built reference on 7 real datasets.

## v0 (NexusML)
- Fixed (CI): `leaderboard.py` used double quotes inside a double-quoted f-string, a SyntaxError on Python 3.11 (our CI matrix) that made every test module fail to import. Rewritten; new `scripts/check_py311.py` and a fast `lint` CI job catch this class of error in seconds; artifact upload now runs even when the eval check fails.
- Added: PDF ingestion (`pdfparse.py`): tables (stitched across pages, numbers coerced), paragraphs/headings with running headers/footers removed, pdfplumber / PyMuPDF / pypdf fallback chain. PDFs with tables become datasets on upload.
- Added: choose the LLM provider in Settings (Groq, Google AI Studio / Gemini, OpenAI, OpenRouter, Together, or your own via `NEXUSML_EXTRA_PROVIDERS`); `--provider/--model` on the CLI. Keys come only from environment variables and are never stored or shown.
- Added: SMOTE, ADASYN and random oversampling (`imbalance.py`, resampling only inside training folds) as strategist levers, alongside class weights; `NEXUSML_IMBALANCE` forces one.
- Added: automatic decision-threshold tuning for binary classifiers (`threshold.py`): chosen on out-of-fold training predictions (F1 / F-beta / PR-distance / Youden / cost), adopted only when it clearly helps, carried into the saved model and exported script.
- Added: agentic data segmentation (`segmentation.py`, assistant tool `segment_analysis`): slice by class, conditions or date range, then EDA, comparison with the rest, modelling and clustering on that slice only. A constant target inside the slice is explained, not crashed on.
- Added: NLP candidates NB-SVM and word+char TF-IDF; `benchmarks/nlp_bench.py` reproduces the comparison on AG News / IMDb / 20 Newsgroups.
- Renamed from "ds_agent" / "Meridian": package `nexusml`, environment variables `NEXUSML_*` (was `DS_AGENT_*`).
- Fixed: the global font rule overrode the icon font, which can turn sidebar icons into plain words. Icons are now excluded.
- Removed unused imports, the unreferenced `overview.py` and `chatstore.py`, and the per-version notes (history is below).
- Matplotlib boxplot uses `orientation=` (needs matplotlib 3.10+).
- Plainer wording in the interface and reports ("CV score" instead of "honest score").
- One README, one architecture note.
- Fixed (images): an uploaded image zip stored only its CSV, with relative image paths pointing at nothing, so every image became missing data and models scored at chance (F1 0.17 on a 3-class set) without any error. The stored table now holds absolute paths, datasets saved the old way are located again automatically, and a run stops with a clear message if most image files cannot be found.
- Images: common loader for all backends (EXIF rotation, transparency on white, 16-bit and palette files), a 4x4 colour layout feature, and a new rotation-invariant outline-shape backend (segmentation, compactness, solidity, Hu moments, outline harmonics) offered as separate candidates. On a shapes-only test set held-out F1 went from about 0.5 to about 0.97.
- Added: image preprocessing recipes (resize with squash / fit / fill, blur, sharpen, median denoise, histogram equalise, contrast stretch, grayscale, Sobel edge detection). Chosen on New analysis > Image options with a before/after preview, stored with the model, and applied identically to new photos. CLI/API: `image_prep=["blur:1.5", "edges"]`.
- Added: deep-network features. ResNet-18 and ResNet-50 (ImageNet weights) feed LogisticRegression / SVC / Ridge candidates next to the hand-crafted ones; batched, cached, GPU used when present. Choose Automatic / Off / ResNet-18 / ResNet-50 / Both per run. Offline: set `NEXUSML_DEEP_WEIGHTS` to a saved `.pth` file. Needs torch + torchvision (`Dockerfile.multimodal`); without them the hand-crafted models run as before.
- Added: Google Gemini as the LLM provider (OpenAI-compatible endpoint). `GEMINI_API_KEY` selects it and the low-cost `gemini-3.1-flash-lite` model; Groq still works with `GROQ_API_KEY`.
- Added: Data page > Combine tables. Pick two or more datasets, review the proposed joins, approve, and get one combined dataset. Before this, multi-table joins only worked from the CLI.

## Earlier work (as ds_agent)
- v7: multi-page app, assistant with 15 validated tools, workspace (datasets, saved chats), background jobs, per-session user identity.
- v6: parallel agent crew, goal-driven AutoML.
- v5: leaderboard, 13-stage graph, NLP and vision pipelines.
- v4: documents/images as input, MLflow integration, chat history.
