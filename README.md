# NexusML

Upload a table (or documents, images, a zip) and ask a question. NexusML profiles the data, picks suitable statistical tests,
trains and cross-validates models, and writes a report with charts. It also has an assistant for quick questions
("any outliers?", "what drives churn?") that runs one tool instead of the whole pipeline.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # set GEMINI_API_KEY (Google AI Studio) or GROQ_API_KEY
streamlit run app.py          # dark theme: streamlit run app.py --config .streamlit/config.dark.toml
```

CLI:

```bash
python -m nexusml.cli data.csv "Why do customers churn?" --auto-approve
```

Output goes to `.agent_artifacts/` (change with `NEXUSML_ARTIFACTS`).
Dark mode also needs `NEXUSML_THEME=dark` so charts match the page.

## How it works

1. Profile the data: types, quality, missing values, duplicates, outliers.
2. Frame the problem (LLM): business question, target, date column, segments, KPIs.
3. Statistical tests chosen by data type and question.
4. Models are compared by cross-validation on the training split; the test split is scored once, never used for selection. Tuned models need nested CV to beat the default.
5. Findings, root causes and recommendations must cite evidence ids; uncited claims are rejected.
6. Report (Markdown and HTML), model bundle, optional Power BI package, MLflow tracking when installed.

All numbers come from code. The LLM sees column names, summary statistics and category counts, not rows
(`NEXUSML_SEND_SAMPLES=1` allows a few example values). If the LLM is unreachable the run stops with a message;
there is no rule-based fallback. The default model is `gemini-3.1-flash-lite` with a Gemini key, or `openai/gpt-oss-120b` with Groq (`NEXUSML_LLM_MODEL` overrides both).

## Images

Upload a zip of images (folder names become the class labels). On **New analysis > Image options** you can add preprocessing steps
(resize, blur, sharpen, denoise, equalise, contrast, grayscale, edge detection), see a before/after preview, and choose whether ResNet features are used.
Models compared: hand-crafted (colour, HOG, LBP, outline shape) and, when torch is installed, ResNet embeddings with a linear/SVM head.
This is transfer learning, not fine-tuning of the network.

## Several tables

Add each CSV on the Data page, open **Combine tables**, select the tables, and review the proposed joins. Child tables are summarised to one row per record of the main table; the result must keep the main table's row count. After you approve, the combined table appears as a new dataset you can analyse. The CLI does the same with `--approve-joins`.

## App pages

Home, Assistant, Data (library of uploaded datasets), New analysis (runs in the background, can be cancelled),
Analyses, Models, Settings. Set `NEXUSML_APP_PASSWORD` to require sign-in; each name gets its own workspace.

## Deploy

```bash
docker build -t nexusml . && docker run -p 8501:8501 -e GEMINI_API_KEY=... -v nexusml-data:/data nexusml
```

`Dockerfile.multimodal` adds torch for the ResNet backends (ResNet-18/50; weights download on first use, or set `NEXUSML_DEEP_WEIGHTS=/path/to/weights.pth` offline). Optional extras are in `requirements-extras.txt`.
Model files are pickles; only load ones you trust.

## Tests

```bash
python -m unittest discover -s tests
python -m nexusml.evaluation.harness --check     # deterministic benchmark with a stub LLM
```

The full suite takes several minutes. The UI tests run against a stand-in for Streamlit (`tests/fake_st.py`),
so layout and styling still need a look in a real browser.

More: `docs/ARCHITECTURE.md`, `docs/EVALUATION.md`.
