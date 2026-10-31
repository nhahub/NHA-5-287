# Evaluation protocol

The harness (`python -m nexusml.evaluation.harness`) runs the whole pipeline on datasets whose correct answers are KNOWN.

## What it measures (offline, stub LLM, deterministic)
| Measure | How |
|---|---|
| Framing plumbing | target and problem type reach the plan unchanged |
| Cleaning | leaky / constant / mostly-missing columns are dropped |
| Driver recovery | planted drivers appear in the top-k permutation importances |
| Interaction search | a planted two-factor effect (overtime x tenure) is surfaced |
| Model quality | held-out score vs a plain tabular scikit-learn baseline (best of linear / boosted) on the same split |
| Clustering | adjusted Rand index against planted groups |
| Multi-table | joins keep the base row count; many-to-many / low-match links are rejected |
| Text / image / multimodal | score vs a tabular-only baseline |
| Honesty | the bootstrap CI contains the point estimate |
| Reproducibility | same seed twice, and parallel vs sequential, give identical evidence |
| Cost / speed | seconds and LLM calls per case |

`--check` compares the summary with `nexusml/evaluation/thresholds.json` and exits 1 on a regression (CI does this on every push).

## What the stub mode cannot measure
The stub LLM writes fixed text, so **claim faithfulness and framing quality are only meaningful with `--live`** (real LLM, run on a schedule).
Also: planted data is easier than real data; a perfect score here is not a promise about your data.

## Comparing with ChatGPT / Gemini / Claude (manual protocol)
1. Generate the datasets: `python -c "from nexusml.evaluation import datasets as D; import pathlib; p=pathlib.Path('bench'); p.mkdir(exist_ok=True); [g(p) for g in D.GENERATORS.values()]"`.
2. Give each tool the same CSV and the same prompt ("Why does X happen and what should we do? Predict X and report held-out performance.") with a fixed random seed in the instructions. Repeat 3-5 times per case.
3. Score blind with the same functions in `evaluation/metrics.py` (driver recall from the tool's ranked list, held-out metric it reports vs a re-computation, claims that cite no number you can verify).
4. Put the rows in a JSON list `[{"tool":..., "case":..., "metric":..., "value":...}]` and pass it with `--external file.json` to include them in the table.
