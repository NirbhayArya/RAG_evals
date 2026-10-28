# RAG Evaluation Dashboard

This project evaluates a Retrieval-Augmented Generation (RAG) question-answering
pipeline against a small, reproducible question set about penguins. It indexes
the included `Penguins_ACL.pdf`, retrieves relevant passages, generates
structured answers with an LLM, and scores the results with DeepEval metrics.

The project also includes a Streamlit dashboard for comparing saved evaluation
runs.

## What the project demonstrates

- Extracting text blocks from a PDF with PyMuPDF.
- Creating a persistent Chroma vector store with Mistral embeddings.
- Retrieving evidence for a question with LangChain.
- Generating an answer and the source context used by the answer.
- Evaluating RAG outputs for:
  - Answer relevancy
  - Faithfulness
  - Context relevancy
  - Correctness
- Comparing retrieval experiments in a Streamlit dashboard.

## Project structure

```text
.
├── app.py                    # Streamlit evaluation dashboard
├── eval_dataset.jsonl        # Questions and expected answers
├── experiment_v1.jsonl       # Summary for the k=3 experiment
├── experiment_v2.jsonl       # Summary for the k=1 experiment
├── experiment_v3.jsonl       # Summary for the k=2 experiment
├── GroqJudge.py              # DeepEval judge backed by Groq
├── pdfs/
│   └── Penguins_ACL.pdf      # Knowledge-base document
├── RAG1/
│   ├── agent.py              # RAG agent and ask() entry point
│   ├── file_reader.py        # PDF paragraph extraction
│   ├── indexing.py           # Embedding and Chroma indexing
│   ├── objects.py            # Structured answer schema
│   ├── prompts.py            # Agent instructions
│   └── tools.py              # Retrieval tool
└── versions/
    ├── v1/                   # k=3 evaluation scripts and outputs
    ├── v2/                   # k=1 evaluation scripts and outputs
    └── v3/                   # k=2 evaluation scripts and outputs
```

The `db/` directory contains the local Chroma database created by the indexing
pipeline. The `.env` file is intentionally ignored by Git and should contain
local credentials only.

## How the RAG pipeline works

1. `RAG1/file_reader.py` extracts ordered text blocks from the PDF.
2. `RAG1/indexing.py` embeds those blocks with the `mistral-embed` model and
   stores them in Chroma.
3. `RAG1/tools.py` exposes similarity search as an agent tool.
4. `RAG1/agent.py` uses a Groq-hosted model to answer questions based only on
   retrieved context and returns a structured `Answer`.
5. The evaluation scripts run the same question set, save per-question results,
   and calculate aggregate metrics with DeepEval.

## Requirements

- Python 3.14 or newer
- A Mistral API key for embeddings
- A Groq API key for answer generation and evaluation
- [`uv`](https://docs.astral.sh/uv/) recommended for dependency management

Set the required credentials in a local `.env` file:

```dotenv
MISTRALAI_API_KEY=your-mistral-api-key
GROQ_API_KEY=your-groq-api-key
```

Do not commit this file or expose the keys in logs, notebooks, or screenshots.

## Installation

From the project root:

```powershell
uv sync
```

The dependencies and Python requirement are defined in `pyproject.toml`.

## Running the dashboard

Start the Streamlit dashboard with:

```powershell
uv run streamlit run app.py
```

The dashboard loads `experiment_v1.jsonl`, `experiment_v2.jsonl`, and
`experiment_v3.jsonl`. It displays each run's aggregate metrics, charts metric
comparisons, and lets you inspect an individual run.

## Running an evaluation

The evaluation is split into three stages:

1. Generate answers for every question in `eval_dataset.jsonl`.
2. Score the generated answers with DeepEval.
3. Save an aggregate experiment summary.

Each version directory contains the corresponding runner, metric evaluator,
and summary script. Run a version from its directory so its relative paths
resolve correctly. For example:

```powershell
Set-Location versions\v2
uv run python eval_runner2.py
uv run python eval_all_metrics2.py
uv run python evaluation_summary2.py
```

The generated per-question and summary files are JSONL files. The summary files
used by the dashboard are kept at the project root.

## Evaluation results

Each experiment contains 20 test cases. The saved aggregate results are:

| Experiment | Retriever `k` | Answer relevancy | Faithfulness | Context relevancy | Correctness |
|---|---:|---:|---:|---:|---:|
| `experiment_v1_k3` | 3 | 0.988 | 1.000 | 0.733 | 0.900 |
| `experiment_v2_k1` | 1 | 1.000 | 1.000 | 0.781 | 0.915 |
| `experiment_v3_k2` | 2 | 1.000 | 1.000 | 0.787 | 0.890 |

These are observed results for this dataset and configuration, not guarantees
of general RAG performance. Scores can change with the model, prompt,
embedding model, document, judge, or test set.

## Retrieval decision

**Choose `k = 1` for now**, because it offers the best observed correctness and
answer relevancy with less retrieval overhead. Treat this as a provisional
decision, not proof that `k = 1` is universally optimal. If `k = 2` materially
improves retrieval recall on questions requiring multiple pieces of evidence,
it may be the better choice for those questions.

This decision should be revisited when the evaluation set includes more
multi-hop questions and when retrieval recall is measured directly. Historical
experiments for `k = 1`, `k = 2`, and `k = 3` are retained so that future
changes can be compared against the same baseline.

## Limitations and next steps

- The evaluation set is small and focused on one document.
- Aggregate metrics do not show which individual questions fail.
- Retrieval recall is not currently measured as a separate metric.
- The judge-based correctness score depends on the selected evaluation model.
- Future experiments should include multi-evidence questions and compare
  per-question retrieval recall before changing the default `k`.
