# Memora Evaluation

Two evaluation tracks ship in this release. They produce **identical result
schemas** so a single aggregator (`evals/model_eval/aggregate_results.py`) can
ingest both and reproduce Table 3 of the paper.

```
data/<period>/<persona>/                       <- released data layout
├── conversations/session_NNNN.json
├── evaluation_questions_<persona>.json
└── eval_results/                              <- written by both tracks
    ├── <model>_no_reasoning/                  <- Track 1 (LLM)
    ├── <model>_reasoning/                     <- Track 1 (LLM)
    └── <agent>/                               <- Track 2 (agent)
```

`<period>` is one of `weekly` (15 questions / persona), `monthly`
(15 / persona), `quarterly` (30 / persona — 10 per task). 10 personas total.

## The headline metric: FAMA

**FAMA** (Forgetting-Aware Memory Accuracy, paper §4.2) is the single number
reported in Table 3. It rewards a model both for *recalling* information that
should be remembered and for *forgetting* information that has since been
deleted or updated. Per-question:

```
FAMA = max(0, MPA − λ · (1 − FAA))
  where  MPA = correct memory_presence sub-q / total memory_presence sub-q
         FAA = correct forgetting_absence sub-q / total forgetting_absence sub-q
         λ   = N_forget / (N_presence + N_forget)
```

Per-question FAMA is bounded `[0, 1]`. Each Table 3 cell `(model, task, period)`
is the mean of per-question FAMAs in that bucket × 100 — bounded `[0, 100]` and
directly comparable across periods.

## Track 1 — LLM evaluation (`model_eval/`)

Stuffs the entire conversation history into the LLM's context, asks each
question, and uses an LLM-as-judge (multi-judge by default) to score the answer
against per-question yes/no sub-questions. Each model is evaluated **twice**:
once with reasoning OFF and once with reasoning ON.

Reported models (Table 3):

- `qwen/qwen3-32b`
- `anthropic/claude-sonnet-4.5`
- `google/gemini-3-pro-preview`
- `openai/gpt-5.2`

All routed through OpenRouter — same provider used in `conversation_generation/`.

```bash
# Single run (sanity check)
uv run python evals/model_eval/model_based_evaluator.py \
    --sessions-dir data/weekly/academic_researcher \
    --model anthropic/claude-sonnet-4.5 \
    --output-dir   data/weekly/academic_researcher/eval_results/anthropic_claude-sonnet-4.5_no_reasoning \
    --limit 5

# Full sweep (4 models × 2 modes × periods × personas)
./evals/model_eval/run_model.sh
```

Edit the arrays at the top of `run_model.sh` to widen `PERIODS` / `PERSONAS`
or to override `QUESTION_LIMIT` (also accepted via env: `QUESTION_LIMIT=15 ./run_model.sh`).

The judges always run via OpenRouter (`openai/gpt-4.1`,
`anthropic/claude-haiku-4.5`, `google/gemini-2.5-flash`); a majority vote is
taken per sub-question. Pass `--no-multi-judge` to fall back to a single judge.

## Track 2 — Memory-agent evaluation (`agent_eval/`)

A two-step pipeline:

1. **`conversation_to_memory.py`** — ingest every session for one persona under
   `user_id="<persona>_<period>"` in the chosen memory system.
2. **`memory_to_answer.py`** — for each evaluation question, retrieve memories
   under that same `user_id`, generate an answer with `gpt-4o-mini`, and run
   the same multi-judge as Track 1.

Reported agents (Table 3):

- `a_mem` — local ChromaDB
- `langmem` — local LangGraph store
- `mem_0` — Mem0 cloud
- `memobase` — Memobase cloud
- `memos` — MemOS (cloud or self-hosted)
- `nemori` — local

```bash
# All six agents, sequentially
./evals/agent_eval/run_all_agents.sh

# All six in parallel
./evals/agent_eval/run_all_agents.sh --parallel

# Single agent
./evals/agent_eval/run_a_mem.sh

# Resume after a partial run
./evals/agent_eval/run_mem_0.sh --skip-store    # answer only
./evals/agent_eval/run_mem_0.sh --skip-answer   # store only
```

The same `user_id` (`<persona>_<period>`) is used in both steps — never change
it between the two scripts or retrieval will return nothing. Mem0 in
particular needs ~60s between store and answer for backend processing; the
provided `run_mem_0.sh` waits automatically.

## Result schema (uniform across both tracks)

Both tracks write two files per run, in the per-run output directory:

- `eval_results_<TIMESTAMP>.json` — detailed per-question results (model
  response, retrieved memories for Track 2, every sub-question's per-judge
  result, per-question FAMA)
- `eval_report_<TIMESTAMP>.json` — aggregated report

Top-level keys of `eval_report.json`:

```
metadata
    evaluation_timestamp
    model_name        (Track 1) | memory_system  (Track 2)
    sessions_directory                          (Track 1)
    user_id, model                              (Track 2)
    use_multi_judge, judge_models

overall_metrics
    total_questions, total_evaluation_questions, total_correct_evaluations
    overall_accuracy
    fama                              <- headline number
    memory_presence_total / _correct / _accuracy
    forgetting_absence_total / _correct / _accuracy

per_judge_overall
    openai, anthropic, google → {correct, total, accuracy}

per_judge_by_task_type
    {task → judge → {correct, total, accuracy}}

by_task_type
    Remembering, Reasoning, Recommending → {
        total_questions, total_eval_questions, correct_count,
        memory_presence_total / _correct / _accuracy,
        forgetting_absence_total / _correct / _accuracy,
        accuracy, avg_question_accuracy,
        fama
    }

detailed_results [...]                          # per-question rows
context_metadata                                # Track 1 only
```

The three task buckets (Remembering / Reasoning / Recommending) are always
present in `by_task_type`, even if a `--limit` smoke run only populated one of
them.

## Aggregating runs into a Table-3-style summary

```bash
# Walk the entire data tree and print the 9-cell grid
uv run python evals/model_eval/aggregate_results.py data/ --print

# Or aggregate a custom set of report files
uv run python evals/model_eval/aggregate_results.py \
    data/weekly/academic_researcher/eval_results/*/eval_report_*.json --print

# Save the aggregated rows + nested table as JSON
uv run python evals/model_eval/aggregate_results.py data/ --output table3.json
```

`aggregate_results.py` distinguishes Track 1 reasoning ON vs OFF using the
run-id suffix (`_reasoning` vs `_no_reasoning`) and reports them as separate
rows, matching Table 3.

## Setup

We use [`uv`](https://github.com/astral-sh/uv) — see the top-level
[`README.md`](../README.md#quick-start-uv) for the full bootstrap. In short:

```bash
uv sync                                  # one-shot from pyproject.toml
# or:
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt       # union of both tracks
```

Per-track installs are also available:

```bash
uv pip install -r evals/model_eval/requirements.txt    # Track 1 only
uv pip install -r evals/agent_eval/requirements.txt    # Track 2 only
```

API keys go in `.env` (see top-level `.env.example`). The judges always run
via OpenRouter, so `OPENROUTER_API_KEY` is required for both tracks. The
evaluator accepts both `OPENROUTER_API_KEY` and the underscore variant
`OPEN_ROUTER_API_KEY` (the latter is the spelling used by the data-generation
code in `conversation_generation/`).
