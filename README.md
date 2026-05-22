<p align="center"><a href="https://genies.com"><img src="assets/genies-logo.svg" alt="Genies" height="32"></a></p>

# From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents

**Memora** · Accepted to ACL 2026 · [arXiv:2604.20006](https://arxiv.org/abs/2604.20006)

[![arXiv](https://img.shields.io/badge/arXiv-2604.20006-b31b1b.svg)](https://arxiv.org/abs/2604.20006)
[![Conference](https://img.shields.io/badge/ACL-2026-4b44ce.svg)](https://2026.aclweb.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![ASU](https://img.shields.io/badge/ASU-8C1D40.svg)](https://www.asu.edu/)
[![Genies](https://img.shields.io/badge/Genies-312A37.svg)](https://genies.com/)

This repository contains the released dataset and the evaluation code used to
produce **Table 3** of the paper. The benchmark measures how well an LLM (or a
memory-augmented agent) can answer memory-grounded questions over a long,
realistic conversation history — with explicit credit for both *recalling*
information that should be remembered and *forgetting* information that has
been deleted or updated. Both behaviours are combined into a single score:
**FAMA** (Forgetting-Aware Memory Accuracy, paper §4.2).

<p align="center"><img src="assets/teaser.png" alt="The three tasks of the Memora benchmark: Remembering, Reasoning, Recommending" width="100%"></p>

## What's here

```
data/                               Released conversations + evaluation questions
├── weekly/<persona>/               7 days, ~150 sessions / persona
├── monthly/<persona>/              30 days, ~600 sessions / persona
└── quarterly/<persona>/            90 days, ~2000 sessions / persona

evals/
├── model_eval/                     Track 1 — direct LLM evaluation
└── agent_eval/                     Track 2 — long-term memory agents
```

10 personas: `academic_researcher`, `business_executive`, `content_writer`,
`creative_designer`, `financial_analyst`, `management_consultant`,
`marketing_manager`, `sales_manager`, `software_engineer`, `startup_founder`.

Each persona-period carries **15 evaluation questions** for `weekly` /
`monthly` (5 each in *Remembering*, *Reasoning*, *Recommending*) and **30**
for `quarterly` (10 each), reflecting the larger evidence horizon. Every
question carries an `evaluation` block of yes/no sub-questions tagged
`memory_presence` (information that should be recalled) or `forgetting_absence`
(information that has since been forgotten and should *not* surface). The
judge scores each sub-question independently; FAMA combines the two into one
number per question.

See [`data/README.md`](data/README.md) for the full dataset spec.

## Quick start (uv)

We use [`uv`](https://github.com/astral-sh/uv) — install it first if you don't
have it (`brew install uv`, or
`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
# 1. Bootstrap the venv from pyproject.toml — one command, fully reproducible
uv sync

# (or, if you prefer the requirements.txt path:)
# uv venv --python 3.11 .venv
# source .venv/bin/activate
# uv pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env: OPENROUTER_API_KEY (required), OPENAI_API_KEY (Track 2),
# plus the per-agent keys for any agents you plan to run.

# 3. Track 1 — single LLM, single persona/period (sanity check)
uv run python evals/model_eval/model_based_evaluator.py \
    --sessions-dir data/weekly/academic_researcher \
    --model anthropic/claude-sonnet-4.5 \
    --limit 5

# 4. Track 1 — full sweep (edit MODELS/PERIODS/PERSONAS/QUESTION_LIMIT first)
./evals/model_eval/run_model.sh

# 5. Track 2 — single memory agent
./evals/agent_eval/run_a_mem.sh

# 6. Track 2 — all six agents
./evals/agent_eval/run_all_agents.sh

# 7. Aggregate results into a Table-3-style summary
uv run python evals/model_eval/aggregate_results.py data/ --print
```

`uv run` automatically activates the project venv; if you'd rather work in an
activated shell, run `source .venv/bin/activate` once and drop the `uv run`
prefix from each command.

See [`evals/README.md`](evals/README.md) for the full evaluation guide,
including the result-file schema (uniform across both tracks) and FAMA
definition.

## Models and agents (Table 3)

**LLMs** (each evaluated with reasoning ON and OFF, all via OpenRouter):

| Model                        | OpenRouter id                  |
| ---------------------------- | ------------------------------ |
| Qwen3-32B                    | `qwen/qwen3-32b`               |
| Claude Sonnet 4.5            | `anthropic/claude-sonnet-4.5`  |
| Gemini 3 Pro Preview         | `google/gemini-3-pro-preview`  |
| GPT-5.2                      | `openai/gpt-5.2`               |

**Memory agents**: A-Mem, LangMem, Mem-0, Memobase, MemOS, Nemori. All use
`gpt-4o-mini` for answer generation (per paper).

## Citation

If you use Memora in your research, please cite:

```bibtex
@inproceedings{uddin2026memora,
  title     = {From Recall to Forgetting: Benchmarking Long-Term Memory
               for Personalized Agents},
  author    = {Uddin, Md Nayem and Shubham, Kumar and Blanco, Eduardo
               and Baral, Chitta and Wang, Gengyu},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for
               Computational Linguistics (ACL 2026)},
  publisher = {Association for Computational Linguistics},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.20006}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
