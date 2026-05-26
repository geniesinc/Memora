<p align="center"><a href="https://genies.com"><img src="assets/genies-logo.svg" alt="Genies" height="32"></a></p>

# From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents

<p align="center"><b>Memora</b> · Accepted to ACL 2026 · <a href="https://arxiv.org/abs/2604.20006">arXiv:2604.20006</a></p>

<p align="center">
<a href="https://arxiv.org/abs/2604.20006"><img src="https://img.shields.io/badge/arXiv-2604.20006-b31b1b.svg" alt="arXiv"></a>
<a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/ACL-2026-4b44ce.svg" alt="ACL 2026"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
<a href="https://www.asu.edu/"><img src="https://img.shields.io/badge/ASU-8C1D40.svg" alt="ASU"></a>
<a href="https://genies.com/"><img src="https://img.shields.io/badge/Genies-312A37.svg" alt="Genies"></a>
</p>

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

### Results

Task-level FAMA scores (higher is better, scaled to [0, 100]), reproduced
from Table 3 of the paper. **W** = weekly, **M** = monthly, **Q** = quarterly.

<table>
<thead>
<tr>
  <th rowspan="2">Model / Agent</th>
  <th colspan="3">Remembering</th>
  <th colspan="3">Recommending</th>
  <th colspan="3">Reasoning</th>
</tr>
<tr>
  <th>W</th><th>M</th><th>Q</th>
  <th>W</th><th>M</th><th>Q</th>
  <th>W</th><th>M</th><th>Q</th>
</tr>
</thead>
<tbody>
<tr><td colspan="10"><i>Language models (without reasoning tokens)</i></td></tr>
<tr><td>Qwen3-32B</td>            <td>26.12</td><td>21.14</td><td>19.24</td><td>50.16</td><td>50.30</td><td>48.88</td><td>6.00</td><td>2.00</td><td>6.00</td></tr>
<tr><td>Claude Sonnet 4.5</td>    <td>27.50</td><td>19.42</td><td>21.25</td><td>43.62</td><td>39.00</td><td>44.02</td><td>6.66</td><td>3.00</td><td>5.50</td></tr>
<tr><td>Gemini 3 Pro Preview</td> <td>20.36</td><td>21.44</td><td>17.28</td><td>45.12</td><td>45.94</td><td>52.56</td><td>6.66</td><td>4.00</td><td>4.00</td></tr>
<tr><td>GPT-5.2</td>              <td>25.32</td><td>19.92</td><td>23.39</td><td>54.80</td><td>51.12</td><td>53.36</td><td>4.66</td><td>0.00</td><td>1.00</td></tr>
<tr><td colspan="10"><i>Language models (with reasoning tokens)</i></td></tr>
<tr><td>Qwen3-32B</td>            <td>23.86</td><td>25.62</td><td>17.14</td><td>50.04</td><td>53.06</td><td>47.71</td><td>6.66</td><td>9.00</td><td>3.00</td></tr>
<tr><td>Claude Sonnet 4.5</td>    <td>26.56</td><td>21.40</td><td>19.13</td><td>52.40</td><td>60.90</td><td>51.78</td><td>4.00</td><td>0.00</td><td>2.50</td></tr>
<tr><td>Gemini 3 Pro Preview</td> <td>21.02</td><td>23.26</td><td>18.12</td><td>43.36</td><td>44.92</td><td>50.83</td><td>6.00</td><td>10.00</td><td>8.50</td></tr>
<tr><td>GPT-5.2</td>              <td>25.70</td><td>19.22</td><td>22.16</td><td>53.40</td><td>51.60</td><td>53.36</td><td>4.66</td><td>0.00</td><td>2.00</td></tr>
<tr><td colspan="10"><i>Long-term memory agents</i></td></tr>
<tr><td>A-Mem</td>    <td>71.82</td><td>41.90</td><td>40.78</td><td>35.04</td><td>37.52</td><td>34.95</td><td>2.00</td><td>2.00</td><td>5.00</td></tr>
<tr><td>LangMem</td>  <td>71.16</td><td>42.00</td><td>39.14</td><td>48.88</td><td>44.08</td><td>33.85</td><td>30.00</td><td>14.00</td><td>11.00</td></tr>
<tr><td>Mem-0</td>    <td>40.42</td><td>21.08</td><td>19.90</td><td>52.58</td><td>36.20</td><td>38.47</td><td>16.00</td><td>0.00</td><td>2.00</td></tr>
<tr><td>MemoBase</td> <td>43.60</td><td>20.08</td><td>15.18</td><td>68.94</td><td>58.46</td><td>45.62</td><td>18.00</td><td>7.00</td><td>1.00</td></tr>
<tr><td>MemoryOS</td> <td>51.84</td><td>29.78</td><td>25.05</td><td>62.64</td><td>48.54</td><td>44.02</td><td>20.66</td><td>6.00</td><td>5.50</td></tr>
<tr><td>Nemori</td>   <td>65.06</td><td>44.08</td><td>33.83</td><td>52.84</td><td>45.90</td><td>41.66</td><td>18.66</td><td>0.00</td><td>6.50</td></tr>
</tbody>
</table>

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
