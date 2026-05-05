# Memora Dataset

Released conversations and evaluation questions for the Memora benchmark.

## Layout

```
data/
├── weekly/<persona>/
│   ├── conversations/session_NNNN.json
│   └── evaluation_questions_<persona>.json
├── monthly/<persona>/
│   └── ...
└── quarterly/<persona>/
    └── ...
```

The three `<period>` buckets correspond to the three temporal durations in the
paper. They share the same persona set; what changes is the number of sessions
generated and the evaluation horizon for the questions:

| Period      | Days | Sessions / persona | Questions / persona |
| ----------- | ---: | -----------------: | ------------------: |
| `weekly`    |    7 |               ~150 |                  15 |
| `monthly`   |   30 |               ~600 |                  15 |
| `quarterly` |   90 |              ~2000 |                  30 |

Quarterly questions are doubled (10 per task instead of 5) because the larger
evidence horizon supports a wider range of memory probes.

The 10 personas are: `academic_researcher`, `business_executive`,
`content_writer`, `creative_designer`, `financial_analyst`,
`management_consultant`, `marketing_manager`, `sales_manager`,
`software_engineer`, `startup_founder`.

## Session file (`conversations/session_NNNN.json`)

```jsonc
{
  "session_id": 1,
  "session_type": "no_memory" | "memory_introduction" | "memory_update" | ...,
  "operation": "...",                  // what happened in this session
  "operation_details": { ... },
  "date": "2025-06-01",
  "persona": "academic_researcher",
  "conversation": [
    {
      "turn": 1,
      "speaker": "ai_agent" | "user",
      "message": "...",
      "share_memory": false             // true = this turn carries new memory
                                        //        the agent should retain
    },
    ...
  ]
}
```

Sessions are stored in chronological order (`session_0001.json` is the
earliest). `share_memory` flags the turns that introduce information the user
later expects the agent to remember (or, for forgetting sessions, to forget).

## Evaluation file (`evaluation_questions_<persona>.json`)

```jsonc
{
  "persona": "academic_researcher",
  "date_range": { "start_date": "...", "end_date": "..." },
  "questions": {
    "remembering":  [ <Question>, ... ],   // 5 (weekly/monthly) or 10 (quarterly)
    "reasoning":    [ <Question>, ... ],
    "recommending": [ <Question>, ... ]
  }
}
```

The three keys map to the paper's three task types. Each `<Question>` looks
like:

```jsonc
{
  "question_id": "activity_todos_158",
  "question": "What tasks remain on my todo list this week?",
  "question_date": "2025-06-07",

  // Ground-truth evidence — included so dataset users can audit the
  // evaluation; not consumed by the evaluators directly.
  "memory_evidence":     { ... },          // info that should be recalled
  "forgetting_evidence": { ... },          // info that has been deleted/updated

  // The actual scoring rubric — yes/no sub-questions the judge answers.
  "evaluation": {
    "evaluation_questions": [
      {
        "evaluation_question_id": "activity_todos_158_eval_memory_presence_0",
        "evaluation_question": "Does the response mention the task: ...?",
        "expected_answer": "yes",
        "evaluation_type": "memory_presence"      // or "forgetting_absence"
      },
      ...
    ],
    "total_evaluation_questions":   N,
    "memory_presence_questions":    M,
    "forgetting_absence_questions": F   // M + F == N
  }
}
```

### `evaluation_type` semantics

| Type                 | What it measures                                                                | `expected_answer` |
| -------------------- | ------------------------------------------------------------------------------- | ----------------- |
| `memory_presence`    | The model **should** mention this fact (it's in the conversation history)       | `"yes"`           |
| `forgetting_absence` | The model **should not** mention this fact (it was deleted or updated)          | `"no"`            |

The judge is asked the sub-question against the model's free-form answer and
returns `yes` / `no`; the sub-question is scored correct iff its answer
matches `expected_answer`. **FAMA** (see [`evals/README.md`](../evals/README.md))
combines the two rates into the Table 3 number.

### Reasoning-task caveat

Most *Reasoning* questions ask the model to draw a conclusion from a body of
remembered facts; they have only `memory_presence` sub-questions and no
`forgetting_absence` sub-questions. In that case `λ = 0` in the FAMA formula
and FAMA collapses to memory-presence accuracy.

## Where eval results land

When you run an evaluator, results are written next to the conversation data:

```
data/<period>/<persona>/eval_results/<run_id>/
    eval_results_<TIMESTAMP>.json     # per-question detail
    eval_report_<TIMESTAMP>.json      # aggregated metrics (FAMA + breakdowns)
```

`<run_id>` is `<model>_reasoning` / `<model>_no_reasoning` for Track 1, and the
agent name (`a_mem`, `langmem`, `mem_0`, `memobase`, `memos`, `nemori`) for
Track 2. See [`evals/README.md`](../evals/README.md) for the full report
schema.

## License

The dataset is released under Apache License 2.0 (see
[`../LICENSE`](../LICENSE)).
