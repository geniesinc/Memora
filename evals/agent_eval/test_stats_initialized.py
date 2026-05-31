"""Regression test: ``self.stats`` must be initialized on EVERY ``__init__`` path.

Before the fix, ``self.stats = {...}`` lived inside ``_handle_multi_judge_failure``,
so it was only created on the failure path. A successful multi-judge initialization
left ``self.stats`` undefined, and the first stats update in ``answer_question`` /
``process_questions`` / ``print_summary`` crashed with ``AttributeError``.

This test asserts the source-level contract without needing API keys or network:
the single ``self.stats`` assignment lives in ``__init__`` (before the failure
handler), so it runs on the successful-init path too, and it carries every key the
rest of the module increments.

Run:  python -m pytest evals/agent_eval/test_stats_initialized.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

AGENT_EVAL_DIR = Path(__file__).resolve().parent
if str(AGENT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_EVAL_DIR))

import memory_to_answer as mta  # noqa: E402

# Every key the module's stats updates touch; a missing key would KeyError at runtime.
REQUIRED_STATS_KEYS = {
    "total_questions",
    "answered",
    "failed",
    "no_memories_found",
    "with_memories_found",
    "total_evaluations",
    "passed_evaluations",
    "questions_with_evaluations",
    "memory_presence_total",
    "memory_presence_passed",
    "forgetting_absence_total",
    "forgetting_absence_passed",
}


def test_stats_initialized_outside_failure_handler():
    """self.stats must be assigned exactly once, in __init__, not in the handler."""
    src = Path(mta.__file__).read_text(encoding="utf-8")
    assert src.count("self.stats = {") == 1, (
        "self.stats must be assigned exactly once (in __init__, not in the "
        "multi-judge-failure handler)."
    )
    assign_idx = src.index("self.stats = {")
    handler_idx = src.index("def _handle_multi_judge_failure")
    assert assign_idx < handler_idx, (
        "self.stats assignment must live in __init__ (before the failure-handler "
        "definition) so it runs on the successful multi-judge init path too."
    )


def test_stats_contract_has_all_keys():
    """The stats dict must contain every key the module later increments."""
    src = Path(mta.__file__).read_text(encoding="utf-8")
    for key in REQUIRED_STATS_KEYS:
        assert f"'{key}'" in src, f"stats key missing from contract: {key}"
