#!/usr/bin/env python3
"""
Regression test for the agent-eval "passed evaluations" counter.

No network or API keys required: we stub the optional `openai`/`dotenv`
imports so memory_to_answer can be imported, then exercise the pure
counting helper that the per-question summary depends on.

Bug being guarded against: answer_question() previously counted passed
evaluations via the non-existent top-level key `evaluation_passed`, so the
count was ALWAYS 0 and the run summary printed "Passed evaluations: 0"
regardless of judge results. Correctness actually lives in the nested
`evaluation_result['is_correct']` field (the same schema the Track-1
model-based evaluator uses).
"""

import sys
import types
from pathlib import Path

# Make the agent_eval dir importable and stub optional deps so import works
# without installing openai / python-dotenv.
sys.path.insert(0, str(Path(__file__).resolve().parent))
if 'openai' not in sys.modules:
    _openai = types.ModuleType('openai')
    _openai.OpenAI = lambda *a, **k: None
    sys.modules['openai'] = _openai

from memory_to_answer import count_passed_evaluations  # noqa: E402


def _eval_result(is_correct, evaluation_type):
    """Build one entry shaped exactly like evaluate_answer_with_llm output."""
    return {
        'evaluation_question_id': 'e',
        'evaluation_question': 'q?',
        'expected_answer': 'yes' if evaluation_type == 'memory_presence' else 'no',
        'evaluation_type': evaluation_type,
        'evaluation_result': {
            'is_correct': is_correct,
            'confidence': 1.0,
            'explanation': '',
            'llm_answer': 'yes' if is_correct else 'no',
        },
    }


def test_counts_nested_is_correct():
    results = [
        _eval_result(True, 'memory_presence'),
        _eval_result(False, 'memory_presence'),
        _eval_result(True, 'forgetting_absence'),
    ]
    # 2 of 3 sub-questions are correct. The buggy implementation that read a
    # top-level 'evaluation_passed' key would return 0 here.
    assert count_passed_evaluations(results) == 2


def test_all_correct():
    results = [
        _eval_result(True, 'memory_presence'),
        _eval_result(True, 'forgetting_absence'),
    ]
    assert count_passed_evaluations(results) == 2


def test_none_correct():
    results = [
        _eval_result(False, 'memory_presence'),
        _eval_result(False, 'forgetting_absence'),
    ]
    assert count_passed_evaluations(results) == 0


def test_empty():
    assert count_passed_evaluations([]) == 0


def test_does_not_read_legacy_evaluation_passed_key():
    # An entry whose ONLY truthy flag is the legacy/wrong key must NOT be
    # counted as passed — guards against regressing to eval.get('evaluation_passed').
    legacy = {
        'evaluation_type': 'memory_presence',
        'evaluation_passed': True,           # wrong/legacy key
        'evaluation_result': {'is_correct': False},
    }
    assert count_passed_evaluations([legacy]) == 0


if __name__ == '__main__':
    test_counts_nested_is_correct()
    test_all_correct()
    test_none_correct()
    test_empty()
    test_does_not_read_legacy_evaluation_passed_key()
    print("All passed_evaluations regression tests passed.")
