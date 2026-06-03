#!/usr/bin/env python3
"""Regression test for the multi-judge client discovery path.

Context
-------
Multi-judge evaluation (majority vote of three OpenRouter judges) is the
protocol behind the published Table 3 numbers. ``memory_to_answer.py`` loads
the judge backend (``api_client.OpenRouterClient``) by appending a directory to
``sys.path`` and importing it. That directory must be ``evals/model_eval`` —
i.e. ``Path(__file__).parent.parent / "model_eval"`` relative to
``evals/agent_eval/memory_to_answer.py``.

A prior version used ``Path(__file__).parent / "model_eval"``, which points at
the non-existent ``evals/agent_eval/model_eval``. The resulting ``ImportError``
was caught and the code silently set ``use_multi_judge = False`` — so every
agent run degraded to a SINGLE judge with no error, producing FAMA numbers that
do not match Table 3. This test pins the correct discovery path and guards
against a silent re-regression.

Runs with plain ``pytest``. No API keys, no network: importing
``OpenRouterClient`` only reads the API key inside its ``__init__``, never at
import time, so the class is importable offline. ``openai`` need not be
installed for the import either.

    cd evals/agent_eval && python -m pytest test_judge_import.py -v
"""

import importlib
import sys
from pathlib import Path

import pytest

# Make the agent_eval directory importable when pytest is invoked from elsewhere.
AGENT_EVAL_DIR = Path(__file__).resolve().parent
if str(AGENT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_EVAL_DIR))

mta = importlib.import_module("memory_to_answer")


def test_model_eval_dir_resolves_to_existing_directory():
    """The judge backend dir must actually exist and contain api_client.py.

    Fails on the buggy `.parent` path (resolves to a non-existent
    agent_eval/model_eval); passes once it is `.parent.parent` (evals/model_eval).
    """
    assert hasattr(mta, "model_eval_dir"), (
        "memory_to_answer must expose a single source of truth for the "
        "model_eval directory (model_eval_dir())."
    )
    judge_dir = mta.model_eval_dir()
    assert judge_dir.is_dir(), (
        f"Resolved model_eval directory does not exist: {judge_dir}. "
        f"The multi-judge import would raise ImportError here and the run "
        f"would silently degrade to a single judge."
    )
    assert (judge_dir / "api_client.py").is_file(), (
        f"api_client.py not found in {judge_dir}; OpenRouterClient is "
        f"unreachable and multi-judge would silently fall back."
    )


def test_openrouter_client_imports_without_keys():
    """The multi-judge client must import via the production code path.

    This calls the SAME helper memory_to_answer.__init__ uses to discover and
    import OpenRouterClient. No OPENROUTER_API_KEY is set here; the import must
    still succeed (the key is only read in OpenRouterClient.__init__).
    """
    assert hasattr(mta, "import_openrouter_client"), (
        "memory_to_answer must expose import_openrouter_client() so the "
        "judge-discovery path is testable and reused, not duplicated."
    )
    client_cls = mta.import_openrouter_client()
    assert client_cls.__name__ == "OpenRouterClient"


def test_requested_multi_judge_does_not_silently_degrade(monkeypatch):
    """A requested-multi-judge run must NOT quietly become single-judge.

    We drive the real failure-handling path (_handle_multi_judge_failure) on a
    bare instance. With strict_judges=True it must raise rather than silently
    set use_multi_judge=False — guaranteeing no one unknowingly reports
    single-judge FAMA as if it were the 3-judge Table 3 protocol.
    """
    assert hasattr(mta.MemoryQuestionAnswering, "_handle_multi_judge_failure"), (
        "Multi-judge failures must be handled by a dedicated, loud path."
    )

    # Build a minimal instance without running the real (key-requiring) __init__.
    qa = mta.MemoryQuestionAnswering.__new__(mta.MemoryQuestionAnswering)
    qa.use_multi_judge = True
    qa.strict_judges = True

    with pytest.raises(RuntimeError):
        qa._handle_multi_judge_failure("simulated judge init failure")

    # Non-strict path is allowed to degrade, but must flip the flag explicitly
    # (i.e. the degradation is an explicit, observable decision, not an accident).
    qa2 = mta.MemoryQuestionAnswering.__new__(mta.MemoryQuestionAnswering)
    qa2.use_multi_judge = True
    qa2.strict_judges = False
    qa2._handle_multi_judge_failure("simulated judge init failure")
    assert qa2.use_multi_judge is False
