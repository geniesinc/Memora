"""Offline regression tests for model-evaluation prompt construction."""

from model_based_evaluator import PromptBuilder


def test_published_user_agent_turns_are_included_in_context():
    conversations = [{
        "session_id": 1,
        "date": "2025-06-01",
        "conversation": [
            {"speaker": "user_agent", "message": "My favorite color is teal."},
            {"speaker": "ai_agent", "message": "I'll remember that."},
        ],
    }]

    context, _ = PromptBuilder().build_conversation_context(conversations)

    assert "User: My favorite color is teal." in context
    assert "Assistant: I'll remember that." in context


def test_unknown_speakers_are_not_mislabeled_during_truncation(monkeypatch):
    monkeypatch.setattr("model_based_evaluator.count_tokens", lambda text, model: len(text))
    conversations = [{
        "session_id": 1,
        "date": "2025-06-01",
        "conversation": [
            {"speaker": "unknown", "message": "must not become a user turn"},
            {"speaker": "user_agent", "message": "published user turn"},
        ],
    }]

    context, _ = PromptBuilder().build_conversation_context(conversations, max_tokens=300)

    assert "must not become a user turn" not in context
    assert "User: published user turn" in context
