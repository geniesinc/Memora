"""Offline regression test for temporal question context."""

from model_based_evaluator import EvaluationRunner


class RecordingPromptBuilder:
    def __init__(self):
        self.current_date = None

    def build_question_prompt(self, context, question, current_date=None):
        self.current_date = current_date
        return "system prompt", "user prompt"


class StaticApiClient:
    def generate_response(self, **kwargs):
        return "answer"


def test_question_date_is_used_for_prompt_and_result():
    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.prompt_builder = RecordingPromptBuilder()
    runner.model_name = None
    runner.api_client = StaticApiClient()
    runner.reasoning_config = None
    runner.evaluator = None
    runner.use_multi_judge = False
    runner.judge_models = {}

    [result] = runner._evaluate_questions("context", [{
        "question_id": "q1",
        "question": "What happened this week?",
        "question_date": "2025-06-07",
        "task_type": "Remembering",
        "evaluation": {"evaluation_questions": []},
    }])

    assert runner.prompt_builder.current_date == "2025-06-07"
    assert result["question_date"] == "2025-06-07"
    assert "session_date" not in result
