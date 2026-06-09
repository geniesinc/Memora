#!/usr/bin/env python3
"""
Configuration settings for model evaluation
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    
    # Model settings — all models route through OpenRouter (matches data-gen provider)
    model_name: str = "anthropic/claude-sonnet-4.5"
    temperature: float = 0.7
    judge_temperature: float = 0.0  # Use 0 for consistent evaluation
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0  # Base delay for exponential backoff
    
    # Evaluation settings
    save_frequency: int = 10  # Save results every N questions
    verbose: bool = True
    
    # Context settings
    max_context_length: Optional[int] = None  # Limit context size if needed
    include_system_prompts: bool = False  # Include full prompts in conversation context
    
    # Output settings
    save_detailed_results: bool = True
    save_summary_report: bool = True
    save_analysis: bool = True


# Frontier models under evaluation.
RELEASE_MODELS = [
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.7",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-fable-5",
]

# Model context limits (in tokens)
MODEL_CONTEXT_LIMITS = {
    # Google Models
    "google/gemini-3.1-pro-preview": 1_048_576,
    "gemini-3.1-pro-preview": 1_048_576,
    "google/gemini-3-pro-preview": 1_048_576,
    "gemini-3-pro-preview": 1_048_576,

    # OpenAI Models
    "openai/gpt-5.5": 400_000,
    "gpt-5.5": 400_000,
    "openai/gpt-5.2": 400_000,
    "gpt-5.2": 400_000,

    # Anthropic Models
    "anthropic/claude-fable-5": 1_000_000,
    "claude-fable-5": 1_000_000,
    "anthropic/claude-opus-4.7": 1_000_000,
    "claude-opus-4.7": 1_000_000,
    "anthropic/claude-sonnet-4.5": 1_000_000,
    "claude-sonnet-4.5": 1_000_000,

    # Qwen Models
    "qwen/qwen3-32b": 40_960,
    "qwen3-32b": 40_960,

}

# Evaluation type descriptions
EVALUATION_TYPE_DESCRIPTIONS = {
    "memory_presence": "Checks if the model correctly uses information it should remember",
    "forgetting_absence": "Checks if the model correctly avoids using information it should have forgotten"
}

