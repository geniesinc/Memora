#!/usr/bin/env python3
"""
Model-Based Memory Evaluation System for Memora Framework
==========================================================

This script evaluates a model's ability to answer memory-based questions
by providing conversation history as context and comparing responses against
evaluation questions.

Pipeline Overview:
1. Load all successful conversations from a session directory
2. Build conversation context prompt
3. Load questions from evaluation_questions_*.json (preferred) or unified_questions_*.json (fallback)
4. For each question:
   - Send conversation context + question to model
   - Get model response
   - Evaluate response using evaluation questions (yes/no)
5. Generate comprehensive evaluation report

The evaluator automatically detects and supports both file formats:
- evaluation_questions_*.json: Selected 15 questions (5 per task type)
- unified_questions_*.json: All generated questions (legacy format)

Usage:
    python model_based_evaluator.py \\
        --sessions-dir data/weekly/academic_researcher \\
        --model anthropic/claude-sonnet-4.5 \\
        --output-dir data/weekly/academic_researcher/eval_results/claude_no_reasoning

    # With reasoning enabled
    python model_based_evaluator.py \\
        --sessions-dir data/monthly/business_executive \\
        --model anthropic/claude-sonnet-4.5 \\
        --reasoning-max-tokens 10000

    # Limit number of questions for testing
    python model_based_evaluator.py \\
        --sessions-dir data/weekly/academic_researcher \\
        --limit 5
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import glob
from tqdm import tqdm

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠ tiktoken not installed. Using approximate token counting (chars/4).")
    print("  For accurate token counting, install with: pip install tiktoken")

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Try to load from project root first
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"✓ Loaded environment variables from {env_path}")
        # Verify the key is loaded (check both variants)
        api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPEN_ROUTER_API_KEY')
        if api_key:
            print("✓ OpenRouter API key found")
        else:
            print("⚠ OpenRouter API key not found in .env file")
            print("  Please ensure your .env file contains: OPENROUTER_API_KEY=your_key_here")
            print("  (or OPEN_ROUTER_API_KEY=your_key_here)")
    else:
        load_dotenv(override=True)  # Load from current directory or parents
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
    print("  Or set the API key manually: export OPENROUTER_API_KEY='your_key'")
except Exception as e:
    print(f"⚠ Error loading .env file: {e}")


# The benchmark uses three tasks: Remembering, Reasoning, Recommending
# (paper §3.5 / Figure 1). No further question_type taxonomy is reported.
TASK_TYPES = ("Remembering", "Reasoning", "Recommending")


def fama_score(memory_presence_correct: int, memory_presence_total: int,
               forgetting_absence_correct: int, forgetting_absence_total: int) -> float:
    """
    Forgetting-Aware Memory Accuracy (paper §4.2).

        FAMA = max(0, MPA − λ·(1 − FAA))
        MPA  = memory_presence_correct / memory_presence_total
        FAA  = forgetting_absence_correct / forgetting_absence_total
        λ    = N_forget / (N_presence + N_forget)

    Per-question FAMA ∈ [0, 1]. If a question has no memory_presence criteria
    we return 0; if it has no forgetting_absence criteria, λ=0 → FAMA = MPA.
    """
    n_p = memory_presence_total
    n_f = forgetting_absence_total
    if n_p == 0 and n_f == 0:
        return 0.0
    mpa = (memory_presence_correct / n_p) if n_p > 0 else 0.0
    faa = (forgetting_absence_correct / n_f) if n_f > 0 else 1.0
    lam = n_f / (n_p + n_f) if (n_p + n_f) > 0 else 0.0
    return max(0.0, mpa - lam * (1.0 - faa))


# Import model context limits from config
try:
    from config import MODEL_CONTEXT_LIMITS
except ImportError:
    # Fallback defaults if config not available
    MODEL_CONTEXT_LIMITS = {
        "google/gemini-3.1-pro-preview": 1_000_000,
        "gemini-3.1-pro-preview": 1_000_000,
        "google/gemini-3-pro-preview": 1_000_000,
        "gemini-3-pro-preview": 1_000_000,
        "qwen/qwen3-32b": 40_000,
        "qwen3-32b": 40_000,
        "openai/gpt-5.5": 400_000,
        "gpt-5.5": 400_000,
        "openai/gpt-5.2": 400_000,
        "gpt-5.2": 400_000,
        "anthropic/claude-opus-4.7": 1_000_000,
        "claude-opus-4.7": 1_000_000,
        "anthropic/claude-sonnet-4.5": 1_000_000,
        "claude-sonnet-4.5": 1_000_000,
    }


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken if available, otherwise approximate.
    
    Args:
        text: Text to count tokens for
        model_name: Model name to determine encoding (default: gpt-4)
    
    Returns:
        Approximate token count
    """
    if TIKTOKEN_AVAILABLE:
        try:
            # Map model names to tiktoken encodings
            # For OpenRouter models, try to extract base model name
            encoding_name = "cl100k_base"  # Default for GPT-4, GPT-3.5
            
            # Try to determine encoding based on model name
            if "gpt-4" in model_name.lower() or "gpt-3.5" in model_name.lower() or "gpt-5" in model_name.lower():
                encoding_name = "cl100k_base"
            elif "gpt-3" in model_name.lower():
                encoding_name = "p50k_base"
            elif "qwen" in model_name.lower():
                # Qwen models (Qwen2, Qwen3) use GPT-style tokenization (cl100k_base)
                # Qwen has tiktoken-compatible tokenizer but uses cl100k_base encoding
                encoding_name = "cl100k_base"
            elif "claude" in model_name.lower() or "anthropic" in model_name.lower():
                # Claude uses cl100k_base similar to GPT-4
                encoding_name = "cl100k_base"
            elif "gemini" in model_name.lower() or "google" in model_name.lower():
                # Gemini doesn't have a tiktoken encoding, use approximation
                # But try cl100k_base as a reasonable approximation
                encoding_name = "cl100k_base"
            else:
                # Default to cl100k_base for unknown models
                encoding_name = "cl100k_base"
            
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            # Fall back to approximation if encoding fails
            print(f"⚠ Token counting error: {e}, using approximation")
            return len(text) // 4
    else:
        # Approximate: ~4 characters per token (conservative estimate)
        return len(text) // 4


# Hard cap on context size to avoid API issues with very large prompts
# Even if model supports more, we cap at 200k tokens for practical reliability
MAX_CONTEXT_TOKENS_HARD_CAP = 200_000

def get_model_context_limit(model_name: str, safety_margin: float = 0.3, prompt_overhead: int = 10000) -> Optional[int]:
    """
    Get the context limit for a model with safety margin and prompt overhead reserved.
    
    Args:
        model_name: Model name (may include provider prefix like "openai/")
        safety_margin: Fraction of context to reserve (default: 0.3 = 30%)
        prompt_overhead: Additional tokens to reserve for system prompt, user prompt formatting, 
                        question text, and API overhead (default: 10000 tokens for extra safety)
    
    Returns:
        Maximum tokens available for conversation context, or None if unknown
        Note: Result is capped at MAX_CONTEXT_TOKENS_HARD_CAP for reliability
    """
    # Try exact match first
    if model_name in MODEL_CONTEXT_LIMITS:
        limit = MODEL_CONTEXT_LIMITS[model_name]
    else:
        # Try to match without provider prefix
        model_base = model_name.split("/")[-1] if "/" in model_name else model_name
        
        # Try various matches
        limit = None
        for key, value in MODEL_CONTEXT_LIMITS.items():
            if model_base in key or key in model_base:
                limit = value
                break
        
        # If still not found, try common patterns
        if limit is None:
            if "gpt-5" in model_base.lower():
                limit = (MODEL_CONTEXT_LIMITS.get("openai/gpt-5.5")
                         or MODEL_CONTEXT_LIMITS.get("gpt-5.5")
                         or MODEL_CONTEXT_LIMITS.get("openai/gpt-5.2")
                         or MODEL_CONTEXT_LIMITS.get("gpt-5.2", 400_000))
            elif "gpt-4" in model_base.lower():
                limit = MODEL_CONTEXT_LIMITS.get("gpt-4o", 128_000)
            elif "gpt-3.5" in model_base.lower():
                limit = MODEL_CONTEXT_LIMITS.get("gpt-3.5-turbo", 16_385)
            elif "claude-opus-4" in model_base.lower():
                limit = MODEL_CONTEXT_LIMITS.get("anthropic/claude-opus-4.7") or MODEL_CONTEXT_LIMITS.get("claude-opus-4.7", 1_000_000)
            elif "claude-sonnet-4.5" in model_base.lower() or "claude-sonnet-4" in model_base.lower():
                limit = MODEL_CONTEXT_LIMITS.get("anthropic/claude-sonnet-4.5") or MODEL_CONTEXT_LIMITS.get("claude-sonnet-4.5", 1_000_000)
            elif "claude" in model_base.lower():
                limit = MODEL_CONTEXT_LIMITS.get("claude-3-5-sonnet-20241022", 200_000)
            elif "gemini-3" in model_base.lower():
                limit = (MODEL_CONTEXT_LIMITS.get("google/gemini-3.1-pro-preview")
                         or MODEL_CONTEXT_LIMITS.get("gemini-3.1-pro-preview")
                         or MODEL_CONTEXT_LIMITS.get("google/gemini-3-pro-preview")
                         or MODEL_CONTEXT_LIMITS.get("gemini-3-pro-preview", 1_048_576))
            elif "gemini" in model_base.lower():
                limit = MODEL_CONTEXT_LIMITS.get("google/gemini-2.5-flash", 1_000_000)
            elif "qwen3" in model_base.lower() or "qwen" in model_base.lower():
                limit = MODEL_CONTEXT_LIMITS.get("qwen/qwen3-32b") or MODEL_CONTEXT_LIMITS.get("qwen3-32b", 40_960)
    
    if limit is None:
        return None
    
    # Reserve safety margin AND prompt overhead for system prompt, question, and response
    # This ensures the final prompt (system + user + context + question) fits within the limit
    available_tokens = int(limit * (1 - safety_margin)) - prompt_overhead
    
    # Apply hard cap to avoid API issues with very large contexts
    # Even if model supports more, cap at MAX_CONTEXT_TOKENS_HARD_CAP for reliability
    available_tokens = min(available_tokens, MAX_CONTEXT_TOKENS_HARD_CAP)
    
    return max(available_tokens, 0)  # Ensure non-negative


class ConversationLoader:
    """Load and organize conversations from session directory"""
    
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = Path(sessions_dir)
        # Released layout: data/<period>/<persona>/conversations/session_*.json
        # Legacy layout (data-gen output): conversations/successful/session_*.json
        # Try the released layout first, fall back to the legacy one.
        released_dir = self.sessions_dir / "conversations"
        legacy_dir = released_dir / "successful"
        if legacy_dir.exists():
            self.conversations_dir = legacy_dir
        else:
            self.conversations_dir = released_dir

    def load_all_conversations(self) -> List[Dict[str, Any]]:
        """Load all conversations sorted by session_id"""
        if not self.conversations_dir.exists():
            raise FileNotFoundError(
                f"Conversations directory not found: {self.conversations_dir}"
            )
        
        conversations = []
        json_files = sorted(self.conversations_dir.glob("*.json"))
        
        print(f"Loading {len(json_files)} conversation files...")
        for json_file in tqdm(json_files, desc="Loading conversations"):
            try:
                with open(json_file, 'r') as f:
                    conv_data = json.load(f)
                    conversations.append(conv_data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        # Sort by session_id to ensure chronological order
        conversations.sort(key=lambda x: x.get('session_id', 0))
        
        print(f"Successfully loaded {len(conversations)} conversations")
        return conversations
    
    def load_unified_questions(self) -> Dict[str, Any]:
        """Load questions file from the sessions directory
        
        Prioritizes evaluation_questions_*.json, falls back to unified_questions_*.json
        Normalizes both formats to a common structure for evaluation
        """
        # First, try to find evaluation_questions_*.json (new format with 15 selected questions)
        eval_question_files = list(self.sessions_dir.glob("evaluation_questions_*.json"))
        
        if eval_question_files:
            # Use the most recent one if multiple exist
            question_file = max(eval_question_files, key=lambda p: p.stat().st_mtime)
            print(f"Loading evaluation questions from: {question_file}")
            
            with open(question_file, 'r') as f:
                data = json.load(f)
            
            # Normalize evaluation_questions format to match unified_questions structure
            return self._normalize_evaluation_questions(data)
        
        # Fallback to unified_questions*.json (old format)
        unified_question_files = list(self.sessions_dir.glob("unified_questions*.json"))
        
        if unified_question_files:
            # Use the most recent one if multiple exist
            question_file = max(unified_question_files, key=lambda p: p.stat().st_mtime)
            print(f"Loading unified questions from: {question_file}")
            
            with open(question_file, 'r') as f:
                return json.load(f)
        
        raise FileNotFoundError(
            f"No evaluation_questions_*.json or unified_questions*.json file found in {self.sessions_dir}"
        )
    
    def _normalize_evaluation_questions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize the released ``questions: {remembering, reasoning, recommending}``
        format into a flat per-task structure used by the rest of the pipeline."""
        normalized = {
            "metadata": data.get("metadata", {}),
            "questions_by_task_type": {t: [] for t in TASK_TYPES},
        }

        questions_by_task = data.get("questions", {})
        task_type_mapping = {t.lower(): t for t in TASK_TYPES}

        all_questions = []
        for task_type, questions in questions_by_task.items():
            if not isinstance(questions, list):
                continue
            proper_task_type = task_type_mapping.get(task_type.lower(), task_type)
            for question in questions:
                if not isinstance(question, dict):
                    continue
                question["task_type"] = proper_task_type
                all_questions.append(question)
                if proper_task_type in normalized["questions_by_task_type"]:
                    normalized["questions_by_task_type"][proper_task_type].append(question)

        # Carry through optional top-level metadata fields.
        for key in ("persona", "session_directory", "generation_date",
                    "selection_criteria", "date_range", "config_file"):
            if key in data:
                normalized["metadata"][key] = data[key]

        print(f"Normalized {len(all_questions)} questions from evaluation_questions format")
        print(f"  Remembering: {len(normalized['questions_by_task_type']['Remembering'])}")
        print(f"  Reasoning: {len(normalized['questions_by_task_type']['Reasoning'])}")
        print(f"  Recommending: {len(normalized['questions_by_task_type']['Recommending'])}")
        
        return normalized


class PromptBuilder:
    """Build prompts for model evaluation"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize PromptBuilder
        
        Args:
            model_name: Model name for token limit checking (optional)
        """
        self.model_name = model_name
    
    def build_conversation_context(
        self, 
        conversations: List[Dict[str, Any]], 
        max_tokens: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a formatted conversation context from conversations, with token limit handling.
        
        Args:
            conversations: List of conversation dictionaries
            max_tokens: Maximum tokens allowed (if None, uses model limit with safety margin)
        
        Returns:
            Tuple of (context_string, metadata_dict) where metadata contains:
                - total_conversations: Total number of conversations
                - included_conversations: Number of conversations included
                - truncated_conversations: Number of conversations truncated/omitted
                - total_tokens: Total tokens in final context
                - was_truncated: Whether truncation occurred
        """
        metadata = {
            "total_conversations": len(conversations),
            "included_conversations": 0,
            "truncated_conversations": 0,
            "total_tokens": 0,
            "was_truncated": False
        }
        
        # Determine max tokens if not provided
        if max_tokens is None and self.model_name:
            max_tokens = get_model_context_limit(self.model_name)
            # Apply hard cap even if get_model_context_limit returns something larger
            if max_tokens:
                max_tokens = min(max_tokens, MAX_CONTEXT_TOKENS_HARD_CAP)
        
        # Build header
        header = "# Conversation History\nThe following contains all your previous conversations with this user, in chronological order:\n\n"
        header_tokens = count_tokens(header, self.model_name or "gpt-4")
        
        # Start with header tokens
        current_tokens = header_tokens
        available_tokens = max_tokens - header_tokens if max_tokens else None
        
        # Store formatted conversations with their session_id for sorting
        included_convs = []  # List of (session_id, formatted_text, tokens)
        
        # Process conversations in reverse order (most recent first) to prioritize recent ones
        conversations_reversed = list(reversed(conversations))
        
        for conv_data in conversations_reversed:
            # Format this conversation
            session_id = conv_data.get('session_id', 'unknown')
            session_date = conv_data.get('date', 'unknown')
            
            conv_text_parts = [f"\n## Session {session_id} - {session_date}\n"]
            
            conversation = conv_data.get('conversation', [])
            for turn in conversation:
                speaker = turn.get('speaker', 'unknown')
                message = turn.get('message', '')
                
                if speaker == 'ai_agent':
                    conv_text_parts.append(f"Assistant: {message}")
                elif speaker == 'user':
                    conv_text_parts.append(f"User: {message}")
            
            conv_text_parts.append("")  # Add blank line between sessions
            conv_text = "\n".join(conv_text_parts)
            conv_tokens = count_tokens(conv_text, self.model_name or "gpt-4")
            
            # Check if we can fit this conversation
            if available_tokens is None or current_tokens + conv_tokens <= available_tokens:
                included_convs.append((session_id, conv_text, conv_tokens))
                current_tokens += conv_tokens
                metadata["included_conversations"] += 1
            else:
                # Try to include partial conversation if there's space
                if available_tokens and current_tokens < available_tokens:
                    remaining_tokens = available_tokens - current_tokens
                    # Try to include at least the session header and some turns
                    if conv_tokens > remaining_tokens and remaining_tokens > 100:
                        # Include partial conversation
                        partial_text = f"\n## Session {session_id} - {session_date}\n"
                        partial_tokens = count_tokens(partial_text, self.model_name or "gpt-4")
                        
                        # Add turns until we run out of space
                        for turn in conversation:
                            speaker = turn.get('speaker', 'unknown')
                            message = turn.get('message', '')
                            
                            turn_text = f"{'Assistant' if speaker == 'ai_agent' else 'User'}: {message}\n"
                            turn_tokens = count_tokens(turn_text, self.model_name or "gpt-4")
                            
                            if partial_tokens + turn_tokens <= remaining_tokens:
                                partial_text += turn_text
                                partial_tokens += turn_tokens
                            else:
                                break
                        
                        if partial_tokens > count_tokens(partial_text.split('\n')[0], self.model_name or "gpt-4"):
                            partial_text += "\n[Conversation truncated due to token limit...]"
                            included_convs.append((session_id, partial_text, partial_tokens))
                            current_tokens += partial_tokens
                            metadata["included_conversations"] += 1
                            metadata["was_truncated"] = True
                
                metadata["truncated_conversations"] += 1
                metadata["was_truncated"] = True
        
        # Sort included conversations chronologically by session_id for presentation
        included_convs.sort(key=lambda x: x[0] if isinstance(x[0], (int, float)) else 0)
        
        # Build final context in chronological order
        context_parts = [header]
        for _, conv_text, _ in included_convs:
            context_parts.append(conv_text)
        
        # If we truncated, add a note at the beginning
        if metadata["was_truncated"]:
            truncation_note = f"[Note: Showing {metadata['included_conversations']} of {metadata['total_conversations']} conversations due to token limits. Most recent conversations are prioritized.]\n\n"
            context_parts.insert(1, truncation_note)
            current_tokens += count_tokens(truncation_note, self.model_name or "gpt-4")
        
        context = "\n".join(context_parts)
        metadata["total_tokens"] = current_tokens
        
        # Warn if context is getting very large (approaching hard cap)
        if current_tokens > MAX_CONTEXT_TOKENS_HARD_CAP * 0.8:  # 80% of hard cap
            print(f"⚠ WARNING: Context size ({current_tokens:,} tokens) is approaching hard cap ({MAX_CONTEXT_TOKENS_HARD_CAP:,}). "
                  f"Consider reducing conversation history to avoid API issues.")
        
        return context, metadata
    
    def build_question_prompt(self, context: str, question: str, current_date: Optional[str] = None) -> Tuple[str, str]:
        """
        Build the full prompt with context + question.
        Verifies total prompt size and truncates context if needed to fit within model limits.
        
        Args:
            context: Conversation history context
            question: The question to answer
            current_date: Current date (YYYY-MM-DD format) for temporal context. If None, not included.
        """
        # Build date context if provided
        date_context = ""
        if current_date:
            date_context = f"\n\nIMPORTANT: Today's date is {current_date}. When answering questions about time periods (e.g., 'last 3 months', 'recent', 'this week'), calculate them relative to today's date ({current_date})."
        
        system_prompt = f"""You are an AI assistant with memory capabilities. The conversation history below contains ALL your previous conversations with this user - this history is PROVIDED TO YOU in this prompt.

Your task is to answer the user's question based EXCLUSIVELY on the conversation history that is provided below. The conversation history contains all the information you need to answer questions about the user's preferences, activities, goals, and past interactions.{date_context}

CRITICAL INSTRUCTIONS:
1. The conversation history is PROVIDED IN THIS PROMPT - you have full access to it
2. Answer the question using ONLY information from the conversation history provided below
3. Do NOT say "I don't have access" or "I cannot access" - you DO have access because the history is right here in this prompt
4. Extract relevant information from the conversation history and use it to answer the question
5. If you learned about user preferences, activities, content, or goals from the conversations below, use them in your response
6. If information was mentioned in earlier conversations but later updated or deleted, use only the most recent information
7. Be specific and reference relevant details from the conversations
8. If the conversation history doesn't contain enough information to answer the question, say so clearly (but do NOT say you don't have access)
9. Do not make up information that wasn't in the conversations"""
        
        # Build user prompt template (without context)
        user_prompt_template = f"""
---

# Current Question
User: {{question}}

Based on the conversation history above, please provide your answer:"""
        
        # Calculate tokens for system prompt and user prompt template (without context)
        system_tokens = count_tokens(system_prompt, self.model_name or "gpt-4")
        question_text = user_prompt_template.format(question=question)
        question_tokens = count_tokens(question_text, self.model_name or "gpt-4")
        prompt_overhead = system_tokens + question_tokens
        
        # Get model limit
        model_limit = None
        if self.model_name:
            # Get the raw limit (without safety margin) to check against
            if self.model_name in MODEL_CONTEXT_LIMITS:
                model_limit = MODEL_CONTEXT_LIMITS[self.model_name]
            else:
                model_base = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
                for key, value in MODEL_CONTEXT_LIMITS.items():
                    if model_base in key or key in model_base:
                        model_limit = value
                        break
        
        # Check if we need to truncate context further
        context_tokens = count_tokens(context, self.model_name or "gpt-4")
        total_tokens = prompt_overhead + context_tokens
        
        # Apply hard cap check - truncate if context exceeds hard cap regardless of model limit
        # This prevents API issues with very large contexts
        hard_cap_check = context_tokens > MAX_CONTEXT_TOKENS_HARD_CAP
        model_limit_check = model_limit and total_tokens > model_limit
        
        if hard_cap_check or model_limit_check:
            # Need to truncate context more aggressively
            # For hard cap, use the cap directly; for model limit, use calculated value
            if hard_cap_check:
                # Use hard cap with extra safety margin
                available_for_context = MAX_CONTEXT_TOKENS_HARD_CAP - prompt_overhead - 20000  # Extra 20k buffer for hard cap
                print(f"⚠ WARNING: Context size ({context_tokens:,} tokens) exceeds hard cap ({MAX_CONTEXT_TOKENS_HARD_CAP:,}). Truncating aggressively.")
            else:
                # Use model limit with safety margin
                safety_reserve = int(model_limit * 0.3)  # 30% safety margin
                available_for_context = model_limit - prompt_overhead - safety_reserve - 5000  # Extra 5000 token buffer
            
            available_for_context = max(available_for_context, 1000)  # Ensure at least 1000 tokens
            
            # Truncate context by removing from the beginning (oldest conversations)
            # Context structure: header + optional truncation_note + conversations (oldest first)
            # We need to preserve the header, only remove conversations from the start
            
            if available_for_context < context_tokens:
                # Find where conversations start (first session marker)
                session_marker = "\n## Session "
                first_session_pos = context.find(session_marker)
                
                if first_session_pos > 0:
                    # Extract header (everything before first session)
                    header_text = context[:first_session_pos]
                    header_tokens_actual = count_tokens(header_text, self.model_name or "gpt-4")
                    
                    # Calculate how much space we have for conversations
                    available_for_convs = available_for_context - header_tokens_actual
                    available_for_convs = max(available_for_convs, 500)  # At least 500 tokens for conversations
                    
                    # Get conversations part
                    conversations_text = context[first_session_pos:]
                    conv_tokens = count_tokens(conversations_text, self.model_name or "gpt-4")
                    
                    if conv_tokens > available_for_convs:
                        # Estimate characters to keep for conversations (rough: 4 chars per token)
                        chars_for_convs = available_for_convs * 4
                        
                        # Find where to truncate (keep most recent conversations from the end)
                        truncate_at = len(conversations_text) - chars_for_convs
                        
                        # Look for a session header near the truncation point
                        search_start = max(0, truncate_at - 5000)  # Search within 5k chars
                        marker_pos = conversations_text.rfind(session_marker, search_start, truncate_at + 1000)
                        
                        if marker_pos > 0:
                            # Truncate at the session boundary
                            truncated_convs = conversations_text[marker_pos:]
                        else:
                            # Fallback: truncate at character boundary
                            truncated_convs = conversations_text[-chars_for_convs:]
                        
                        # Add a note about additional truncation
                        additional_note = "\n[Note: Additional truncation applied to fit within model context limits. Showing most recent conversations only.]\n"
                        context = header_text + additional_note + truncated_convs
        
        user_prompt = f"""{context}

---

# Current Question
User: {question}

Based on the conversation history above, please provide your answer:"""
        
        return system_prompt, user_prompt


class ModelEvaluator:
    """Evaluate model responses using LLM-as-judge for yes/no questions"""
    
    def __init__(self, api_client):
        self.api_client = api_client
    
    def evaluate_response(
        self, 
        model_response: str, 
        evaluation_question: str, 
        expected_answer: str,
        question_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a model response against an evaluation question using LLM-as-judge
        
        Returns:
            Dict with keys: is_correct, confidence, explanation, llm_answer
        """
        judge_prompt = self._build_judge_prompt(
            model_response, 
            evaluation_question, 
            question_context
        )
        
        try:
            judge_response = self.api_client.generate_response(
                system_prompt=judge_prompt['system'],
                user_prompt=judge_prompt['user'],
                temperature=0.0  # Use low temperature for consistent evaluation
            )
            
            # Parse the judge response
            result = self._parse_judge_response(judge_response, expected_answer)
            return result
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return {
                "is_correct": False,
                "confidence": 0.0,
                "explanation": f"Evaluation error: {str(e)}",
                "llm_answer": "error",
                "error": str(e)
            }
    
    def _build_judge_prompt(
        self, 
        model_response: str, 
        evaluation_question: str,
        question_context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Build the LLM-as-judge prompt"""
        system_prompt = """You are an expert evaluator assessing AI assistant responses. Your task is to answer a YES/NO evaluation question about a given response.

You must provide your answer in the following JSON format:
{
    "answer": "yes" or "no",
    "confidence": 0.0 to 1.0,
    "explanation": "Brief explanation of your reasoning"
}

Be objective and thorough in your evaluation."""
        
        # Add context information if available
        context_info = ""
        if question_context:
            if 'target_item' in question_context:
                context_info += f"\nTarget item being evaluated: {question_context['target_item']}"
            if 'evaluation_type' in question_context:
                context_info += f"\nEvaluation type: {question_context['evaluation_type']}"
        
        user_prompt = f"""Please evaluate the following AI response against the evaluation question.

AI RESPONSE TO EVALUATE:
{model_response}

EVALUATION QUESTION:
{evaluation_question}
{context_info}

Provide your evaluation in JSON format with answer (yes/no), confidence (0.0-1.0), and explanation."""
        
        return {"system": system_prompt, "user": user_prompt}
    
    def _parse_judge_response(self, judge_response: str, expected_answer: str) -> Dict[str, Any]:
        """Parse the judge LLM response"""
        try:
            # Try to extract JSON from the response
            # Handle cases where the response might have markdown formatting
            response_text = judge_response.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(response_text)
            
            llm_answer = result.get('answer', '').lower().strip()
            confidence = float(result.get('confidence', 0.0))
            explanation = result.get('explanation', '')
            
            # Normalize answer
            if llm_answer not in ['yes', 'no']:
                # Try to infer from explanation
                explanation_lower = explanation.lower()
                if 'yes' in explanation_lower and 'no' not in explanation_lower:
                    llm_answer = 'yes'
                elif 'no' in explanation_lower and 'yes' not in explanation_lower:
                    llm_answer = 'no'
                else:
                    llm_answer = 'unclear'
            
            is_correct = (llm_answer == expected_answer.lower())
            
            return {
                "is_correct": is_correct,
                "confidence": confidence,
                "explanation": explanation,
                "llm_answer": llm_answer
            }
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse judge response as JSON: {judge_response}")
            # Fallback: try to extract yes/no from text
            response_lower = judge_response.lower()
            if 'yes' in response_lower and 'no' not in response_lower:
                llm_answer = 'yes'
            elif 'no' in response_lower and 'yes' not in response_lower:
                llm_answer = 'no'
            else:
                llm_answer = 'unclear'
            
            is_correct = (llm_answer == expected_answer.lower())
            
            return {
                "is_correct": is_correct,
                "confidence": 0.5,
                "explanation": f"Parse error, inferred from text: {judge_response[:200]}",
                "llm_answer": llm_answer,
                "parse_error": str(e)
            }
        except Exception as e:
            return {
                "is_correct": False,
                "confidence": 0.0,
                "explanation": f"Error parsing response: {str(e)}",
                "llm_answer": "error",
                "error": str(e)
            }


# Default judge models for multi-judge evaluation
DEFAULT_JUDGE_MODELS = {
    "openai": "openai/gpt-4.1",
    "anthropic": "anthropic/claude-haiku-4.5",
    "google": "google/gemini-2.5-flash"
}


class MultiJudgeEvaluator:
    """Evaluate model responses using multiple LLM judges and aggregate scores"""
    
    def __init__(self, judge_models: Dict[str, str] = None):
        """
        Initialize multi-judge evaluator
        
        Args:
            judge_models: Dict mapping judge name to model name
                         e.g., {"openai": "openai/gpt-4o", "anthropic": "anthropic/claude-sonnet-4"}
        """
        self.judge_models = judge_models or DEFAULT_JUDGE_MODELS
        self.judges = {}
        
        # Initialize API clients for each judge
        from api_client import OpenRouterClient
        print(f"🔍 Initializing Multi-Judge Evaluation:")
        for judge_name, model_name in self.judge_models.items():
            try:
                client = OpenRouterClient(model=model_name)
                self.judges[judge_name] = ModelEvaluator(client)
                # Verify client type
                client_type = type(client).__name__
                client_base_url = getattr(client.client, 'base_url', 'unknown')
                print(f"  ✓ Initialized judge: {judge_name} ({model_name})")
                print(f"    Client type: {client_type}, Base URL: {client_base_url}")
            except Exception as e:
                print(f"  ✗ Failed to initialize judge {judge_name}: {e}")
        
        if not self.judges:
            raise ValueError("No judges could be initialized")
    
    def evaluate_response(
        self, 
        model_response: str, 
        evaluation_question: str, 
        expected_answer: str,
        question_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a model response using all judges and aggregate results
        
        Returns:
            Dict with keys: 
                - is_correct: bool (majority vote or average threshold)
                - confidence: float (average confidence)
                - per_judge_results: Dict[str, result]
                - agreement_rate: float (how many judges agreed)
        """
        per_judge_results = {}
        
        for judge_name, evaluator in self.judges.items():
            try:
                result = evaluator.evaluate_response(
                    model_response=model_response,
                    evaluation_question=evaluation_question,
                    expected_answer=expected_answer,
                    question_context=question_context
                )
                per_judge_results[judge_name] = result
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"    ⚠ Judge {judge_name} failed: {e}")
                print(f"    Full error traceback:\n{error_details}")
                per_judge_results[judge_name] = {
                    "is_correct": False,
                    "confidence": 0.0,
                    "explanation": f"Judge error: {str(e)}",
                    "llm_answer": "error",
                    "error": str(e),
                    "error_traceback": error_details
                }
        
        # Aggregate results
        valid_results = [r for r in per_judge_results.values() if r.get("llm_answer") != "error"]
        
        if not valid_results:
            return {
                "is_correct": False,
                "confidence": 0.0,
                "explanation": "All judges failed",
                "llm_answer": "error",
                "per_judge_results": per_judge_results,
                "agreement_rate": 0.0,
                "num_judges": len(self.judges),
                "num_valid_judges": 0
            }
        
        # Calculate aggregated metrics
        correct_votes = sum(1 for r in valid_results if r["is_correct"])
        avg_confidence = sum(r["confidence"] for r in valid_results) / len(valid_results)
        
        # Majority vote for is_correct
        is_correct = correct_votes > len(valid_results) / 2
        
        # Agreement rate (how many judges gave the same answer)
        yes_votes = sum(1 for r in valid_results if r["llm_answer"] == "yes")
        no_votes = sum(1 for r in valid_results if r["llm_answer"] == "no")
        max_agreement = max(yes_votes, no_votes)
        agreement_rate = max_agreement / len(valid_results) if valid_results else 0.0
        
        # Determine consensus answer
        if yes_votes > no_votes:
            consensus_answer = "yes"
        elif no_votes > yes_votes:
            consensus_answer = "no"
        else:
            consensus_answer = "tie"
        
        return {
            "is_correct": is_correct,
            "confidence": avg_confidence,
            "explanation": f"Majority vote: {correct_votes}/{len(valid_results)} judges agreed with expected answer",
            "llm_answer": consensus_answer,
            "per_judge_results": per_judge_results,
            "agreement_rate": agreement_rate,
            "num_judges": len(self.judges),
            "num_valid_judges": len(valid_results),
            "correct_votes": correct_votes
        }


class EvaluationRunner:
    """Main evaluation runner"""
    
    def __init__(
        self,
        sessions_dir: Path,
        model_name: str,
        api_client,
        output_dir: Optional[Path] = None,
        limit: Optional[int] = None,
        judge_models: Optional[Dict[str, str]] = None,
        use_multi_judge: bool = True,
        reasoning_config: Optional[Dict[str, Any]] = None
    ):
        self.sessions_dir = Path(sessions_dir)
        self.model_name = model_name
        self.api_client = api_client
        self.output_dir = Path(output_dir) if output_dir else self.sessions_dir / "eval_results"
        self.use_multi_judge = use_multi_judge
        self.judge_models = judge_models or DEFAULT_JUDGE_MODELS
        self.limit = limit
        self.reasoning_config = reasoning_config  # For models with reasoning/thinking mode
        
        self.loader = ConversationLoader(sessions_dir)
        self.prompt_builder = PromptBuilder(model_name=model_name)
        
        # Initialize evaluator(s)
        if self.use_multi_judge:
            print("\n🔍 Initializing Multi-Judge Evaluation:")
            self.evaluator = MultiJudgeEvaluator(judge_models=self.judge_models)
        else:
            self.evaluator = ModelEvaluator(api_client)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the full evaluation pipeline"""
        print("=" * 80)
        print("Model-Based Memory Evaluation")
        print("=" * 80)
        print(f"Sessions directory: {self.sessions_dir}")
        print(f"Model: {self.model_name}")
        print(f"Output directory: {self.output_dir}")
        if self.use_multi_judge:
            print(f"Judge Models: {list(self.judge_models.keys())}")
        print("=" * 80)
        
        # Step 1: Load conversations
        print("\n[Step 1/5] Loading conversations...")
        conversations = self.loader.load_all_conversations()
        
        # Step 2: Build conversation context with token limit handling
        print("\n[Step 2/5] Building conversation context...")
        conversation_context, context_metadata = self.prompt_builder.build_conversation_context(conversations)
        
        # Store context metadata for report
        self.context_metadata = context_metadata
        
        # Display context information
        print(f"Context size: {len(conversation_context)} characters")
        print(f"Estimated tokens: {context_metadata['total_tokens']}")
        print(f"Conversations: {context_metadata['included_conversations']}/{context_metadata['total_conversations']} included")
        
        if context_metadata['was_truncated']:
            print(f"⚠ WARNING: {context_metadata['truncated_conversations']} conversation(s) were truncated due to token limits")
            if self.model_name:
                model_limit = get_model_context_limit(self.model_name)
                if model_limit:
                    print(f"   Model context limit: ~{model_limit} tokens (with safety margin)")
                else:
                    print(f"   Model '{self.model_name}' context limit unknown, using available space")
        else:
            print("✓ All conversations included in context")
        
        # Step 3: Load questions
        print("\n[Step 3/5] Loading questions...")
        questions_data = self.loader.load_unified_questions()
        questions_by_task = questions_data['questions_by_task_type']

        # Flatten questions across the three tasks
        all_questions = []
        for task in TASK_TYPES:
            all_questions.extend(questions_by_task.get(task, []))
        
        if self.limit:
            all_questions = all_questions[:self.limit]
            print(f"Limited to {self.limit} questions")
        
        print(f"Total questions to evaluate: {len(all_questions)}")
        
        # Step 4: Evaluate each question
        print("\n[Step 4/5] Evaluating questions...")
        results = self._evaluate_questions(conversation_context, all_questions)
        
        # Step 5: Generate report
        print("\n[Step 5/5] Generating report...")
        report = self._generate_report(results, questions_data['metadata'])
        
        # Save results
        self._save_results(results, report)
        
        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print("=" * 80)
        self._print_summary(report)
        
        return report
    
    def _evaluate_questions(
        self, 
        context: str, 
        questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Evaluate all questions"""
        results = []
        
        for i, question_data in enumerate(tqdm(questions, desc="Evaluating")):
            question_id = question_data['question_id']
            question_text = question_data['question']
            
            print(f"\n--- Question {i+1}/{len(questions)}: {question_id} ---")
            print(f"Q: {question_text}")
            
            # Get current date from question's session_date (when the question is being asked)
            # Default to None if not available
            current_date = question_data.get('session_date')
            
            # Build prompt with current date for temporal context
            system_prompt, user_prompt = self.prompt_builder.build_question_prompt(
                context, question_text, current_date=current_date
            )
            
            # Verify prompt size before sending (for debugging)
            if self.model_name:
                system_tokens = count_tokens(system_prompt, self.model_name)
                user_tokens = count_tokens(user_prompt, self.model_name)
                total_prompt_tokens = system_tokens + user_tokens
                model_limit = None
                if self.model_name in MODEL_CONTEXT_LIMITS:
                    model_limit = MODEL_CONTEXT_LIMITS[self.model_name]
                else:
                    model_base = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
                    for key, value in MODEL_CONTEXT_LIMITS.items():
                        if model_base in key or key in model_base:
                            model_limit = value
                            break
                
                if model_limit:
                    if total_prompt_tokens > model_limit:
                        print(f"⚠ WARNING: Total prompt tokens ({total_prompt_tokens}) exceeds model limit ({model_limit})")
                    else:
                        print(f"✓ Prompt size: {total_prompt_tokens:,} tokens (limit: {model_limit:,}, remaining: {model_limit - total_prompt_tokens:,})")
            
            # Get model response
            try:
                model_response = self.api_client.generate_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    reasoning=self.reasoning_config  # Pass reasoning config for thinking models
                )
                
                # Check if response is empty or None (shouldn't happen due to API client check, but double-check)
                if not model_response or (isinstance(model_response, str) and len(model_response.strip()) == 0):
                    raise ValueError("Model returned empty response")
                
                print(f"A: {model_response[:200]}{'...' if len(model_response) > 200 else ''}")
            except Exception as e:
                error_msg = str(e)
                print(f"Error generating response: {error_msg}")
                model_response = f"ERROR: {error_msg}"
                # Also print a warning about potential context size issues
                if "context" in error_msg.lower() or "token" in error_msg.lower() or "400" in error_msg:
                    print(f"⚠ WARNING: This error may be due to context size limits. Consider reducing conversation history.")
            
            # Evaluate against evaluation questions
            eval_questions = question_data.get('evaluation', {}).get('evaluation_questions', [])
            evaluation_results = []
            
            for eval_q in eval_questions:
                eval_result = self.evaluator.evaluate_response(
                    model_response=model_response,
                    evaluation_question=eval_q['evaluation_question'],
                    expected_answer=eval_q['expected_answer'],
                    question_context={
                        'target_item': eval_q.get('target_item'),
                        'evaluation_type': eval_q.get('evaluation_type')
                    }
                )
                
                evaluation_results.append({
                    **eval_q,
                    'evaluation_result': eval_result
                })
            
            # Calculate correctness separately for memory_presence and forgetting_absence
            memory_presence_results = [er for er in evaluation_results if er.get('evaluation_type') == 'memory_presence']
            forgetting_absence_results = [er for er in evaluation_results if er.get('evaluation_type') == 'forgetting_absence']
            
            memory_presence_correct = sum(1 for er in memory_presence_results if er['evaluation_result']['is_correct'])
            memory_presence_total = len(memory_presence_results)
            memory_presence_accuracy = memory_presence_correct / memory_presence_total if memory_presence_total > 0 else None
            
            forgetting_absence_correct = sum(1 for er in forgetting_absence_results if er['evaluation_result']['is_correct'])
            forgetting_absence_total = len(forgetting_absence_results)
            forgetting_absence_accuracy = forgetting_absence_correct / forgetting_absence_total if forgetting_absence_total > 0 else None
            
            correct_count = sum(1 for er in evaluation_results if er['evaluation_result']['is_correct'])
            total_count = len(evaluation_results)
            accuracy = correct_count / total_count if total_count > 0 else 0.0

            # Per-question FAMA (paper §4.2). Used for task-level aggregation later.
            fama = fama_score(memory_presence_correct, memory_presence_total,
                              forgetting_absence_correct, forgetting_absence_total)

            task_type = question_data.get('task_type', 'Unknown')

            per_judge_metrics = {}
            if self.use_multi_judge:
                for judge_name in self.judge_models.keys():
                    judge_correct = 0
                    judge_total = 0
                    for er in evaluation_results:
                        per_judge = er['evaluation_result'].get('per_judge_results', {})
                        if judge_name in per_judge:
                            judge_total += 1
                            if per_judge[judge_name].get('is_correct', False):
                                judge_correct += 1
                    per_judge_metrics[judge_name] = {
                        'correct': judge_correct,
                        'total': judge_total,
                        'accuracy': judge_correct / judge_total if judge_total > 0 else 0.0
                    }

            result = {
                'question_id': question_id,
                'question': question_text,
                'task_type': task_type,
                'session_id': question_data.get('session_id'),
                'session_date': question_data.get('session_date'),
                'model_response': model_response,
                'expected_answer': question_data.get('answer', ''),
                'evaluation_questions': evaluation_results,

                'total_eval_questions': total_count,
                'per_judge_metrics': per_judge_metrics,

                'fama': fama,
                'memory_presence_accuracy': memory_presence_accuracy,
                'memory_presence_correct': memory_presence_correct,
                'memory_presence_total': memory_presence_total,
                'forgetting_absence_accuracy': forgetting_absence_accuracy,
                'forgetting_absence_correct': forgetting_absence_correct,
                'forgetting_absence_total': forgetting_absence_total,

                'memory_evidence': question_data.get('memory_evidence', {}),
                'forgetting_evidence': question_data.get('forgetting_evidence', {}),
                'timestamp': datetime.now().isoformat(),
            }

            print(f"Overall Accuracy: {accuracy:.2%} ({correct_count}/{total_count})  FAMA={fama:.2%}")
            if self.use_multi_judge and per_judge_metrics:
                print("  Per-Judge:")
                for judge_name, metrics in per_judge_metrics.items():
                    print(f"    {judge_name}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
            if memory_presence_total > 0:
                print(f"  Memory Presence: {memory_presence_accuracy:.2%} ({memory_presence_correct}/{memory_presence_total})")
            if forgetting_absence_total > 0:
                print(f"  Forgetting Absence: {forgetting_absence_accuracy:.2%} ({forgetting_absence_correct}/{forgetting_absence_total})")
            
            results.append(result)
        
        return results
    
    def _generate_report(self, results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        total_questions = len(results)
        total_eval_questions = sum(r['total_eval_questions'] for r in results)
        
        # Calculate separate metrics for memory_presence and forgetting_absence
        total_memory_presence = sum(r['memory_presence_total'] for r in results)
        total_memory_presence_correct = sum(r['memory_presence_correct'] for r in results)
        memory_presence_accuracy = total_memory_presence_correct / total_memory_presence if total_memory_presence > 0 else None
        
        total_forgetting_absence = sum(r['forgetting_absence_total'] for r in results)
        total_forgetting_absence_correct = sum(r['forgetting_absence_correct'] for r in results)
        forgetting_absence_accuracy = total_forgetting_absence_correct / total_forgetting_absence if total_forgetting_absence > 0 else None

        # Overall accuracy across all evaluation sub-questions (matches agent-eval schema).
        total_correct_evaluations = total_memory_presence_correct + total_forgetting_absence_correct
        overall_accuracy = (total_correct_evaluations / total_eval_questions) if total_eval_questions > 0 else 0.0

        # Per task type analysis (Remembering, Reasoning, Recommending). Only the
        # paper's three tasks are tracked — there is no question_type taxonomy.
        by_task_type = {t: {
            'total_questions': 0,
            'total_eval_questions': 0,
            'correct_count': 0,
            'accuracies': [],
            'memory_presence_total': 0,
            'memory_presence_correct': 0,
            'forgetting_absence_total': 0,
            'forgetting_absence_correct': 0,
            'fama_per_question': [],
        } for t in TASK_TYPES}

        for result in results:
            task_type = result.get('task_type', 'Unknown')
            bucket = by_task_type.setdefault(task_type, {
                'total_questions': 0, 'total_eval_questions': 0, 'correct_count': 0,
                'accuracies': [], 'memory_presence_total': 0, 'memory_presence_correct': 0,
                'forgetting_absence_total': 0, 'forgetting_absence_correct': 0,
                'fama_per_question': [],
            })
            evaluation_questions = result.get('evaluation_questions', [])
            correct_count = sum(1 for eq in evaluation_questions
                                if eq.get('evaluation_result', {}).get('is_correct', False))
            accuracy = (correct_count / result['total_eval_questions']) if result['total_eval_questions'] > 0 else 0.0

            bucket['total_questions'] += 1
            bucket['total_eval_questions'] += result['total_eval_questions']
            bucket['correct_count'] += correct_count
            bucket['accuracies'].append(accuracy)
            bucket['memory_presence_total'] += result['memory_presence_total']
            bucket['memory_presence_correct'] += result['memory_presence_correct']
            bucket['forgetting_absence_total'] += result['forgetting_absence_total']
            bucket['forgetting_absence_correct'] += result['forgetting_absence_correct']
            bucket['fama_per_question'].append(result.get('fama', 0.0))

        # Per-task aggregates: average per-question FAMA, then × 100 (paper §4.2).
        for task_type, stats in by_task_type.items():
            stats['accuracy'] = stats['correct_count'] / stats['total_eval_questions'] if stats['total_eval_questions'] > 0 else 0.0
            stats['avg_question_accuracy'] = sum(stats['accuracies']) / len(stats['accuracies']) if stats['accuracies'] else 0.0
            stats['memory_presence_accuracy'] = stats['memory_presence_correct'] / stats['memory_presence_total'] if stats['memory_presence_total'] > 0 else None
            stats['forgetting_absence_accuracy'] = stats['forgetting_absence_correct'] / stats['forgetting_absence_total'] if stats['forgetting_absence_total'] > 0 else None
            stats['fama'] = (sum(stats['fama_per_question']) / len(stats['fama_per_question'])) * 100 if stats['fama_per_question'] else 0.0
            del stats['fama_per_question']

        # Overall FAMA across all questions (table 3 row aggregate).
        all_fama = [r.get('fama', 0.0) for r in results]
        overall_fama = (sum(all_fama) / len(all_fama)) * 100 if all_fama else 0.0
        
        # Per-judge aggregated statistics
        per_judge_overall = {}
        per_judge_by_task_type = {}
        
        if self.use_multi_judge:
            for judge_name in self.judge_models.keys():
                judge_correct = 0
                judge_total = 0
                
                # Overall stats
                for result in results:
                    judge_metrics = result.get('per_judge_metrics', {}).get(judge_name, {})
                    judge_correct += judge_metrics.get('correct', 0)
                    judge_total += judge_metrics.get('total', 0)
                
                per_judge_overall[judge_name] = {
                    'correct': judge_correct,
                    'total': judge_total,
                    'accuracy': judge_correct / judge_total if judge_total > 0 else 0.0
                }
                
                # Per task type stats for each judge
                per_judge_by_task_type[judge_name] = {}
                for task_type in by_task_type.keys():
                    task_correct = 0
                    task_total = 0
                    for result in results:
                        if result.get('task_type') == task_type:
                            judge_metrics = result.get('per_judge_metrics', {}).get(judge_name, {})
                            task_correct += judge_metrics.get('correct', 0)
                            task_total += judge_metrics.get('total', 0)
                    per_judge_by_task_type[judge_name][task_type] = {
                        'correct': task_correct,
                        'total': task_total,
                        'accuracy': task_correct / task_total if task_total > 0 else 0.0
                    }
        
        report = {
            'metadata': {
                'evaluation_timestamp': datetime.now().isoformat(),
                'model_name': self.model_name,
                'sessions_directory': str(self.sessions_dir),
                'original_metadata': metadata,
                'use_multi_judge': self.use_multi_judge,
                'judge_models': self.judge_models if self.use_multi_judge else None,
            },
            'overall_metrics': {
                'total_questions': total_questions,
                'total_evaluation_questions': total_eval_questions,
                'total_correct_evaluations': total_correct_evaluations,
                'overall_accuracy': overall_accuracy,

                # Forgetting-Aware Memory Accuracy (paper §4.2), normalized to [0, 100].
                'fama': overall_fama,

                'memory_presence_total': total_memory_presence,
                'memory_presence_correct': total_memory_presence_correct,
                'memory_presence_accuracy': memory_presence_accuracy,

                'forgetting_absence_total': total_forgetting_absence,
                'forgetting_absence_correct': total_forgetting_absence_correct,
                'forgetting_absence_accuracy': forgetting_absence_accuracy,
            },
            'per_judge_overall': per_judge_overall,
            'per_judge_by_task_type': per_judge_by_task_type,
            'by_task_type': by_task_type,
            'detailed_results': results,
            'context_metadata': getattr(self, 'context_metadata', {}),
        }

        return report
    
    def _save_results(self, results: List[Dict[str, Any]], report: Dict[str, Any]):
        """Save evaluation results and report (aligned with agent-based evaluation structure)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results with metadata wrapper (same structure as agent-based evaluation)
        results_data = {
            'metadata': {
                'model_name': self.model_name,
                'sessions_directory': str(self.sessions_dir),
                'total_questions': len(results),
                'generation_timestamp': datetime.now().isoformat(),
                'use_multi_judge': self.use_multi_judge,
                'judge_models': self.judge_models if self.use_multi_judge else None,
            },
            'results': results
        }
        
        results_file = self.output_dir / f"eval_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Save summary report
        report_file = self.output_dir / f"eval_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Summary report saved to: {report_file}")
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print evaluation summary"""
        metrics = report['overall_metrics']
        
        print(f"\nTotal Questions: {metrics['total_questions']}")
        print(f"Total Evaluation Questions: {metrics['total_evaluation_questions']}")
        print(f"Overall FAMA: {metrics.get('fama', 0.0):.2f} / 100")

        if self.use_multi_judge and 'per_judge_overall' in report:
            print("\n⚖️  Per-Judge Overall Scores:")
            for judge_name, stats in report['per_judge_overall'].items():
                model_name = self.judge_models.get(judge_name, judge_name)
                print(f"  {judge_name} ({model_name}):")
                print(f"    Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

        print("\n📊 By Evaluation Type:")
        if metrics['memory_presence_accuracy'] is not None:
            print(f"  Memory Presence: {metrics['memory_presence_accuracy']:.2%} ({metrics['memory_presence_correct']}/{metrics['memory_presence_total']})")
        if metrics['forgetting_absence_accuracy'] is not None:
            print(f"  Forgetting Absence: {metrics['forgetting_absence_accuracy']:.2%} ({metrics['forgetting_absence_correct']}/{metrics['forgetting_absence_total']})")

        print("\n🎯 By Task Type:")
        for task_type in TASK_TYPES:
            if task_type not in report.get('by_task_type', {}):
                continue
            stats = report['by_task_type'][task_type]
            if stats['total_questions'] == 0:
                continue
            print(f"  {task_type}:  questions={stats['total_questions']}  FAMA={stats.get('fama', 0.0):.2f}")
            if self.use_multi_judge and 'per_judge_by_task_type' in report:
                for judge_name in self.judge_models.keys():
                    js = report['per_judge_by_task_type'].get(judge_name, {}).get(task_type, {})
                    if js:
                        print(f"    {judge_name}: {js['accuracy']:.2%} ({js['correct']}/{js['total']})")
            if stats['memory_presence_accuracy'] is not None:
                print(f"    Memory Presence: {stats['memory_presence_accuracy']:.2%} ({stats['memory_presence_correct']}/{stats['memory_presence_total']})")
            if stats['forgetting_absence_accuracy'] is not None:
                print(f"    Forgetting Absence: {stats['forgetting_absence_accuracy']:.2%} ({stats['forgetting_absence_correct']}/{stats['forgetting_absence_total']})")


def main():
    parser = argparse.ArgumentParser(
        description="Model-based memory evaluation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--sessions-dir',
        type=str,
        required=True,
        help='Path to a persona directory (e.g., data/weekly/academic_researcher); must contain conversations/ and evaluation_questions_<persona>.json'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='google/gemini-2.5-flash',
        help='Model name to use (default: google/gemini-2.5-flash)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (default: sessions_dir/eval_results)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of questions to evaluate (useful for testing)'
    )
    
    # Multi-judge configuration
    parser.add_argument(
        '--no-multi-judge',
        action='store_true',
        help='Disable multi-judge evaluation (use single judge with same model as answerer)'
    )
    
    parser.add_argument(
        '--judge-openai',
        type=str,
        default='openai/gpt-4.1',
        help='OpenAI judge model (default: openai/gpt-4.1)'
    )
    
    parser.add_argument(
        '--judge-anthropic',
        type=str,
        default='anthropic/claude-haiku-4.5',
        help='Anthropic judge model (default: anthropic/claude-haiku-4.5)'
    )
    
    parser.add_argument(
        '--judge-google',
        type=str,
        default='google/gemini-2.5-flash',
        help='Google judge model (default: google/gemini-2.5-flash)'
    )
    
    # Reasoning mode configuration (for thinking/reasoning models)
    parser.add_argument(
        '--reasoning-effort',
        type=str,
        choices=['high', 'medium', 'low', 'minimal', 'none'],
        help='Reasoning effort level for models that support it (OpenAI o-series, GPT-5, Grok)'
    )
    
    parser.add_argument(
        '--reasoning-max-tokens',
        type=int,
        help='Max tokens for reasoning (Anthropic, Gemini thinking, some Qwen models)'
    )
    
    args = parser.parse_args()

    # All models route through OpenRouter — same provider used in data generation
    # (see conversation_generation/prompt_manager.py).
    from api_client import OpenRouterClient
    api_client = OpenRouterClient(model=args.model)
    
    # Configure judge models
    use_multi_judge = not args.no_multi_judge
    judge_models = {
        "openai": args.judge_openai,
        "anthropic": args.judge_anthropic,
        "google": args.judge_google
    }
    
    # Configure reasoning mode (for thinking/reasoning models)
    # See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    reasoning_config = None
    if args.reasoning_effort:
        reasoning_config = {"effort": args.reasoning_effort}
    elif args.reasoning_max_tokens:
        reasoning_config = {"max_tokens": args.reasoning_max_tokens}
    
    print("\n" + "=" * 80)
    print("Evaluation Configuration")
    print("=" * 80)
    print(f"Model being evaluated: {args.model}")
    print(f"Multi-judge evaluation: {'Enabled' if use_multi_judge else 'Disabled'}")
    if use_multi_judge:
        print("Judge models:")
        for name, model in judge_models.items():
            print(f"  - {name}: {model}")
    if reasoning_config:
        print(f"Reasoning mode: {reasoning_config}")
    print("=" * 80)
    
    # Run evaluation
    runner = EvaluationRunner(
        sessions_dir=args.sessions_dir,
        model_name=args.model,
        api_client=api_client,
        output_dir=args.output_dir,
        limit=args.limit,
        use_multi_judge=use_multi_judge,
        judge_models=judge_models,
        reasoning_config=reasoning_config
    )
    
    runner.run_evaluation()


if __name__ == '__main__':
    main()

