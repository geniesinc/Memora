#!/usr/bin/env python3
"""
STEP 2 OF 2: Question Answering Using Memory System
==================================================

This script answers questions by retrieving relevant memories from any supported
memory system and generating contextual answers using OpenAI.

SUPPORTED MEMORY SYSTEMS (six released agents):
- a_mem      A-Mem (local ChromaDB)
- langmem    LangMem (local LangGraph store)
- mem_0      Mem0 (cloud)
- memobase   Memobase (cloud)
- memos      MemOS (cloud or self-hosted)
- nemori     Nemori (local)

PIPELINE OVERVIEW:
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Store Conversations (conversation_to_memory.py)   │
│  ├─ Input: Conversation JSON files                         │
│  ├─ Process: Send to memory system with user_id             │
│  └─ Output: Memories stored in memory system               │
├─────────────────────────────────────────────────────────────┤
│  STEP 2: Answer Questions (THIS SCRIPT)                     │
│  ├─ Input: Questions JSON file + SAME user_id              │
│  ├─ Process: Retrieve memories & generate answers           │
│  └─ Output: Answers with evaluation metrics                 │
└─────────────────────────────────────────────────────────────┘

IMPORTANT: Use the SAME user_id and memory system that was used in Step 1!

How It Works:
1. Loads questions from unified questions file
2. For each question, searches memory system for relevant memories (using user_id)
3. Generates contextual answer using OpenAI based on retrieved memories
4. Compares generated answer with expected answer
5. Saves results with detailed evaluation metrics

Features:
- Automatic memory search for each question
- OpenAI-based answer generation from memories
- Quality evaluation and scoring (success rate, memory retrieval rate)
- Detailed result tracking with full memory context
- Support for filtering by question type, category, or session range
- Progress tracking and periodic saves

Prerequisites:
- Step 1 must be completed (conversations stored in memory system)
- Same user_id and memory system as Step 1
- OPENAI_API_KEY environment variable set
- Appropriate API keys for the selected memory system

Usage Examples:
    # Released layout: data/<period>/<persona>/, user_id="<persona>_<period>"
    # Use the SAME user_id and memory system that was used in Step 1.
    python memory_to_answer.py \\
        data/weekly/academic_researcher/evaluation_questions_academic_researcher.json \\
        --system mem_0 --user-id academic_researcher_weekly

    # Test with a smaller batch
    python memory_to_answer.py \\
        data/weekly/academic_researcher/evaluation_questions_academic_researcher.json \\
        --system a_mem --user-id academic_researcher_weekly --limit 5

    # Answer questions from a session-id range only
    python memory_to_answer.py \\
        data/weekly/academic_researcher/evaluation_questions_academic_researcher.json \\
        --system mem_0 --user-id academic_researcher_weekly --session-range 1-50
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from base_evaluator import get_memory_system, list_available_systems
except ImportError as e:
    print(f"Error importing base_evaluator: {e}")
    sys.exit(1)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not available. Install with: pip install openai")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Default judge models for multi-judge evaluation (same as model-based evaluation)
DEFAULT_JUDGE_MODELS = {
    "openai": "openai/gpt-4.1",
    "anthropic": "anthropic/claude-haiku-4.5",
    "google": "google/gemini-2.5-flash"
}

# The benchmark uses three tasks: Remembering, Reasoning, Recommending
# (paper §3.5 / Figure 1). No further question_type taxonomy is reported.
TASK_TYPES = ("Remembering", "Reasoning", "Recommending")


def model_eval_dir() -> Path:
    """Filesystem path to the shared ``model_eval`` package.

    ``api_client.OpenRouterClient`` (the multi-judge backend) lives in
    ``evals/model_eval/``. This file is ``evals/agent_eval/memory_to_answer.py``,
    so the directory is one level up from this file's parent:
    ``Path(__file__).parent.parent / "model_eval"`` — NOT
    ``Path(__file__).parent / "model_eval"`` (which would point at the
    non-existent ``evals/agent_eval/model_eval`` and silently break the
    multi-judge import). See ``test_judge_import.py``.
    """
    return Path(__file__).resolve().parent.parent / "model_eval"


def import_openrouter_client():
    """Resolve and import ``OpenRouterClient`` from the shared model_eval dir.

    Adds :func:`model_eval_dir` to ``sys.path`` (if needed) and returns the
    ``OpenRouterClient`` class. Importing the class does NOT require any API
    key or network access (the key is only read in its ``__init__``), so this
    function is safe to call from tests. Raises ``ImportError`` if the path is
    wrong or the module cannot be found.
    """
    judge_path = str(model_eval_dir())
    if judge_path not in sys.path:
        sys.path.append(judge_path)
    from api_client import OpenRouterClient  # noqa: E402
    return OpenRouterClient


def fama_score(memory_presence_correct: int, memory_presence_total: int,
               forgetting_absence_correct: int, forgetting_absence_total: int) -> float:
    """
    Forgetting-Aware Memory Accuracy (paper §4.2).

        FAMA = max(0, MPA − λ·(1 − FAA))
        MPA  = memory_presence_correct / memory_presence_total
        FAA  = forgetting_absence_correct / forgetting_absence_total
        λ    = N_forget / (N_presence + N_forget)

    Per-question FAMA ∈ [0, 1].
    """
    n_p = memory_presence_total
    n_f = forgetting_absence_total
    if n_p == 0 and n_f == 0:
        return 0.0
    mpa = (memory_presence_correct / n_p) if n_p > 0 else 0.0
    faa = (forgetting_absence_correct / n_f) if n_f > 0 else 1.0
    lam = n_f / (n_p + n_f) if (n_p + n_f) > 0 else 0.0
    return max(0.0, mpa - lam * (1.0 - faa))


class MemoryQuestionAnswering:
    """
    STEP 2: Memory-based question answering system using any supported memory system.
    
    This class retrieves memories from the memory system (stored in Step 1) and generates
    answers to questions using OpenAI.
    
    CRITICAL: The user_id and memory system must match the ones used when storing
    conversations in Step 1 (conversation_to_memory.py).
    """
    
    def __init__(self, memory_system_name: str, user_id: str, model: str = "gpt-4o-mini", output_dir: Optional[Path] = None,
                 judge_models: Optional[Dict[str, str]] = None, use_multi_judge: bool = True,
                 strict_judges: bool = False):
        """
        Initialize the question answering system.

        Args:
            memory_system_name: Name of the memory system (e.g., 'mem_0', 'a_mem')
            user_id: User ID for memory access (MUST match Step 1!)
            model: OpenAI model to use for answer generation
            output_dir: Directory to save results (default: current directory)
            judge_models: Dict mapping judge name to model name for multi-judge evaluation
            use_multi_judge: Whether to use multi-judge evaluation (default: True)
            strict_judges: If True, abort (raise RuntimeError) instead of silently
                degrading to a single judge when multi-judge was requested but the
                judge clients cannot be initialized. Results produced by a single
                judge are NOT comparable to the published Table 3 numbers, so repro
                runs should pass strict_judges=True (CLI: --strict-judges).
        """
        self.memory_system_name = memory_system_name
        self.user_id = user_id
        self.model = model
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_multi_judge = use_multi_judge
        self.strict_judges = strict_judges
        self.judge_models = judge_models or DEFAULT_JUDGE_MODELS
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is not available. Please install: pip install openai")
        
        # Load environment variables from .env file if available (same as conversation_to_memory.py)
        if DOTENV_AVAILABLE:
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                print(f"✅ Loaded environment variables from {env_file}")
            else:
                load_dotenv()
        else:
            print("⚠️  python-dotenv not available, using system environment variables only")
        
        # Initialize memory system
        try:
            self.memory_system = get_memory_system(memory_system_name, user_id)
            
            # Validate environment and initialize client
            if not self.memory_system.validate_environment():
                raise RuntimeError(f"Environment validation failed for {memory_system_name}. Check required environment variables.")
            
            if not self.memory_system.initialize_client():
                raise RuntimeError(f"Failed to initialize {memory_system_name} client. Check API keys and configuration.")
            
            print(f"✅ Connected to {memory_system_name} with user ID: {user_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {memory_system_name} connection: {e}")
        
        # Initialize OpenAI client (for answer generation)
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        print(f"✅ Connected to OpenAI with model: {model}")
        
        # Initialize judge clients for evaluation (same as model-based evaluation)
        self.judge_clients = {}
        if self.use_multi_judge:
            print(f"\n🔍 Initializing Multi-Judge Evaluation:")
            # Import OpenRouter client from the shared model_eval package.
            # NOTE: api_client lives in evals/model_eval/, i.e. one level ABOVE
            # this file's directory — see import_openrouter_client()/model_eval_dir().
            try:
                OpenRouterClient = import_openrouter_client()

                for judge_name, judge_model in self.judge_models.items():
                    try:
                        # Use OpenRouterClient properly (same as model_based_evaluator.py)
                        client = OpenRouterClient(model=judge_model)
                        self.judge_clients[judge_name] = client

                        # Verify client configuration
                        client_type = type(client).__name__
                        client_base_url = getattr(client.client, 'base_url', 'unknown')
                        print(f"  ✓ Initialized judge: {judge_name} ({judge_model})")
                        print(f"    Client type: {client_type}, Base URL: {client_base_url}")
                    except Exception as e:
                        print(f"  ✗ Failed to initialize judge {judge_name}: {e}")

                if not self.judge_clients:
                    self._handle_multi_judge_failure(
                        "No judge clients could be initialized (check OPENROUTER_API_KEY "
                        "and judge model names)."
                    )
            except ImportError as e:
                self._handle_multi_judge_failure(
                    f"Could not import OpenRouter client from "
                    f"{model_eval_dir()}: {e}"
                )

    def _handle_multi_judge_failure(self, reason: str) -> None:
        """Handle a multi-judge initialization failure LOUDLY.

        Multi-judge is the published Table 3 protocol (majority vote of three
        judges). Silently degrading to a single judge produces numbers that are
        NOT comparable to Table 3, so we make the failure impossible to miss:
        a prominent banner is always printed, and if ``strict_judges`` is set
        the run aborts instead of continuing with one judge.
        """
        banner = (
            "\n" + "!" * 72 + "\n"
            "!!  MULTI-JUDGE UNAVAILABLE\n"
            f"!!  Reason: {reason}\n"
            "!!  Requested multi-judge evaluation could NOT be initialized.\n"
            "!!  Results below would use a SINGLE judge and are NOT comparable\n"
            "!!  to the published Table 3 (3-judge majority vote).\n"
            + "!" * 72
        )
        print(banner, file=sys.stderr)
        if self.strict_judges:
            raise RuntimeError(
                "Multi-judge evaluation was requested but is unavailable, and "
                "--strict-judges is set. Aborting rather than silently producing "
                f"single-judge (non-Table-3-comparable) results. Reason: {reason}"
            )
        print(
            "  ⚠️  Falling back to a SINGLE OpenAI judge "
            "(pass --strict-judges to make this a hard failure instead).",
            file=sys.stderr,
        )
        self.use_multi_judge = False
        
        # Statistics tracking
        self.stats = {
            'total_questions': 0,
            'answered': 0,
            'failed': 0,
            'no_memories_found': 0,
            'with_memories_found': 0,
            'total_evaluations': 0,
            'passed_evaluations': 0,
            'questions_with_evaluations': 0,
            # New accuracy metrics
            'memory_presence_total': 0,
            'memory_presence_passed': 0,
            'forgetting_absence_total': 0,
            'forgetting_absence_passed': 0
        }
    
    def load_questions(self, questions_file: str) -> Dict[str, Any]:
        """
        Load questions from evaluation_questions or unified_questions file.
        Supports both formats and normalizes to a common structure.
        
        Args:
            questions_file: Path to questions JSON file
            
        Returns:
            Dict containing questions data (normalized format)
        """
        questions_path = Path(questions_file)
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        try:
            with open(questions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if it's evaluation_questions format
            if 'questions' in data and isinstance(data['questions'], dict):
                # This is evaluation_questions format - normalize it
                print(f"📂 Loaded evaluation_questions from: {questions_file}")
                normalized = self._normalize_evaluation_questions(data)
                print(f"   Total questions: {len(self._extract_all_questions_from_normalized(normalized))}")
                print(f"   Question types: {list(normalized.get('questions_by_type', {}).keys())}")
                return normalized
            else:
                # This is unified_questions format (or already normalized)
                print(f"📂 Loaded unified_questions from: {questions_file}")
                print(f"   Total questions: {data['metadata'].get('total_questions', 0)}")
                print(f"   Question types: {list(data.get('questions_by_type', {}).keys())}")
                return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load questions file: {e}")
    
    def _normalize_evaluation_questions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize evaluation_questions format to match unified_questions structure.
        
        evaluation_questions format:
        {
            "questions": {
                "remembering": [...],
                "reasoning": [...],
                "recommending": [...]
            },
            "persona": "...",
            "session_directory": "...",
            ...
        }
        
        unified_questions format:
        {
            "questions_by_type": { question_type: [...] },
            "questions_by_task_type": { task_type: [...] },
            "metadata": {...}
        }
        """
        normalized = {
            "metadata": data.get("metadata", {}),
            "questions_by_type": {},
            "questions_by_task_type": {
                "Remembering": [],
                "Reasoning": [],
                "Recommending": []
            }
        }
        
        # Extract questions from evaluation_questions format
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
                # Stamp task_type onto the question so downstream code (the per-question
                # result dict, by_task_type aggregation) can read it.
                question['task_type'] = proper_task_type
                all_questions.append(question)

                # Released data is task-keyed; questions_by_type is kept only for
                # backwards-compatibility with the legacy unified_questions format.
                question_type = question.get("question_type", "unknown")
                if question_type not in normalized["questions_by_type"]:
                    normalized["questions_by_type"][question_type] = []
                normalized["questions_by_type"][question_type].append(question)

                if proper_task_type in normalized["questions_by_task_type"]:
                    normalized["questions_by_task_type"][proper_task_type].append(question)
        
        # Add metadata from evaluation_questions if available
        if "persona" in data:
            normalized["metadata"]["persona"] = data["persona"]
        if "session_directory" in data:
            normalized["metadata"]["session_directory"] = data["session_directory"]
        if "generation_date" in data:
            normalized["metadata"]["generation_date"] = data["generation_date"]
        if "selection_criteria" in data:
            normalized["metadata"]["selection_criteria"] = data["selection_criteria"]
        if "date_range" in data:
            normalized["metadata"]["date_range"] = data["date_range"]
        if "config_file" in data:
            normalized["metadata"]["config_file"] = data["config_file"]
        
        # Add total_questions count
        normalized["metadata"]["total_questions"] = len(all_questions)
        
        print(f"   Normalized {len(all_questions)} questions")
        print(f"   Remembering: {len(normalized['questions_by_task_type']['Remembering'])}")
        print(f"   Reasoning: {len(normalized['questions_by_task_type']['Reasoning'])}")
        print(f"   Recommending: {len(normalized['questions_by_task_type']['Recommending'])}")
        
        return normalized
    
    def _extract_all_questions_from_normalized(self, normalized_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Helper to extract all questions from normalized format"""
        all_questions = []
        if 'questions_by_type' in normalized_data:
            for question_type, questions in normalized_data['questions_by_type'].items():
                for question in questions:
                    if 'question_type' not in question:
                        question['question_type'] = question_type
                    all_questions.append(question)
        return all_questions
    
    def extract_all_questions(self, questions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all questions from the questions data structure.
        
        Args:
            questions_data: Loaded questions data
            
        Returns:
            List of all questions
        """
        all_questions = []
        
        # Handle unified questions format
        if 'questions_by_type' in questions_data:
            for question_type, questions in questions_data['questions_by_type'].items():
                for question in questions:
                    # Ensure question_type is set
                    if 'question_type' not in question:
                        question['question_type'] = question_type
                    all_questions.append(question)
        
        # Also check for 'all_questions' key (alternative format)
        # elif 'all_questions' in questions_data:
        #     all_questions = questions_data['all_questions']
        
        return all_questions
    
    def filter_questions(self,
                        questions: List[Dict[str, Any]],
                        session_range: Optional[Tuple[int, int]] = None,
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Filter questions by session range and/or limit."""
        filtered = questions

        if session_range:
            start, end = session_range
            filtered = [q for q in filtered
                        if start <= q.get('session_id', 0) <= end]
            print(f"🎯 Filtered by session range [{start}, {end}]: {len(filtered)} questions")

        if limit:
            filtered = filtered[:limit]
            print(f"🎯 Limited to first {limit} questions")

        return filtered
    
    def search_memories(self, question: str, limit: int = 50, session_date: Optional[str] = None,
                      date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Search memory system for relevant memories using the system's search method.
        
        Note: session_date is passed as context to help semantic search understand temporal
        queries (like "this week", "last month"), but we do NOT filter by date.
        Semantic search handles temporal queries naturally through the question text itself.
        
        Args:
            question: Question to search for (may include temporal references like "this week")
            limit: Maximum number of memories to return
            session_date: Not used - kept for API compatibility
            date_range: Not used - kept for API compatibility
            
        Returns:
            List of relevant memories in format: [{'memory': str, 'score': float, ...}]
        """
        try:
            # Use the memory system's search_memories method
            # session_date is passed for context (to help understand temporal queries), not for filtering
            memories = self.memory_system.search_memories(
                question, 
                limit=limit,
                session_date=session_date,
                date_range=date_range
            )
            date_info = ""
            if session_date:
                date_info = f" (context: {session_date})"
            print(f"   🔍 {self.memory_system_name} search found {len(memories)} memories{date_info}")
            return memories if memories else []
            
        except Exception as e:
            print(f"⚠️  Memory search failed: {e}")
            print(f"   Error details: {str(e)}")
            return []
    
    def generate_answer(self, 
                       question: str, 
                       memories: List[Dict[str, Any]],
                       question_context: Dict[str, Any]) -> Optional[str]:
        """
        Generate an answer using OpenAI based on retrieved memories.
        
        Args:
            question: The question to answer
            memories: Retrieved memories from Mem0
            question_context: Additional context about the question
            
        Returns:
            Generated answer or None if generation fails
        """
        if not memories:
            return "I don't have enough information in my memory to answer this question accurately."
        
        # Format memories for the prompt (use all extracted memories)
        memory_context = []
        for i, memory in enumerate(memories, 1):  # Use all memories
            memory_text = memory.get('memory', str(memory))
            score = memory.get('score', 0)
            memory_context.append(f"{i}. {memory_text} (relevance: {score:.2f})")
        
        memory_text = "\n".join(memory_context)

        system_prompt = f"""You are a helpful AI assistant with access to a user's personal memory system.
        
Your task is to answer questions based ONLY on the user's stored memories and preferences.

Guidelines:
1. Use ONLY the information from the provided memories
2. Be specific and reference actual items/preferences from memory
3. If the question is about recommendations, suggest based on similar items in memory
4. Be conversational and helpful
5. Don't make up information not in the memories"""

        user_prompt = f"""User's Question: {question}

User's Relevant Memories:
{memory_text}

Please provide a helpful answer based on these memories."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            generated_answer = response.choices[0].message.content.strip()
            return generated_answer
            
        except Exception as e:
            print(f"⚠️  Answer generation failed: {e}")
            return None
    
    def _evaluate_with_single_judge(self, generated_answer: str, eval_question: str, 
                                     expected_answer: str, client, model: str) -> Dict[str, Any]:
        """Evaluate with a single judge using OpenRouterClient"""
        system_prompt = """You are an expert evaluator assessing AI assistant responses. Your task is to answer a YES/NO evaluation question about a given response.

You must provide your answer in the following JSON format:
{
    "answer": "yes" or "no",
    "confidence": 0.0 to 1.0,
    "explanation": "Brief explanation of your reasoning"
}

Be objective and thorough in your evaluation."""

        user_prompt = f"""Please evaluate the following AI response against the evaluation question.

AI RESPONSE TO EVALUATE:
{generated_answer}

EVALUATION QUESTION:
{eval_question}

Provide your evaluation in JSON format with answer (yes/no), confidence (0.0-1.0), and explanation."""

        try:
            # Use OpenRouterClient's generate_response method (same as model_based_evaluator.py)
            if hasattr(client, 'generate_response'):
                # This is an OpenRouterClient
                response_text = client.generate_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.0
                )
            else:
                # Fallback to raw OpenAI client (for single judge mode)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.0
                )
                response_text = response.choices[0].message.content
            
            # Parse JSON response (same as model_based_evaluator.py)
            response_text = response_text.strip()
            
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
                'llm_answer': llm_answer,
                'is_correct': is_correct,
                'confidence': confidence,
                'explanation': explanation
            }
            
        except json.JSONDecodeError as e:
            # Fallback: try to extract yes/no from text
            if 'response_text' in locals():
                response_lower = response_text.lower()
                if 'yes' in response_lower and 'no' not in response_lower:
                    llm_answer = 'yes'
                elif 'no' in response_lower and 'yes' not in response_lower:
                    llm_answer = 'no'
                else:
                    llm_answer = 'unclear'
                
                is_correct = (llm_answer == expected_answer.lower())
                
                return {
                    'llm_answer': llm_answer,
                    'is_correct': is_correct,
                    'confidence': 0.5,
                    'explanation': f"Parse error, inferred from text: {response_text[:200]}",
                    'parse_error': str(e)
                }
            else:
                return {
                    'llm_answer': 'error',
                    'is_correct': False,
                    'confidence': 0.0,
                    'error': str(e)
                }
        except Exception as e:
            return {
                'llm_answer': 'error',
                'is_correct': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def evaluate_answer_with_llm(self, 
                                generated_answer: str, 
                                evaluation_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use LLM judges to evaluate the generated answer against evaluation questions.
        Uses multi-judge evaluation with majority voting (same as model-based evaluation).
        
        Args:
            generated_answer: The generated answer to evaluate
            evaluation_questions: List of evaluation questions (expecting yes/no answers)
            
        Returns:
            List of evaluation results (same format as model-based evaluation)
        """
        evaluation_results = []
        
        for eval_q in evaluation_questions:
            eval_question = eval_q.get('evaluation_question', '')
            expected_answer = eval_q.get('expected_answer', 'yes').lower()
            eval_id = eval_q.get('evaluation_question_id', 'unknown')
            
            if self.use_multi_judge and self.judge_clients:
                # Multi-judge evaluation with majority voting
                per_judge_results = {}
                
                for judge_name, judge_client in self.judge_clients.items():
                    judge_model = self.judge_models[judge_name]
                    result = self._evaluate_with_single_judge(
                        generated_answer, eval_question, expected_answer,
                        judge_client, judge_model  # model param kept for fallback to raw client
                    )
                    per_judge_results[judge_name] = result
                
                # Aggregate results (majority vote)
                valid_results = [r for r in per_judge_results.values() if r['llm_answer'] != 'error']
                
                if valid_results:
                    # Majority vote
                    correct_votes = sum(1 for r in valid_results if r['is_correct'])
                    is_correct = correct_votes > len(valid_results) / 2
                    
                    # Consensus answer
                    yes_votes = sum(1 for r in valid_results if r['llm_answer'] == 'yes')
                    no_votes = sum(1 for r in valid_results if r['llm_answer'] == 'no')
                    consensus_answer = 'yes' if yes_votes > no_votes else 'no'
                    
                    # Agreement rate
                    max_agreement = max(yes_votes, no_votes)
                    agreement_rate = max_agreement / len(valid_results) if valid_results else 0.0
                else:
                    is_correct = False
                    consensus_answer = 'error'
                    agreement_rate = 0.0
                
                # Build evaluation_result nested structure (matching model-based format)
                eval_result = {
                    'is_correct': is_correct,
                    'confidence': sum(r['confidence'] for r in valid_results) / len(valid_results) if valid_results else 0.0,
                    'explanation': f"Majority vote: {correct_votes}/{len(valid_results)} judges agreed with expected answer" if valid_results else "All judges failed",
                    'llm_answer': consensus_answer,
                    'per_judge_results': per_judge_results,
                    'agreement_rate': agreement_rate,
                    'num_judges': len(self.judge_clients),
                    'num_valid_judges': len(valid_results),
                    'correct_votes': correct_votes if valid_results else 0
                }
                
                # Use nested structure matching model-based format
                evaluation_results.append({
                    **eval_q,  # Include all original eval_q fields (evaluation_question_id, evaluation_question, expected_answer, evaluation_type, target_item, context)
                    'evaluation_result': eval_result
                })
            else:
                # Single judge evaluation (fallback)
                result = self._evaluate_with_single_judge(
                    generated_answer, eval_question, expected_answer,
                    self.openai_client, self.model
                )
                
                # Build evaluation_result nested structure (matching model-based format)
                eval_result = {
                    'is_correct': result['is_correct'],
                    'confidence': result.get('confidence', 0.0),
                    'explanation': result.get('explanation', ''),
                    'llm_answer': result['llm_answer']
                }
                if result.get('error'):
                    eval_result['error'] = result['error']
                
                # Use nested structure matching model-based format
                evaluation_results.append({
                    **eval_q,  # Include all original eval_q fields
                    'evaluation_result': eval_result
                })
        
        return evaluation_results
    
    def answer_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer a single question using memory retrieval and generation.
        
        Args:
            question_data: Question data dictionary
            
        Returns:
            Result dictionary with answer and evaluation
        """
        question_id = question_data.get('question_id', 'unknown')
        question = question_data.get('question', '')
        expected_answer = question_data.get('answer', '')
        
        # Extract evaluation questions - they're nested under "evaluation" key
        evaluation_data = question_data.get('evaluation', {})
        evaluation_questions = evaluation_data.get('evaluation_questions', [])
        
        # Debug: Print evaluation questions info
        print(f"   📋 Found {len(evaluation_questions)} evaluation questions for {question_id}")
        
        # Get session_date from question_data to pass as context (not for filtering)
        # This helps semantic search understand temporal context for queries like "this week"
        session_date = question_data.get('session_date')
        
        # Search memories using semantic search (session_date passed for context, not filtering)
        memories = self.search_memories(question, session_date=session_date)
        
        # Generate answer
        generated_answer = self.generate_answer(question, memories, question_data)
        
        # Evaluate answer with LLM judge if evaluation questions exist
        evaluation_results = []
        if generated_answer and evaluation_questions:
            print(f"   🤖 Running LLM evaluation on {len(evaluation_questions)} questions...")
            evaluation_results = self.evaluate_answer_with_llm(generated_answer, evaluation_questions)
            print(f"   ✅ Completed {len(evaluation_results)} evaluations")
        
        # Calculate evaluation metrics (same as model-based evaluation)
        total_evaluations = len(evaluation_results)
        passed_evaluations = sum(1 for eval_result in evaluation_results if eval_result.get('evaluation_passed', False))
        evaluation_score = (passed_evaluations / total_evaluations) if total_evaluations > 0 else 0.0
        
        # Calculate memory retention and forgetting accuracy metrics
        memory_presence_total = 0
        memory_presence_passed = 0
        forgetting_absence_total = 0
        forgetting_absence_passed = 0
        
        for eval_result in evaluation_results:
            eval_type = eval_result.get('evaluation_type', '')
            # Extract is_correct from nested evaluation_result structure
            eval_result_data = eval_result.get('evaluation_result', {})
            eval_passed = eval_result_data.get('is_correct', False)
            
            if eval_type == 'memory_presence':
                memory_presence_total += 1
                if eval_passed:
                    memory_presence_passed += 1
            elif eval_type == 'forgetting_absence':
                forgetting_absence_total += 1
                if eval_passed:
                    forgetting_absence_passed += 1
        
        # Calculate individual accuracy scores
        memory_retention_accuracy = (memory_presence_passed / memory_presence_total) if memory_presence_total > 0 else None
        forgetting_accuracy = (forgetting_absence_passed / forgetting_absence_total) if forgetting_absence_total > 0 else None
        
        # Calculate per-judge metrics (same as model-based evaluation)
        per_judge_metrics = {}
        if self.use_multi_judge and self.judge_clients:
            for judge_name in self.judge_models.keys():
                judge_correct = 0
                judge_total = 0
                for eval_result in evaluation_results:
                    # Extract per_judge_results from nested evaluation_result structure
                    eval_result_data = eval_result.get('evaluation_result', {})
                    per_judge = eval_result_data.get('per_judge_results', {})
                    if judge_name in per_judge:
                        judge_total += 1
                        if per_judge[judge_name].get('is_correct', False):
                            judge_correct += 1
                per_judge_metrics[judge_name] = {
                    'correct': judge_correct,
                    'total': judge_total,
                    'accuracy': judge_correct / judge_total if judge_total > 0 else 0.0
                }
        
        task_type = question_data.get('task_type', 'Unknown')
        fama = fama_score(memory_presence_passed, memory_presence_total,
                          forgetting_absence_passed, forgetting_absence_total)

        result = {
            'question_id': question_id,
            'question': question,
            'task_type': task_type,
            'session_id': question_data.get('session_id'),
            'session_date': question_data.get('session_date'),
            'model_response': generated_answer,
            'expected_answer': expected_answer,
            'evaluation_questions': evaluation_results,

            # Agent-specific fields (memory retrieval info)
            'memories_retrieved': len(memories),
            'memories': memories,

            'total_eval_questions': total_evaluations,
            'per_judge_metrics': per_judge_metrics,

            'fama': fama,
            'memory_presence_accuracy': memory_retention_accuracy,
            'memory_presence_correct': memory_presence_passed,
            'memory_presence_total': memory_presence_total,
            'forgetting_absence_accuracy': forgetting_accuracy,
            'forgetting_absence_correct': forgetting_absence_passed,
            'forgetting_absence_total': forgetting_absence_total,

            'memory_evidence': question_data.get('memory_evidence', {}),
            'forgetting_evidence': question_data.get('forgetting_evidence', {}),

            'timestamp': datetime.now().isoformat(),
        }
        
        # Update statistics
        self.stats['total_questions'] += 1
        if generated_answer:
            self.stats['answered'] += 1
        else:
            self.stats['failed'] += 1
        
        if len(memories) == 0:
            self.stats['no_memories_found'] += 1
        else:
            self.stats['with_memories_found'] += 1
        
        # Update evaluation statistics
        if evaluation_results:
            self.stats['questions_with_evaluations'] += 1
            self.stats['total_evaluations'] += total_evaluations
            self.stats['passed_evaluations'] += passed_evaluations
            # Update accuracy-specific statistics
            self.stats['memory_presence_total'] += memory_presence_total
            self.stats['memory_presence_passed'] += memory_presence_passed
            self.stats['forgetting_absence_total'] += forgetting_absence_total
            self.stats['forgetting_absence_passed'] += forgetting_absence_passed
        
        return result
    
    def process_questions(self,
                         questions: List[Dict[str, Any]],
                         save_interval: int = 20) -> List[Dict[str, Any]]:
        """
        Process all questions and generate answers.
        
        Args:
            questions: List of questions to process
            save_interval: Save results every N questions
            
        Returns:
            List of results (may be partial if interrupted)
        """
        results = []
        
        print(f"\n🚀 Starting question answering process...")
        print(f"   Total questions to process: {len(questions)}")
        print("=" * 70)
        
        try:
            for i, question_data in enumerate(questions, 1):
                question_id = question_data.get('question_id', f'q_{i}')
                question = question_data.get('question', '')
                
                print(f"\n📝 [{i}/{len(questions)}] Processing: {question_id}")
                print(f"   Question: {question[:80]}{'...' if len(question) > 80 else ''}")
                
                try:
                    result = self.answer_question(question_data)
                    results.append(result)
                    
                    memories_found = result['memories_retrieved']
                    total_evals = result.get('total_eval_questions', 0)
                    # Calculate accuracy and correct_count from evaluation_questions
                    evaluation_questions = result.get('evaluation_questions', [])
                    correct_count = sum(1 for eq in evaluation_questions 
                                      if eq.get('evaluation_result', {}).get('is_correct', False))
                    accuracy = (correct_count / total_evals) if total_evals > 0 else 0.0
                    memory_ret_acc = result.get('memory_presence_accuracy')
                    forgetting_acc = result.get('forgetting_absence_accuracy')
                    per_judge_metrics = result.get('per_judge_metrics', {})
                    
                    print(f"   ✅ Answered | Memories: {memories_found} | Overall: {accuracy:.2%} ({correct_count}/{total_evals})")
                    
                    # Print per-judge metrics if using multi-judge
                    if self.use_multi_judge and per_judge_metrics:
                        print("      Per-Judge:")
                        for judge_name, metrics in per_judge_metrics.items():
                            print(f"        {judge_name}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
                    
                    # Print memory metrics
                    if memory_ret_acc is not None:
                        memory_correct = result.get('memory_presence_correct', 0)
                        memory_total = result.get('memory_presence_total', 0)
                        print(f"      Memory Presence: {memory_ret_acc:.2%} ({memory_correct}/{memory_total})")
                    if forgetting_acc is not None:
                        forgetting_correct = result.get('forgetting_absence_correct', 0)
                        forgetting_total = result.get('forgetting_absence_total', 0)
                        print(f"      Forgetting Absence: {forgetting_acc:.2%} ({forgetting_correct}/{forgetting_total})")
                
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    results.append({
                        'question_id': question_id,
                        'question': question,
                        'task_type': question_data.get('task_type', 'Unknown'),
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                    })
                    self.stats['failed'] += 1
                
                # Periodic save (outside inner try-except)
                if i % save_interval == 0:
                    self.save_results(results, suffix='_partial')
                    print(f"\n💾 Progress saved ({i}/{len(questions)} completed)")
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Processing interrupted at question {i}/{len(questions)}")
            # Save partial results before re-raising
            if results:
                print(f"💾 Saving partial results...")
                self.save_results(results, suffix='_interrupted')
            raise
        except Exception as e:
            print(f"\n❌ Error during processing at question {i}/{len(questions)}: {e}")
            # Save partial results before re-raising
            if results:
                print(f"💾 Saving partial results...")
                self.save_results(results, suffix='_error')
            raise
        
        return results
    
    def _clean_for_json(self, obj: Any) -> Any:
        """Recursively clean data structure to ensure JSON serializability."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert non-serializable types to string
            try:
                return str(obj)
            except:
                return None
    
    def save_results(self, results: List[Dict[str, Any]], suffix: str = '') -> str:
        """
        Save results to JSON file.
        
        Args:
            results: List of question-answer results
            suffix: Optional suffix for filename
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include memory system name in filename (same format as model-based evaluation)
        output_file = self.output_dir / f"eval_results_{timestamp}{suffix}.json"
        report_file = self.output_dir / f"eval_report_{timestamp}{suffix}.json"
        
        # Save detailed results (aligned with model-based evaluation structure)
        results_data = {
            'metadata': {
                'memory_system': self.memory_system_name,
                'user_id': self.user_id,
                'model': self.model,
                'total_questions': len(results),
                'generation_timestamp': datetime.now().isoformat(),
                'use_multi_judge': self.use_multi_judge,
                'judge_models': self.judge_models if self.use_multi_judge else None,
                'statistics': self.stats,
            },
            'results': results
        }
        
        # Clean data to ensure JSON serializability
        try:
            results_data = self._clean_for_json(results_data)
        except Exception as clean_error:
            print(f"⚠️  Warning: Error cleaning data for JSON: {clean_error}")
        
        # Generate report similar to model-based evaluation format
        try:
            report_data = self._generate_report({
                'metadata': results_data['metadata'],
                'results': results_data['results']
            })
            report_data = self._clean_for_json(report_data)
        except Exception as report_error:
            print(f"⚠️  Warning: Error generating report: {report_error}")
            report_data = {'error': str(report_error)}
        
        try:
            # Save detailed results with atomic write (write to temp file first, then rename)
            temp_output_file = output_file.with_suffix('.tmp.json')
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            temp_output_file.replace(output_file)
            
            # Save summary report with atomic write
            temp_report_file = report_file.with_suffix('.tmp.json')
            with open(temp_report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            temp_report_file.replace(report_file)
            
            print(f"💾 Detailed results saved to: {output_file}")
            print(f"💾 Summary report saved to: {report_file}")
            return str(output_file)
            
        except (TypeError, ValueError) as json_error:
            print(f"⚠️  JSON serialization error: {json_error}")
            print(f"   Attempting to save with minimal data...")
            # Try saving with minimal data structure
            try:
                minimal_data = {
                    'metadata': {
                        'memory_system': self.memory_system_name,
                        'user_id': self.user_id,
                        'model': self.model,
                        'total_questions': len(results),
                        'generation_timestamp': datetime.now().isoformat(),
                        'error': f"Full save failed: {str(json_error)}"
                    },
                    'results': [
                        {
                            'question_id': r.get('question_id', 'unknown'),
                            'question': r.get('question', ''),
                            'error': 'Data serialization failed'
                        }
                        for r in results[:10]  # Save first 10 as sample
                    ]
                }
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal_data, f, indent=2, ensure_ascii=False)
                print(f"💾 Minimal results saved to: {output_file}")
                return str(output_file)
            except Exception as minimal_error:
                print(f"❌ Failed to save even minimal results: {minimal_error}")
                return ""
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _generate_report(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evaluation report matching model-based evaluation format EXACTLY"""
        results = results_data['results']
        
        # Calculate overall metrics (directly from results, same as model-based)
        total_questions = len(results)
        total_evaluations = sum(r.get('total_eval_questions', 0) for r in results)
        
        # Memory presence and forgetting absence metrics
        memory_presence_total = sum(r.get('memory_presence_total', 0) for r in results)
        memory_presence_passed = sum(r.get('memory_presence_correct', 0) for r in results)
        memory_presence_accuracy = (memory_presence_passed / memory_presence_total) if memory_presence_total > 0 else None
        
        forgetting_absence_total = sum(r.get('forgetting_absence_total', 0) for r in results)
        forgetting_absence_passed = sum(r.get('forgetting_absence_correct', 0) for r in results)
        forgetting_absence_accuracy = (forgetting_absence_passed / forgetting_absence_total) if forgetting_absence_total > 0 else None
        
        # Overall accuracy + FAMA across all sub-questions.
        total_correct_evaluations = memory_presence_passed + forgetting_absence_passed
        overall_accuracy = (total_correct_evaluations / total_evaluations) if total_evaluations > 0 else 0.0

        # Per task type analysis (Remembering, Reasoning, Recommending). The paper
        # tracks only these three tasks — no question_type taxonomy.
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
            accuracy = (correct_count / result.get('total_eval_questions', 1)) if result.get('total_eval_questions', 0) > 0 else 0.0
            bucket['total_questions'] += 1
            bucket['total_eval_questions'] += result.get('total_eval_questions', 0)
            bucket['correct_count'] += correct_count
            bucket['accuracies'].append(accuracy)
            bucket['memory_presence_total'] += result.get('memory_presence_total', 0)
            bucket['memory_presence_correct'] += result.get('memory_presence_correct', 0)
            bucket['forgetting_absence_total'] += result.get('forgetting_absence_total', 0)
            bucket['forgetting_absence_correct'] += result.get('forgetting_absence_correct', 0)
            bucket['fama_per_question'].append(result.get('fama', 0.0))

        for task_type, stats in by_task_type.items():
            stats['accuracy'] = stats['correct_count'] / stats['total_eval_questions'] if stats['total_eval_questions'] > 0 else 0.0
            stats['avg_question_accuracy'] = sum(stats['accuracies']) / len(stats['accuracies']) if stats['accuracies'] else 0.0
            stats['memory_presence_accuracy'] = stats['memory_presence_correct'] / stats['memory_presence_total'] if stats['memory_presence_total'] > 0 else None
            stats['forgetting_absence_accuracy'] = stats['forgetting_absence_correct'] / stats['forgetting_absence_total'] if stats['forgetting_absence_total'] > 0 else None
            stats['fama'] = (sum(stats['fama_per_question']) / len(stats['fama_per_question'])) * 100 if stats['fama_per_question'] else 0.0
            del stats['fama_per_question']

        all_fama = [r.get('fama', 0.0) for r in results]
        overall_fama = (sum(all_fama) / len(all_fama)) * 100 if all_fama else 0.0
        
        # Per-judge aggregated statistics (same structure as model-based evaluation)
        per_judge_overall = {}
        per_judge_by_task_type = {}
        
        if self.use_multi_judge and self.judge_clients:
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
        
        return {
            'metadata': {
                'evaluation_timestamp': datetime.now().isoformat(),
                'memory_system': self.memory_system_name,
                'user_id': self.user_id,
                'model': self.model,
                'use_multi_judge': self.use_multi_judge,
                'judge_models': self.judge_models if self.use_multi_judge else None,
            },
            'overall_metrics': {
                'total_questions': total_questions,
                'total_evaluation_questions': total_evaluations,
                'total_correct_evaluations': total_correct_evaluations,
                'overall_accuracy': overall_accuracy,
                # Forgetting-Aware Memory Accuracy (paper §4.2), normalized to [0, 100].
                'fama': overall_fama,
                'memory_presence_total': memory_presence_total,
                'memory_presence_correct': memory_presence_passed,
                'memory_presence_accuracy': memory_presence_accuracy,
                'forgetting_absence_total': forgetting_absence_total,
                'forgetting_absence_correct': forgetting_absence_passed,
                'forgetting_absence_accuracy': forgetting_absence_accuracy,
            },
            'per_judge_overall': per_judge_overall,
            'per_judge_by_task_type': per_judge_by_task_type,
            'by_task_type': by_task_type,
            'detailed_results': results,
        }
    
    def print_summary(self, report: Optional[Dict[str, Any]] = None):
        """Print processing summary statistics (matching model-based evaluation format)"""
        print(f"\n📊 Processing Summary")
        print("=" * 70)
        print(f"Total questions processed: {self.stats['total_questions']}")
        print(f"Successfully answered: {self.stats['answered']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Questions with memories found: {self.stats['with_memories_found']}")
        print(f"Questions with no memories: {self.stats['no_memories_found']}")
        print(f"Questions with evaluations: {self.stats['questions_with_evaluations']}")
        print(f"Total evaluation questions: {self.stats['total_evaluations']}")
        print(f"Passed evaluations: {self.stats['passed_evaluations']}")
        
        if self.stats['total_questions'] > 0:
            success_rate = (self.stats['answered'] / self.stats['total_questions']) * 100
            memory_rate = (self.stats['with_memories_found'] / self.stats['total_questions']) * 100
            print(f"\nSuccess rate: {success_rate:.1f}%")
            print(f"Memory retrieval rate: {memory_rate:.1f}%")
            
        # Per-judge overall scores (same as model-based evaluation)
        if report and self.use_multi_judge and 'per_judge_overall' in report:
            print(f"\n⚖️  Per-Judge Overall Scores:")
            for judge_name, stats in report['per_judge_overall'].items():
                model_name = self.judge_models.get(judge_name, judge_name)
                print(f"  {judge_name} ({model_name}):")
                print(f"    Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
            
        # Memory retention and forgetting accuracy
        print(f"\n📊 By Evaluation Type:")
        if self.stats['memory_presence_total'] > 0:
            memory_retention_accuracy = (self.stats['memory_presence_passed'] / self.stats['memory_presence_total'])
            print(f"  Memory Presence: {memory_retention_accuracy:.2%} ({self.stats['memory_presence_passed']}/{self.stats['memory_presence_total']})")
            
        if self.stats['forgetting_absence_total'] > 0:
            forgetting_accuracy = (self.stats['forgetting_absence_passed'] / self.stats['forgetting_absence_total'])
            print(f"  Forgetting Absence: {forgetting_accuracy:.2%} ({self.stats['forgetting_absence_passed']}/{self.stats['forgetting_absence_total']})")
        
        # By task type breakdown (same as model-based evaluation)
        if report and 'by_task_type' in report:
            print(f"\n📋 By Task Type:")
            task_type_order = ["Remembering", "Reasoning", "Recommending"]
            for task_type in task_type_order:
                if task_type in report['by_task_type']:
                    stats = report['by_task_type'][task_type]
                    if stats['total_questions'] == 0:
                        continue
                    print(f"  {task_type}:  questions={stats['total_questions']}  FAMA={stats.get('fama', 0.0):.2f}")
                    if stats['memory_presence_accuracy'] is not None:
                        print(f"    Memory Presence: {stats['memory_presence_accuracy']:.2%} ({stats['memory_presence_correct']}/{stats['memory_presence_total']})")
                    if stats['forgetting_absence_accuracy'] is not None:
                        print(f"    Forgetting Absence: {stats['forgetting_absence_accuracy']:.2%} ({stats['forgetting_absence_correct']}/{stats['forgetting_absence_total']})")
                    
                    # Per-judge by task type
                    if self.use_multi_judge and 'per_judge_by_task_type' in report:
                        print(f"    Per-Judge Accuracy:")
                        for judge_name in self.judge_models.keys():
                            judge_stats = report['per_judge_by_task_type'].get(judge_name, {}).get(task_type, {})
                            if judge_stats:
                                print(f"      {judge_name}: {judge_stats['accuracy']:.2%} ({judge_stats['correct']}/{judge_stats['total']})")
            
        if self.stats['questions_with_evaluations'] > 0:
            avg_evals_per_question = self.stats['total_evaluations'] / self.stats['questions_with_evaluations']
            print(f"\nAverage evaluations per question: {avg_evals_per_question:.1f}")


def extract_persona_and_timeline(questions_file: str) -> Tuple[str, str]:
    """
    Extract (persona, timeline) for a questions file in the released layout
    ``data/<timeline>/<persona>/evaluation_questions_<persona>.json``.

    Falls back to parsing the filename if the parent dirs do not match the layout.
    """
    TIMELINES = {"weekly", "monthly", "quarterly"}
    questions_path = Path(questions_file).resolve()
    persona_dir = questions_path.parent
    timeline_dir = persona_dir.parent

    persona = persona_dir.name
    timeline = timeline_dir.name if timeline_dir.name in TIMELINES else "weekly"

    if persona == "" or persona == "unknown_persona":
        m = re.search(r"evaluation_questions_(.+)\.json", questions_path.name)
        if m:
            persona = m.group(1)
        else:
            persona = "unknown_persona"

    return persona, timeline


def parse_range(range_str: str) -> Tuple[int, int]:
    """Parse a range string like '10-50' into a tuple (10, 50)."""
    try:
        start, end = range_str.split('-')
        return (int(start.strip()), int(end.strip()))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Range must be in format 'start-end', got: {range_str}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Memory-based question answering using any supported memory system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Released layout: data/<period>/<persona>/, user_id="<persona>_<period>".
  # Use the SAME user_id and memory system that was used in Step 1.
  python memory_to_answer.py \\
      data/weekly/academic_researcher/evaluation_questions_academic_researcher.json \\
      --system mem_0 --user-id academic_researcher_weekly

  # First 5 questions only (smoke test)
  python memory_to_answer.py \\
      data/weekly/academic_researcher/evaluation_questions_academic_researcher.json \\
      --system a_mem --user-id academic_researcher_weekly --limit 5

  # Restrict to a session-id range
  python memory_to_answer.py \\
      data/weekly/academic_researcher/evaluation_questions_academic_researcher.json \\
      --system mem_0 --user-id academic_researcher_weekly --session-range 1-50

Note: Requires OPENAI_API_KEY and appropriate API keys for the memory system.
      Available systems: """ + ", ".join(list_available_systems())
    )

    # Required arguments
    parser.add_argument(
        'questions_file',
        help='Path to evaluation_questions_<persona>.json file (released format)'
    )

    # Memory system - normalize to lowercase for validation
    available_systems = [s.lower() for s in list_available_systems()]
    parser.add_argument(
        '--system',
        type=str,
        required=True,
        choices=available_systems,
        help='Memory system to use (a_mem, langmem, mem_0, memobase, memos, nemori). '
             'Must match the system used in Step 1.'
    )
    
    # User ID
    parser.add_argument(
        '--user-id',
        type=str,
        help='User ID for memory access (default: from MEMORA_USER_ID env var, or "memora_user")'
    )
    
    # Filtering options
    parser.add_argument(
        '--session-range',
        type=parse_range,
        metavar='START-END',
        help='Filter by session range (e.g., --session-range 1-50)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of questions to process (e.g., --limit 5 to test with 5 questions)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model to use (default: gpt-4o-mini)'
    )
    
    # Multi-judge configuration (same as model-based evaluation)
    parser.add_argument(
        '--no-multi-judge',
        action='store_true',
        help='Disable multi-judge evaluation (use single judge)'
    )

    parser.add_argument(
        '--strict-judges',
        action='store_true',
        help='Abort instead of silently degrading to a single judge if '
             'multi-judge initialization fails. Use this for reproduction runs: '
             'single-judge results are NOT comparable to the published Table 3.'
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
    
    args = parser.parse_args()
    
    # Normalize system name to lowercase (matching get_memory_system behavior)
    args.system = args.system.lower()
    
    try:
        # Extract persona and timeline from questions file path
        persona, timeline = extract_persona_and_timeline(args.questions_file)
        
        # Generate user_id: {persona}_{timeline}
        user_id = args.user_id or os.getenv('MEMORA_USER_ID') or f"{persona}_{timeline}"
        
        # Determine output directory (same structure as model-based evaluation)
        questions_path = Path(args.questions_file).resolve()
        session_dir = questions_path.parent
        output_dir = session_dir / "eval_results" / args.system
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure judge models (same as model-based evaluation)
        use_multi_judge = not args.no_multi_judge
        judge_models = {
            "openai": args.judge_openai,
            "anthropic": args.judge_anthropic,
            "google": args.judge_google
        }
        
        print(f"🚀 STEP 2: Memory-Based Question Answering")
        print("=" * 70)
        print(f"This script answers questions using memories from {args.system}.")
        print(f"")
        print(f"⚠️  PREREQUISITE: Step 1 must be completed first!")
        print(f"   Run: python conversation_to_memory.py --system {args.system} --user-id {user_id}")
        print(f"")
        print("=" * 70)
        print(f"Memory System: {args.system}")
        print(f"Persona: {persona}")
        print(f"Timeline: {timeline}")
        print(f"User ID: {user_id} (must match Step 1!)")
        print(f"Model: {args.model}")
        print(f"Multi-judge evaluation: {'Enabled' if use_multi_judge else 'Disabled'}")
        if use_multi_judge:
            print(f"Judge Models:")
            for name, model in judge_models.items():
                print(f"  - {name}: {model}")
        print(f"Questions file: {args.questions_file}")
        print(f"Output directory: {output_dir}")
        print(f"")
        
        # Initialize QA system
        qa_system = MemoryQuestionAnswering(
            memory_system_name=args.system,
            user_id=user_id,
            model=args.model,
            output_dir=output_dir,
            judge_models=judge_models,
            use_multi_judge=use_multi_judge,
            strict_judges=args.strict_judges
        )
        
        # Load questions
        questions_data = qa_system.load_questions(args.questions_file)
        all_questions = qa_system.extract_all_questions(questions_data)
        
        # Filter questions
        filtered_questions = qa_system.filter_questions(
            all_questions,
            session_range=args.session_range,
            limit=args.limit,
        )
        
        if not filtered_questions:
            print("❌ No questions match the filter criteria")
            sys.exit(1)
        
        # Process questions with error handling to save partial results
        results = []
        try:
            results = qa_system.process_questions(filtered_questions)
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Processing interrupted by user!")
            if results:
                print(f"💾 Saving partial results ({len(results)}/{len(filtered_questions)} questions)...")
                try:
                    partial_file = qa_system.save_results(results, suffix='_interrupted')
                    print(f"✅ Partial results saved to: {partial_file}")
                except Exception as save_error:
                    print(f"⚠️  Failed to save partial results: {save_error}")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            if results:
                print(f"💾 Attempting to save partial results ({len(results)}/{len(filtered_questions)} questions)...")
                try:
                    partial_file = qa_system.save_results(results, suffix='_error')
                    print(f"✅ Partial results saved to: {partial_file}")
                except Exception as save_error:
                    print(f"⚠️  Failed to save partial results: {save_error}")
            sys.exit(1)
        
        # Save final results
        if results:
            try:
                output_file = qa_system.save_results(results)
                
                # Load report for summary
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = output_dir / f"eval_report_{timestamp}.json"
                report = None
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        report = json.load(f)
                
                # Print summary (with report for per-judge metrics)
                qa_system.print_summary(report)
                
                print(f"\n✅ Question answering completed!")
                print(f"📄 Results saved to: {output_file}")
            except Exception as save_error:
                print(f"❌ Failed to save final results: {save_error}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        else:
            print("❌ No results to save")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Processing interrupted by user!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

