"""
LLM-based Conversation Evaluator

This module uses OpenRouter to evaluate conversation quality by asking
specific yes/no questions about transition quality, memory operations,
and post-memory behavior.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import requests

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class EvaluationResult:
    """Result of LLM-based evaluation"""
    memory_operation_quality: bool
    overall_pass: bool
    llm_responses: List[str]
    evaluation_time: float
    evaluation_questions: List[str] = field(default_factory=list)
    evaluation_answers: List[str] = field(default_factory=list)

class LLMEvaluator:
    """
    Evaluates conversation quality using OpenRouter LLM.
    
    This evaluator:
    1. Generates specific evaluation questions
    2. Sends them to OpenRouter LLM
    3. Parses yes/no responses
    4. Determines overall quality pass/fail
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "anthropic/claude-sonnet-4.5"):
        self.api_key = api_key or os.getenv('OPEN_ROUTER_API_KEY')
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPEN_ROUTER_API_KEY environment variable.")
    
    def evaluate_conversation(self, session_data: Dict[str, Any], memory_segment: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate conversation quality using LLM.
        
        Args:
            session_data: Session metadata
            memory_segment: Extracted memory segment with context
            
        Returns:
            EvaluationResult with quality assessment
        """
        start_time = time.time()
        
        try:
            # Generate evaluation questions
            from evaluation_question_generator import EvaluationQuestionGenerator
            question_generator = EvaluationQuestionGenerator()
            questions = question_generator.generate_all_evaluation_questions(session_data, memory_segment)
            
            # Format questions for LLM
            formatted_questions = question_generator.format_questions_for_llm(questions)
            
            # Get conversation context
            conversation_context = self._extract_conversation_context(memory_segment)
            
            # Create evaluation prompt
            evaluation_prompt = self._create_evaluation_prompt(formatted_questions, conversation_context)
            
            # Send to LLM
            llm_response = self._query_llm(evaluation_prompt)
            
            # Parse response
            evaluation_results = question_generator.parse_llm_response(llm_response, questions)
            
            # Determine overall pass - now handles multiple questions
            overall_pass = all(evaluation_results.values())
            
            evaluation_time = time.time() - start_time
            
            # Extract questions and answers for debugging
            question_texts = [q.question for q in questions]
            answer_texts = [evaluation_results.get(f"question_{i}", False) for i, q in enumerate(questions)]
            
            return EvaluationResult(
                memory_operation_quality=overall_pass,  # Overall pass for memory operations
                overall_pass=overall_pass,
                llm_responses=[llm_response],
                evaluation_time=evaluation_time,
                evaluation_questions=question_texts,
                evaluation_answers=answer_texts
            )
            
        except Exception as e:
            print(f"❌ Error in LLM evaluation: {e}")
            return EvaluationResult(
                memory_operation_quality=False,
                overall_pass=False,
                llm_responses=[f"Error: {str(e)}"],
                evaluation_time=time.time() - start_time,
                evaluation_questions=[],
                evaluation_answers=[]
            )
    
    def _extract_conversation_context(self, memory_segment: Dict[str, Any]) -> str:
        """Extract relevant conversation context for evaluation"""
        context_parts = []
        
        # Context before
        if memory_segment.get('context_before'):
            before = memory_segment['context_before']
            agent = before.get('speaker', 'unknown').replace('_agent', '')
            context_parts.append(f"Context Before: {agent} - {before.get('message', '')}")
        
        # Memory turns
        memory_turns = memory_segment.get('memory_turns', [])
        for turn in memory_turns:
            agent = turn.get('speaker', 'unknown').replace('_agent', '')
            context_parts.append(f"Memory Turn: {agent} - {turn.get('message', '')}")
        
        # Context after
        if memory_segment.get('context_after'):
            after = memory_segment['context_after']
            agent = after.get('speaker', 'unknown').replace('_agent', '')
            context_parts.append(f"Context After: {agent} - {after.get('message', '')}")
        
        return "\n".join(context_parts)
    
    def _create_evaluation_prompt(self, questions: str, conversation_context: str) -> str:
        """Create the evaluation prompt for the LLM"""
        return f"""You are evaluating a conversation for quality. Please analyze the conversation and answer the questions with 'yes' or 'no' only.

IMPORTANT EVALUATION GUIDELINES:
- Look for the MEANING and INTENT of what the user said, not just exact phrasing
- Accept natural conversational language and paraphrasing
- If the user clearly communicated the required information, answer "yes" even if they didn't use specific keywords
- For example, "I just finished reading X" clearly indicates they have read X
- "I recently completed X" or "I've read X" are all valid ways to express having read something

CONVERSATION CONTEXT:
{conversation_context}

EVALUATION QUESTIONS:
{questions}

Please provide your answers in the format:
1. yes/no
2. yes/no
3. yes/no
etc.

Answer each question with 'yes' or 'no' only, based on whether the user CLEARLY COMMUNICATED the required information (not exact wording)."""
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM via OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://memora-conversation-evaluator",
            "X-Title": "Memora Conversation Evaluator"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert conversation evaluator. Analyze conversations and provide clear yes/no answers to evaluation questions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.0,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenRouter API error: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected API response format: {e}")


def test_llm_evaluator_with_file(conversation_file: str):
    """
    Test LLM evaluator with an existing conversation file.
    
    Args:
        conversation_file: Path to the conversation JSON file
    """
    import json
    from memory_slice import extract_memory_segment
    
    print(f"🔍 Testing LLM evaluator with: {conversation_file}")
    print("=" * 60)
    
    # Load the conversation file
    try:
        with open(conversation_file, 'r') as f:
            conversation_data = json.load(f)
        print(f"✅ Loaded conversation file")
    except Exception as e:
        print(f"❌ Error loading conversation file: {e}")
        return None
    
    # Display basic conversation info
    print(f"📊 Conversation Info:")
    print(f"   - Session ID: {conversation_data.get('session_id', 'N/A')}")
    print(f"   - Session Type: {conversation_data.get('session_type', 'N/A')}")
    print(f"   - Operation: {conversation_data.get('operation', 'N/A')}")
    print(f"   - Item: {conversation_data.get('item', 'N/A')}")
    print(f"   - Total turns: {len(conversation_data.get('conversation', []))}")
    
    # Extract session data
    session_data = {
        "session_id": conversation_data.get("session_id"),
        "session_type": conversation_data.get("session_type"),
        "category": conversation_data.get("category"),
        "subcategory": conversation_data.get("subcategory"),
        "preference_type": conversation_data.get("preference_type"),
        "item": conversation_data.get("item"),
        "operation": conversation_data.get("operation"),
        "operation_details": conversation_data.get("operation_details")
    }
    
    # Extract memory segment
    print(f"\n🔪 Extracting memory segment...")
    memory_segment = extract_memory_segment(conversation_data)
    
    if memory_segment is None:
        print("❌ No memory segment found in conversation")
        return None
    
    print(f"✅ Memory segment extracted successfully!")
    print(f"   - Memory turns: {len(memory_segment['memory_turns'])}")
    
    # Generate evaluation questions
    print(f"\n❓ Generating evaluation questions...")
    from evaluation_question_generator import EvaluationQuestionGenerator
    question_generator = EvaluationQuestionGenerator()
    questions = question_generator.generate_all_evaluation_questions(session_data, memory_segment)
    
    print(f"✅ Generated {len(questions)} evaluation question(s):")
    for i, question in enumerate(questions, 1):
        print(f"   {i}. [{question.evaluation_type}] {question.question}")
        print(f"      Expected: {question.expected_answer}")
    
    # Run LLM evaluation
    print(f"\n🧠 Running LLM evaluation...")
    try:
        llm_evaluator = LLMEvaluator()
        evaluation_result = llm_evaluator.evaluate_conversation(session_data, memory_segment)
        
        print(f"✅ LLM evaluation completed!")
        print(f"\n📊 Evaluation Results:")
        print(f"   - Overall Pass: {'✅ PASS' if evaluation_result.overall_pass else '❌ FAIL'}")
        print(f"   - Memory Operation Quality: {'✅ PASS' if evaluation_result.memory_operation_quality else '❌ FAIL'}")
        print(f"   - Evaluation Time: {evaluation_result.evaluation_time:.2f}s")
        
        print(f"\n📝 LLM Response:")
        print("-" * 50)
        print(evaluation_result.llm_responses[0])
        print("-" * 50)
        
        print(f"\n📋 Questions and Answers:")
        for i, (question, answer) in enumerate(zip(evaluation_result.evaluation_questions, evaluation_result.evaluation_answers), 1):
            status = "✅ YES" if answer else "❌ NO"
            print(f"   {i}. {status} - {question}")
        
        return evaluation_result
        
    except Exception as e:
        print(f"❌ Error in LLM evaluation: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 2:
        # Test with provided JSON file
        conversation_file = sys.argv[1]
        if not os.path.exists(conversation_file):
            print(f"❌ Conversation file not found: {conversation_file}")
            sys.exit(1)
        
        result = test_llm_evaluator_with_file(conversation_file)
        
        if result:
            print(f"\n🎉 LLM evaluation test completed successfully!")
            print(f"✅ Overall result: {'PASS' if result.overall_pass else 'FAIL'}")
        else:
            print(f"\n❌ LLM evaluation test failed")
    else:
        # Show usage
        print("Usage: python llm_evaluator.py <conversation_file>")
        print("Example: python llm_evaluator.py ../output/sessions_<timestamp>_<persona>/conversations/session_0002_activity_memory.json")
        print("\nThis will test the complete evaluation pipeline:")
        print("1. Extract memory segment from conversation")
        print("2. Generate evaluation questions")
        print("3. Send to LLM for evaluation")
        print("4. Parse and display results")