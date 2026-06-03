"""
Memory-Grounded Conversation Generator

Main system that orchestrates the complete memory-grounded conversation generation process.
Combines session processing, prompt management, and turn-by-turn generation.
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from session_processor import SessionProcessor, SessionInfo
from prompt_manager import PromptManager
from memory_slice import extract_memory_segment
from evaluation_question_generator import EvaluationQuestionGenerator
from llm_evaluator import LLMEvaluator

load_dotenv()


class MemoryGroundedGenerator:
    """Memory-grounded conversation generator with intent-based turn-by-turn control"""
    
    def __init__(self, model_name: str = "google/gemini-2.5-flash", enable_evaluation: bool = False, max_evaluation_iterations: int = 3): 
        self.model_name = model_name
        self.session_processor = None
        # Use the prompt manager that includes FlowManager and LLM generation
        self.prompt_manager = PromptManager(model_name=model_name)
        
        # Quality evaluation components
        self.enable_evaluation = enable_evaluation
        self.max_evaluation_iterations = max_evaluation_iterations
        self.question_generator = EvaluationQuestionGenerator()
        self.llm_evaluator = None
        
        # Initialize LLM evaluator if evaluation is enabled
        if self.enable_evaluation:
            try:
                self.llm_evaluator = LLMEvaluator()
                print("🔍 Quality evaluation enabled")
            except Exception as e:
                print(f"⚠️  Quality evaluation disabled: {e}")
                self.enable_evaluation = False
    
    def _evaluate_conversation_quality(self, conversation_data: Dict[str, Any], session: SessionInfo) -> Dict[str, Any]:
        """
        Evaluate conversation quality using LLM-based evaluation.
        
        Args:
            conversation_data: Generated conversation data
            session: Session information
            
        Returns:
            Dict with evaluation results and feedback
        """
        if not self.enable_evaluation or not self.llm_evaluator:
            return {
                "evaluation_enabled": False,
                "passed": True,
                "feedback": "Evaluation disabled"
            }
        
        # Skip evaluation for no_memory sessions since they don't have memory operations
        if session.memory_type == "no_memory":
            return {
                "evaluation_enabled": True,
                "passed": True,
                "feedback": "No evaluation needed for no_memory sessions",
                "evaluation_questions": [],
                "evaluation_answers": [],
                "evaluation_model": self.llm_evaluator.model,
                "evaluation_result": {
                    "memory_operation_quality": True,
                    "overall_pass": True,
                    "evaluation_time": 0.0
                }
            }
        
        try:
            print("🔍 Evaluating conversation quality...")
            
            # Extract memory segment
            memory_segment = extract_memory_segment(conversation_data)
            if not memory_segment:
                return {
                    "evaluation_enabled": True,
                    "passed": False,
                    "feedback": "No memory segment found for evaluation",
                    "error": "No memory segment"
                }
            
            # Create session data for evaluation
            session_data = {
                "session_id": session.session_id,
                "session_type": session.memory_type,
                "category": session.category,
                "subcategory": session.subcategory,
                "preference_type": session.preference_type,
                "item": session.item,
                "operation": session.operation,
                "operation_details": session.operation_details
            }
            
            # Evaluate conversation quality
            evaluation_result = self.llm_evaluator.evaluate_conversation(session_data, memory_segment)
            
            # Determine if passed
            passed = evaluation_result.overall_pass
            
            # Generate detailed feedback
            feedback = ""
            if not passed:
                # Convert boolean answers to string format for feedback generation
                answer_strings = ['yes' if ans else 'no' for ans in evaluation_result.evaluation_answers]
                feedback = self._generate_detailed_feedback(evaluation_result.evaluation_questions, answer_strings, session)
            
            return {
                "evaluation_enabled": True,
                "passed": passed,
                "feedback": feedback,
                "evaluation_questions": evaluation_result.evaluation_questions,
                "evaluation_answers": evaluation_result.evaluation_answers,
                "evaluation_model": self.llm_evaluator.model,
                "evaluation_result": {
                    "memory_operation_quality": evaluation_result.memory_operation_quality,
                    "overall_pass": evaluation_result.overall_pass,
                    "evaluation_time": evaluation_result.evaluation_time
                }
            }
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {
                "evaluation_enabled": True,
                "passed": False,
                "feedback": f"Evaluation failed: {str(e)}",
                "error": str(e)
            }
    
    def _generate_with_evaluation_feedback(self, session: SessionInfo, feedback: str, flow_ids: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Generate conversation with evaluation feedback.
        
        Args:
            session: Session information
            feedback: Evaluation feedback from previous attempt
            flow_ids: Optional flow IDs to use
            
        Returns:
            Generated conversation data
        """
        print(f"🔄 Re-generating with feedback: {feedback}")
        
        # Generate conversation with feedback integration
        return self.prompt_manager.generate_conversation(session, use_llm=True, flow_ids=flow_ids, feedback=feedback)
    
    def _generate_detailed_feedback(self, questions: List[str], answers: List[str], session: SessionInfo) -> str:
        """
        Generate detailed feedback using LLM analysis of evaluation questions and answers.
        
        Args:
            questions: List of evaluation questions
            answers: List of evaluation answers (yes/no)
            session: Session information
            
        Returns:
            Detailed feedback string generated by LLM
        """
        if not self.llm_evaluator:
            return "EVALUATION FEEDBACK - LLM evaluator not available for feedback generation."
        
        # Prepare the feedback generation prompt
        failed_questions = []
        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            if answer.lower() == "no":  # If answer is "no"
                failed_questions.append(f"Question {i}: {question}")
        
        if not failed_questions:
            return "EVALUATION FEEDBACK - All questions passed, but evaluation failed for unknown reasons."
        
        # Create feedback generation prompt
        feedback_prompt = f"""
You are an expert conversation quality evaluator. Analyze the failed evaluation questions and generate specific, actionable feedback for improving the conversation generation.

SESSION CONTEXT:
- Memory Type: {session.memory_type}
- Category: {session.category}
- Subcategory: {session.subcategory}
- Operation: {session.operation}
- Item: {session.item}

FAILED EVALUATION QUESTIONS:
{chr(10).join(failed_questions)}

TASK: Generate specific, actionable feedback that will help the LLM generate better conversations in the next attempt. Focus on:
1. What specific information the user should mention
2. How to make the conversation more natural and specific
3. What details are missing that caused the evaluation to fail

Provide concise, actionable feedback that directly addresses the evaluation failures.
"""

        try:
            # Use the same LLM method as the evaluator
            feedback = self.llm_evaluator._query_llm(feedback_prompt)
            return f"EVALUATION FEEDBACK:\n{feedback}"
            
        except Exception as e:
            # Fallback to basic feedback if LLM fails
            return f"EVALUATION FEEDBACK - LLM feedback generation failed: {str(e)}\nFailed questions: {'; '.join(failed_questions)}"
    
    def load_sessions(self, session_file_path: str) -> bool:
        """Load sessions from file"""
        self.session_processor = SessionProcessor(session_file_path)
        return self.session_processor.load_sessions()
    
    # Remove _call_llm and generate_turn methods since IntegratedPromptManager handles all LLM generation
    
    def generate_conversation(self, session: SessionInfo, flow_ids: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Generate complete memory-grounded conversation using integrated pipeline with quality evaluation
        
        Args:
            session: Session information for conversation generation
            flow_ids: Optional dict specifying exact flow IDs to use for each phase
                     Format: {"opening": 1, "exploration": 5, "memory": 3, "closing": 2}
        """
        
        print(f"🎭 Generating conversation for Session {session.session_id}")
        print(f"   Type: {session.memory_type} | Operation: {session.operation}")
        
        # Generate context display based on memory type
        if session.memory_type == "activity_memory":
            if session.operation_details:
                item_details = session.operation_details.get('item', {})
                if isinstance(item_details, dict) and 'expense_type' in item_details:
                    expense_type = item_details['expense_type']
                    amount = item_details.get('amount', '')
                    print(f"   Context: {session.category} ${amount} on {expense_type}")
                else:
                    print(f"   Context: {session.category} '{session.item}'")
            else:
                print(f"   Context: {session.category} '{session.item}'")
        else:
            # For preference/content/goal memory, use preference_type
            print(f"   Context: {session.category}/{session.subcategory} {session.preference_type} '{session.item}'")
        
        if flow_ids:
            print(f"🎯 Using specified flow IDs: {flow_ids}")
        
        # Quality evaluation loop
        evaluation_iterations = 0
        feedback_history = []
        
        # Initialize variables that will be used in the return statement
        result = None
        conversation_turns = []
        clean_conversation = []
        evaluation_result = None
        
        try:
            # If evaluation is disabled, just generate once
            if not self.enable_evaluation:
                result = self.prompt_manager.generate_conversation(session, use_llm=True, flow_ids=flow_ids)
                if not result.get('success'):
                    return {
                        "session_id": session.session_id,
                        "error": "Failed to generate conversation",
                        "success": False
                    }
                
                # Transform result to match expected format
                conversation_turns = []
                for turn in result.get('conversation_turns', []):
                    turn_record = {
                        "turn": turn['turn_number'],
                        "speaker": f"{turn['agent']}_agent",
                        "message": turn['content'],
                        "intent": turn['intent_id'],
                        "description": turn.get('description', ''),
                        "agent": turn['agent'],
                        "phase": turn['phase'],
                        "share_memory": turn['share_memory'],
                        "conversation_type": turn.get('conversation_type', 'general'),
                        "requires_instruction": turn.get('requires_instruction', False),
                        "instruction": turn['instruction'],
                        "prompt": turn.get('prompt', {})
                    }
                    conversation_turns.append(turn_record)
                
                # Create clean conversation-only section
                clean_conversation = []
                for turn in conversation_turns:
                    clean_turn = {
                        "turn": turn["turn"],
                        "speaker": turn["speaker"],
                        "message": turn["message"]
                    }
                    clean_conversation.append(clean_turn)
            else:
                # Evaluation enabled - use the evaluation loop
                while evaluation_iterations < self.max_evaluation_iterations:
                    evaluation_iterations += 1
                    
                    # Generate conversation
                    if evaluation_iterations == 1:
                        # First attempt - use original generation
                        result = self.prompt_manager.generate_conversation(session, use_llm=True, flow_ids=flow_ids)
                    else:
                        # Subsequent attempts - use feedback
                        result = self._generate_with_evaluation_feedback(session, feedback_history[-1], flow_ids)
                
                    if not result.get('success'):
                        return {
                            "session_id": session.session_id,
                            "error": "Failed to generate conversation",
                            "success": False
                        }
                    
                    # Transform result to match expected format (keep all prompt details)
                    conversation_turns = []
                    for turn in result.get('conversation_turns', []):
                        turn_record = {
                            "turn": turn['turn_number'],
                            "speaker": f"{turn['agent']}_agent",
                            "message": turn['content'],  # Fixed: use 'content' not 'text'
                            "intent": turn['intent_id'],
                            "description": turn.get('description', ''),
                            "agent": turn['agent'],
                            "phase": turn['phase'],
                            "share_memory": turn['share_memory'],
                            "conversation_type": turn.get('conversation_type', 'general'),
                            "requires_instruction": turn.get('requires_instruction', False),
                            "instruction": turn['instruction'],
                            "prompt": turn.get('prompt', {})  # Include full prompt details
                        }
                        conversation_turns.append(turn_record)
                    
                    # Create clean conversation-only section
                    clean_conversation = []
                    for turn in conversation_turns:
                        clean_turn = {
                            "turn": turn["turn"],
                            "speaker": turn["speaker"],
                            "message": turn["message"]
                        }
                        clean_conversation.append(clean_turn)
                    
                    # Create conversation data for evaluation
                    conversation_data = {
                        "session_id": session.session_id,
                        "session_type": session.memory_type,
                        "category": session.category,
                        "subcategory": session.subcategory,
                        "preference_type": session.preference_type,
                        "item": session.item,
                        "operation": session.operation,
                        "operation_details": session.operation_details,
                        "conversation": conversation_turns
                    }
                    
                    # Evaluate conversation quality if enabled
                    print(f"🔍 Evaluating conversation quality (attempt {evaluation_iterations}/{self.max_evaluation_iterations})...")
                    evaluation_result = self._evaluate_conversation_quality(conversation_data, session)
                    
                    if evaluation_result["passed"]:
                        print("✅ Quality evaluation passed!")
                        # Quality passed - break out of loop and return successful result
                        break
                    else:
                        print(f"❌ Quality evaluation failed: {evaluation_result['feedback']}")
                        feedback_history.append(evaluation_result['feedback'])
                        
                        if evaluation_iterations >= self.max_evaluation_iterations:
                            print(f"❌ Max evaluation iterations ({self.max_evaluation_iterations}) reached")
                            # Return failed result with evaluation details AND conversation data
                            return {
                                # Session metadata
                                "session_id": session.session_id,
                                "session_type": session.memory_type,
                                "category": session.category,
                                "subcategory": session.subcategory,
                                "preference_type": session.preference_type,
                                "item": session.item,
                                "operation": session.operation,
                                "operation_details": session.operation_details,
                                "date": session.date,
                                "persona": getattr(self.session_processor, 'persona', 'unknown') if self.session_processor else 'unknown',
                                
                                # Conversation data (for human review)
                                "conversation": conversation_turns,
                                "conversation_only": clean_conversation,
                                "total_turns": len(conversation_turns),
                                
                                # Generation metadata
                                "conversation_flow": result.get('flow_instructions', []),
                                "flow_metadata": result.get('flow_metadata', {}),
                                "model_used": self.model_name,
                                "generation_timestamp": datetime.now().isoformat(),
                                
                                # Evaluation failure details
                                "error": f"Quality evaluation failed after {self.max_evaluation_iterations} attempts",
                                "success": False,
                                "evaluation_failed": True,
                                "evaluation_enabled": True,
                                "evaluation_iterations": evaluation_iterations,
                                "evaluation_model": evaluation_result.get("evaluation_model", "unknown"),
                                "evaluation_feedback": feedback_history,
                                "evaluation_result": evaluation_result.get("evaluation_result", {}),
                                "evaluation_questions": evaluation_result.get("evaluation_questions", []),
                                "evaluation_answers": evaluation_result.get("evaluation_answers", [])
                            }
                        else:
                            print(f"🔄 Re-generating conversation with feedback...")
                            continue
                
            # Include comprehensive session metadata
            session_metadata = {
                "session_id": session.session_id,
                "session_type": session.memory_type,
                "category": session.category,
                "subcategory": session.subcategory,
                "preference_type": session.preference_type,
                "item": session.item,
                "operation": session.operation,
                "operation_details": session.operation_details,
                "date": session.date,
                "persona": getattr(self.session_processor, 'persona', 'unknown') if self.session_processor else 'unknown'
            }
            
            # Add evaluation metadata if evaluation was performed
            evaluation_metadata = {}
            if self.enable_evaluation:
                evaluation_metadata = {
                    "evaluation_enabled": True,
                    "evaluation_iterations": evaluation_iterations,
                    "evaluation_feedback_history": feedback_history,
                    "evaluation_model": evaluation_result.get("evaluation_model", "unknown") if evaluation_result else "unknown",
                    "evaluation_result": evaluation_result.get("evaluation_result", {}) if evaluation_result and evaluation_result.get("passed") else None,
                    "evaluation_questions": evaluation_result.get("evaluation_questions", []) if evaluation_result else [],
                    "evaluation_answers": evaluation_result.get("evaluation_answers", []) if evaluation_result else []
                }
            else:
                evaluation_metadata = {
                    "evaluation_enabled": False
                }
            
            return {
                # Session metadata (comprehensive)
                **session_metadata,
                
                # Conversation generation metadata
                "conversation_flow": result.get('flow_instructions', []),
                "flow_metadata": result.get('flow_metadata', {}),
                "model_used": self.model_name,
                "generation_timestamp": datetime.now().isoformat(),
                
                # Conversation data
                "conversation": conversation_turns,
                "conversation_only": clean_conversation,
                "total_turns": len(conversation_turns),
                
                # Evaluation metadata
                **evaluation_metadata,
                
                # Status
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Conversation generation failed: {e}")
            return {
                "session_id": session.session_id,
                "session_type": getattr(session, 'memory_type', 'unknown'),
                "error": str(e),
                "success": False
            }
    
    def generate_multiple_conversations(self, 
                                      session_file_path: str, 
                                      num_sessions: Optional[int] = None,
                                      specific_session_id: Optional[int] = None,
                                      flow_ids: Optional[Dict[str, int]] = None,
                                      memory_type_filter: Optional[str] = None,
                                      save_individual: bool = True,
                                      output_folder: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate conversations for multiple sessions
        
        Args:
            session_file_path: Path to sessions file
            num_sessions: Number of sessions to process (ignored if specific_session_id is provided)
            specific_session_id: Generate conversation for this specific session ID only
            flow_ids: Optional dict specifying exact flow IDs to use for each phase
            memory_type_filter: Optional memory type to filter sessions (e.g., 'no_memory', 'preference_memory')
            save_individual: Whether to save each conversation as individual file (default: True)
            output_folder: Optional output folder path to save conversations (if None, creates timestamped folder)
        """
        
        # Load sessions
        if not self.load_sessions(session_file_path):
            return []
        
        # Get sessions to process
        if specific_session_id is not None:
            # Find specific session
            sessions = self.session_processor.get_session_by_id(specific_session_id)
            if sessions is None:
                print(f"❌ Session ID {specific_session_id} not found")
                return []
            sessions = [sessions]  # Make it a list for consistent processing
            print(f"🎯 Processing specific session ID: {specific_session_id}")
        else:
            # If memory type filter is specified, get all sessions first, then filter, then limit
            if memory_type_filter:
                all_sessions = self.session_processor.get_sessions()  # Get all sessions
                filtered_sessions = [s for s in all_sessions if s.memory_type == memory_type_filter]
                print(f"🔍 Found {len(filtered_sessions)} {memory_type_filter} sessions (from {len(all_sessions)} total)")
                
                # Now limit to the requested number
                if num_sessions and len(filtered_sessions) > num_sessions:
                    sessions = filtered_sessions[:num_sessions]
                    print(f"📊 Limited to first {num_sessions} {memory_type_filter} sessions")
                else:
                    sessions = filtered_sessions
            else:
                if num_sessions is None:
                    # Process all sessions
                    sessions = self.session_processor.get_sessions()
                    print(f"📊 Processing ALL {len(sessions)} sessions from the file")
                else:
                    sessions = self.session_processor.get_sessions(num_sessions)
        
        # Determine output folder for individual files with success/failed separation
        conversation_output_folder = None
        successful_folder = None
        failed_folder = None
        
        if save_individual:
            if output_folder:
                # Use provided output folder and create conversations subdirectories
                successful_folder = os.path.join(output_folder, "conversations", "successful")
                failed_folder = os.path.join(output_folder, "conversations", "failed")
                os.makedirs(successful_folder, exist_ok=True)
                os.makedirs(failed_folder, exist_ok=True)
                conversation_output_folder = os.path.join(output_folder, "conversations")
                print(f"📁 Using provided output folder: {output_folder}")
                print(f"📁 Successful conversations: {successful_folder}")
                print(f"📁 Failed conversations: {failed_folder}")
            else:
                # Auto-detect metadata directory from session file path
                metadata_dir = self._detect_metadata_directory(session_file_path)
                if metadata_dir:
                    # Create conversations subdirectories within metadata directory
                    successful_folder = os.path.join(metadata_dir, "conversations", "successful")
                    failed_folder = os.path.join(metadata_dir, "conversations", "failed")
                    os.makedirs(successful_folder, exist_ok=True)
                    os.makedirs(failed_folder, exist_ok=True)
                    conversation_output_folder = os.path.join(metadata_dir, "conversations")
                    print(f"📁 Auto-detected metadata directory: {metadata_dir}")
                    print(f"📁 Successful conversations: {successful_folder}")
                    print(f"📁 Failed conversations: {failed_folder}")
                else:
                    # Fallback: Create timestamped folder in conversation_generation directory (legacy behavior)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filter_suffix = f"_{memory_type_filter}" if memory_type_filter else ""
                    successful_folder = f"conversations_{timestamp}{filter_suffix}_successful"
                    failed_folder = f"conversations_{timestamp}{filter_suffix}_failed"
                    os.makedirs(successful_folder, exist_ok=True)
                    os.makedirs(failed_folder, exist_ok=True)
                    conversation_output_folder = f"conversations_{timestamp}{filter_suffix}"
                    print(f"📁 Created timestamped output folders:")
                    print(f"📁 Successful: {successful_folder}")
                    print(f"📁 Failed: {failed_folder}")
        
        print(f"🎯 Memory-Grounded Conversation Generation")
        print(f"📁 Source: {session_file_path}")
        print(f"🤖 Model: {self.model_name}")
        print(f"📊 Processing {len(sessions)} sessions...")
        if save_individual and conversation_output_folder:
            print(f"💾 Individual files: {conversation_output_folder}/")
        print("=" * 80)
        
        conversations = []
        
        for i, session in enumerate(sessions, 1):
            print(f"\n📝 Session {i}/{len(sessions)}")
            
            try:
                conversation = self.generate_conversation(session, flow_ids=flow_ids)
                conversations.append(conversation)
                
                if conversation.get('success', False):
                    print(f"✅ Generated {conversation.get('total_turns', 0)} turns")
                    # Show memory grounding
                    print(f"🎯 Memory: {conversation.get('operation', '')} {conversation.get('item', '')}")
                else:
                    print(f"❌ Generation failed: {conversation.get('error', 'Unknown error')}")
                
                # Save individual file immediately (successful vs failed in separate folders)
                if save_individual:
                    # Ensure both folders exist
                    if successful_folder and failed_folder:
                        individual_filename = f"session_{session.session_id:04d}_{session.memory_type}.json"
                        
                        # Choose appropriate folder based on success/failure
                        if conversation.get('success', False):
                            target_folder = successful_folder
                            print(f"💾 Saved: {individual_filename}")
                        else:
                            target_folder = failed_folder
                            individual_filename = f"session_{session.session_id:04d}_{session.memory_type}_FAILED.json"
                            print(f"💾 Saved failed: {individual_filename}")
                        
                        if target_folder:
                            individual_path = os.path.join(target_folder, individual_filename)
                            try:
                                with open(individual_path, 'w', encoding='utf-8') as f:
                                    json.dump(conversation, f, indent=2, ensure_ascii=False)
                            except Exception as save_error:
                                print(f"⚠️  Failed to save individual file: {save_error}")
                    else:
                        print(f"⚠️  Warning: Folders not initialized - cannot save conversation")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                failed_conversation = {
                    "session_id": session.session_id,
                    "session_type": session.memory_type,
                    "error": str(e),
                    "success": False
                }
                conversations.append(failed_conversation)
                
                # Save failed conversation in failed folder
                if save_individual and failed_folder:
                    individual_filename = f"session_{session.session_id:04d}_{session.memory_type}_FAILED.json"
                    individual_path = os.path.join(failed_folder, individual_filename)
                    try:
                        with open(individual_path, 'w', encoding='utf-8') as f:
                            json.dump(failed_conversation, f, indent=2, ensure_ascii=False)
                        print(f"💾 Saved failed: {individual_filename}")
                    except Exception as save_error:
                        print(f"⚠️  Failed to save failed conversation file: {save_error}")
        
        return conversations
    
    def find_failed_sessions(self, output_folder: str) -> List[int]:
        """Find failed sessions from generation report (source of truth)
        
        Args:
            output_folder: Path to output folder containing failed sessions
            
        Returns:
            List of failed session IDs
        """
        report_path = os.path.join(output_folder, "conversations", "generation_report.json")
        
        # Try to load from generation report first
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    failed_sessions = report.get("failed_sessions", [])
                    if failed_sessions:
                        print(f"📋 Found {len(failed_sessions)} failed sessions from generation report")
                        return sorted(failed_sessions)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Fallback: scan failed folder if no report exists
        print("📋 No generation report found, scanning failed folder...")
        failed_folder = os.path.join(output_folder, "conversations", "failed")
        failed_sessions = []
        
        if os.path.exists(failed_folder):
            for filename in os.listdir(failed_folder):
                if filename.endswith("_FAILED.json"):
                    try:
                        session_id = int(filename.split("_")[1])
                        failed_sessions.append(session_id)
                    except (IndexError, ValueError):
                        continue
        
        return sorted(failed_sessions)
    
    def rerun_failed_sessions(self, session_file_path: str, output_folder: str, 
                              flow_ids: Optional[Dict[str, int]] = None,
                              enable_evaluation: bool = True, max_evaluation_iterations: int = 3,
                              max_retries: int = 3) -> Dict[str, Any]:
        """Rerun failed sessions with evaluation and retry logic
        
        Args:
            session_file_path: Path to sessions file
            output_folder: Path to output folder containing failed sessions
            flow_ids: Optional flow IDs to use
            enable_evaluation: Whether to enable evaluation for rerun
            max_evaluation_iterations: Maximum number of evaluation attempts per session
            max_retries: Maximum number of times to retry a failed session
            
        Returns:
            Dictionary with rerun results
        """
        print("🔍 Finding failed sessions...")
        failed_session_ids = self.find_failed_sessions(output_folder)
        
        if not failed_session_ids:
            print("✅ No failed sessions found!")
            return {"failed_sessions": [], "rerun_results": {}}
        
        print(f"📋 Found {len(failed_session_ids)} failed sessions: {failed_session_ids}")
        
        # Load sessions
        if not self.load_sessions(session_file_path):
            return {"failed_sessions": failed_session_ids, "rerun_results": {}}
        
        # Set evaluation settings for rerun
        original_max_iterations = self.max_evaluation_iterations
        original_enable_evaluation = self.enable_evaluation
        
        self.max_evaluation_iterations = max_evaluation_iterations
        self.enable_evaluation = enable_evaluation
        
        if enable_evaluation:
            print(f"🔍 Evaluation enabled with {max_evaluation_iterations} max iterations for rerun")
        else:
            print(f"⚠️  Evaluation disabled for rerun - sessions will not be quality checked")
        
        rerun_results = {}
        successful_folder = os.path.join(output_folder, "conversations", "successful")
        failed_folder = os.path.join(output_folder, "conversations", "failed")
        
        for session_id in failed_session_ids:
            print(f"\n🔄 Rerunning session {session_id} with evaluation...")
            
            # Get session
            session = self.session_processor.get_session_by_id(session_id)
            if not session:
                print(f"❌ Session {session_id} not found in session file")
                continue
            
            # Retry logic: try up to max_retries times
            conversation = None
            try:
                for retry_attempt in range(max_retries):
                    try:
                        if retry_attempt > 0:
                            print(f"🔄 Retry attempt {retry_attempt + 1}/{max_retries} for session {session_id}")
                        
                        # Generate conversation with evaluation
                        conversation = self.generate_conversation(session, flow_ids=flow_ids)
                        
                        # If successful, break out of retry loop
                        if conversation.get('success', False):
                            break
                            
                    except Exception as e:
                        print(f"❌ Error in retry attempt {retry_attempt + 1}: {e}")
                        if retry_attempt == max_retries - 1:
                            print(f"💥 All {max_retries} retry attempts failed for session {session_id}")
                            conversation = {
                                'success': False,
                                'error': f'Failed after {max_retries} retry attempts: {str(e)}',
                                'session_id': session_id,
                                'retry_attempts': retry_attempt + 1
                            }
                            break
                
                # Save in appropriate folder (keep all files for reference)
                if conversation and conversation.get('success', False):
                    target_folder = successful_folder
                    filename = f"session_{session_id:04d}_{session.memory_type}.json"
                    print(f"✅ Session {session_id} succeeded on retry!")
                else:
                    target_folder = failed_folder
                    filename = f"session_{session_id:04d}_{session.memory_type}_FAILED.json"
                    print(f"❌ Session {session_id} still failed after {max_retries} retries")
                
                # Save the conversation
                if target_folder:
                    file_path = os.path.join(target_folder, filename)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(conversation, f, indent=2, ensure_ascii=False)
                    print(f"💾 Saved: {filename}")
                
                rerun_results[session_id] = {
                    "success": conversation.get('success', False),
                    "turns": conversation.get('total_turns', 0),
                    "error": conversation.get('error', '') if not conversation.get('success', False) else None
                }
                
            except Exception as e:
                print(f"❌ Error rerunning session {session_id}: {e}")
                rerun_results[session_id] = {
                    "success": False,
                    "turns": 0,
                    "error": str(e)
                }
        
        # Update generation report with rerun results
        successful_rerun_sessions = []
        failed_rerun_sessions = []
        
        for session_id, result in rerun_results.items():
            if result.get("success", False):
                successful_rerun_sessions.append(session_id)
            else:
                failed_rerun_sessions.append(session_id)
        
        print(f"🔍 DEBUG: Rerun results dictionary: {rerun_results}")
        print(f"🔍 DEBUG: Rerun results - {len(successful_rerun_sessions)} successful, {len(failed_rerun_sessions)} failed")
        print(f"🔍 DEBUG: Successful sessions: {successful_rerun_sessions}")
        print(f"🔍 DEBUG: Failed sessions: {failed_rerun_sessions}")
        
        self.update_generation_report(output_folder, successful_rerun_sessions, failed_rerun_sessions)
        
        # Restore original evaluation settings
        self.max_evaluation_iterations = original_max_iterations
        self.enable_evaluation = original_enable_evaluation
        
        return {
            "failed_sessions": failed_session_ids,
            "rerun_results": rerun_results
        }
    
    def _update_report_after_rerun(self, output_folder: str, successful_sessions: List[int], failed_sessions: List[int]) -> None:
        """Update the generation report after rerun by moving successful sessions from failed to successful
        
        Args:
            output_folder: Path to the output folder
            successful_sessions: List of session IDs that succeeded in rerun
            failed_sessions: List of session IDs that failed in rerun
        """
        report_path = os.path.join(output_folder, "conversations", "generation_report.json")
        
        # Load existing report
        if not os.path.exists(report_path):
            print("❌ No generation report found")
            return
            
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("❌ Could not load generation report")
            return
        
        # Get current state
        current_successful = set(report.get("successful_sessions", []))
        current_failed = set(report.get("failed_sessions", []))
        
        print(f"📊 Before update: {len(current_successful)} successful, {len(current_failed)} failed")
        
        # Move successful sessions from failed to successful
        for session_id in successful_sessions:
            if session_id in current_failed:
                current_failed.remove(session_id)
                print(f"✅ Moved session {session_id} from failed to successful")
            if session_id not in current_successful:
                current_successful.add(session_id)
                print(f"✅ Added session {session_id} to successful")
        
        # Keep failed sessions in failed (they stay failed)
        for session_id in failed_sessions:
            if session_id not in current_failed:
                current_failed.add(session_id)
                print(f"❌ Session {session_id} remains failed")
        
        # Update report
        report["last_update_timestamp"] = datetime.now().isoformat()
        report["successful_sessions"] = sorted(list(current_successful))
        report["failed_sessions"] = sorted(list(current_failed))
        report["successful_count"] = len(current_successful)
        report["failed_count"] = len(current_failed)
        report["total_conversations"] = len(current_successful) + len(current_failed)
        total_sessions = len(current_successful) + len(current_failed)
        report["success_rate"] = (len(current_successful) / total_sessions * 100) if total_sessions > 0 else 0
        
        # Save updated report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save timestamped copy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_report_path = os.path.join(output_folder, "conversations", f"generation_report_{timestamp}.json")
        with open(timestamped_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Updated report: {len(current_successful)} successful, {len(current_failed)} failed")
        print(f"📊 Timestamped report saved: {timestamped_report_path}")
    
    def rerun_failed_sessions_simple(self, session_file_path: str, output_folder: str, 
                                   flow_ids: Optional[Dict[str, int]] = None,
                                   enable_evaluation: bool = True, max_evaluation_iterations: int = 3,
                                   max_retries: int = 3, generation_report_path: Optional[str] = None) -> Dict[str, Any]:
        """Simplified rerun method that reads failed sessions from generation report
        
        Args:
            session_file_path: Path to the session file
            output_folder: Path to the output folder containing conversations
            flow_ids: Optional flow IDs for conversation generation
            enable_evaluation: Whether to enable evaluation
            max_evaluation_iterations: Maximum evaluation attempts per session
            max_retries: Maximum number of times to retry a failed session
            
        Returns:
            Dictionary with rerun results
        """
        print(f"🔄 Rerunning failed sessions (simplified approach)...")
        print(f"📊 Max evaluation iterations: {max_evaluation_iterations}")
        print(f"📊 Max retries per session: {max_retries}")
        
        # Store original settings
        original_max_iterations = self.max_evaluation_iterations
        original_enable_evaluation = self.enable_evaluation
        
        # Set evaluation parameters for rerun
        self.max_evaluation_iterations = max_evaluation_iterations
        self.enable_evaluation = enable_evaluation
        
        # Load the generation report to get failed sessions
        if generation_report_path:
            report_path = generation_report_path
        else:
            report_path = os.path.join(output_folder, "conversations", "generation_report.json")
        
        if not os.path.exists(report_path):
            print(f"❌ No generation report found at {report_path}. Cannot determine failed sessions.")
            return {"successful": [], "failed": [], "total": 0}
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("❌ Could not load generation report")
            return {"successful": [], "failed": [], "total": 0}
        
        # Get failed sessions from report
        failed_sessions = report.get("failed_sessions", [])
        if not failed_sessions:
            print("✅ No failed sessions found in report")
            return {"successful": [], "failed": [], "total": 0}
        
        print(f"📋 Found {len(failed_sessions)} failed sessions to rerun: {failed_sessions}")
        
        # Load session data
        if not self.load_sessions(session_file_path):
            print("❌ Failed to load sessions from file")
            return {"successful": [], "failed": [], "total": 0}
        
        # Set up output folders
        successful_folder = os.path.join(output_folder, "conversations", "successful")
        failed_folder = os.path.join(output_folder, "conversations", "failed")
        
        # Track results
        successful_rerun_sessions = []
        failed_rerun_sessions = []
        
        for session_id in failed_sessions:
            print(f"\n🔄 Rerunning session {session_id}...")
            
            # Get session data
            session = self.session_processor.get_session_by_id(session_id)
            if not session:
                print(f"❌ Session {session_id} not found in session file")
                failed_rerun_sessions.append(session_id)
                continue
            
            # Retry logic: try up to max_retries times
            conversation = None
            success = False
            
            try:
                for retry_attempt in range(max_retries):
                    try:
                        if retry_attempt > 0:
                            print(f"🔄 Retry attempt {retry_attempt + 1}/{max_retries} for session {session_id}")
                        
                        # Generate conversation with evaluation
                        conversation = self.generate_conversation(session, flow_ids=flow_ids)
                        
                        # If successful, break out of retry loop
                        if conversation.get('success', False):
                            success = True
                            break
                            
                    except Exception as e:
                        print(f"❌ Error in retry attempt {retry_attempt + 1}: {e}")
                        if retry_attempt == max_retries - 1:
                            print(f"💥 All {max_retries} retry attempts failed for session {session_id}")
                            conversation = {
                                'success': False,
                                'error': f'Failed after {max_retries} retry attempts: {str(e)}',
                                'session_id': session_id,
                                'retry_attempts': retry_attempt + 1
                            }
                            break
                
                # Save in appropriate folder
                if success:
                    target_folder = successful_folder
                    filename = f"session_{session_id:04d}_{session.memory_type}.json"
                    print(f"✅ Session {session_id} succeeded on retry!")
                    successful_rerun_sessions.append(session_id)
                else:
                    target_folder = failed_folder
                    filename = f"session_{session_id:04d}_{session.memory_type}_FAILED.json"
                    print(f"❌ Session {session_id} still failed after {max_retries} retries")
                    failed_rerun_sessions.append(session_id)
                
                # Save the conversation
                if target_folder and conversation:
                    file_path = os.path.join(target_folder, filename)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(conversation, f, indent=2, ensure_ascii=False)
                    print(f"💾 Saved: {filename}")
                
            except Exception as e:
                print(f"❌ Error rerunning session {session_id}: {e}")
                failed_rerun_sessions.append(session_id)
        
        # Update the generation report
        self._update_report_after_rerun(output_folder, successful_rerun_sessions, failed_rerun_sessions)
        
        # Restore original evaluation settings
        self.max_evaluation_iterations = original_max_iterations
        self.enable_evaluation = original_enable_evaluation
        
        # Print summary
        print(f"\n📊 Rerun Summary:")
        print(f"   Total failed sessions: {len(failed_sessions)}")
        print(f"   Successfully rerun: {len(successful_rerun_sessions)}")
        print(f"   Still failed: {len(failed_rerun_sessions)}")
        
        return {
            "successful": successful_rerun_sessions,
            "failed": failed_rerun_sessions,
            "total": len(failed_sessions)
        }
    
    def generate_conversation_report(self, output_folder: str, max_evaluation_iterations: int = 3) -> Dict[str, Any]:
        """Generate a report of conversation generation results
        
        Args:
            output_folder: Path to the output folder containing conversations
            max_evaluation_iterations: Maximum evaluation iterations used
            
        Returns:
            Dictionary with report data
        """
        successful_folder = os.path.join(output_folder, "conversations", "successful")
        failed_folder = os.path.join(output_folder, "conversations", "failed")
        
        # Count successful conversations
        successful_count = 0
        successful_sessions = []
        if os.path.exists(successful_folder):
            for filename in os.listdir(successful_folder):
                if filename.endswith(".json") and not filename.endswith("_FAILED.json"):
                    successful_count += 1
                    # Extract session ID
                    try:
                        session_id = int(filename.split("_")[1])
                        successful_sessions.append(session_id)
                    except (IndexError, ValueError):
                        continue
        
        # Count failed conversations
        failed_count = 0
        failed_sessions = []
        if os.path.exists(failed_folder):
            for filename in os.listdir(failed_folder):
                if filename.endswith("_FAILED.json"):
                    failed_count += 1
                    # Extract session ID
                    try:
                        session_id = int(filename.split("_")[1])
                        failed_sessions.append(session_id)
                    except (IndexError, ValueError):
                        continue
        
        total_conversations = successful_count + failed_count
        success_rate = (successful_count / total_conversations * 100) if total_conversations > 0 else 0
        
        report = {
            "generation_timestamp": datetime.now().isoformat(),
            "total_conversations": total_conversations,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "success_rate": round(success_rate, 2),
            "max_evaluation_iterations": max_evaluation_iterations,
            "successful_sessions": sorted(successful_sessions),
            "failed_sessions": sorted(failed_sessions)
        }
        
        # Save report to conversations directory
        report_path = os.path.join(output_folder, "conversations", "generation_report.json")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📊 Generation report saved: {report_path}")
            
            # Also save a timestamped copy for history
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_report_path = os.path.join(output_folder, "conversations", f"generation_report_{timestamp}.json")
            with open(timestamped_report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📊 Timestamped report saved: {timestamped_report_path}")
        except Exception as e:
            print(f"⚠️  Failed to save generation report: {e}")
        
        return report
    
    def update_generation_report(self, output_folder: str, successful_sessions: List[int], failed_sessions: List[int]) -> None:
        """Update the generation report with new success/failure status
        
        Args:
            output_folder: Path to the output folder
            successful_sessions: List of session IDs that succeeded in this rerun
            failed_sessions: List of session IDs that failed in this rerun
        """
        report_path = os.path.join(output_folder, "conversations", "generation_report.json")
        
        # Load existing report if it exists
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                report = {}
        else:
            report = {}
        
        # Get current state from existing report
        existing_successful = set(report.get("successful_sessions", []))
        existing_failed = set(report.get("failed_sessions", []))
        
        print(f"📊 Before update: {len(existing_successful)} successful, {len(existing_failed)} failed")
        print(f"📊 Rerun results: {len(successful_sessions)} succeeded, {len(failed_sessions)} failed")
        
        # Update based on rerun results
        # Sessions that succeeded in rerun move from failed to successful
        for session_id in successful_sessions:
            if session_id in existing_failed:
                existing_failed.remove(session_id)
                print(f"✅ Moved session {session_id} from failed to successful")
            existing_successful.add(session_id)
        
        # Sessions that failed in rerun stay in failed (or add if new)
        for session_id in failed_sessions:
            existing_failed.add(session_id)
            print(f"❌ Session {session_id} remains failed")
        
        # Clean up: Remove any sessions that appear in both lists (shouldn't happen, but safety check)
        for session_id in list(existing_successful):
            if session_id in existing_failed:
                existing_failed.remove(session_id)
                print(f"⚠️  Cleaned up session {session_id} that appeared in both successful and failed lists")
        
        # Use the updated state from rerun results
        final_successful = existing_successful
        final_failed = existing_failed
        
        # Update report with final state
        report["last_update_timestamp"] = datetime.now().isoformat()
        report["successful_sessions"] = sorted(list(final_successful))
        report["failed_sessions"] = sorted(list(final_failed))
        report["successful_count"] = len(final_successful)
        report["failed_count"] = len(final_failed)
        report["total_conversations"] = len(final_successful) + len(final_failed)
        total_sessions = len(final_successful) + len(final_failed)
        report["success_rate"] = (len(final_successful) / total_sessions * 100) if total_sessions > 0 else 0
        
        # Save updated report
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also save a timestamped copy for history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_report_path = os.path.join(output_folder, "conversations", f"generation_report_{timestamp}.json")
        with open(timestamped_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📊 Timestamped report saved: {timestamped_report_path}")
        
        print(f"📊 Updated generation report: {len(existing_successful)} successful, {len(existing_failed)} failed")
    
    def _detect_metadata_directory(self, session_file_path: str) -> Optional[str]:
        """Auto-detect metadata directory from session file path
        
        Args:
            session_file_path: Path to the session file (e.g., '../output/sessions_<timestamp>_<persona>/new_sessions.json')
            
        Returns:
            Directory path where conversations should be saved, or None if cannot detect
        """
        try:
            # Get the directory containing the session file
            session_dir = os.path.dirname(os.path.abspath(session_file_path))
            
            # Check if this looks like a metadata directory (contains 'sessions_' pattern)
            dir_name = os.path.basename(session_dir)
            if 'sessions_' in dir_name and ('_' in dir_name or dir_name.startswith('sessions_')):
                return session_dir
            
            # If session file is directly in output folder, look for session directories
            if os.path.basename(session_dir) == 'output':
                # Look for the most recent sessions directory
                session_dirs = []
                for item in os.listdir(session_dir):
                    item_path = os.path.join(session_dir, item)
                    if os.path.isdir(item_path) and item.startswith('sessions_'):
                        session_dirs.append(item_path)
                
                if session_dirs:
                    # Return the most recently modified session directory
                    latest_dir = max(session_dirs, key=os.path.getmtime)
                    return latest_dir
            
            return None
            
        except Exception as e:
            print(f"⚠️  Could not auto-detect metadata directory: {e}")
            return None
    
    def save_conversations(self, conversations: List[Dict[str, Any]], output_file: str):
        """Save conversations to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(conversations, file, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved {len(conversations)} conversations to {output_file}")
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    def analyze_conversations(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze generated conversations"""
        if not conversations:
            return {"error": "No conversations to analyze"}
        
        successful = [c for c in conversations if c.get("success", False)]
        failed = [c for c in conversations if not c.get("success", False)]
        
        # Memory type distribution
        memory_types = {}
        operation_types = {}
        flow_patterns = {}
        total_turns = 0
        
        for conv in successful:
            # Memory types
            mem_type = conv.get("session_type", "unknown")
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
            
            # Operations
            operation = conv.get("operation", "unknown")
            operation_types[operation] = operation_types.get(operation, 0) + 1
            
            # Flow patterns
            flow_pattern = " → ".join(conv.get("conversation_flow", []))
            flow_patterns[flow_pattern] = flow_patterns.get(flow_pattern, 0) + 1
            
            # Turn counts
            total_turns += conv.get("total_turns", 0)
        
        avg_turns = total_turns / len(successful) if successful else 0
        
        return {
            "total_conversations": len(conversations),
            "successful": len(successful),
            "failed": len(failed),
            "memory_types": memory_types,
            "operations": operation_types,
            "flow_patterns": flow_patterns,
            "total_turns": total_turns,
            "average_turns": round(avg_turns, 1)
        }
    

    # Removed _analyze_prompts method since it's no longer needed with the integrated system


def main():
    """Test the memory-grounded conversation generator with enhanced argument support"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory-Grounded Conversation Generator')
    parser.add_argument('--num-sessions', type=int, default=3, 
                       help='Number of sessions to process (default: 3)')
    parser.add_argument('--generate-all', action='store_true',
                       help='Generate all sessions from the session file (overrides --num-sessions and --session-id)')
    parser.add_argument('--session-id', type=int, 
                       help='Specific session ID to generate conversation for')
    parser.add_argument('--opening-flow', type=int, 
                       help='Specific opening flow ID to use')
    parser.add_argument('--exploration-flow', type=int, 
                       help='Specific exploration flow ID to use')
    parser.add_argument('--memory-flow', type=int, 
                       help='Specific memory flow ID to use')
    parser.add_argument('--closing-flow', type=int, 
                       help='Specific closing flow ID to use')
    parser.add_argument('--session-file', type=str, default="../output/new_sessions.json",
                       help='Path to sessions file (default: ../output/new_sessions.json)')
    parser.add_argument('--output-folder', type=str,
                       help='Output folder path to save conversations. If not provided, will auto-detect from session file path or create timestamped folder')
    parser.add_argument('--memory-type', type=str, 
                       help='Filter sessions by memory type (e.g., no_memory, preference_memory, activity_memory, goal_memory, content_memory)')
    parser.add_argument('--test-no-memory', action='store_true',
                       help='Quick test with no_memory sessions only')
    parser.add_argument('--enable-evaluation', action='store_true',
                       help='Enable LLM-based quality evaluation')
    parser.add_argument('--max-evaluation-iterations', type=int, default=3,
                       help='Maximum number of evaluation attempts per session (default: 3)')
    parser.add_argument('--rerun-failed', action='store_true',
                       help='Rerun failed sessions with evaluation')
    parser.add_argument('--generation-report', type=str,
                       help='Path to generation report file (required for --rerun-failed)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum number of times to retry a failed session (default: 3)')
    
    args = parser.parse_args()
    
    # Build flow_ids dict if any flow IDs are specified
    flow_ids = None
    if any([args.opening_flow, args.exploration_flow, args.memory_flow, args.closing_flow]):
        flow_ids = {}
        if args.opening_flow is not None:
            flow_ids['opening'] = args.opening_flow
        if args.exploration_flow is not None:
            flow_ids['exploration'] = args.exploration_flow
        if args.memory_flow is not None:
            flow_ids['memory'] = args.memory_flow
        if args.closing_flow is not None:
            flow_ids['closing'] = args.closing_flow
        
        print(f"🎯 Using specified flow IDs: {flow_ids}")
    
    # Initialize generator with evaluation settings
    generator = MemoryGroundedGenerator(
        enable_evaluation=args.enable_evaluation,
        max_evaluation_iterations=args.max_evaluation_iterations
    )
    
    # Handle special test options
    memory_type_filter = args.memory_type
    if args.test_no_memory:
        memory_type_filter = 'no_memory'
        if args.num_sessions == 3:  # Default value
            args.num_sessions = 2  # Test with just 2 no_memory sessions
        print("🧪 Testing no_memory conversation generation...")
    
    # Handle --generate-all flag
    if args.generate_all:
        print("🔄 Generating ALL sessions from the session file...")
        num_sessions = None  # Process all sessions
        specific_session_id = None  # Override specific session ID
    else:
        num_sessions = args.num_sessions
        specific_session_id = args.session_id
    
    # Handle rerun failed sessions
    if args.rerun_failed:
        # Validate required arguments for rerun
        if not args.generation_report:
            print("❌ --generation-report is required when using --rerun-failed")
            print("   Please specify the path to the generation report file")
            sys.exit(1)
        
        if not os.path.exists(args.generation_report):
            print(f"❌ Generation report file not found: {args.generation_report}")
            sys.exit(1)
        
        # Auto-detect output folder from session file path
        if args.output_folder:
            output_folder = args.output_folder
        else:
            # Auto-detect from session file path
            metadata_dir = generator._detect_metadata_directory(args.session_file)
            if not metadata_dir:
                print("❌ Could not auto-detect output directory from session file path")
                print("   Please provide --output-folder or ensure session file is in a valid metadata directory")
                sys.exit(1)
            output_folder = metadata_dir
            print(f"📁 Auto-detected output directory: {output_folder}")
        
        print("🔄 Rerunning failed sessions...")
        print(f"📊 Using generation report: {args.generation_report}")
        rerun_results = generator.rerun_failed_sessions_simple(
            session_file_path=args.session_file,
            output_folder=output_folder,
            flow_ids=flow_ids,
            enable_evaluation=args.enable_evaluation,
            max_evaluation_iterations=args.max_evaluation_iterations,
            max_retries=args.max_retries,
            generation_report_path=args.generation_report
        )
        
        # Print summary
        successful_reruns = len(rerun_results["successful"])
        total_reruns = rerun_results["total"]
        
        print(f"\n📊 Rerun Summary:")
        print(f"   Total failed sessions: {total_reruns}")
        print(f"   Successfully rerun: {successful_reruns}")
        print(f"   Still failed: {len(rerun_results['failed'])}")
        
        # Report was already updated by _update_report_after_rerun
        print("\n📊 Report already updated with rerun results")
        
        sys.exit(0)
    
    # Generate conversations
    conversations = generator.generate_multiple_conversations(
        session_file_path=args.session_file,
        num_sessions=num_sessions,
        specific_session_id=specific_session_id,
        flow_ids=flow_ids,
        memory_type_filter=memory_type_filter,
        save_individual=True,  # Always save individual files for batch processing
        output_folder=args.output_folder
    )
    
    # Summary file removed - individual files are already saved
    # No need to save large summary file with all conversations included
    
    # Generate conversation report for regular runs
    if (num_sessions is None or num_sessions > 1):
        # Auto-detect output folder if not provided
        if not args.output_folder:
            metadata_dir = generator._detect_metadata_directory(args.session_file)
            if metadata_dir:
                output_folder = metadata_dir
            else:
                print("⚠️  Cannot generate report: output folder not specified and cannot auto-detect")
                output_folder = None
        else:
            output_folder = args.output_folder
        
        if output_folder:
            print("\n📊 Generating conversation report...")
            report = generator.generate_conversation_report(output_folder, args.max_evaluation_iterations)
            print(f"📈 Results: {report['successful_count']} successful, {report['failed_count']} failed ({report['success_rate']}% success rate)")
    
    # Analyze results
    if conversations:
        analysis = generator.analyze_conversations(conversations)
        
        print(f"\n📊 Generation Analysis:")
        print(f"✅ Successful: {analysis.get('successful', 0)}")
        print(f"❌ Failed: {analysis.get('failed', 0)}")
        print(f"🔄 Average turns: {analysis.get('average_turns', 0)}")
        
        print(f"\n🧠 Memory Types:")
        for mem_type, count in analysis.get('memory_types', {}).items():
            print(f"  • {mem_type}: {count}")
        
        print(f"\n⚙️ Operations:")
        for operation, count in analysis.get('operations', {}).items():
            print(f"  • {operation}: {count}")
    else:
        print(f"\n❌ No conversations generated!")
    
    # Show sample conversation with memory validation
    if conversations and conversations[0].get("success"):
        print(f"\n📖 Sample Memory-Grounded Conversation:")
        print("=" * 60)
        sample = conversations[0]
        
        # Show memory context
        print(f"🎯 Session Context:")
        print(f"   Type: {sample['session_type']}")
        print(f"   Operation: {sample['operation']}")
        print(f"   Category: {sample['category']}")
        print(f"   Subcategory: {sample.get('subcategory', '')}")
        print(f"   Preference: {sample.get('preference_type', '')}")
        print(f"   Item: {sample.get('item', '')}")
        
        print(f"\n💬 Conversation Flow:")
        for turn in sample["conversation"]:
            speaker = "🤖 AI" if turn["speaker"] == "ai_agent" else "👤 USER"
            intent = turn.get("intent", "unknown")
            message = turn["message"]
            
            # Truncate long messages
            if len(message) > 100:
                message = message[:97] + "..."
            
            print(f"[{intent:20}] {speaker}: {message}")
        
        # Memory grounding validation disabled
        print(f"\n✅ Conversation generated successfully")


if __name__ == "__main__":
    main()
