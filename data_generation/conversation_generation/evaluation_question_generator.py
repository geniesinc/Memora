"""
Evaluation Question Generator

This module generates specific evaluation questions based on memory type,
operation, and item to verify that the user actually performed the expected
memory operation in the conversation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os

@dataclass
class EvaluationQuestion:
    """Represents an evaluation question with expected answer"""
    question: str
    expected_answer: str
    evaluation_type: str
    memory_type: str
    operation: str
    item: str

class EvaluationQuestionGenerator:
    """
    Generates evaluation questions for different memory types and operations.
    
    The questions are designed to verify that:
    1. The user actually performed the expected memory operation
    2. The correct item was involved in the operation
    3. The operation was performed as intended (add/update/delete)
    """
    
    def __init__(self):
        pass
    
    def generate_transition_question(self, memory_segment: Dict[str, Any], session_data: Dict[str, Any]) -> EvaluationQuestion:
        """Generate question to evaluate transition quality"""
        context_before = memory_segment.get('context_before', {})
        before_message = context_before.get('message', '')
        
        # Get memory type context for more specific question
        memory_type = session_data.get('session_type', '')
        category = session_data.get('category', '')
        
        question = f"""Did the conversation flow naturally from the previous topic into sharing {memory_type} information about {category}?

Previous message: "{before_message[:200]}..."

The transition should feel like something a real person would say in conversation."""
        
        return EvaluationQuestion(
            question=question,
            expected_answer="yes",
            evaluation_type="transition_quality",
            memory_type=memory_type,
            operation="all",
            item=category
        )
    
    def generate_memory_operation_question(self, session_data: Dict[str, Any], memory_segment: Dict[str, Any]) -> List[EvaluationQuestion]:
        """Generate question(s) to evaluate if the expected memory operation was performed"""
        memory_type = session_data.get('session_type', '')
        operation = session_data.get('operation', '')
        item = session_data.get('item', '')
        category = session_data.get('category', '')
        subcategory = session_data.get('subcategory', '')
        preference_type = session_data.get('preference_type', '')
        operation_details = session_data.get('operation_details', {})
        
        questions = []
        
        # Generate specific question(s) based on memory type and operation
        if memory_type == "preference_memory":
            question_text = self._generate_preference_question(operation, item, category, subcategory, operation_details, preference_type)
            questions.append(EvaluationQuestion(
                question=question_text,
                expected_answer="yes",
                evaluation_type="memory_operation_quality",
                memory_type=memory_type,
                operation=operation,
                item=item
            ))
        elif memory_type == "activity_memory":
            question_text = self._generate_activity_question(operation, item, category, operation_details)
            questions.append(EvaluationQuestion(
                question=question_text,
                expected_answer="yes",
                evaluation_type="memory_operation_quality",
                memory_type=memory_type,
                operation=operation,
                item=item
            ))
        elif memory_type == "content_memory":
            # For content memory, generate questions based on operation type
            if operation == "add" and operation_details and 'content_data' in operation_details:
                # For add operations, generate questions based on content_data
                content_data = operation_details['content_data']
                for key, value in content_data.items():
                    if isinstance(value, list) and value:
                        # Handle list values (like hashtags, recipients, etc.)
                        if all(isinstance(item, str) for item in value):
                            # String list - join all values
                            values_str = ', '.join(value)
                            question_text = f"Did the user mention the {key}: {values_str}?"
                        else:
                            # Mixed list - show first few items
                            first_items = ', '.join([str(item) for item in value[:3]])
                            if len(value) > 3:
                                first_items += f" and {len(value) - 3} more"
                            question_text = f"Did the user mention the {key}: {first_items}?"
                    elif isinstance(value, str) and value:
                        # Handle string values
                        question_text = f"Did the user mention the {key}: '{value}'?"
                    elif isinstance(value, (int, float)) and value:
                        # Handle numeric values
                        question_text = f"Did the user mention the {key}: {value}?"
                    elif isinstance(value, dict) and value:
                        # Handle nested dictionaries
                        nested_items = ', '.join([f"{k}: {v}" for k, v in value.items()][:3])
                        question_text = f"Did the user mention the {key}: {nested_items}?"
                    elif value:  # Handle other truthy values
                        question_text = f"Did the user mention the {key}: {value}?"
                    else:
                        continue  # Skip empty values
                    
                    questions.append(EvaluationQuestion(
                        question=question_text,
                        expected_answer="yes",
                        evaluation_type="memory_operation_quality",
                        memory_type=memory_type,
                        operation=operation,
                        item=item
                    ))
            elif operation == "delete" and operation_details and 'memory_deletes' in operation_details:
                # For delete operations, generate questions based on memory_deletes
                memory_deletes = operation_details['memory_deletes']
                
                # First, check if user mentioned the content identifier
                content_identifier = self._get_content_identifier_for_evaluation(operation_details)
                if content_identifier:
                    questions.append(EvaluationQuestion(
                        question=f"Did the user mention '{content_identifier}' (or refer to it naturally) when discussing which content item they want to remove information from?",
                        expected_answer="yes",
                        evaluation_type="memory_operation_quality",
                        memory_type=memory_type,
                        operation=operation,
                        item=item
                    ))
                
                # Then check for specific deletions
                for delete_item in memory_deletes:
                    field = delete_item.get('field', 'unknown')
                    action = delete_item.get('action', '')
                    
                    # Handle different types of delete operations
                    if action == 'budget_reverted':
                        # For budget reversion, check if user mentioned the reversion
                        reverted_from = delete_item.get('reverted_from', '')
                        reverted_to = delete_item.get('reverted_to', '')
                        question_text = f"Did the user mention reverting the budget from {reverted_from} to {reverted_to}?"
                    else:
                        # For regular removals, check if user mentioned the removed item
                        removed_item = delete_item.get('removed_item', '')
                        question_text = f"Did the user mention removing the {field}: '{removed_item}'?"
                    
                    questions.append(EvaluationQuestion(
                        question=question_text,
                        expected_answer="yes",
                        evaluation_type="memory_operation_quality",
                        memory_type=memory_type,
                        operation=operation,
                        item=item
                    ))
            elif operation == "update" and operation_details and 'memory_updates' in operation_details:
                # For update operations, generate questions based on memory_updates
                memory_updates = operation_details['memory_updates']
                
                # First, check if user mentioned the content identifier
                content_identifier = self._get_content_identifier_for_evaluation(operation_details)
                if content_identifier:
                    questions.append(EvaluationQuestion(
                        question=f"Did the user mention '{content_identifier}' (or refer to it naturally) to identify which content they want to update?",
                        expected_answer="yes",
                        evaluation_type="memory_operation_quality",
                        memory_type=memory_type,
                        operation=operation,
                        item=item
                    ))
                
                # Then check for specific updates
                for update_item in memory_updates:
                    field = update_item.get('field', 'unknown')
                    action = update_item.get('action', '')
                    
                    if action == "added":
                        # For additions to lists (most common case)
                        added_item = update_item.get('added_item', '')
                        question_text = f"Did the user mention adding to the {field}: '{added_item}'?"
                    elif action == "budget_revised":
                        # For budget updates (only true "from X to Y" updates)
                        updated_from = update_item.get('updated_from', '')
                        updated_to = update_item.get('updated_to', '')
                        question_text = f"Did the user mention updating the {field} from {updated_from} to {updated_to}?"
                    else:
                        # Fallback for other update types
                        question_text = f"Did the user mention updating the {field}?"
                    
                    questions.append(EvaluationQuestion(
                        question=question_text,
                        expected_answer="yes",
                        evaluation_type="memory_operation_quality",
                        memory_type=memory_type,
                        operation=operation,
                        item=item
                    ))
            else:
                # Fallback for other operations or missing data
                question_text = self._generate_content_question(operation, item, category, subcategory, operation_details)
                questions.append(EvaluationQuestion(
                    question=question_text,
                    expected_answer="yes",
                    evaluation_type="memory_operation_quality",
                    memory_type=memory_type,
                    operation=operation,
                    item=item
                ))
        elif memory_type == "goal_memory":
            question_text = self._generate_goal_question(operation, item, category, subcategory, operation_details)
            questions.append(EvaluationQuestion(
                question=question_text,
                expected_answer="yes",
                evaluation_type="memory_operation_quality",
                memory_type=memory_type,
                operation=operation,
                item=item
            ))
        else:
            # Fallback to generic question
            question_text = f"Did the user {operation} {memory_type} for '{item}' in this conversation?"
            questions.append(EvaluationQuestion(
                question=question_text,
                expected_answer="yes",
                evaluation_type="memory_operation_quality",
                memory_type=memory_type,
                operation=operation,
                item=item
            ))
        
        return questions
    
    def _generate_preference_question(self, operation: str, item: str, category: str, subcategory: str, operation_details: Dict[str, Any], preference_type: str = '') -> str:
        """Generate preference memory question"""
        if operation == "add":
            # Special handling for already_read_list - must confirm they've READ it
            if subcategory == "already_read_list":
                return f"Did the user mention that they have read, finished, or completed reading '{item}'?"
            
            # Handle preference type (like/dislike)
            if preference_type == "dislike":
                return f"Did the user express that they dislike or do not prefer '{item}' in the {category}/{subcategory} category?"
            else:
                # Default to "like" or general preference
                return f"Did the user express a new preference for '{item}' in the {category}/{subcategory} category?"
        elif operation == "update":
            # Check update_type in operation_details
            if operation_details and operation_details.get('update_type') == "value_update":
                # Value update: changing from one item to another
                old_item = operation_details.get('old_item', 'previous item')
                pref_word = "dislike" if preference_type == "dislike" else "preference for"
                return f"Did the user change their {pref_word} from '{old_item}' to '{item}' in the {category}/{subcategory} category?"
            elif operation_details and operation_details.get('update_type') == "preference_update":
                # Preference update: same item, different preference (like to dislike or vice versa)
                old_preference = operation_details.get('old_preference', 'previous preference')
                new_preference = operation_details.get('preference', preference_type) or 'new preference'
                return f"Did the user change their preference for '{item}' from '{old_preference}' to '{new_preference}' in the {category}/{subcategory} category?"
            else:
                # Generic update
                return f"Did the user change their preference for '{item}' in the {category}/{subcategory} category?"
        elif operation == "delete":
            # Handle both like and dislike deletions
            if preference_type == "dislike":
                return f"Did the user express that they no longer dislike '{item}' in the {category}/{subcategory} category?"
            else:
                return f"Did the user express that they no longer like or prefer '{item}' in the {category}/{subcategory} category?"
        else:
            return f"Did the user {operation} preference for '{item}' in the {category}/{subcategory} category?"
    
    def _generate_activity_question(self, operation: str, item: str, category: str, operation_details: Dict[str, Any]) -> str:
        """Generate activity memory question"""
        if operation == "add":
            if category == "food_expenses" and operation_details:
                item_details = operation_details.get('item', {})
                if isinstance(item_details, dict) and 'expense_type' in item_details:
                    expense_type = item_details['expense_type']
                    amount = item_details.get('amount', '')
                    return f"Did the user mention spending ${amount} on {expense_type} (adding to their {category} tracking)?"
            elif category == "calendar_event" and operation_details:
                # For calendar events, check for both event name and date
                item_details = operation_details.get('item', {})
                if isinstance(item_details, dict):
                    event_date_str = self._calculate_calendar_event_date_for_evaluation(item_details)
                    if event_date_str:
                        return f"Did the user mention both the event '{item}' AND when it's scheduled ({event_date_str})?"
            return f"Did the user mention or share information about '{item}' in their {category} activities?"
        elif operation == "update":
            return f"Did the user mention updating or modifying something related to '{item}' in their {category} activities?"
        elif operation == "delete":
            if category == "calendar_event":
                return f"Did the user mention missing, not attending, or canceling something related to '{item}' from their {category} activities?"
            else:
                return f"Did the user mention completing, finishing, or removing something related to '{item}' from their {category} activities?"
        else:
            return f"Did the user {operation} activity for '{item}' in the {category} category?"
    
    def _generate_content_question(self, operation: str, item: str, category: str, subcategory: str, operation_details: Dict[str, Any]) -> str:
        """Generate content memory question based on content_data dictionary keys and values"""
        if not operation_details or 'content_data' not in operation_details:
            return f"Did the user {operation} content for '{item}' in the {category}/{subcategory} category?"
        
        content_data = operation_details['content_data']
        
        if operation == "add":
            # For content memory, we'll generate one question per field
            # This will be handled in the main question generation method
            return f"Did the user share content information for '{item}' in the {category}/{subcategory} category?"
        elif operation == "update":
            return f"Did the user update or modify content information about '{item}' for their {category}/{subcategory}?"
        elif operation == "delete":
            return f"Did the user mention removing or deleting content information about '{item}' for their {category}/{subcategory}?"
        else:
            return f"Did the user {operation} content for '{item}' in the {category}/{subcategory} category?"
    
    def _generate_goal_question(self, operation: str, item: str, category: str, subcategory: str, operation_details: Dict[str, Any]) -> str:
        """Generate goal memory question"""
        if operation == "add":
            return f"Did the user set or establish a new goal related to '{item}' for their {category}/{subcategory}?"
        elif operation == "update":
            return f"Did the user modify or change their goal related to '{item}' for their {category}/{subcategory}?"
        elif operation == "delete":
            return f"Did the user mention abandoning or removing their goal related to '{item}' for their {category}/{subcategory}?"
        else:
            return f"Did the user {operation} goal for '{item}' in the {category}/{subcategory} category?"
    
    def generate_post_memory_question(self, memory_segment: Dict[str, Any]) -> EvaluationQuestion:
        """Generate question to evaluate post-memory conversation quality"""
        context_after = memory_segment.get('context_after', {})
        after_message = context_after.get('message', '') if context_after else ""
        
        question = f"""Did the AI appropriately acknowledge the user's memory sharing and continue the conversation naturally?

AI response after memory sharing: "{after_message[:200]}..."

The AI should respond in a way that feels natural and engaging."""
        
        return EvaluationQuestion(
            question=question,
            expected_answer="yes",
            evaluation_type="post_memory_quality",
            memory_type="all",
            operation="all",
            item="ai_acknowledgment"
        )
    
    def generate_all_evaluation_questions(self, session_data: Dict[str, Any], memory_segment: Dict[str, Any]) -> List[EvaluationQuestion]:
        """Generate all evaluation questions for a conversation"""
        questions = []
        
        # Only evaluate the memory operation quality - whether user performed expected operation
        memory_questions = self.generate_memory_operation_question(session_data, memory_segment)
        questions.extend(memory_questions)
        
        return questions
    
    def format_questions_for_llm(self, questions: List[EvaluationQuestion]) -> str:
        """Format questions for LLM evaluation"""
        formatted = "Please evaluate the following conversation aspects. Answer each question with 'yes' or 'no' only.\n\n"
        
        for i, question in enumerate(questions, 1):
            formatted += f"Question {i} ({question.evaluation_type}):\n"
            formatted += f"{question.question}\n\n"
        
        return formatted
    
    def parse_llm_response(self, response: str, questions: List[EvaluationQuestion]) -> Dict[str, bool]:
        """Parse LLM response and map to evaluation results"""
        # Extract yes/no answers from response
        response_lower = response.lower()
        answers = []
        
        # Look for yes/no patterns
        import re
        yes_no_pattern = r'\b(yes|no)\b'
        matches = re.findall(yes_no_pattern, response_lower)
        
        if len(matches) >= len(questions):
            answers = matches[:len(questions)]
        else:
            # Fallback: assume all "no" if we can't parse properly
            answers = ['no'] * len(questions)
        
        # Map answers to evaluation results
        results = {}
        for i, question in enumerate(questions):
            if i < len(answers):
                # Use question index as key to avoid overwriting
                results[f"question_{i}"] = answers[i] == 'yes'
            else:
                results[f"question_{i}"] = False
        
        return results
    
    def _get_content_identifier_for_evaluation(self, operation_details: Dict[str, Any]) -> str:
        """Get content identifier for evaluation questions"""
        if not operation_details:
            return None
        
        content_data = operation_details.get('content_data', {})
        
        # Use content-specific identifier based on category
        if 'meeting_title' in content_data:
            return content_data['meeting_title']
        elif 'subject' in content_data:
            return content_data['subject']
        elif 'project_title' in content_data:
            return content_data['project_title']
        elif 'post_title' in content_data:
            return content_data['post_title']
        else:
            # Fallback to first available title field
            for field in ['title', 'name', 'subject', 'project_title', 'meeting_title']:
                if field in content_data and content_data[field]:
                    return content_data[field]
            return None
    
    def _calculate_calendar_event_date_for_evaluation(self, item_details: Dict[str, Any]) -> Optional[str]:
        """Calculate and format calendar event date for evaluation question
        
        Args:
            item_details: Dictionary containing date and created_at fields
            
        Returns:
            Formatted date string (e.g., "June 18th", "in 17 days", "tomorrow") or None
        """
        date_offset = item_details.get('date')
        created_at = item_details.get('created_at')
        
        if not date_offset or not created_at:
            return None
        
        try:
            from datetime import datetime, timedelta
            
            # Parse the creation date
            created_datetime = datetime.strptime(created_at, "%Y-%m-%d")
            
            # Parse the offset (e.g., "+17 days", "+1 week")
            if date_offset.startswith("+"):
                offset_str = date_offset[1:].strip()
                
                if "day" in offset_str:
                    days = int(offset_str.split()[0])
                    event_datetime = created_datetime + timedelta(days=days)
                    
                    # Format naturally based on days
                    if days == 0:
                        return "today"
                    elif days == 1:
                        return "tomorrow"
                    elif days <= 7:
                        return f"in {days} days"
                    else:
                        # Format as actual date (e.g., "June 18th")
                        return event_datetime.strftime("%B %d")
                elif "week" in offset_str:
                    weeks = int(offset_str.split()[0])
                    event_datetime = created_datetime + timedelta(weeks=weeks)
                    if weeks == 1:
                        return "next week"
                    else:
                        return event_datetime.strftime("%B %d")
                else:
                    # Unknown format, return None
                    return None
            else:
                # If it's not a relative date, return as-is
                return date_offset
                
        except (ValueError, AttributeError, IndexError):
            return None


def main():
    """Test the question generator"""
    generator = EvaluationQuestionGenerator()
    
    # Test different memory types with realistic session data
    test_cases = [
        {
            "name": "Preference Memory - Add",
            "session_data": {
                "session_type": "preference_memory",
                "operation": "add",
                "item": "techno music",
                "category": "music",
                "subcategory": "electronic"
            },
            "memory_segment": {
                "context_before": {
                    "message": "That's really interesting about quantum computing!"
                },
                "context_after": {
                    "message": "Thanks for sharing your music preference!"
                }
            }
        },
        {
            "name": "Preference Memory - Update (Generic)",
            "session_data": {
                "session_type": "preference_memory",
                "operation": "update",
                "item": "jazz music",
                "category": "music",
                "subcategory": "jazz",
                "operation_details": {
                    "preference": "like",
                    "subcategory": "genres"
                }
            },
            "memory_segment": {
                "context_before": {
                    "message": "I used to really like rock music, but now I'm more into jazz."
                },
                "context_after": {
                    "message": "That's a great musical evolution! Jazz has such rich complexity."
                }
            }
        },
        {
            "name": "Preference Memory - Value Update",
            "session_data": {
                "session_type": "preference_memory",
                "operation": "update",
                "item": "New Zealand",
                "category": "travel",
                "subcategory": "regions",
                "operation_details": {
                    "preference": "like",
                    "subcategory": "regions",
                    "update_type": "value_update",
                    "old_preference": "like",
                    "old_item": "Iceland"
                }
            },
            "memory_segment": {
                "context_before": {
                    "message": "I used to be interested in Iceland, but now I'm more drawn to New Zealand."
                },
                "context_after": {
                    "message": "That's wonderful! New Zealand has such diverse landscapes."
                }
            }
        },
        {
            "name": "Preference Memory - Preference Update",
            "session_data": {
                "session_type": "preference_memory",
                "operation": "update",
                "item": "horror movies",
                "category": "movies",
                "subcategory": "genres",
                "operation_details": {
                    "preference": "dislike",
                    "subcategory": "genres",
                    "update_type": "preference_update",
                    "old_preference": "like"
                }
            },
            "memory_segment": {
                "context_before": {
                    "message": "I used to like horror movies, but now I can't stand them."
                },
                "context_after": {
                    "message": "That's understandable! Horror can be too intense for some people."
                }
            }
        },
        {
            "name": "Activity Memory - Add (Food Expense)",
            "session_data": {
                "session_type": "activity_memory",
                "operation": "add",
                "item": "$7.31 coffee",
                "category": "food_expenses",
                "subcategory": "beverages",
                "operation_details": {
                    "item": {
                        "expense_type": "coffee",
                        "amount": "7.31"
                    }
                }
            },
            "memory_segment": {
                "context_before": {
                    "message": "Speaking of daily habits, I just spent $7.31 on my usual coffee this morning."
                },
                "context_after": {
                    "message": "Ah, a classic start to the day! A good coffee can certainly set the tone."
                }
            }
        },
        {
            "name": "Content Memory - Add",
            "session_data": {
                "session_type": "content_memory",
                "operation": "add",
                "item": "social_media_post_1",
                "category": "social_media_post",
                "subcategory": "professional",
                "operation_details": {
                    "item": "social_media_post_1",
                    "content_data": {
                        "platform": "LinkedIn",
                        "content_type": "Insight Post",
                        "main_message": "Optimizing legacy system integration is crucial for driving cost reduction in healthcare. Focusing on modular architecture and secure API development can yield significant savings and enhance data interoperability across North American healthcare providers.",
                        "hashtags": [
                            "#CostReduction",
                            "#SoftwareEngineering",
                            "#HealthTech",
                            "#DigitalTransformation"
                        ]
                    }
                }
            },
            "memory_segment": {
                "context_before": {
                    "message": "I need some help organizing and structuring information for a social media post I'm working on."
                },
                "context_after": {
                    "message": "Great, I have all the key details for your LinkedIn Insight Post!"
                }
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {test_case['name']}")
        print(f"{'='*60}")
        
        # Generate questions
        questions = generator.generate_all_evaluation_questions(test_case['session_data'], test_case['memory_segment'])
        
        print(f"\n📋 Generated Evaluation Questions:")
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i} ({question.evaluation_type}):")
            print(f"  {question.question}")
            print(f"  Expected: {question.expected_answer}")
        
        # Format for LLM
        formatted = generator.format_questions_for_llm(questions)
        print(f"\n🤖 Formatted for LLM:")
        print(formatted[:500] + "..." if len(formatted) > 500 else formatted)

def test_question_generator_with_file(conversation_file: str):
    """
    Test question generation with an existing conversation file.
    
    Args:
        conversation_file: Path to the conversation JSON file
    """
    import json
    import os
    from memory_slice import extract_memory_segment
    
    print(f"🔍 Testing question generator with: {conversation_file}")
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
    print(f"   - Category: {conversation_data.get('category', 'N/A')}")
    print(f"   - Subcategory: {conversation_data.get('subcategory', 'N/A')}")
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
    generator = EvaluationQuestionGenerator()
    questions = generator.generate_all_evaluation_questions(session_data, memory_segment)
    
    print(f"✅ Generated {len(questions)} evaluation question(s):")
    for i, question in enumerate(questions, 1):
        print(f"\n   {i}. [{question.evaluation_type}] {question.question}")
        print(f"      Expected: {question.expected_answer}")
        print(f"      Memory Type: {question.memory_type} | Operation: {question.operation} | Item: {question.item}")
    
    # Format questions for LLM
    print(f"\n🤖 Formatting questions for LLM...")
    formatted_questions = generator.format_questions_for_llm(questions)
    print(f"✅ Formatted questions for LLM")
    print(f"\n📝 Formatted Questions Preview:")
    print("-" * 50)
    print(formatted_questions[:800] + "..." if len(formatted_questions) > 800 else formatted_questions)
    print("-" * 50)
    
    # Show conversation context that would be sent to LLM
    print(f"\n📋 Conversation Context for LLM:")
    print("-" * 30)
    
    # Context before
    if memory_segment['context_before']:
        before = memory_segment['context_before']
        print(f"Context Before: {before['speaker']} - {before['message'][:100]}...")
    
    # Memory turns
    for turn in memory_segment['memory_turns']:
        print(f"Memory Turn: {turn['speaker']} - {turn['message'][:100]}...")
    
    # Context after
    if memory_segment['context_after']:
        after = memory_segment['context_after']
        print(f"Context After: {after['speaker']} - {after['message'][:100]}...")
    
    print("-" * 30)
    
    return {
        "session_data": session_data,
        "memory_segment": memory_segment,
        "questions": questions,
        "formatted_questions": formatted_questions
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 2:
        # Test with provided JSON file
        conversation_file = sys.argv[1]
        if not os.path.exists(conversation_file):
            print(f"❌ Conversation file not found: {conversation_file}")
            sys.exit(1)
        
        result = test_question_generator_with_file(conversation_file)
        
        if result:
            print(f"\n🎉 Question generation test completed successfully!")
            print(f"✅ Generated {len(result['questions'])} evaluation question(s)")
        else:
            print(f"\n❌ Question generation test failed")
    else:
        # Run the original test with sample data
        print("Usage: python evaluation_question_generator.py <conversation_file>")
        print("Example: python evaluation_question_generator.py ../output/sessions_<timestamp>_<persona>/conversations/session_0002_activity_memory.json")
        print("\nRunning sample test instead...")
        main()
