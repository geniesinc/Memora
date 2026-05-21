#!/usr/bin/env python3
"""
Memory Evidence Evaluation Generator for Memora Framework

This module generates yes/no evaluation questions to validate whether AI responses
correctly include memory evidence and exclude forgotten evidence.

Key Features:
1. Memory Evidence Validation: Check if current memory items are present in responses
2. Forgetting Evidence Validation: Check if forgotten items are correctly excluded
3. Template-based Question Generation: Consistent evaluation questions across memory types
4. Category-specific Validation: Tailored checks for different memory categories
"""

import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationQuestion:
    """Represents a single evaluation question"""
    question_id: str
    evaluation_question: str
    expected_answer: str  # "yes" or "no"
    evaluation_type: str  # "memory_presence" or "forgetting_absence"
    target_item: Any
    context: Dict[str, Any]  # Contains check_type: "goal_mention", "goal_progress", "goal_status"


class MemoryEvaluationGenerator:
    """Generate evaluation questions to validate memory and forgetting evidence"""
    
    def __init__(self):
        self.evaluation_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize evaluation question templates for different memory types and categories"""
        return {
            "preference_memory": {
                "movies": {
                    "memory_presence": [
                        "Does the response reflect or align with the user's preference for {item}?"
                    ],
                    "forgetting_absence": [
                        "Does the response reflect or mention the user's preference for {item}?"
                    ]
                },
                "books": {
                    "memory_presence": [
                        "Does the response reflect or align with the user's preference for {item}?"
                    ],
                    "forgetting_absence": [
                        "Does the response reflect or mention the user's preference for {item}?"
                    ]
                },
                "music": {
                    "memory_presence": [
                        "Does the response reflect or align with the user's preference for {item}?"
                    ],
                    "forgetting_absence": [
                        "Does the response reflect or mention the user's preference for {item}?"
                    ]
                },
                "travel": {
                    "memory_presence": [
                        "Does the response reflect or align with the user's preference for {item}?"
                    ],
                    "forgetting_absence": [
                        "Does the response reflect or mention the user's preference for {item}?"
                    ]
                }
            },
            "activity_memory": {
                "food_expenses": {
                    "memory_presence": [
                        "Does the response include the expense amount of ${amount} for {item}?",
                        "Is the {expense_type} expense of ${amount} mentioned in the calculation?",
                        "Does the response correctly account for the {item} expense?",
                        "Is the spending on {item} (${amount}) included in the total?"
                    ],
                    "forgetting_absence": [
                        "Does the response include the deleted expense for {item}?",
                        "Is the removed {expense_type} expense mentioned in the calculation?",
                        "Does the response mention the {item} expense that was deleted?",
                        "Is the deleted spending record for {item} included?"
                    ]
                },
                "step_tracker": {
                    "memory_presence": [
                        "Does the response include the {step_count} steps from {date}?",
                        "Is the step count of {step_count} mentioned in the calculation?",
                        "Does the response correctly account for {step_count} steps?",
                        "Is the daily step record of {step_count} included in the total?"
                    ],
                    "forgetting_absence": [
                        "Does the response include the deleted step data for {date}?",
                        "Is the removed step count of {step_count} mentioned in the calculation?",
                        "Does the response mention step data from {date} that was deleted?",
                        "Is the deleted step record for {date} included in calculations?"
                    ]
                },
                "todo_list": {
                    "memory_presence": [
                        "Does the response mention the task: {item}?",
                        "Is '{item}' listed as a remaining todo item?",
                        "Does the response include {item} in the current task list?",
                        "Is the todo item '{item}' correctly shown as active?"
                    ],
                    "forgetting_absence": [
                        "Does the response mention the deleted task: {item}?",
                        "Is '{item}' listed as a remaining todo item?",
                        "Does the response include {item} in the task list?",
                        "Is the completed/deleted task '{item}' mentioned?"
                    ]
                },
                "calendar_event": {
                    "memory_presence": [
                        "Does the response mention the upcoming event: {event_title}?",
                        "Is the event '{event_title}' on {event_date} included in upcoming events?",
                        "Does the response mention both the event '{event_title}' AND when it's scheduled ({event_date})?",
                        "Does the response correctly list {event_title} as a future event with the date {event_date}?",
                        "Is the calendar event '{event_title}' scheduled for {event_date} appropriately mentioned?"
                    ],
                    "forgetting_absence": [
                        "Does the response mention the past event: {event_title}?",
                        "Is the past event '{event_title}' listed as an upcoming event?",
                        "Does the response include {event_title} (past event)?",
                        "Is the expired calendar event '{event_title}' on {event_date} mentioned?"
                    ]
                }
            },
            "content_memory": {
                "project_proposal": {
                    "memory_presence": [
                        "Does the response include the {field}: {value}?",
                        "Is the {field} information '{value}' correctly included in the proposal?",
                        "Does the response mention the {field} details: {value}?",
                        "Is the project's {field} '{value}' appropriately included?"
                    ],
                    "forgetting_absence": [
                        "Does the response include the deleted {field}: {value}?",
                        "Is the removed {field} information '{value}' mentioned in the proposal?",
                        "Does the response mention the {field} '{value}' that was deleted?",
                        "Is the deleted project {field} '{value}' included?"
                    ]
                },
                "email_writeup": {
                    "memory_presence": [
                        "Does the response include the {field}: {value}?",
                        "Is the {field} information '{value}' correctly included in the email?",
                        "Does the response mention the {field} details: {value}?",
                        "Is the email's {field} '{value}' appropriately included?"
                    ],
                    "forgetting_absence": [
                        "Does the response include the deleted {field}: {value}?",
                        "Is the removed {field} information '{value}' mentioned in the email?",
                        "Does the response mention the {field} '{value}' that was deleted?",
                        "Is the deleted email {field} '{value}' included?"
                    ]
                },
                "social_media_post": {
                    "memory_presence": [
                        "Does the response include the {field}: {value}?",
                        "Is the {field} information '{value}' correctly included in the post?",
                        "Does the response mention the {field} details: {value}?",
                        "Is the post's {field} '{value}' appropriately included?"
                    ],
                    "forgetting_absence": [
                        "Does the response include the deleted {field}: {value}?",
                        "Is the removed {field} information '{value}' mentioned in the post?",
                        "Does the response mention the {field} '{value}' that was deleted?",
                        "Is the deleted post {field} '{value}' included?"
                    ]
                },
                "meeting_notes": {
                    "memory_presence": [
                        "Does the response include the {field}: {value}?",
                        "Is the {field} information '{value}' correctly included in the notes?",
                        "Does the response mention the {field} details: {value}?",
                        "Is the meeting's {field} '{value}' appropriately included?"
                    ],
                    "forgetting_absence": [
                        "Does the response include the deleted {field}: {value}?",
                        "Is the removed {field} information '{value}' mentioned in the notes?",
                        "Does the response mention the {field} '{value}' that was deleted?",
                        "Is the deleted meeting {field} '{value}' included?"
                    ]
                }
            },
            "goal_memory": {
                "food_expenses": {
                    "memory_presence": [
                        "Does the response mention the {subcategory} budget goal of ${goal_value}?",
                        "Is the goal amount of ${goal_value} for {subcategory} correctly referenced?",
                        "Does the response include the budget target: ${goal_value}?",
                        "Is the {subcategory} spending goal appropriately mentioned?"
                    ],
                    "forgetting_absence": [
                        "Does the response mention a deleted budget goal?",
                        "Is a removed spending target referenced in the response?",
                        "Does the response include goal information that was deleted?",
                        "Is a deleted {subcategory} goal mentioned?"
                    ]
                },
                "step_tracker": {
                    "memory_presence": [
                        "Does the response mention the {subcategory} goal of {goal_value} steps?",
                        "Is the step target of {goal_value} correctly referenced?",
                        "Does the response include the step goal: {goal_value}?",
                        "Is the {subcategory} step target appropriately mentioned?"
                    ],
                    "forgetting_absence": [
                        "Does the response mention a deleted step goal?",
                        "Is a removed step target referenced in the response?",
                        "Does the response include goal information that was deleted?",
                        "Is a deleted {subcategory} goal mentioned?"
                    ]
                }
            }
        }
    
    def generate_memory_evaluation_questions(self, question_data: Dict[str, Any]) -> List[EvaluationQuestion]:
        """Generate evaluation questions for memory evidence validation"""
        evaluation_questions = []
        
        question_type = question_data.get("question_type", "")
        memory_type = question_data.get("memory_type")
        category = question_data.get("category")
        subcategory = question_data.get("subcategory", "general")
        question_id = question_data.get("question_id")
        memory_evidence = question_data.get("memory_evidence", {})
        
        # Handle temporal comparative analysis questions (e.g., "Which week had the most steps?")
        if question_type == "temporal_comparative_analysis":
            temporal_questions = self._generate_temporal_comparative_evaluation_questions(
                question_id, memory_type, category, subcategory, memory_evidence, question_data
            )
            evaluation_questions.extend(temporal_questions)
        # Handle temporal goal analysis questions (e.g., "How many weeks met the budget?")
        elif question_type == "temporal_goal_analysis":
            temporal_goal_questions = self._generate_temporal_goal_evaluation_questions(
                question_id, memory_type, category, subcategory, memory_evidence, question_data
            )
            evaluation_questions.extend(temporal_goal_questions)
        else:
            # Generate questions for memory presence validation (standard flow)
            memory_questions = self._generate_memory_presence_questions(
                question_id, memory_type, category, subcategory, memory_evidence
            )
            evaluation_questions.extend(memory_questions)
        
        return evaluation_questions
    
    def generate_forgetting_evaluation_questions(self, question_data: Dict[str, Any]) -> List[EvaluationQuestion]:
        """Generate evaluation questions for forgetting evidence validation"""
        evaluation_questions = []
        
        memory_type = question_data.get("memory_type")
        category = question_data.get("category")
        subcategory = question_data.get("subcategory", "general")
        question_id = question_data.get("question_id")
        forgetting_evidence = question_data.get("forgetting_evidence", {})
        
        # Skip forgetting questions for categories that only have ADD operations
        if memory_type == "activity_memory" and category in ["food_expenses", "step_tracker"]:
            return evaluation_questions
        
        # Generate questions for forgetting absence validation
        forgetting_questions = self._generate_forgetting_absence_questions(
            question_id, memory_type, category, subcategory, forgetting_evidence
        )
        evaluation_questions.extend(forgetting_questions)
        
        return evaluation_questions
    
    def _generate_memory_presence_questions(self, base_question_id: str, memory_type: str, 
                                         category: str, subcategory: str, memory_evidence: Dict) -> List[EvaluationQuestion]:
        """Generate questions to validate that memory items are present in the response"""
        questions = []
        
        if memory_type == "preference_memory":
            questions.extend(self._generate_preference_memory_questions(
                base_question_id, category, subcategory, memory_evidence, "memory_presence"
            ))
        elif memory_type == "activity_memory":
            questions.extend(self._generate_activity_memory_questions(
                base_question_id, category, memory_evidence, "memory_presence"
            ))
        elif memory_type == "content_memory":
            questions.extend(self._generate_content_memory_questions(
                base_question_id, category, memory_evidence, "memory_presence"
            ))
        elif memory_type == "goal_memory":
            questions.extend(self._generate_goal_memory_questions(
                base_question_id, category, subcategory, memory_evidence, "memory_presence"
            ))
        
        return questions
    
    def _generate_forgetting_absence_questions(self, base_question_id: str, memory_type: str,
                                             category: str, subcategory: str, forgetting_evidence: Dict) -> List[EvaluationQuestion]:
        """Generate questions to validate that forgotten items are absent from the response"""
        questions = []
        
        forgotten_items = forgetting_evidence.get("forgotten_items", [])
        if not forgotten_items:
            return questions
        
        for i, forgotten_item_data in enumerate(forgotten_items):
            forgotten_item = forgotten_item_data.get("forgotten_item")
            operation_type = forgotten_item_data.get("operation_type")
            session_id = forgotten_item_data.get("session_id")
            session_date = forgotten_item_data.get("session_date")
            
            if memory_type == "preference_memory":
                question = self._create_preference_forgetting_question(
                    base_question_id, category, forgotten_item, i, operation_type, session_id, session_date
                )
            elif memory_type == "activity_memory":
                question = self._create_activity_forgetting_question(
                    base_question_id, category, forgotten_item, i, operation_type, session_id, session_date
                )
            elif memory_type == "content_memory":
                question = self._create_content_forgetting_question(
                    base_question_id, category, forgotten_item, i, operation_type, session_id, session_date
                )
            elif memory_type == "goal_memory":
                question = self._create_goal_forgetting_question(
                    base_question_id, category, forgotten_item, i, operation_type, session_id, session_date
                )
            else:
                continue
            
            if question:
                questions.append(question)
        
        return questions
    
    def _generate_preference_memory_questions(self, base_question_id: str, category: str, 
                                            subcategory: str, memory_evidence: Dict, eval_type: str) -> List[EvaluationQuestion]:
        """Generate evaluation questions for preference memory"""
        questions = []
        
        if subcategory == "general":
            # General preference questions - check overall memory items
            memory_items = memory_evidence.get("memory_items", {})
            item_count = 0
            
            for subcat, subcat_data in memory_items.items():
                if isinstance(subcat_data, dict):
                    # Handle likes
                    if "likes" in subcat_data:
                        likes = subcat_data.get("likes", [])
                        for i, like_item in enumerate(likes):
                            item_value = like_item.get("item") if isinstance(like_item, dict) else like_item
                            question = self._create_preference_evaluation_question(
                                base_question_id, category, item_value, item_count, eval_type, subcat, "like", like_item
                            )
                            questions.append(question)
                            item_count += 1
                    
                    # Handle dislikes
                    if "dislikes" in subcat_data:
                        dislikes = subcat_data.get("dislikes", [])
                        for i, dislike_item in enumerate(dislikes):
                            item_value = dislike_item.get("item") if isinstance(dislike_item, dict) else dislike_item
                            question = self._create_preference_evaluation_question(
                                base_question_id, category, item_value, item_count, eval_type, subcat, "dislike", dislike_item
                            )
                            questions.append(question)
                            item_count += 1
        else:
            # Subcategory-specific questions
            specific_items = memory_evidence.get("specific_items", [])
            subcategory_data = memory_evidence.get("subcategory_data", {})
            
            # Check likes in subcategory
            if isinstance(subcategory_data, dict) and "likes" in subcategory_data:
                likes = subcategory_data.get("likes", [])
                for i, item in enumerate(likes):
                    item_value = item.get("item") if isinstance(item, dict) else item
                    question = self._create_preference_evaluation_question(
                        base_question_id, category, item_value, i, eval_type, subcategory, "like", item
                    )
                    questions.append(question)
            
            # Check dislikes in subcategory  
            if isinstance(subcategory_data, dict) and "dislikes" in subcategory_data:
                dislikes = subcategory_data.get("dislikes", [])
                for i, item in enumerate(dislikes):
                    item_value = item.get("item") if isinstance(item, dict) else item
                    question = self._create_preference_evaluation_question(
                        base_question_id, category, item_value, len(subcategory_data.get("likes", [])) + i, eval_type, subcategory, "dislike", item
                    )
                    questions.append(question)
        
        return questions
    
    def _generate_activity_memory_questions(self, base_question_id: str, category: str, 
                                          memory_evidence: Dict, eval_type: str) -> List[EvaluationQuestion]:
        """Generate evaluation questions for activity memory"""
        questions = []
        
        if category == "food_expenses":
            if "food_expenses" in memory_evidence:
                # For total food expenses, check only the aggregated value
                if "total_amount" in memory_evidence:
                    total_amount = memory_evidence["total_amount"]
                    question = self._create_food_total_evaluation_question(
                        base_question_id, total_amount, 0, eval_type
                    )
                    questions.append(question)
            elif "expense_items" in memory_evidence:
                # Category-specific expenses - check only the category total
                if "category_total" in memory_evidence:
                    category_total = memory_evidence["category_total"]
                    expense_type = memory_evidence.get("expense_type", "unknown")
                    question = self._create_food_category_evaluation_question(
                        base_question_id, expense_type, category_total, 0, eval_type
                    )
                    questions.append(question)
        
        elif category == "step_tracker":
            if "step_data" in memory_evidence:
                # For total steps, check only the aggregated value
                if "total_steps" in memory_evidence:
                    total_steps = memory_evidence["total_steps"]
                    question = self._create_steps_total_evaluation_question(
                        base_question_id, total_steps, 0, eval_type
                    )
                    questions.append(question)
        
        elif category == "todo_list":
            if "remaining_tasks" in memory_evidence:
                tasks = memory_evidence["remaining_tasks"]
                for i, task in enumerate(tasks):
                    question = self._create_activity_evaluation_question(
                        base_question_id, category, task, i, eval_type
                    )
                    questions.append(question)
        
        elif category == "calendar_event":
            if "calendar_events" in memory_evidence:
                events = memory_evidence["calendar_events"]
                for i, event in enumerate(events):
                    question = self._create_activity_evaluation_question(
                        base_question_id, category, event, i, eval_type
                    )
                    questions.append(question)
        
        return questions
    
    def _generate_content_memory_questions(self, base_question_id: str, category: str,
                                         memory_evidence: Dict, eval_type: str) -> List[EvaluationQuestion]:
        """Generate evaluation questions for content memory"""
        questions = []
        
        content_data = memory_evidence.get("content_data", {})
        item_id = memory_evidence.get("item_id")
        
        # Generate questions for key content fields
        key_fields = self._get_content_key_fields(category, content_data)
        
        for i, (field, value) in enumerate(key_fields.items()):
            question = self._create_content_evaluation_question(
                base_question_id, category, field, value, i, eval_type, item_id
            )
            questions.append(question)
        
        return questions
    
    def _generate_goal_memory_questions(self, base_question_id: str, category: str, subcategory: str,
                                       memory_evidence: Dict, eval_type: str) -> List[EvaluationQuestion]:
        """Generate evaluation questions for goal memory"""
        questions = []
        
        goal_value = memory_evidence.get("goal_value")
        actual_value = memory_evidence.get("actual_value")
        goal_status = memory_evidence.get("goal_status", {})
        goal_session = memory_evidence.get("goal_session")
        activity_sessions = memory_evidence.get("activity_sessions", [])
        
        if goal_value is not None:
            # Question 1: Check if goal is mentioned
            question = self._create_goal_evaluation_question(
                base_question_id, category, subcategory, goal_value, actual_value, 0, eval_type,
                goal_session=goal_session, activity_sessions=activity_sessions
            )
            questions.append(question)
            
            # Question 2: Check if actual progress/amount is mentioned
            if actual_value is not None:
                progress_question = self._create_goal_progress_evaluation_question(
                    base_question_id, category, subcategory, goal_value, actual_value, 1, eval_type,
                    goal_session=goal_session, activity_sessions=activity_sessions
                )
                questions.append(progress_question)
            
            # Question 3: Check if goal status/decision is mentioned
            if goal_status:
                status_question = self._create_goal_status_evaluation_question(
                    base_question_id, category, subcategory, goal_value, actual_value, goal_status, 2, eval_type,
                    goal_session=goal_session, activity_sessions=activity_sessions
                )
                if status_question:
                    questions.append(status_question)
        
        return questions
    
    def _generate_temporal_comparative_evaluation_questions(self, base_question_id: str, memory_type: str,
                                                           category: str, subcategory: str, memory_evidence: Dict,
                                                           question_data: Dict[str, Any]) -> List[EvaluationQuestion]:
        """Generate evaluation questions for temporal comparative analysis questions
        
        These questions compare values across time periods (e.g., "Which week had the most steps?")
        Memory evidence structure:
        - max_period: The period number with the maximum value
        - max_value: The maximum value
        - period_breakdown: Dict with period numbers as keys, containing period_ref (e.g., "2nd week")
        """
        questions = []
        
        max_period = memory_evidence.get("max_period")
        max_value = memory_evidence.get("max_value")
        period_breakdown = memory_evidence.get("period_breakdown", {})
        temporal_context = question_data.get("temporal_context", {})
        comparison_type = temporal_context.get("comparison_type", "maximum")
        
        # Get period reference (e.g., "2nd week", "3rd month")
        period_ref = None
        if max_period is not None:
            # Try both string and integer keys for max_period
            period_key = str(max_period)
            if period_key in period_breakdown:
                period_ref = period_breakdown[period_key].get("period_ref")
            elif max_period in period_breakdown:
                period_ref = period_breakdown[max_period].get("period_ref")
        
        # Question 1: Check if max period is mentioned (ALWAYS generate if max_period exists)
        if max_period is not None:
            # If period_ref is missing, try to construct it from max_period
            if not period_ref:
                period_type = temporal_context.get("period_type", "period")
                # Generate period_ref from max_period (e.g., 1 -> "1st week", 2 -> "2nd week")
                ordinal_map = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th", 8: "8th", 9: "9th", 10: "10th"}
                ordinal = ordinal_map.get(max_period, f"{max_period}th")
                period_ref = f"{ordinal} {period_type}"
            
            if category == "step_tracker":
                question_text = f"Does the response mention that the {period_ref} had the most steps?"
            elif category == "food_expenses":
                expense_type = subcategory if subcategory != "general" else temporal_context.get("expense_type", "expenses")
                question_text = f"Does the response mention that the {period_ref} had the most {expense_type} spending?"
            else:
                question_text = f"Does the response mention that the {period_ref} had the maximum value?"
            
            question = EvaluationQuestion(
                question_id=f"{base_question_id}_eval_memory_presence_max_period_0",
                evaluation_question=question_text,
                expected_answer="yes",
                evaluation_type="memory_presence",
                target_item={
                    "max_period": max_period,
                    "period_ref": period_ref
                },
                context={
                    "category": category,
                    "subcategory": subcategory,
                    "memory_type": memory_type,
                    "comparison_type": comparison_type
                }
            )
            questions.append(question)
        
        # Question 2: Check if max value is mentioned
        if max_value is not None:
            if category == "step_tracker":
                question_text = f"Does the response mention the maximum step count of {max_value:,} steps?"
            elif category == "food_expenses":
                expense_type = subcategory if subcategory != "general" else temporal_context.get("expense_type", "expenses")
                question_text = f"Does the response mention the maximum {expense_type} spending amount of ${max_value:.2f}?"
            else:
                question_text = f"Does the response mention the maximum value of {max_value}?"
            
            question = EvaluationQuestion(
                question_id=f"{base_question_id}_eval_memory_presence_max_value_0",
                evaluation_question=question_text,
                expected_answer="yes",
                evaluation_type="memory_presence",
                target_item={
                    "max_value": max_value
                },
                context={
                    "category": category,
                    "subcategory": subcategory,
                    "memory_type": memory_type,
                    "comparison_type": comparison_type
                }
            )
            questions.append(question)
        
        return questions
    
    def _generate_temporal_goal_evaluation_questions(self, base_question_id: str, memory_type: str,
                                                    category: str, subcategory: str, memory_evidence: Dict,
                                                    question_data: Dict[str, Any]) -> List[EvaluationQuestion]:
        """Generate evaluation questions for temporal goal analysis questions
        
        These questions analyze goal compliance across time periods (e.g., "How many weeks met the budget?")
        Memory evidence structure:
        - met_count: Number of periods that met the goal
        - exceeded_count: Number of periods that exceeded the goal
        - total_periods: Total number of periods analyzed
        """
        questions = []
        
        met_count = memory_evidence.get("met_count")
        exceeded_count = memory_evidence.get("exceeded_count")
        total_periods = memory_evidence.get("total_periods")
        temporal_context = question_data.get("temporal_context", {})
        comparison_type = temporal_context.get("comparison_type", "budget_compliance")
        
        # Check if met_count is mentioned
        if met_count is not None:
            period_type = temporal_context.get("period_type", "period")
            period_plural = f"{period_type}s" if met_count != 1 else period_type
            
            if category == "food_expenses":
                question_text = f"Does the response mention that you met your {subcategory} budget in {met_count} {period_plural}?"
            elif category == "step_tracker":
                # For step tracker, use appropriate wording based on subcategory
                if subcategory == "daily_steps":
                    question_text = f"Does the response mention that you met your daily step goal in {met_count} {period_plural}?"
                else:
                    question_text = f"Does the response mention that you met your {subcategory} step goal in {met_count} {period_plural}?"
            else:
                question_text = f"Does the response mention that you met your goal in {met_count} {period_plural}?"
            
            question = EvaluationQuestion(
                question_id=f"{base_question_id}_eval_memory_presence_met_count_0",
                evaluation_question=question_text,
                expected_answer="yes",
                evaluation_type="memory_presence",
                target_item={
                    "met_count": met_count
                },
                context={
                    "category": category,
                    "subcategory": subcategory,
                    "memory_type": memory_type,
                    "comparison_type": comparison_type
                }
            )
            questions.append(question)
        
        return questions
    
    def _create_preference_evaluation_question(self, base_question_id: str, category: str, 
                                             item_value: str, index: int, eval_type: str, subcategory: str, preference_type: str = "like", item_data: Dict = None) -> EvaluationQuestion:
        """Create a preference memory evaluation question"""
        templates = self.evaluation_templates["preference_memory"][category][eval_type]
        template = templates[index % len(templates)]
        
        # Modify question based on preference type
        if preference_type == "dislike" and eval_type == "memory_presence":
            # For dislikes, we want to check that the AI AVOIDS recommending them
            if "mention or consider" in template:
                question_text = template.replace("mention or consider", "avoid recommending").format(item=item_value)
            elif "referenced" in template:
                question_text = template.replace("referenced", "avoided").format(item=item_value)
            elif "show awareness" in template:
                question_text = template.replace("show awareness", "avoid suggesting").format(item=item_value)
            else:
                question_text = f"Does the response avoid recommending {item_value} (which the user dislikes)?"
        else:
            question_text = template.format(item=item_value)
        
        # For dislikes in memory presence, we expect "yes" (the AI should avoid them)
        # For likes in memory presence, we expect "yes" (the AI should consider them)
        # For forgetting absence, we expect "no" (forgotten items should not be mentioned)
        if eval_type == "memory_presence":
            expected_answer = "yes"  # Both likes (should consider) and dislikes (should avoid)
        else:
            expected_answer = "no"  # Forgotten items should not be mentioned
        
        # Build context with session linking information
        context = {
            "category": category,
            "subcategory": subcategory,
            "memory_type": "preference_memory",
            "preference_type": preference_type
        }
        
        # Add session linking information to context
        if item_data and isinstance(item_data, dict):
            if "session_id" in item_data:
                context["session_id"] = item_data["session_id"]
            if "created_at" in item_data:
                context["created_at"] = item_data["created_at"]
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_{eval_type}_{preference_type}_{index}",
            evaluation_question=question_text,
            expected_answer=expected_answer,
            evaluation_type=eval_type,
            target_item=item_value,
            context=context
        )
    
    def _create_preference_forgetting_question(self, base_question_id: str, category: str,
                                             forgotten_item: Any, index: int, operation_type: str, session_id: int = None, session_date: str = None) -> EvaluationQuestion:
        """Create a preference forgetting evaluation question"""
        templates = self.evaluation_templates["preference_memory"][category]["forgetting_absence"]
        template = templates[index % len(templates)]
        
        question_text = template.format(item=forgotten_item)
        
        # Build context with session linking information
        context = {
            "category": category,
            "memory_type": "preference_memory",
            "operation_type": operation_type
        }
        
        # Add session information for linking to specific deletion/update
        if session_id is not None:
            context["session_id"] = session_id
        if session_date:
            context["session_date"] = session_date
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_forgetting_{index}",
            evaluation_question=question_text,
            expected_answer="no",  # Forgotten items should not be mentioned
            evaluation_type="forgetting_absence",
            target_item=forgotten_item,
            context=context
        )
    
    def _create_activity_evaluation_question(self, base_question_id: str, category: str,
                                           item_data: Dict, index: int, eval_type: str) -> EvaluationQuestion:
        """Create an activity memory evaluation question"""
        templates = self.evaluation_templates["activity_memory"][category][eval_type]
        template = templates[index % len(templates)]
        
        # Extract relevant fields based on category
        if category == "food_expenses":
            amount = item_data.get("amount", 0)
            expense_type = item_data.get("expense_type", "unknown")
            item_desc = item_data.get("item", "expense")
            question_text = template.format(amount=amount, expense_type=expense_type, item=item_desc)
        elif category == "step_tracker":
            step_count = item_data.get("step_count", 0)
            date = item_data.get("date", "unknown date")
            question_text = template.format(step_count=step_count, date=date)
        elif category == "todo_list":
            item_desc = item_data.get("description", item_data.get("task", str(item_data))) if isinstance(item_data, dict) else str(item_data)
            question_text = template.format(item=item_desc)
        elif category == "calendar_event":
            event_title = item_data.get("event_name", item_data.get("event_title", "unknown event"))
            event_date = item_data.get("calculated_event_date", item_data.get("date", "unknown date"))
            question_text = template.format(event_title=event_title, event_date=event_date)
        else:
            question_text = template.format(item=str(item_data))
        
        # memory_presence expects "yes" (item should be mentioned)
        # forgetting_absence expects "no" (item should not be mentioned)
        expected_answer = "yes" if eval_type == "memory_presence" else "no"
        
        # Build context with session linking information
        context = {
            "category": category,
            "memory_type": "activity_memory"
        }
        
        # Add session linking information to context
        if isinstance(item_data, dict):
            if "session_id" in item_data:
                context["session_id"] = item_data["session_id"]
            if "created_at" in item_data:
                context["created_at"] = item_data["created_at"]
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_{eval_type}_{index}",
            evaluation_question=question_text,
            expected_answer=expected_answer,
            evaluation_type=eval_type,
            target_item=item_data,
            context=context
        )
    
    def _create_activity_forgetting_question(self, base_question_id: str, category: str,
                                           forgotten_item: Any, index: int, operation_type: str, session_id: int = None, session_date: str = None) -> EvaluationQuestion:
        """Create an activity forgetting evaluation question"""
        templates = self.evaluation_templates["activity_memory"][category]["forgetting_absence"]
        template = templates[index % len(templates)]
        
        if isinstance(forgotten_item, dict):
            if category == "food_expenses":
                item_desc = forgotten_item.get("item", "expense")
                question_text = template.format(item=item_desc)
            elif category == "step_tracker":
                date = forgotten_item.get("date", "unknown date")
                step_count = forgotten_item.get("step_count", 0)
                question_text = template.format(date=date, step_count=step_count)
            elif category == "todo_list":
                item_desc = forgotten_item.get("description", forgotten_item.get("task", str(forgotten_item)))
                question_text = template.format(item=item_desc)
            elif category == "calendar_event":
                event_title = forgotten_item.get("event_name", forgotten_item.get("event_title", "unknown event"))
                event_date = forgotten_item.get("calculated_event_date", forgotten_item.get("date", "unknown date"))
                question_text = template.format(event_title=event_title, event_date=event_date)
            else:
                question_text = template.format(item=str(forgotten_item))
        else:
            question_text = template.format(item=str(forgotten_item))
        
        # Build context with session linking information
        context = {
            "category": category,
            "memory_type": "activity_memory",
            "operation_type": operation_type
        }
        
        # Add session information for linking to specific deletion
        if session_id is not None:
            context["session_id"] = session_id
        if session_date:
            context["session_date"] = session_date
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_forgetting_{index}",
            evaluation_question=question_text,
            expected_answer="no",  # Forgotten items should not be mentioned
            evaluation_type="forgetting_absence",
            target_item=forgotten_item,
            context=context
        )
    
    def _create_content_evaluation_question(self, base_question_id: str, category: str,
                                          field: str, value: Any, index: int, eval_type: str, item_id: str = None) -> EvaluationQuestion:
        """Create a content memory evaluation question"""
        templates = self.evaluation_templates["content_memory"][category][eval_type]
        template = templates[index % len(templates)]
        
        # Get category-specific field names for better questions
        field_display = self._get_content_field_display_name(category, field)
        
        # Format value appropriately for different types
        if isinstance(value, list):
            # For list fields, format as comma-separated items - show all items
            formatted_value = ", ".join(str(item) for item in value)
        else:
            # For non-list fields, use complete text
            formatted_value = str(value)
        
        question_text = template.format(field=field_display, value=formatted_value)
        # memory_presence expects "yes" (item should be mentioned)
        # forgetting_absence expects "no" (item should not be mentioned)
        expected_answer = "yes" if eval_type == "memory_presence" else "no"
        
        # Build context with item linking information
        context = {
            "category": category,
            "memory_type": "content_memory",
            "field": field
        }
        
        # Add item_id to context for linking to specific content item
        if item_id is not None:
            context["item_id"] = item_id
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_{eval_type}_{index}",
            evaluation_question=question_text,
            expected_answer=expected_answer,
            evaluation_type=eval_type,
            target_item={"field": field, "value": value},
            context=context
        )
    
    def _create_content_forgetting_question(self, base_question_id: str, category: str,
                                          forgotten_item: Dict, index: int, operation_type: str, session_id: int = None, session_date: str = None) -> EvaluationQuestion:
        """Create a content forgetting evaluation question"""
        templates = self.evaluation_templates["content_memory"][category]["forgetting_absence"]
        template = templates[index % len(templates)]
        
        field = forgotten_item.get("field", "unknown field")
        value = forgotten_item.get("value", "unknown value")
        item_id = forgotten_item.get("item_id")
        field_display = self._get_content_field_display_name(category, field)
        
        # Format value - handle lists and strings
        if isinstance(value, list):
            formatted_value = ", ".join(str(item) for item in value)
        else:
            formatted_value = str(value)
        
        question_text = template.format(field=field_display, value=formatted_value)
        
        # Build context with session linking information
        context = {
            "category": category,
            "memory_type": "content_memory",
            "operation_type": operation_type,
            "field": field
        }
        
        # Add item_id to context for linking to specific content item
        if item_id is not None:
            context["item_id"] = item_id
        
        # Add session information for linking to specific deletion
        if session_id is not None:
            context["session_id"] = session_id
        if session_date:
            context["session_date"] = session_date
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_forgetting_{index}",
            evaluation_question=question_text,
            expected_answer="no",  # Forgotten items should not be mentioned
            evaluation_type="forgetting_absence",
            target_item=forgotten_item,
            context=context
        )
    
    def _create_goal_evaluation_question(self, base_question_id: str, category: str, subcategory: str,
                                       goal_value: float, actual_value: float, index: int, eval_type: str,
                                       goal_session: Dict = None, activity_sessions: List[Dict] = None) -> EvaluationQuestion:
        """Create a goal memory evaluation question - checks if goal is mentioned"""
        templates = self.evaluation_templates["goal_memory"][category][eval_type]
        template = templates[index % len(templates)]
        
        question_text = template.format(
            subcategory=subcategory,
            goal_value=goal_value,
            actual_value=actual_value
        )
        # memory_presence expects "yes" (item should be mentioned)
        # forgetting_absence expects "no" (item should not be mentioned)
        expected_answer = "yes" if eval_type == "memory_presence" else "no"
        
        # Build context with session evidence
        context = {
            "category": category,
            "subcategory": subcategory,
            "memory_type": "goal_memory",
            "check_type": "goal_mention"
        }
        
        # Add goal session evidence
        if goal_session and goal_session.get("session_id"):
            context["goal_session"] = {
                "session_id": goal_session.get("session_id"),
                "session_date": goal_session.get("session_date")
            }
        
        # Add activity sessions evidence
        if activity_sessions:
            context["activity_sessions"] = activity_sessions
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_{eval_type}_{index}",
            evaluation_question=question_text,
            expected_answer=expected_answer,
            evaluation_type=eval_type,
            target_item={"goal_value": goal_value, "actual_value": actual_value},
            context=context
        )
    
    def _create_goal_progress_evaluation_question(self, base_question_id: str, category: str, subcategory: str,
                                                 goal_value: float, actual_value: float, index: int, eval_type: str,
                                                 goal_session: Dict = None, activity_sessions: List[Dict] = None) -> EvaluationQuestion:
        """Create evaluation question for goal progress (actual vs goal)"""
        if category == "step_tracker":
            # For step tracker: check if actual steps vs goal is mentioned
            question_text = f"Does the response mention the actual step count ({actual_value:,.0f} steps) compared to the goal ({goal_value:,.0f} steps)?"
        elif category == "food_expenses":
            # For food expenses: check if actual spending vs budget is mentioned
            question_text = f"Does the response mention the actual spending amount (${actual_value:.2f}) compared to the budget goal (${goal_value:.2f})?"
        else:
            # Generic fallback
            question_text = f"Does the response mention the actual value ({actual_value}) compared to the goal ({goal_value})?"
        
        expected_answer = "yes" if eval_type == "memory_presence" else "no"
        
        # Build context with session evidence
        context = {
            "category": category,
            "subcategory": subcategory,
            "memory_type": "goal_memory",
            "check_type": "goal_progress"
        }
        
        # Add goal session evidence
        if goal_session and goal_session.get("session_id"):
            context["goal_session"] = {
                "session_id": goal_session.get("session_id"),
                "session_date": goal_session.get("session_date")
            }
        
        # Add activity sessions evidence
        if activity_sessions:
            context["activity_sessions"] = activity_sessions
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_{eval_type}_progress_{index}",
            evaluation_question=question_text,
            expected_answer=expected_answer,
            evaluation_type=eval_type,
            target_item={
                "goal_value": goal_value,
                "actual_value": actual_value,
                "difference": actual_value - goal_value
            },
            context=context
        )
    
    def _create_goal_status_evaluation_question(self, base_question_id: str, category: str, subcategory: str,
                                                goal_value: float, actual_value: float, goal_status: Dict, 
                                                index: int, eval_type: str,
                                                goal_session: Dict = None, activity_sessions: List[Dict] = None) -> EvaluationQuestion:
        """Create evaluation question for goal status/decision"""
        is_goal_met = goal_status.get("is_goal_met", False)
        status = goal_status.get("status", "")
        progress_percentage = goal_status.get("progress_percentage", 0)
        
        if category == "step_tracker":
            # For step tracker: check if goal_met status is mentioned
            if is_goal_met:
                question_text = f"Does the response indicate that the step goal was MET (actual: {actual_value:,.0f} steps vs goal: {goal_value:,.0f} steps)?"
            else:
                question_text = f"Does the response indicate that the step goal was NOT MET (actual: {actual_value:,.0f} steps vs goal: {goal_value:,.0f} steps)?"
        
        elif category == "food_expenses":
            # For food expenses: check if under_budget/over_budget status is mentioned
            if actual_value <= goal_value:
                question_text = f"Does the response indicate that spending is UNDER BUDGET (actual: ${actual_value:.2f} vs budget: ${goal_value:.2f})?"
            else:
                question_text = f"Does the response indicate that spending is OVER BUDGET (actual: ${actual_value:.2f} vs budget: ${goal_value:.2f})?"
        
        else:
            # Generic fallback
            if is_goal_met:
                question_text = f"Does the response indicate that the goal was MET?"
            else:
                question_text = f"Does the response indicate that the goal was NOT MET?"
        
        expected_answer = "yes" if eval_type == "memory_presence" else "no"
        
        # Build context with session evidence
        context = {
            "category": category,
            "subcategory": subcategory,
            "memory_type": "goal_memory",
            "check_type": "goal_status"
        }
        
        # Add goal session evidence
        if goal_session and goal_session.get("session_id"):
            context["goal_session"] = {
                "session_id": goal_session.get("session_id"),
                "session_date": goal_session.get("session_date")
            }
        
        # Add activity sessions evidence
        if activity_sessions:
            context["activity_sessions"] = activity_sessions
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_{eval_type}_status_{index}",
            evaluation_question=question_text,
            expected_answer=expected_answer,
            evaluation_type=eval_type,
            target_item={
                "goal_value": goal_value,
                "actual_value": actual_value,
                "is_goal_met": is_goal_met,
                "status": status,
                "progress_percentage": progress_percentage
            },
            context=context
        )
    
    def _create_goal_forgetting_question(self, base_question_id: str, category: str,
                                       forgotten_item: Any, index: int, operation_type: str, session_id: int = None, session_date: str = None) -> EvaluationQuestion:
        """Create a goal forgetting evaluation question"""
        templates = self.evaluation_templates["goal_memory"][category]["forgetting_absence"]
        template = templates[index % len(templates)]
        
        # Extract subcategory from forgotten item if available
        subcategory = "unknown"
        if isinstance(forgotten_item, dict):
            subcategory = forgotten_item.get("subcategory", "unknown")
        
        question_text = template.format(subcategory=subcategory)
        
        # Build context with session linking information
        context = {
            "category": category,
            "memory_type": "goal_memory",
            "operation_type": operation_type
        }
        
        # Add session information for linking to specific deletion
        if session_id is not None:
            context["session_id"] = session_id
        if session_date:
            context["session_date"] = session_date
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_forgetting_{index}",
            evaluation_question=question_text,
            expected_answer="no",  # Forgotten items should not be mentioned
            evaluation_type="forgetting_absence",
            target_item=forgotten_item,
            context=context
        )
    
    def _get_content_key_fields(self, category: str, content_data: Dict) -> Dict[str, Any]:
        """Get ALL fields for content evaluation - generate questions for every field in content_data"""
        key_fields = {}
        
        # Include ALL fields from content_data, not just a subset
        for field, value in content_data.items():
            # Skip internal metadata fields that shouldn't be evaluated
            if not field.startswith('_'):
                key_fields[field] = value
        
        return key_fields
    
    def _get_content_field_display_name(self, category: str, field: str) -> str:
        """Get user-friendly display name for content fields"""
        display_names = {
            "project_proposal": {
                "project_title": "project title",
                "project_description": "project description",
                "budget": "budget",
                "timeline": "timeline",
                "stakeholders": "stakeholders",
                "objectives": "objectives"
            },
            "email_writeup": {
                "email_purpose": "email purpose",
                "recipient_list": "recipient list",
                "key_points": "key points",
                "email_subject": "email subject"
            },
            "social_media_post": {
                "platform": "platform",
                "main_message": "main message",
                "hashtags": "hashtags",
                "content_type": "content type"
            },
            "meeting_notes": {
                "meeting_title": "meeting title",
                "agenda_items": "agenda items",
                "key_decisions": "key decisions",
                "action_items": "action items",
                "attendees": "attendees"
            }
        }
        
        return display_names.get(category, {}).get(field, field)
    
    def generate_all_evaluation_questions(self, question_data: Dict[str, Any]) -> List[EvaluationQuestion]:
        """Generate all evaluation questions (memory + forgetting) for a given question"""
        all_evaluations = []
        
        # Generate memory presence questions
        memory_evaluations = self.generate_memory_evaluation_questions(question_data)
        all_evaluations.extend(memory_evaluations)
        
        # Generate forgetting absence questions
        forgetting_evaluations = self.generate_forgetting_evaluation_questions(question_data)
        all_evaluations.extend(forgetting_evaluations)
        
        return all_evaluations
    
    def format_evaluation_questions_for_output(self, evaluation_questions: List[EvaluationQuestion]) -> Dict[str, Any]:
        """Format evaluation questions for JSON output"""
        formatted_questions = []
        
        for eval_q in evaluation_questions:
            formatted_q = {
                "evaluation_question_id": eval_q.question_id,
                "evaluation_question": eval_q.evaluation_question,
                "expected_answer": eval_q.expected_answer,
                "evaluation_type": eval_q.evaluation_type,
                "target_item": eval_q.target_item,
                "context": eval_q.context
            }
            formatted_questions.append(formatted_q)
        
        return {
            "evaluation_questions": formatted_questions,
            "total_evaluation_questions": len(formatted_questions),
            "memory_presence_questions": len([q for q in evaluation_questions if q.evaluation_type == "memory_presence"]),
            "forgetting_absence_questions": len([q for q in evaluation_questions if q.evaluation_type == "forgetting_absence"])
        }
    
    def _create_food_total_evaluation_question(self, base_question_id: str, total_amount: float, 
                                             index: int, eval_type: str) -> EvaluationQuestion:
        """Create evaluation question for total food expenses"""
        if eval_type == "memory_presence":
            question_text = f"Does the response mention the total food spending amount of ${total_amount:.2f}?"
            expected_answer = "yes"
        else:  # forgetting_absence
            question_text = f"Does the response avoid mentioning the incorrect total of ${total_amount:.2f}?"
            expected_answer = "yes"
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_{eval_type}_total_{index}",
            evaluation_question=question_text,
            expected_answer=expected_answer,
            evaluation_type=eval_type,
            target_item={"total_amount": total_amount},
            context={
                "category": "food_expenses",
                "memory_type": "activity_memory"
            }
        )
    
    def _create_food_category_evaluation_question(self, base_question_id: str, expense_type: str, 
                                                category_total: float, index: int, eval_type: str) -> EvaluationQuestion:
        """Create evaluation question for category-specific food expenses"""
        if eval_type == "memory_presence":
            question_text = f"Does the response mention the {expense_type} spending amount of ${category_total:.2f}?"
            expected_answer = "yes"
        else:  # forgetting_absence
            question_text = f"Does the response avoid mentioning the incorrect {expense_type} total of ${category_total:.2f}?"
            expected_answer = "yes"
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_{eval_type}_{expense_type}_{index}",
            evaluation_question=question_text,
            expected_answer=expected_answer,
            evaluation_type=eval_type,
            target_item={"expense_type": expense_type, "category_total": category_total},
            context={
                "category": "food_expenses",
                "memory_type": "activity_memory"
            }
        )
    
    def _create_steps_total_evaluation_question(self, base_question_id: str, total_steps: int, 
                                              index: int, eval_type: str) -> EvaluationQuestion:
        """Create evaluation question for total step count"""
        if eval_type == "memory_presence":
            question_text = f"Does the response mention the total step count of {total_steps:,} steps?"
            expected_answer = "yes"
        else:  # forgetting_absence
            question_text = f"Does the response avoid mentioning the incorrect step count of {total_steps:,}?"
            expected_answer = "yes"
        
        return EvaluationQuestion(
            question_id=f"{base_question_id}_eval_{eval_type}_total_{index}",
            evaluation_question=question_text,
            expected_answer=expected_answer,
            evaluation_type=eval_type,
            target_item={"total_steps": total_steps},
            context={
                "category": "step_tracker",
                "memory_type": "activity_memory"
            }
        )
