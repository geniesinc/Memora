#!/usr/bin/env python3
"""
Question Generation Pipeline Validator

This script validates the question generation pipeline to ensure:
1. Sessions are traversed correctly
2. Evidence is collected properly (memory, forgetting, session evidence)
3. Questions are formed with complete structure
4. Session evidence matches across activity and goal questions
5. No inconsistencies or missing data
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict


class QuestionGenerationValidator:
    """Validate question generation pipeline for bugs and inconsistencies"""
    
    def __init__(self, session_directory: str):
        self.session_directory = Path(session_directory)
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
        
        # Load data
        self.questions_data = None
        self.memory_states = None
        self.new_sessions = None
        
        self._load_data()
    
    def _load_data(self):
        """Load all required data files"""
        print("📂 Loading data files...")
        
        # Load questions - prefer evaluation_questions if available, otherwise unified_questions
        questions_file = list(self.session_directory.glob("evaluation_questions_*.json"))
        file_type = "evaluation"
        if not questions_file:
            questions_file = list(self.session_directory.glob("unified_questions_*.json"))
            file_type = "unified"
        if not questions_file:
            raise FileNotFoundError(f"No questions file found in {self.session_directory}")
        
        with open(questions_file[0]) as f:
            raw_data = json.load(f)
            
            # Handle different formats
            if file_type == "evaluation":
                # evaluation_questions: {"questions": {"remembering": [...], "reasoning": [...], "recommending": [...]}}
                self.questions_data = raw_data
            else:
                # unified_questions: {"questions_by_task_type": {"Remembering": [...], ...}}
                # Convert to evaluation format for consistent processing
                if "questions_by_task_type" in raw_data:
                    self.questions_data = {
                        "questions": {
                            "remembering": raw_data["questions_by_task_type"].get("Remembering", []),
                            "reasoning": raw_data["questions_by_task_type"].get("Reasoning", []),
                            "recommending": raw_data["questions_by_task_type"].get("Recommending", [])
                        }
                    }
                elif "all_questions" in raw_data:
                    # Group by task_type
                    grouped = {"remembering": [], "reasoning": [], "recommending": []}
                    for q in raw_data["all_questions"]:
                        task_type = q.get("task_type", "").lower()
                        if task_type in grouped:
                            grouped[task_type].append(q)
                    self.questions_data = {"questions": grouped}
                else:
                    # Fallback
                    self.questions_data = {"questions": {"remembering": [], "reasoning": [], "recommending": []}}
        
        print(f"  ✅ Loaded questions: {questions_file[0].name} ({file_type} format)")
        
        # Load memory states
        memory_states_file = self.session_directory / "memory_states_by_session.json"
        if memory_states_file.exists():
            with open(memory_states_file) as f:
                data = json.load(f)
                # Handle nested structure (new format) or flat structure (old format)
                if isinstance(data, dict) and 'memory_states' in data:
                    self.memory_states = data['memory_states']
                else:
                    self.memory_states = data
            print(f"  ✅ Loaded memory states: {len(self.memory_states)} sessions")
        else:
            print(f"  ⚠️  Memory states file not found")
        
        # Load new sessions
        new_sessions_file = self.session_directory / "new_sessions.json"
        if new_sessions_file.exists():
            with open(new_sessions_file) as f:
                data = json.load(f)
                # Handle nested structure (new format) or list structure (old format)
                if isinstance(data, dict) and 'sessions' in data:
                    sessions_list = data['sessions']
                elif isinstance(data, list):
                    sessions_list = data
                else:
                    sessions_list = []
                # Convert to dict for quick lookup
                self.new_sessions = {s["id"]: s for s in sessions_list if isinstance(s, dict) and "id" in s}
            print(f"  ✅ Loaded sessions: {len(self.new_sessions)} sessions")
        else:
            print(f"  ⚠️  New sessions file not found")
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks"""
        print("\n" + "="*70)
        print("🔍 VALIDATING QUESTION GENERATION PIPELINE")
        print("="*70)
        
        # 1. Validate question structure
        self._validate_question_structure()
        
        # 2. Validate memory evidence
        self._validate_memory_evidence()
        
        # 3. Validate session evidence (the new feature)
        self._validate_session_evidence()
        
        # 4. Validate session traversal consistency
        self._validate_session_traversal()
        
        # 5. Validate activity-goal session consistency
        self._validate_activity_goal_consistency()
        
        # 6. Validate evaluation questions completeness
        self._validate_evaluation_questions()
        
        # Print results
        self._print_results()
        
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": dict(self.stats),
            "success": len(self.errors) == 0
        }
    
    def _validate_question_structure(self):
        """Validate basic question structure"""
        print("\n📋 Validating question structure...")
        
        # evaluation_questions structure: {"questions": {"remembering": [...], "reasoning": [...], "recommending": [...]}}
        questions_by_category = self.questions_data.get("questions", {})
        
        for category, questions in questions_by_category.items():
            self.stats[f"questions_{category}"] = len(questions)
            
            for q in questions:
                # Required fields
                required_fields = ["question_id", "question", "answer", "question_type", 
                                 "category", "memory_type", "session_id", "session_date", 
                                 "memory_evidence"]
                
                for field in required_fields:
                    if field not in q:
                        self.errors.append({
                            "type": "MISSING_REQUIRED_FIELD",
                            "question_id": q.get("question_id", "unknown"),
                            "field": field,
                            "category": category
                        })
                
                # Validate memory_evidence is not empty
                if "memory_evidence" in q:
                    if not q["memory_evidence"]:
                        self.errors.append({
                            "type": "EMPTY_MEMORY_EVIDENCE",
                            "question_id": q.get("question_id"),
                            "category": category
                        })
        
        total_questions = sum(self.stats.get(f"questions_{cat}", 0) for cat in ["remembering", "reasoning", "recommending"])
        print(f"  ✅ Validated structure for {total_questions} questions")
    
    def _validate_memory_evidence(self):
        """Validate memory evidence completeness"""
        print("\n🧠 Validating memory evidence...")
        
        questions_by_type = self.questions_data.get("questions", {})
        
        for task_type, questions in questions_by_type.items():
            for q in questions:
                memory_evidence = q.get("memory_evidence", {})
                memory_type = q.get("memory_type")
                category = q.get("category")
                question_id = q.get("question_id")
                question_type = q.get("question_type", "")  # Get question_type for comparative detection
                
                # Validate based on memory type
                if memory_type == "activity_memory":
                    if category == "food_expenses":
                        if "expense_items" not in memory_evidence and "food_expenses" not in memory_evidence:
                            self.warnings.append({
                                "type": "MISSING_ACTIVITY_ITEMS",
                                "question_id": question_id,
                                "category": category,
                                "message": "No expense_items or food_expenses in memory_evidence"
                            })
                    elif category == "step_tracker":
                        if "step_data" not in memory_evidence:
                            self.warnings.append({
                                "type": "MISSING_ACTIVITY_ITEMS",
                                "question_id": question_id,
                                "category": category,
                                "message": "No step_data in memory_evidence"
                            })
                
                elif memory_type == "goal_memory":
                    # Check if this is a comparative question (different structure)
                    is_comparative = (
                        "comparative_goal" in question_id or 
                        question_type in ["temporal_goal_analysis", "temporal_comparative_analysis"]
                    )
                    
                    if is_comparative:
                        # Comparative questions have different memory_evidence structure
                        # They should have: budget_timeline, periods_met, periods_exceeded, etc.
                        # OR: max_period, max_value, period_breakdown (for activity comparisons)
                        has_comparative_fields = (
                            "budget_timeline" in memory_evidence or 
                            "periods_met" in memory_evidence or
                            "period_breakdown" in memory_evidence
                        )
                        if not has_comparative_fields:
                            self.warnings.append({
                                "type": "MISSING_COMPARATIVE_FIELDS",
                                "question_id": question_id,
                                "message": "Comparative question missing expected fields (budget_timeline, periods_met, or period_breakdown)"
                            })
                    else:
                        # Regular goal questions require: goal_value, actual_value, goal_status
                        required_goal_fields = ["goal_value", "actual_value", "goal_status"]
                        for field in required_goal_fields:
                            if field not in memory_evidence:
                                self.errors.append({
                                    "type": "MISSING_GOAL_FIELD",
                                    "question_id": question_id,
                                    "field": field
                                })
        
        print(f"  ✅ Validated memory evidence")
    
    def _validate_session_evidence(self):
        """Validate session evidence (goal_session and activity_sessions)"""
        print("\n📍 Validating session evidence...")
        
        questions_by_type = self.questions_data.get("questions", {})
        
        for task_type, questions in questions_by_type.items():
            for q in questions:
                memory_evidence = q.get("memory_evidence", {})
                memory_type = q.get("memory_type")
                question_id = q.get("question_id")
                
                # Check activity memory has session_ids
                if memory_type == "activity_memory":
                    category = q.get("category")
                    
                    if category == "food_expenses":
                        expense_items = memory_evidence.get("expense_items", [])
                        for i, item in enumerate(expense_items):
                            if isinstance(item, dict) and "session_id" not in item:
                                self.errors.append({
                                    "type": "MISSING_SESSION_ID_IN_ACTIVITY",
                                    "question_id": question_id,
                                    "category": category,
                                    "item_index": i,
                                    "message": "Activity item missing session_id"
                                })
                                self.stats["activity_items_missing_session_id"] += 1
                            else:
                                self.stats["activity_items_with_session_id"] += 1
                    
                    elif category == "step_tracker":
                        step_data = memory_evidence.get("step_data", [])
                        for i, item in enumerate(step_data):
                            if isinstance(item, dict) and "session_id" not in item:
                                self.errors.append({
                                    "type": "MISSING_SESSION_ID_IN_ACTIVITY",
                                    "question_id": question_id,
                                    "category": category,
                                    "item_index": i,
                                    "message": "Step data missing session_id"
                                })
                                self.stats["activity_items_missing_session_id"] += 1
                            else:
                                self.stats["activity_items_with_session_id"] += 1
                
                # Check goal memory has session evidence
                elif memory_type == "goal_memory":
                    # Skip session evidence checks for comparative questions (they have different structure)
                    question_type = q.get("question_type", "")
                    is_comparative = (
                        "comparative_goal" in question_id or 
                        question_type in ["temporal_goal_analysis", "temporal_comparative_analysis"]
                    )
                    
                    if is_comparative:
                        # Comparative questions don't have goal_session/activity_sessions
                        # They have budget_timeline, periods_met, periods_exceeded, etc.
                        continue
                    
                    goal_session = memory_evidence.get("goal_session")
                    activity_sessions = memory_evidence.get("activity_sessions", [])
                    
                    # Check goal_session
                    if goal_session is None:
                        self.warnings.append({
                            "type": "MISSING_GOAL_SESSION",
                            "question_id": question_id,
                            "message": "goal_session is None"
                        })
                        self.stats["goals_missing_session"] += 1
                    else:
                        if "session_id" not in goal_session or "session_date" not in goal_session:
                            self.errors.append({
                                "type": "INCOMPLETE_GOAL_SESSION",
                                "question_id": question_id,
                                "message": "goal_session missing session_id or session_date"
                            })
                        self.stats["goals_with_session"] += 1
                    
                    # Check activity_sessions
                    actual_value = memory_evidence.get("actual_value", 0)
                    if actual_value > 0 and len(activity_sessions) == 0:
                        self.errors.append({
                            "type": "EMPTY_ACTIVITY_SESSIONS",
                            "question_id": question_id,
                            "actual_value": actual_value,
                            "message": f"actual_value={actual_value} but activity_sessions is empty"
                        })
                        self.stats["goals_with_empty_activity_sessions"] += 1
                    elif len(activity_sessions) > 0:
                        self.stats["goals_with_activity_sessions"] += 1
                        self.stats["total_activity_sessions"] += len(activity_sessions)
                        
                        # Validate each activity session
                        for i, sess in enumerate(activity_sessions):
                            if not isinstance(sess, dict):
                                self.errors.append({
                                    "type": "INVALID_ACTIVITY_SESSION",
                                    "question_id": question_id,
                                    "session_index": i,
                                    "message": "activity_session is not a dict"
                                })
                                continue
                            
                            if "session_id" not in sess:
                                self.errors.append({
                                    "type": "ACTIVITY_SESSION_MISSING_SESSION_ID",
                                    "question_id": question_id,
                                    "session_index": i
                                })
                            if "session_date" not in sess:
                                self.errors.append({
                                    "type": "ACTIVITY_SESSION_MISSING_SESSION_DATE",
                                    "question_id": question_id,
                                    "session_index": i
                                })
        
        print(f"  ✅ Validated session evidence")
        print(f"     - Activity items with session_id: {self.stats.get('activity_items_with_session_id', 0)}")
        print(f"     - Activity items missing session_id: {self.stats.get('activity_items_missing_session_id', 0)}")
        print(f"     - Goals with session: {self.stats.get('goals_with_session', 0)}")
        print(f"     - Goals missing session: {self.stats.get('goals_missing_session', 0)}")
        print(f"     - Goals with activity_sessions: {self.stats.get('goals_with_activity_sessions', 0)}")
        print(f"     - Goals with empty activity_sessions: {self.stats.get('goals_with_empty_activity_sessions', 0)}")
    
    def _validate_session_traversal(self):
        """Validate session IDs are consistent and properly sequenced"""
        print("\n🔄 Validating session traversal...")
        
        if not self.memory_states:
            print("  ⚠️  Skipping: No memory states available")
            return
        
        questions_by_type = self.questions_data.get("questions", {})
        all_question_session_ids = set()
        
        # Collect all session IDs from questions
        for task_type, questions in questions_by_type.items():
            for q in questions:
                session_id = q.get("session_id")
                if session_id:
                    all_question_session_ids.add(session_id)
        
        # Get session IDs from memory states (skip 'metadata' key)
        memory_state_session_ids = set()
        for sid in self.memory_states.keys():
            if sid != 'metadata':
                try:
                    memory_state_session_ids.add(int(sid))
                except ValueError:
                    # Skip non-numeric keys
                    continue
        
        # Check if question sessions are valid
        invalid_sessions = all_question_session_ids - memory_state_session_ids
        if invalid_sessions:
            self.errors.append({
                "type": "INVALID_QUESTION_SESSION_IDS",
                "invalid_sessions": sorted(list(invalid_sessions)),
                "message": f"Questions reference {len(invalid_sessions)} non-existent sessions"
            })
        
        self.stats["total_memory_sessions"] = len(memory_state_session_ids)
        self.stats["total_question_sessions"] = len(all_question_session_ids)
        
        print(f"  ✅ Validated session traversal")
        print(f"     - Memory states: {len(memory_state_session_ids)} sessions")
        print(f"     - Question sessions: {len(all_question_session_ids)} unique sessions")
    
    def _validate_activity_goal_consistency(self):
        """Validate that activity sessions in goals match actual activity questions"""
        print("\n🔗 Validating activity-goal session consistency...")
        
        questions_by_type = self.questions_data.get("questions", {})
        
        # Collect activity sessions by category/subcategory
        activity_sessions_map = defaultdict(set)
        
        for task_type, questions in questions_by_type.items():
            for q in questions:
                memory_type = q.get("memory_type")
                category = q.get("category")
                subcategory = q.get("subcategory")
                
                if memory_type == "activity_memory":
                    memory_evidence = q.get("memory_evidence", {})
                    
                    if category == "food_expenses":
                        expense_items = memory_evidence.get("expense_items", [])
                        for item in expense_items:
                            if isinstance(item, dict) and "session_id" in item:
                                expense_type = item.get("expense_type", subcategory)
                                key = f"food_expenses_{expense_type}"
                                activity_sessions_map[key].add(item["session_id"])
                    
                    elif category == "step_tracker":
                        step_data = memory_evidence.get("step_data", [])
                        for item in step_data:
                            if isinstance(item, dict) and "session_id" in item:
                                key = f"step_tracker_{subcategory}"
                                activity_sessions_map[key].add(item["session_id"])
        
        # Check goal questions reference correct activity sessions
        for task_type, questions in questions_by_type.items():
            for q in questions:
                memory_type = q.get("memory_type")
                
                if memory_type == "goal_memory":
                    category = q.get("category")
                    subcategory = q.get("subcategory")
                    memory_evidence = q.get("memory_evidence", {})
                    activity_sessions = memory_evidence.get("activity_sessions", [])
                    question_id = q.get("question_id")
                    
                    key = f"{category}_{subcategory}"
                    expected_sessions = activity_sessions_map.get(key, set())
                    
                    goal_session_ids = set(s["session_id"] for s in activity_sessions if isinstance(s, dict) and "session_id" in s)
                    
                    # Check if goal's activity sessions are a subset of actual activity sessions
                    if goal_session_ids and expected_sessions:
                        extra_sessions = goal_session_ids - expected_sessions
                        if extra_sessions:
                            self.errors.append({
                                "type": "EXTRA_ACTIVITY_SESSIONS_IN_GOAL",
                                "question_id": question_id,
                                "category": category,
                                "subcategory": subcategory,
                                "extra_sessions": sorted(list(extra_sessions)),
                                "message": "Goal references activity sessions not found in activity questions"
                            })
        
        print(f"  ✅ Validated activity-goal consistency")
        print(f"     - Activity categories tracked: {len(activity_sessions_map)}")
    
    def _validate_evaluation_questions(self):
        """Validate evaluation questions completeness"""
        print("\n✅ Validating evaluation questions...")
        
        questions_by_type = self.questions_data.get("questions", {})
        
        for task_type, questions in questions_by_type.items():
            for q in questions:
                evaluation = q.get("evaluation")
                question_id = q.get("question_id")
                memory_type = q.get("memory_type")
                question_type = q.get("question_type", "")
                
                # Skip evaluation checks for comparative questions (they don't have evaluation sections)
                is_comparative = (
                    "comparative_" in question_id or 
                    question_type in ["temporal_goal_analysis", "temporal_comparative_analysis"]
                )
                
                if is_comparative:
                    # Comparative questions are analytical, not memory recall
                    continue
                
                if not evaluation:
                    self.errors.append({
                        "type": "MISSING_EVALUATION",
                        "question_id": question_id,
                        "message": "Question missing evaluation section"
                    })
                    continue
                
                eval_questions = evaluation.get("evaluation_questions", [])
                if not eval_questions:
                    self.errors.append({
                        "type": "EMPTY_EVALUATION_QUESTIONS",
                        "question_id": question_id,
                        "message": "No evaluation questions generated"
                    })
                    continue
                
                # Validate evaluation question structure
                for i, eval_q in enumerate(eval_questions):
                    required_fields = ["evaluation_question_id", "evaluation_question", 
                                     "expected_answer", "evaluation_type", "context"]
                    
                    for field in required_fields:
                        if field not in eval_q:
                            self.errors.append({
                                "type": "MISSING_EVAL_QUESTION_FIELD",
                                "question_id": question_id,
                                "eval_index": i,
                                "field": field
                            })
                    
                    # Check context for goal questions includes session evidence
                    if memory_type == "goal_memory":
                        context = eval_q.get("context", {})
                        check_type = context.get("check_type")
                        
                        # All goal evaluation questions should have session evidence
                        if "goal_session" not in context and "activity_sessions" not in context:
                            self.warnings.append({
                                "type": "EVAL_QUESTION_MISSING_SESSION_EVIDENCE",
                                "question_id": question_id,
                                "eval_question_id": eval_q.get("evaluation_question_id"),
                                "check_type": check_type,
                                "message": "Goal evaluation question missing session evidence"
                            })
        
        print(f"  ✅ Validated evaluation questions")
    
    def _print_results(self):
        """Print validation results"""
        print("\n" + "="*70)
        print("📊 VALIDATION RESULTS")
        print("="*70)
        
        # Print stats
        print("\n📈 Statistics:")
        for key, value in sorted(self.stats.items()):
            print(f"  {key}: {value}")
        
        # Print errors
        if self.errors:
            print(f"\n❌ ERRORS FOUND: {len(self.errors)}")
            error_types = defaultdict(int)
            for error in self.errors:
                error_types[error["type"]] += 1
            
            print("\nError Summary:")
            for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
                print(f"  {error_type}: {count}")
            
            print("\nFirst 10 errors:")
            for error in self.errors[:10]:
                print(f"\n  Type: {error['type']}")
                for key, value in error.items():
                    if key != "type":
                        print(f"    {key}: {value}")
        else:
            print("\n✅ NO ERRORS FOUND!")
        
        # Print warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            warning_types = defaultdict(int)
            for warning in self.warnings:
                warning_types[warning["type"]] += 1
            
            print("\nWarning Summary:")
            for warning_type, count in sorted(warning_types.items(), key=lambda x: -x[1]):
                print(f"  {warning_type}: {count}")
        else:
            print("\n✅ NO WARNINGS")
        
        # Final verdict
        print("\n" + "="*70)
        if len(self.errors) == 0:
            print("✅ VALIDATION PASSED - Pipeline is bug-free!")
        else:
            print("❌ VALIDATION FAILED - Bugs detected in pipeline")
        print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate question generation pipeline")
    parser.add_argument("session_directory", help="Path to session directory with questions")
    parser.add_argument("--output", "-o", help="Output file for validation report (JSON)")
    
    args = parser.parse_args()
    
    try:
        validator = QuestionGenerationValidator(args.session_directory)
        results = validator.validate_all()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Report saved to: {args.output}")
        
        # Exit with error code if validation failed
        sys.exit(0 if results["success"] else 1)
        
    except Exception as e:
        print(f"\n❌ Error running validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

