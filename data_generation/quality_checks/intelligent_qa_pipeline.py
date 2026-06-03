#!/usr/bin/env python3
"""
Intelligent Question Generation Pipeline with Quality Gates

This script orchestrates the complete pipeline:
1. Generate sessions (session_simulator.py)
2. Validate session quality (memory_state_validator.py)
3. Generate questions (unified_question_generator.py)
4. Validate question pipeline (validate_question_generation_pipeline.py)
5. Analyze question evidence
6. Select questions with right evidence types
7. Retry if validation fails or not enough questions

Quality Gates:
- Session quality: Memory state consistency, operation correctness
- Question pipeline: Structure, evidence collection, session traversal
- Evidence requirements: Memory/forgetting evidence per category

Goal: Get exactly 5 questions per category with proper evidence:
- Remembering: 5 questions with BOTH memory + forgetting evidence
- Reasoning: 5 questions with memory evidence (no forgetting needed)
- Recommending: 5 questions with BOTH memory + forgetting evidence

Usage:
    python quality_checks/intelligent_qa_pipeline.py --persona software_engineer
    python quality_checks/intelligent_qa_pipeline.py --persona all
"""

import subprocess
import json
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import time

PERSONAS = [
    "academic_researcher",
    "business_executive",
    "content_writer",
    "creative_designer",
    "financial_analyst",
    "management_consultant",
    "marketing_manager",
    "sales_manager",
    "software_engineer",
    "startup_founder"
]

class IntelligentQAPipeline:
    def __init__(self, persona: str, start_date: str, end_date: str, config_file: str, max_retries: int = 3):
        self.persona = persona
        self.start_date = start_date
        self.end_date = end_date
        self.config_file = config_file
        self.max_retries = max_retries
        
        # Use a FIXED output directory that will be reused/overwritten on retries
        # Format: output/qa_pipeline_{persona}_{start_date}_{end_date}
        date_str = start_date.replace('-', '') + '_' + end_date.replace('-', '')
        self.output_dir = f"output/qa_pipeline_{persona}_{date_str}"
        self.questions_file = None
    
    def cleanup_failed_run(self):
        """Delete output directory if run was unsuccessful"""
        if self.output_dir and Path(self.output_dir).exists():
            try:
                print(f"\n🧹 Cleaning up failed run: {self.output_dir}")
                shutil.rmtree(self.output_dir)
                print(f"✅ Deleted unsuccessful output directory")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete {self.output_dir}: {e}")
        
    def run_session_simulator(self) -> Dict[str, Any]:
        """Step 1: Generate sessions"""
        print("\n" + "="*80)
        print(f"📝 STEP 1: Generating Sessions for {self.persona}")
        print("="*80)
        
        try:
            result = subprocess.run(
                [
                    "python", "session_simulator.py",
                    "--config", self.config_file,
                    "--persona", self.persona,
                    "--start-date", self.start_date,
                    "--end-date", self.end_date
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Session simulator failed: {result.stderr[:500]}"
                }
            
            # Extract output directory from stdout/stderr
            output_dir = None
            all_lines = result.stdout.split('\n') + result.stderr.split('\n')
            
            for line in all_lines:
                if 'output/sessions_' in line:
                    # Handle file paths - extract directory from file path
                    if '/memory_states_by_session.json' in line:
                        # Extract directory from file path
                        dir_path = line.split('/memory_states_by_session.json')[0]
                        if dir_path.startswith('output/sessions_'):
                            output_dir = dir_path
                            break
                    elif '/new_sessions.json' in line:
                        # Extract directory from file path
                        dir_path = line.split('/new_sessions.json')[0]
                        if dir_path.startswith('output/sessions_'):
                            output_dir = dir_path
                            break
                    elif 'Output folder:' in line or 'output/sessions_' in line:
                        # Try to extract directory path
                        parts = line.split('output/sessions_')
                        if len(parts) > 1:
                            dir_part = parts[1].split()[0].split('/')[0].rstrip('/:,')
                            # Make sure it's not a file extension
                            if not dir_part.endswith('.json'):
                                output_dir = f"output/sessions_{dir_part}"
                                break
            
            # Last resort: check if directory exists by pattern matching timestamp
            if not output_dir:
                # Try to find the most recent directory matching the pattern
                from pathlib import Path
                output_base = Path("output")
                if output_base.exists():
                    # Find directories matching sessions_YYYYMMDD_HHMMSS_persona pattern
                    matching_dirs = [
                        d for d in output_base.iterdir() 
                        if d.is_dir() and d.name.startswith("sessions_") and self.persona in d.name
                    ]
                    if matching_dirs:
                        # Get the most recent one created in the last minute
                        now = time.time()
                        recent_dirs = [d for d in matching_dirs if (now - d.stat().st_mtime) < 60]
                        if recent_dirs:
                            output_dir = str(recent_dirs[0])
                        else:
                            output_dir = str(max(matching_dirs, key=lambda p: p.stat().st_mtime))
            
            if not output_dir:
                return {
                    "success": False,
                    "error": "Could not determine output directory"
                }
            
            self.output_dir = output_dir
            print(f"✅ Sessions generated successfully")
            print(f"📁 Output: {output_dir}")
            print(f"📅 Date Range: {self.start_date} to {self.end_date}")
            print(f"⚙️  Config: {self.config_file}")
            
            return {
                "success": True,
                "output_dir": output_dir
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_sessions(self) -> Dict[str, Any]:
        """Step 2: Validate session quality"""
        print("\n" + "="*80)
        print(f"✅ STEP 2: Validating Session Quality")
        print("="*80)
        
        try:
            result = subprocess.run(
                ["python", "quality_checks/memory_state_validator.py", self.output_dir],
                capture_output=True,
                text=True
            )
            
            # Parse validation results
            passed = result.returncode == 0
            errors_count = 0
            
            for line in result.stdout.split('\n'):
                if 'ERRORS FOUND:' in line:
                    try:
                        errors_count = int(line.split(':')[1].strip().split()[0])
                    except:
                        pass
            
            if passed:
                print(f"✅ Quality validation PASSED")
                print(f"🎯 Zero errors detected")
            else:
                print(f"❌ Quality validation FAILED")
                print(f"🔴 {errors_count} errors found")
            
            return {
                "success": passed,
                "errors_count": errors_count
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_questions(self) -> Dict[str, Any]:
        """Step 3: Generate questions"""
        print("\n" + "="*80)
        print(f"❓ STEP 3: Generating Questions")
        print("="*80)
        
        try:
            # Determine timeline from config file name
            timeline = "weekly"  # default
            if "monthly" in self.config_file.lower():
                timeline = "monthly"
            elif "quarterly" in self.config_file.lower():
                timeline = "quarterly"
            
            result = subprocess.run(
                [
                    "python", "question_generation/unified_question_generator.py",
                    "--session_directory", self.output_dir,
                    "--timeline", timeline
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Question generation failed: {result.stderr[:500]}"
                }
            
            # Find the generated questions file in the fixed output directory
            output_path = Path(self.output_dir)
            # Look for evaluation_questions first (preferred), then unified_questions (fallback)
            question_files = list(output_path.glob("evaluation_questions_*.json"))
            if not question_files:
                question_files = list(output_path.glob("unified_questions_*.json"))
            
            if not question_files:
                return {
                    "success": False,
                    "error": f"No question file generated in {self.output_dir}"
                }
            
            # Use the most recent file
            questions_file = max(question_files, key=lambda p: p.stat().st_mtime)
            self.questions_file = str(questions_file)
            
            print(f"✅ Questions generated successfully")
            print(f"📄 File: {questions_file.name}")
            print(f"📁 Location: {self.output_dir}")
            
            return {
                "success": True,
                "questions_file": str(questions_file)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_question_pipeline(self) -> Dict[str, Any]:
        """Step 4: Validate question generation pipeline integrity"""
        print("\n" + "="*80)
        print(f"✅ STEP 4: Validating Question Pipeline")
        print("="*80)
        
        try:
            result = subprocess.run(
                ["python", "quality_checks/validate_question_generation_pipeline.py", self.output_dir],
                capture_output=True,
                text=True
            )
            
            # Parse validation results
            passed = result.returncode == 0
            errors_count = 0
            warnings_count = 0
            
            for line in result.stdout.split('\n'):
                if 'ERRORS FOUND:' in line:
                    try:
                        errors_count = int(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                elif 'WARNINGS:' in line and 'NO WARNINGS' not in line:
                    try:
                        warnings_count = int(line.split(':')[1].strip().split()[0])
                    except:
                        pass
            
            if passed:
                print(f"✅ Pipeline validation PASSED")
                print(f"🎯 Zero errors detected")
                if warnings_count > 0:
                    print(f"⚠️  {warnings_count} warnings (non-critical)")
            else:
                print(f"❌ Pipeline validation FAILED")
                print(f"🔴 {errors_count} errors found")
                # Print first few errors for debugging
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if 'First 10 errors:' in line:
                        print(f"\n{line}")
                        for j in range(i+1, min(i+20, len(lines))):
                            if lines[j].strip():
                                print(lines[j])
            
            return {
                "success": passed,
                "errors_count": errors_count,
                "warnings_count": warnings_count
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_questions(self) -> Dict[str, Any]:
        """Step 5: Analyze question quality and evidence from the generated file"""
        print("\n" + "="*80)
        print(f"🔍 STEP 5: Analyzing Question Evidence")
        print("="*80)
        
        if not self.questions_file:
            return {
                "success": False,
                "error": "No questions file available for analysis"
            }
        
        try:
            print(f"📖 Reading questions from: {self.questions_file}")
            with open(self.questions_file, 'r') as f:
                data = json.load(f)
            
            # Handle the actual file structure: questions_by_task_type or all_questions
            questions = []
            if isinstance(data, dict):
                # Try questions_by_task_type first (organized by task type)
                if "questions_by_task_type" in data:
                    task_type_questions = data["questions_by_task_type"]
                    # Flatten all questions from all task types
                    for task_type, q_list in task_type_questions.items():
                        if isinstance(q_list, list):
                            questions.extend(q_list)
                    print(f"📊 Loaded from questions_by_task_type")
                # Fallback to all_questions
                elif "all_questions" in data:
                    questions = data["all_questions"] if isinstance(data["all_questions"], list) else []
                    print(f"📊 Loaded from all_questions")
                # Fallback to questions
                elif "questions" in data:
                    questions = data["questions"] if isinstance(data["questions"], list) else []
                    print(f"📊 Loaded from questions")
            elif isinstance(data, list):
                questions = data
                print(f"📊 Loaded from root list")
            
            print(f"📊 Found {len(questions)} total questions in file")
            
            if len(questions) == 0:
                # Debug: show file structure
                print(f"⚠️  Debug: File structure keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                return {
                    "success": False,
                    "error": "No questions found in generated file"
                }
            
            # Determine timeline from config file
            timeline = "weekly"  # default
            if "monthly" in self.config_file.lower():
                timeline = "monthly"
            elif "quarterly" in self.config_file.lower():
                timeline = "quarterly"
            
            # Filter out goal_progress_analysis questions for monthly/quarterly timelines
            # These questions are designed for single-period analysis, not multi-week comparisons
            if timeline in ["monthly", "quarterly"]:
                original_count = len(questions)
                questions = [
                    q for q in questions 
                    if q.get("question_type") != "goal_progress_analysis"
                ]
                filtered_count = original_count - len(questions)
                if filtered_count > 0:
                    print(f"🚫 Filtered out {filtered_count} goal_progress_analysis questions (not suitable for {timeline} timeline)")
                    print(f"📊 Remaining questions: {len(questions)}")
            
            # Debug: show structure of first question
            if len(questions) > 0:
                print(f"📋 Sample question keys: {list(questions[0].keys())[:10]}")
            
            # Categorize questions by reasoning type and evidence
            categorized = {
                "remembering": {"memory_and_forgetting": [], "memory_only": [], "forgetting_only": []},
                "reasoning": {"memory_evidence": [], "no_evidence": []},
                "recommending": {"memory_and_forgetting": [], "memory_only": [], "forgetting_only": []}
            }
            
            for q in questions:
                # Get reasoning type - use task_type field (the actual field name)
                reasoning_type = (
                    q.get("task_type") or  # Primary field
                    q.get("reasoning_type") or 
                    q.get("type") or 
                    ""
                ).lower()
                
                # Check memory evidence - structure varies by memory type
                memory_evidence = q.get("memory_evidence", {})
                has_memory = False
                if isinstance(memory_evidence, dict):
                    # Different memory types use different field names:
                    # - Preference: item_count, memory_items
                    # - Activity: expense_count, food_expenses, expense_items, category_total
                    # - Content: item_count, content_items
                    # - Goal: goal_count, goal_items
                    # Check for any count fields > 0
                    count_fields = [
                        memory_evidence.get("item_count", 0),
                        memory_evidence.get("expense_count", 0),
                        memory_evidence.get("goal_count", 0),
                        memory_evidence.get("total_amount", 0),
                        memory_evidence.get("category_total", 0)
                    ]
                    has_count = any(c > 0 for c in count_fields)
                    
                    # Check for any item/list fields with content
                    item_fields = [
                        memory_evidence.get("memory_items", {}),
                        memory_evidence.get("food_expenses", []),
                        memory_evidence.get("expense_items", []),
                        memory_evidence.get("content_items", []),
                        memory_evidence.get("goal_items", [])
                    ]
                    has_items = False
                    for item_field in item_fields:
                        if isinstance(item_field, dict) and item_field:
                            has_items = True
                            break
                        elif isinstance(item_field, list) and len(item_field) > 0:
                            has_items = True
                            break
                    
                    # Memory evidence exists if there's any count or items
                    has_memory = has_count or has_items
                    
                    # Also check if dict has any meaningful keys (not empty)
                    if not has_memory and memory_evidence:
                        # Check if there are any non-zero numeric values or non-empty collections
                        for key, value in memory_evidence.items():
                            if isinstance(value, (int, float)) and value > 0:
                                has_memory = True
                                break
                            elif isinstance(value, (list, dict)) and value:
                                has_memory = True
                                break
                                
                elif isinstance(memory_evidence, bool):
                    has_memory = memory_evidence
                else:
                    # Fallback to boolean field if it exists
                    has_memory = q.get("has_memory_evidence", False)
                
                # Check forgetting evidence - actual structure uses forgetting_evidence dict
                forgetting_evidence = q.get("forgetting_evidence", {})
                has_forgetting = False
                if isinstance(forgetting_evidence, dict):
                    # Check if there are forgotten items
                    total_forgotten = forgetting_evidence.get("total_forgotten_items", 0)
                    forgotten_items = forgetting_evidence.get("forgotten_items", [])
                    has_forgetting = total_forgotten > 0 or len(forgotten_items) > 0
                elif isinstance(forgetting_evidence, bool):
                    has_forgetting = forgetting_evidence
                else:
                    # Fallback to boolean field if it exists
                    has_forgetting = q.get("has_forgetting_evidence", False)
                
                if reasoning_type == "remembering":
                    if has_memory and has_forgetting:
                        categorized["remembering"]["memory_and_forgetting"].append(q)
                    elif has_memory:
                        categorized["remembering"]["memory_only"].append(q)
                    elif has_forgetting:
                        categorized["remembering"]["forgetting_only"].append(q)
                
                elif reasoning_type == "reasoning":
                    # Reasoning questions ONLY need memory evidence (no forgetting needed)
                    if has_memory:
                        categorized["reasoning"]["memory_evidence"].append(q)
                    else:
                        categorized["reasoning"]["no_evidence"].append(q)
                
                elif reasoning_type == "recommending":
                    if has_memory and has_forgetting:
                        categorized["recommending"]["memory_and_forgetting"].append(q)
                    elif has_memory:
                        categorized["recommending"]["memory_only"].append(q)
                    elif has_forgetting:
                        categorized["recommending"]["forgetting_only"].append(q)
            
            # Count what we have
            remembering_good = len(categorized["remembering"]["memory_and_forgetting"])
            reasoning_good = len(categorized["reasoning"]["memory_evidence"])  # Only memory needed, no forgetting
            recommending_good = len(categorized["recommending"]["memory_and_forgetting"])
            
            print(f"\n📊 Question Analysis:")
            print(f"  Remembering (need 5 with memory + forgetting): {remembering_good} found")
            print(f"  Reasoning (need 5 with memory only, NO forgetting): {reasoning_good} found")
            print(f"  Recommending (need 5 with memory + forgetting): {recommending_good} found")
            
            # Debug: Show breakdown
            print(f"\n📋 Detailed Breakdown:")
            print(f"  Remembering:")
            print(f"    - Memory + Forgetting: {remembering_good}")
            print(f"    - Memory only: {len(categorized['remembering']['memory_only'])}")
            print(f"    - Forgetting only: {len(categorized['remembering']['forgetting_only'])}")
            print(f"  Reasoning:")
            print(f"    - Memory evidence: {reasoning_good}")
            print(f"    - No evidence: {len(categorized['reasoning']['no_evidence'])}")
            print(f"  Recommending:")
            print(f"    - Memory + Forgetting: {recommending_good}")
            print(f"    - Memory only: {len(categorized['recommending']['memory_only'])}")
            print(f"    - Forgetting only: {len(categorized['recommending']['forgetting_only'])}")
            
            # Check if we have enough (need 5 per category)
            sufficient = (remembering_good >= 5 and 
                         reasoning_good >= 5 and 
                         recommending_good >= 5)
            
            if sufficient:
                print(f"\n✅ SUFFICIENT questions found!")
            else:
                print(f"\n⚠️  INSUFFICIENT questions - need to retry")
                if remembering_good < 5:
                    print(f"    Need {5 - remembering_good} more Remembering")
                if reasoning_good < 5:
                    print(f"    Need {5 - reasoning_good} more Reasoning")
                if recommending_good < 5:
                    print(f"    Need {5 - recommending_good} more Recommending")
            
            return {
                "success": sufficient,
                "categorized": categorized,
                "counts": {
                    "remembering": remembering_good,
                    "reasoning": reasoning_good,
                    "recommending": recommending_good
                },
                "target_per_category": 5,
                "total_questions": len(questions)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _count_evidence_sessions(self, question: Dict[str, Any]) -> int:
        """Count total number of unique sessions referenced in memory and forgetting evidence"""
        session_ids = set()
        
        # Extract from memory_evidence
        memory_evidence = question.get("memory_evidence", {})
        if isinstance(memory_evidence, dict):
            # Check for goal_session
            goal_session = memory_evidence.get("goal_session")
            if goal_session and isinstance(goal_session, dict):
                session_id = goal_session.get("session_id")
                if session_id is not None:
                    session_ids.add(int(session_id))
            
            # Check for activity_sessions
            activity_sessions = memory_evidence.get("activity_sessions", [])
            if isinstance(activity_sessions, list):
                for session in activity_sessions:
                    if isinstance(session, dict):
                        session_id = session.get("session_id")
                        if session_id is not None:
                            session_ids.add(int(session_id))
            
            # Check for calendar_events, food_expenses, step_tracker, remaining_tasks
            for key in ["calendar_events", "food_expenses", "step_tracker", "remaining_tasks", "expense_items"]:
                if key in memory_evidence:
                    items = memory_evidence[key]
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                session_id = item.get("session_id")
                                if session_id is not None:
                                    session_ids.add(int(session_id))
            
            # Check for budget_timeline (for comparative goal questions)
            # budget_timeline is {day_offset: budget_value}, not {day_offset: {items: [...]}}
            # For comparative goal questions, we need to count actual expense sessions
            budget_timeline = memory_evidence.get("budget_timeline")
            if isinstance(budget_timeline, dict) and len(budget_timeline) > 0:
                # This is a comparative goal question - need to count expense sessions
                # Load session data to count actual expense sessions
                expense_sessions = self._count_expense_sessions_for_comparative_question(question, memory_evidence)
                session_ids.update(expense_sessions)
                
                # Count goal sessions from budget_timeline
                # Each key represents a day when budget was set - map to session_id
                goal_sessions = self._count_goal_sessions_from_budget_timeline(question, budget_timeline)
                session_ids.update(goal_sessions)
            
            # Check for period_breakdown (for comparative questions)
            period_breakdown = memory_evidence.get("period_breakdown")
            if isinstance(period_breakdown, dict):
                for period_key, period_data in period_breakdown.items():
                    if isinstance(period_data, dict):
                        items = period_data.get("items", [])
                        for item in items:
                            if isinstance(item, dict):
                                session_id = item.get("session_id")
                                if session_id is not None:
                                    session_ids.add(int(session_id))
            
            # Check memory_items for preference memory (nested structure)
            memory_items = memory_evidence.get("memory_items", {})
            if isinstance(memory_items, dict):
                for subcategory_data in memory_items.values():
                    if isinstance(subcategory_data, dict):
                        for pref_type in ["likes", "dislikes"]:
                            items = subcategory_data.get(pref_type, [])
                            if isinstance(items, list):
                                for item in items:
                                    if isinstance(item, dict):
                                        session_id = item.get("session_id")
                                        if session_id is not None:
                                            session_ids.add(int(session_id))
        
        # Extract from forgetting_evidence
        forgetting_evidence = question.get("forgetting_evidence", {})
        if isinstance(forgetting_evidence, dict):
            forgotten_items = forgetting_evidence.get("forgotten_items", [])
            if isinstance(forgotten_items, list):
                for forgotten_item in forgotten_items:
                    if isinstance(forgotten_item, dict):
                        session_id = forgotten_item.get("session_id")
                        if session_id is not None:
                            session_ids.add(int(session_id))
        
        # Also include the question's own session_id
        question_session_id = question.get("session_id")
        if question_session_id is not None:
            session_ids.add(int(question_session_id))
        
        return len(session_ids)
    
    def _count_expense_sessions_for_comparative_question(self, question: Dict[str, Any], memory_evidence: Dict[str, Any]) -> set:
        """Count actual expense sessions for comparative goal questions by looking up session data"""
        expense_sessions = set()
        
        try:
            # Get question metadata
            subcategory = question.get("subcategory", "")
            session_date = question.get("session_date", "")
            
            if not subcategory or not session_date:
                return expense_sessions
            
            # Load session data from output directory
            sessions_file = Path(self.output_dir) / "new_sessions.json"
            if not sessions_file.exists():
                return expense_sessions
            
            with open(sessions_file, 'r') as f:
                sessions_data = json.load(f)
            
            sessions = sessions_data.get("sessions", []) if isinstance(sessions_data, dict) else sessions_data
            
            # Count coffee expense sessions in date range
            for session in sessions:
                if not isinstance(session, dict):
                    continue
                
                session_id = session.get("id")
                session_type = session.get("type", "")
                session_category = session.get("category", "")
                session_date_str = session.get("date", "")
                
                # Check if within date range (from start_date to question session_date)
                if not session_date_str or not (self.start_date <= session_date_str <= session_date):
                    continue
                
                # Count expense sessions matching the subcategory
                if session_type == 'activity_memory' and session_category == 'food_expenses':
                    operation_details = session.get('operation_details', {})
                    if isinstance(operation_details, dict):
                        item = operation_details.get('item', {})
                        if isinstance(item, dict) and item.get('expense_type') == subcategory:
                            expense_sessions.add(session_id)
        
        except Exception as e:
            # If lookup fails, return empty set (conservative)
            pass
        
        return expense_sessions
    
    def _count_goal_sessions_from_budget_timeline(self, question: Dict[str, Any], budget_timeline: Dict) -> set:
        """Count goal sessions from budget_timeline by mapping day offsets to session_ids"""
        goal_sessions = set()
        
        try:
            # Load session data to map day offsets to session_ids
            sessions_file = Path(self.output_dir) / "new_sessions.json"
            if not sessions_file.exists():
                return goal_sessions
            
            with open(sessions_file, 'r') as f:
                sessions_data = json.load(f)
            
            sessions = sessions_data.get("sessions", []) if isinstance(sessions_data, dict) else sessions_data
            
            # Find start date from first session
            start_date = None
            for session in sessions:
                if isinstance(session, dict):
                    session_date_str = session.get("date", "")
                    if session_date_str:
                        start_date = datetime.strptime(session_date_str, "%Y-%m-%d")
                        break
            
            if not start_date:
                return goal_sessions
            
            # Map budget_timeline day offsets to session_ids
            subcategory = question.get("subcategory", "")
            for day_offset_str, budget_value in budget_timeline.items():
                try:
                    day_offset = int(day_offset_str)
                    # Calculate target date
                    target_date = start_date + timedelta(days=day_offset - 1)
                    target_date_str = target_date.strftime("%Y-%m-%d")
                    
                    # Find session on that date that set/updated the goal
                    for session in sessions:
                        if not isinstance(session, dict):
                            continue
                        
                        session_id = session.get("id")
                        session_type = session.get("type", "")
                        session_category = session.get("category", "")
                        session_date_str = session.get("date", "")
                        
                        if (session_type == 'goal_memory' and 
                            session_category == 'food_expenses' and
                            session_date_str == target_date_str):
                            operation_details = session.get('operation_details', {})
                            if isinstance(operation_details, dict):
                                goal_data = operation_details.get('goal_data', {})
                                if isinstance(goal_data, dict) and goal_data.get('subcategory') == subcategory:
                                    goal_sessions.add(session_id)
                                    break
                except (ValueError, TypeError):
                    continue
        
        except Exception as e:
            # If lookup fails, return empty set (conservative)
            pass
        
        return goal_sessions
    
    def _select_questions_stratified(self, pool: List[Dict[str, Any]], target_count: int, 
                                     category_name: str) -> List[Dict[str, Any]]:
        """
        Select questions using stratified approach: group by memory_type/category,
        then select top questions from each group to ensure diversity.
        """
        if len(pool) <= target_count:
            # Not enough questions, just return sorted by evidence
            return sorted(pool, key=lambda q: self._count_evidence_sessions(q), reverse=True)
        
        # Group questions by memory_type and category
        groups = defaultdict(list)
        
        for q in pool:
            memory_type = q.get("memory_type", "unknown")
            category = q.get("category", "unknown")
            group_key = f"{memory_type}/{category}"
            groups[group_key].append(q)
        
        # Sort each group by evidence count (descending)
        for group_key in groups:
            groups[group_key] = sorted(
                groups[group_key],
                key=lambda q: self._count_evidence_sessions(q),
                reverse=True
            )
        
        # Select questions: try to get at least 1 from each group, then fill remaining slots
        selected = []
        groups_list = list(groups.items())
        
        # First pass: take top 1 from each group (if we have enough groups)
        if len(groups_list) <= target_count:
            for group_key, group_questions in groups_list:
                if group_questions:
                    selected.append(group_questions[0])
                    group_questions.pop(0)
        else:
            # More groups than target - take top from largest groups first
            groups_list_sorted = sorted(
                groups_list,
                key=lambda x: len(x[1]),
                reverse=True
            )
            for group_key, group_questions in groups_list_sorted[:target_count]:
                if group_questions:
                    selected.append(group_questions[0])
                    group_questions.pop(0)
        
        # Second pass: fill remaining slots by taking next highest evidence questions
        # across all groups
        remaining_slots = target_count - len(selected)
        if remaining_slots > 0:
            # Collect remaining questions from all groups
            remaining_questions = []
            for group_key, group_questions in groups.items():
                remaining_questions.extend(group_questions)
            
            # Sort by evidence count and take top remaining
            remaining_questions_sorted = sorted(
                remaining_questions,
                key=lambda q: self._count_evidence_sessions(q),
                reverse=True
            )
            selected.extend(remaining_questions_sorted[:remaining_slots])
        
        # Show selection breakdown
        selected_groups = defaultdict(int)
        for q in selected:
            memory_type = q.get("memory_type", "unknown")
            category = q.get("category", "unknown")
            group_key = f"{memory_type}/{category}"
            selected_groups[group_key] += 1
        
        print(f"✅ Selected {len(selected)} {category_name} questions from {len(selected_groups)} groups:")
        for group_key, count in sorted(selected_groups.items()):
            print(f"   - {group_key}: {count} question(s)")
        
        return selected
    
    def select_and_save_questions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Select exactly 5 questions per category using stratified selection (diversity + evidence)"""
        print("\n" + "="*80)
        print(f"💾 STEP 6: Selecting and Saving Final Questions")
        print("="*80)
        
        try:
            categorized = analysis["categorized"]
            
            # Select exactly 5 questions per category using stratified approach
            selected = {}
            
            # Remembering: Select 5 from memory_and_forgetting
            remembering_pool = categorized["remembering"]["memory_and_forgetting"]
            if len(remembering_pool) >= 5:
                selected["remembering"] = self._select_questions_stratified(
                    remembering_pool, 5, "remembering"
                )
            else:
                selected["remembering"] = sorted(
                    remembering_pool,
                    key=lambda q: self._count_evidence_sessions(q),
                    reverse=True
                )
                print(f"⚠️  Warning: Only {len(remembering_pool)} remembering questions available (need 5)")
            
            # Reasoning: Select 5 from memory_evidence
            reasoning_pool = categorized["reasoning"]["memory_evidence"]
            if len(reasoning_pool) >= 5:
                selected["reasoning"] = self._select_questions_stratified(
                    reasoning_pool, 5, "reasoning"
                )
            else:
                selected["reasoning"] = sorted(
                    reasoning_pool,
                    key=lambda q: self._count_evidence_sessions(q),
                    reverse=True
                )
                print(f"⚠️  Warning: Only {len(reasoning_pool)} reasoning questions available (need 5)")
            
            # Recommending: Select 5 from memory_and_forgetting
            recommending_pool = categorized["recommending"]["memory_and_forgetting"]
            if len(recommending_pool) >= 5:
                selected["recommending"] = self._select_questions_stratified(
                    recommending_pool, 5, "recommending"
                )
            else:
                selected["recommending"] = sorted(
                    recommending_pool,
                    key=lambda q: self._count_evidence_sessions(q),
                    reverse=True
                )
                print(f"⚠️  Warning: Only {len(recommending_pool)} recommending questions available (need 5)")
            
            # Create output structure
            output = {
                "persona": self.persona,
                "session_directory": self.output_dir,
                "config_file": self.config_file,
                "date_range": {
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "generation_date": datetime.now().isoformat(),
                "selection_criteria": {
                    "remembering": "5 questions with both memory AND forgetting evidence",
                    "reasoning": "5 questions with memory evidence ONLY (forgetting NOT required)",
                    "recommending": "5 questions with both memory AND forgetting evidence"
                },
                "questions": {
                    "remembering": selected["remembering"],
                    "reasoning": selected["reasoning"],
                    "recommending": selected["recommending"]
                },
                "summary": {
                    "total_selected": len(selected["remembering"]) + len(selected["reasoning"]) + len(selected["recommending"]),
                    "by_category": {
                        "remembering": len(selected["remembering"]),
                        "reasoning": len(selected["reasoning"]),
                        "recommending": len(selected["recommending"])
                    },
                    "selection_method": "stratified_evidence_based",
                    "available_pool_sizes": {
                        "remembering": len(categorized["remembering"]["memory_and_forgetting"]),
                        "reasoning": len(categorized["reasoning"]["memory_evidence"]),
                        "recommending": len(categorized["recommending"]["memory_and_forgetting"])
                    }
                }
            }
            
            # Save to file
            output_file = f"{self.output_dir}/evaluation_questions_{self.persona}.json"
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            total_selected = len(selected['remembering']) + len(selected['reasoning']) + len(selected['recommending'])
            print(f"✅ Selected {total_selected} questions")
            print(f"💾 Saved to: {output_file}")
            print(f"\n📊 Final Selection:")
            print(f"  Remembering: {len(selected['remembering'])} questions (target: 5)")
            print(f"  Reasoning: {len(selected['reasoning'])} questions (target: 5)")
            print(f"  Recommending: {len(selected['recommending'])} questions (target: 5)")
            
            # Show question IDs and evidence counts for verification
            if len(selected['remembering']) > 0:
                print(f"\n  Remembering question IDs (sorted by evidence count):")
                for q in selected['remembering']:
                    evidence_count = self._count_evidence_sessions(q)
                    print(f"    - {q.get('question_id', 'unknown')} ({evidence_count} sessions)")
            if len(selected['reasoning']) > 0:
                print(f"\n  Reasoning question IDs (sorted by evidence count):")
                for q in selected['reasoning']:
                    evidence_count = self._count_evidence_sessions(q)
                    print(f"    - {q.get('question_id', 'unknown')} ({evidence_count} sessions)")
            if len(selected['recommending']) > 0:
                print(f"\n  Recommending question IDs (sorted by evidence count):")
                for q in selected['recommending']:
                    evidence_count = self._count_evidence_sessions(q)
                    print(f"    - {q.get('question_id', 'unknown')} ({evidence_count} sessions)")
            
            return {
                "success": True,
                "output_file": output_file,
                "selected_questions": output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline with retries - regenerates everything until quality is met"""
        print("="*80)
        print(f"🚀 INTELLIGENT QA PIPELINE: {self.persona}")
        print(f"📅 Date Range: {self.start_date} to {self.end_date}")
        print(f"⚙️  Config: {self.config_file}")
        print(f"📁 Fixed Output Directory: {self.output_dir}")
        print("="*80)
        
        for attempt in range(1, self.max_retries + 1):
            print(f"\n{'='*80}")
            print(f"🔄 ATTEMPT {attempt}/{self.max_retries}")
            print(f"{'='*80}")
            
            # Step 1: Generate sessions (to fixed directory)
            gen_result = self.run_session_simulator()
            if not gen_result["success"]:
                print(f"❌ Session generation failed: {gen_result['error']}")
                self.cleanup_failed_run()
                if attempt < self.max_retries:
                    print(f"⏭️  Retrying entire pipeline...")
                    continue
                else:
                    return False
            
            # Step 2: Validate quality
            val_result = self.validate_sessions()
            if not val_result["success"]:
                print(f"❌ Quality validation failed")
                self.cleanup_failed_run()
                if attempt < self.max_retries:
                    print(f"⏭️  Retrying entire pipeline...")
                    continue
                else:
                    return False
            
            # Step 3: Generate questions (to same fixed directory)
            qa_result = self.generate_questions()
            if not qa_result["success"]:
                print(f"❌ Question generation failed: {qa_result['error']}")
                self.cleanup_failed_run()
                if attempt < self.max_retries:
                    print(f"⏭️  Retrying entire pipeline...")
                    continue
                else:
                    return False
            
            # Step 4: Validate question pipeline integrity
            pipeline_val_result = self.validate_question_pipeline()
            if not pipeline_val_result["success"]:
                print(f"❌ Question pipeline validation failed")
                self.cleanup_failed_run()
                if attempt < self.max_retries:
                    print(f"⏭️  Retrying entire pipeline...")
                    continue
                else:
                    return False
            
            # Step 5: Analyze questions from the ACTUAL generated file
            analysis_result = self.analyze_questions()
            if not analysis_result["success"]:
                if "error" in analysis_result:
                    print(f"❌ Question analysis failed: {analysis_result['error']}")
                    self.cleanup_failed_run()
                    if attempt < self.max_retries:
                        print(f"⏭️  Retrying entire pipeline...")
                        continue
                    else:
                        return False
                
                # Not enough questions - retry entire pipeline
                if attempt < self.max_retries:
                    print(f"⏭️  Insufficient questions, regenerating sessions and questions...")
                    self.cleanup_failed_run()
                    continue
                else:
                    print(f"❌ Failed to get sufficient questions after {self.max_retries} attempts")
                    self.cleanup_failed_run()
                    return False
            
            # Step 6: Select and save
            save_result = self.select_and_save_questions(analysis_result)
            if not save_result["success"]:
                print(f"❌ Failed to save questions: {save_result['error']}")
                self.cleanup_failed_run()
                return False
            
            # Get selected questions from save_result
            selected_questions = save_result.get("selected_questions", {})
            questions_by_category = selected_questions.get("questions", {})
            
            # Success!
            print("\n" + "="*80)
            print(f"🎉 SUCCESS: Pipeline completed for {self.persona}")
            print("="*80)
            print(f"\n📁 Output Directory: {self.output_dir}")
            print(f"📄 Questions File: {save_result['output_file']}")
            print(f"✅ Quality: PASSED")
            
            total_selected = (
                len(questions_by_category.get("remembering", [])) +
                len(questions_by_category.get("reasoning", [])) +
                len(questions_by_category.get("recommending", []))
            )
            print(f"✅ Questions: {total_selected} selected (5 per category)")
            print(f"✅ Attempt: {attempt}/{self.max_retries}")
            
            return True
        
        return False


def main():
    parser = argparse.ArgumentParser(description='Intelligent QA Pipeline with Quality Gates')
    parser.add_argument('--persona', required=True, 
                       help='Persona to generate questions for (or "all" for all personas)')
    parser.add_argument('--start-date', required=True,
                       help='Start date for session generation (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True,
                       help='End date for session generation (YYYY-MM-DD)')
    parser.add_argument('--config', required=True,
                       help='Memory config file (e.g., meta_data/memory_configs/memory_config_weekly.json)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts per persona (default: 3)')
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {args.config}")
        return 1
    
    # Validate date format
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"❌ Error: Invalid date format. Use YYYY-MM-DD format.")
        print(f"   Start date: {args.start_date}")
        print(f"   End date: {args.end_date}")
        return 1
    
    # Validate date range
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    if start >= end:
        print(f"❌ Error: Start date must be before end date")
        print(f"   Start: {args.start_date}")
        print(f"   End: {args.end_date}")
        return 1
    
    # Determine personas to process
    if args.persona.lower() == 'all':
        personas = PERSONAS
        print(f"\n🎯 Processing ALL {len(personas)} personas")
    else:
        if args.persona not in PERSONAS:
            print(f"❌ Error: Unknown persona '{args.persona}'")
            print(f"Available personas: {', '.join(PERSONAS)}")
            return 1
        personas = [args.persona]
    
    # Process each persona
    results = {}
    success_count = 0
    
    start_time = time.time()
    
    for i, persona in enumerate(personas, 1):
        print(f"\n{'='*80}")
        print(f"👤 PERSONA {i}/{len(personas)}: {persona}")
        print(f"{'='*80}")
        
        pipeline = IntelligentQAPipeline(
            persona, 
            start_date=args.start_date,
            end_date=args.end_date,
            config_file=args.config,
            max_retries=args.max_retries
        )
        success = pipeline.run_pipeline()
        
        results[persona] = {
            "success": success,
            "output_dir": pipeline.output_dir
        }
        
        if success:
            success_count += 1
    
    # Final summary
    duration = time.time() - start_time
    
    print("\n" + "="*80)
    print("📊 FINAL PIPELINE SUMMARY")
    print("="*80)
    print(f"\n⏱️  Total Duration: {duration/60:.1f} minutes")
    print(f"✅ Successful: {success_count}/{len(personas)}")
    print(f"❌ Failed: {len(personas) - success_count}/{len(personas)}")
    
    if success_count < len(personas):
        print(f"\n⚠️  Failed personas:")
        for persona, result in results.items():
            if not result["success"]:
                print(f"  - {persona}")
    
    # Save summary report in dedicated reports directory
    reports_dir = Path("quality_checks/reports")
    reports_dir.mkdir(exist_ok=True)
    
    report_file = reports_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config_file": args.config,
            "date_range": {
                "start_date": args.start_date,
                "end_date": args.end_date
            },
            "duration_seconds": duration,
            "total_personas": len(personas),
            "successful": success_count,
            "failed": len(personas) - success_count,
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Pipeline report saved: {report_file}")
    print("="*80)
    
    return 0 if success_count == len(personas) else 1


if __name__ == "__main__":
    sys.exit(main())

