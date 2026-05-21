#!/usr/bin/env python3
"""
Unified Question Generator for Memora Framework

This script combines all question generation approaches into a single unified system:
- Operation-based questions for preference memory (movies, books, music, travel)
- Activity aggregation questions (food expenses, step tracker, todo list, calendar)
- Content regeneration questions (project proposals, emails, social media, meeting notes)
- Goal-based questions for goal memory

Key Features:
1. Session-based intervals: Extract memory states at regular session intervals
2. Time-based context: Consider temporal patterns for question relevance
3. Memory type analysis: Generate appropriate questions based on memory content
4. Unified output format: Consistent question structure across all types

Usage:
    python unified_question_generator.py --session_directory <path> [--session_interval <N>]
    
    If session_interval is provided: Generate questions at regular intervals
    If session_interval is omitted: Generate questions only from final memory state
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Import the memory evaluation generator
from memory_evaluation_generator import MemoryEvaluationGenerator


@dataclass
class QuestionGenerationConfig:
    """Configuration for question generation"""
    memory_states_file: str = ""
    output_dir: str = ""
    timeline_type: str = "weekly"  # Timeline type: "weekly", "monthly", or "quarterly"


@dataclass
class MemorySnapshot:
    """Represents a memory state snapshot at a specific session"""
    session_id: int
    session_date: str
    memory_state: Dict[str, Any]
    sessions_since_start: int
    time_since_start: timedelta


class UnifiedQuestionGenerator:
    """Unified question generator for all memory types"""
    
    # Mapping from question_type to task_type (new framing)
    # Task types: Remembering, Reasoning, Recommending
    QUESTION_TYPE_TO_TASK_TYPE = {
        "preference_based_recommendation": "Recommending",
        "activity_aggregation": "Reasoning",
        "activity_status": "Remembering",
        "content_regeneration": "Remembering",
        "goal_progress_analysis": "Reasoning",
        "temporal_comparative_analysis": "Reasoning",
        "temporal_goal_analysis": "Reasoning",
    }
    
    def __init__(self, config: QuestionGenerationConfig):
        self.config = config
        self.memory_states = {}
        self.metadata = {}  # Metadata from memory_states file (persona, date range, etc.)
        self.questions = []
        self.new_sessions = []  # List of sessions from new_sessions.json
        
        # Initialize memory evaluation generator
        self.evaluation_generator = MemoryEvaluationGenerator()
        
        # Track what content questions have been asked to avoid repeats
        self.asked_content_items = {
            "project_proposal": set(),
            "email_writeup": set(), 
            "social_media_post": set(),
            "meeting_notes": set()
        }
        
        # Load data
        self._load_memory_states()
        self._load_new_sessions()
        
        # Question templates for different types
        self.preference_templates = {
            "movies": [
                "Can you suggest me a movie?",
                "What movie would you recommend?",
                "I'm looking for a good movie to watch. Any suggestions?",
                "Suggest me a movie based on my preferences."
            ],
            "books": [
                "Can you suggest me a book?",
                "What book would you recommend?",
                "I'm looking for a good book to read. Any suggestions?",
                "Suggest me a book based on my preferences."
            ],
            "music": [
                "Can you suggest me some music?",
                "What music would you recommend?",
                "I'm looking for new music to listen to. Any suggestions?",
                "Suggest me some music based on my preferences."
            ],
            "travel": [
                "Can you suggest me a travel destination?",
                "Where should I travel next?",
                "I'm planning a trip. Any destination suggestions?",
                "Suggest me a place to visit based on my preferences."
            ]
        }
    
    def _load_memory_states(self):
        """Load memory states from file"""
        print(f"Loading memory states from: {self.config.memory_states_file}")
        
        if not os.path.exists(self.config.memory_states_file):
            raise FileNotFoundError(f"Memory states file not found: {self.config.memory_states_file}")
        
        with open(self.config.memory_states_file, 'r') as f:
            data = json.load(f)
        
        self.memory_states = data["memory_states"]
        self.metadata = data["metadata"]
        
        print(f"✅ Loaded {len(self.memory_states)} memory states")
        print(f"📊 Persona: {self.metadata.get('persona', 'unknown')}")
        print(f"📅 Date range: {self._get_date_range()}")
    
    def _load_new_sessions(self):
        """Load new_sessions.json file for easier session lookup"""
        # Derive new_sessions.json path from memory_states_file
        session_dir = os.path.dirname(self.config.memory_states_file)
        new_sessions_file = os.path.join(session_dir, "new_sessions.json")
        
        if os.path.exists(new_sessions_file):
            with open(new_sessions_file, 'r') as f:
                data = json.load(f)
                self.new_sessions = data.get("sessions", [])
            print(f"✅ Loaded {len(self.new_sessions)} sessions from new_sessions.json")
        else:
            print(f"⚠️  new_sessions.json not found: {new_sessions_file}")
            self.new_sessions = []
    
    def _calculate_intermediate_snapshots_count(self) -> int:
        """Calculate how many intermediate snapshots to extract based on timeline type
        
        Returns:
            Number of intermediate snapshots (not including final)
        """
        if not self.memory_states:
            return 0
        
        session_ids = sorted([int(sid) for sid in self.memory_states.keys()])
        start_date_str = self.memory_states[str(session_ids[0])]["session_date"]
        end_date_str = self.memory_states[str(session_ids[-1])]["session_date"]
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        total_days = (end_date - start_date).days + 1
        
        if self.config.timeline_type == "weekly":
            # Weekly: No intermediate snapshots
            return 0
        elif self.config.timeline_type == "monthly":
            # Monthly: ~1 snapshot per week (approximately total_days / 7)
            return max(0, (total_days // 7) - 1)  # -1 because final is separate
        elif self.config.timeline_type == "quarterly":
            # Quarterly: ~1 snapshot per month (approximately total_days / 30)
            return max(0, (total_days // 30) - 1)  # -1 because final is separate
        
        return 0
    
    def _get_date_range(self) -> str:
        """Get the date range of sessions"""
        if not self.memory_states:
            return "No sessions"
        
        session_ids = [int(sid) for sid in self.memory_states.keys()]
        min_session = min(session_ids)
        max_session = max(session_ids)
        
        start_date = self.memory_states[str(min_session)]["session_date"]
        end_date = self.memory_states[str(max_session)]["session_date"]
        
        return f"{start_date} to {end_date} ({len(session_ids)} sessions)"
    
    def _get_relative_time_reference(self, session_date: str, days_since_start: int) -> str:
        """Generate relative time reference based on session duration
        
        Args:
            session_date: Current session date
            days_since_start: Days since the start of the timeline
        """
        # Use timeline type to determine appropriate time reference
        if self.config.timeline_type == "weekly":
            return "this week"
        elif self.config.timeline_type == "monthly":
            return "this month"
        elif self.config.timeline_type == "quarterly":
            return "in the last 3 months"
        
        # Fallback for unknown timeline types (shouldn't happen)
        if days_since_start <= 1:
            return "today"
        elif days_since_start <= 3:
            return "in the last few days"
        elif days_since_start <= 7:
            return "this week"
        elif days_since_start <= 14:
            return "in the last 2 weeks"
        elif days_since_start <= 30:
            return "this month"
        elif days_since_start <= 60:
            return "in the last 2 months"
        elif days_since_start <= 90:
            return "in the last 3 months"
        else:
            return "recently"
    
    def _get_lookback_period_reference(self, snapshot: MemorySnapshot, lookback_days: int) -> str:
        """Generate lookback period reference (e.g., '2 weeks ago', 'last week')
        
        Args:
            snapshot: Current memory snapshot
            lookback_days: Number of days to look back
            
        Returns:
            Human-readable lookback reference
        """
        if lookback_days <= 7:
            if lookback_days == 7:
                return "last week"
            else:
                return f"{lookback_days} days ago"
        elif lookback_days <= 14:
            return "2 weeks ago"
        elif lookback_days <= 21:
            return "3 weeks ago"
        elif lookback_days <= 30:
            return "last month"
        else:
            weeks = lookback_days // 7
            return f"{weeks} weeks ago"
    
    def _extract_forgetting_evidence(self, snapshot: MemorySnapshot, category: str, memory_type: str) -> Dict[str, Any]:
        """Extract evidence of what should have been forgotten through delete and update operations"""
        forgetting_evidence = {
            "forgotten_items": [],  # Items that should be completely forgotten
            "total_forgotten_items": 0
        }
        
        # Track all operations for this category to detect re-additions after deletions
        operations_timeline = []
        
        # Analyze all sessions up to this snapshot to find operations
        for session_id_str, session_data in self.memory_states.items():
            session_id = int(session_id_str)
            
            # Only look at sessions up to the current snapshot
            if session_id > snapshot.session_id:
                continue
            
            session_type = session_data.get("session_type")
            session_category = session_data.get("session_category")
            operation = session_data.get("operation_performed")
            operation_details = session_data.get("operation_details", {})
            
            # Track all add/delete/update operations for this category
            if (session_type == memory_type and 
                session_category == category and 
                operation in ["add", "delete", "update"]):
                
                item = operation_details.get("item")
                old_item = operation_details.get("old_item")
                
                operations_timeline.append({
                    "session_id": session_id,
                    "operation": operation,
                    "item": item,
                    "old_item": old_item,
                    "session_date": session_data.get("session_date"),
                    "operation_details": operation_details
                })
        
        # Sort operations by session_id to get chronological order
        operations_timeline.sort(key=lambda x: x["session_id"])
        
        # Find items that were deleted/replaced and NOT re-added later
        deleted_items = {}  # item_str -> {item, last_delete_session_info}
        
        for op in operations_timeline:
            item = op["item"]
            old_item = op["old_item"]
            
            if op["operation"] == "delete":
                # Mark this item as deleted
                item_str = str(item) if item is not None else "None"
                deleted_items[item_str] = {
                    "item": item,
                    "session_id": op["session_id"],
                    "session_date": op["session_date"],
                    "operation_type": "delete"
                }
                
            elif op["operation"] == "update":
                # For updates, the OLD item/value is forgotten
                if old_item is not None:
                    old_item_str = str(old_item)
                    deleted_items[old_item_str] = {
                        "item": old_item,
                        "session_id": op["session_id"],
                        "session_date": op["session_date"],
                        "operation_type": "update_forgotten",
                        "replaced_by": item
                    }
                
                # The NEW item from update is effectively "added"
                # If this new item was previously forgotten, remove it from deleted_items
                item_str = str(item) if item is not None else "None"
                if item_str in deleted_items:
                    del deleted_items[item_str]
                
            elif op["operation"] == "add":
                # If this item was previously deleted/updated, remove it from deleted_items
                # because it was re-added
                item_str = str(item) if item is not None else "None"
                if item_str in deleted_items:
                    del deleted_items[item_str]
        
        # Only items that remain in deleted_items should be in forgetting evidence
        for item_str, delete_info in deleted_items.items():
            forgotten_item = {
                "forgotten_item": delete_info["item"],
                "session_id": delete_info["session_id"],
                "session_date": delete_info["session_date"],
                "operation_type": delete_info["operation_type"]
            }
            
            # Add replacement info for update operations
            if delete_info["operation_type"] == "update_forgotten":
                forgotten_item["replaced_by"] = delete_info["replaced_by"]
            
            forgetting_evidence["forgotten_items"].append(forgotten_item)
            forgetting_evidence["total_forgotten_items"] += 1
        
        return forgetting_evidence
    
    def extract_memory_snapshots(self) -> List[MemorySnapshot]:
        """Extract memory snapshots based on timeline type
        
        All timeline types extract ONLY the final snapshot.
        The timeline type affects what questions are generated, not when.
        """
        print(f"📅 Timeline type: {self.config.timeline_type.upper()}")
        print(f"🔍 Extracting final memory snapshot only (all questions generated at end)...")
        
        snapshots = []
        session_ids = sorted([int(sid) for sid in self.memory_states.keys()])
        
        if not session_ids:
            print("❌ No sessions found")
            return snapshots
        
        start_session = session_ids[0]
        final_session = session_ids[-1]
        start_date = datetime.strptime(self.memory_states[str(start_session)]["session_date"], "%Y-%m-%d")
        end_date = datetime.strptime(self.memory_states[str(final_session)]["session_date"], "%Y-%m-%d")
        total_days = (end_date - start_date).days
        
        # Always extract only final snapshot
        session_data = self.memory_states[str(final_session)]
        session_date = datetime.strptime(session_data["session_date"], "%Y-%m-%d")
        
        final_snapshot = MemorySnapshot(
            session_id=final_session,
            session_date=session_data["session_date"],
            memory_state=session_data["memory_state_after_session"],
            sessions_since_start=final_session - start_session,
            time_since_start=session_date - start_date
        )
        snapshots.append(final_snapshot)
        
        # Calculate breakdown info for question generation
        if self.config.timeline_type == "monthly":
            weeks = [d for d in [7, 14, 21, 28] if d <= total_days]
            print(f"📸 Final snapshot at session {final_session} ({session_data['session_date']}) - {total_days} days total")
            print(f"   Will generate comparative questions across {len(weeks)} weeks (e.g., 'Which week had most expenses?')")
        elif self.config.timeline_type == "quarterly":
            months = [d for d in [30, 60, 90] if d <= total_days]
            print(f"📸 Final snapshot at session {final_session} ({session_data['session_date']}) - {total_days} days total")
            print(f"   Will generate comparative questions across {len(months)} months (e.g., 'Which month had most steps?')")
        else:
            print(f"📸 Final snapshot at session {final_session} ({session_data['session_date']}) - {total_days} days total")
        
        print(f"✅ Extracted {len(snapshots)} memory snapshot")
        return snapshots
    
    def _count_actual_memory_items(self, category_items: Any) -> int:
        """Count actual memory items, not just empty structures"""
        if not category_items:
            return 0
        
        if isinstance(category_items, list):
            return len(category_items)
        elif isinstance(category_items, dict):
            total_items = 0
            for key, value in category_items.items():
                if isinstance(value, dict):
                    # Count items in nested dictionaries (likes/dislikes structure)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, list):
                            total_items += len(subvalue)
                        elif subvalue:  # Non-empty value
                            total_items += 1
                elif isinstance(value, list):
                    total_items += len(value)
                elif value:  # Non-empty value
                    total_items += 1
            return total_items
        else:
            return 1 if category_items else 0
    
    def _find_session_id_for_item(self, item_value: str, created_at: str, category: str, 
                                   subcategory: str = None, memory_type: str = None) -> Optional[int]:
        """Find the session_id where a specific item was created using new_sessions.json
        
        Args:
            item_value: The item to search for (e.g., "Terry Gilliam")
            created_at: The creation date (e.g., "2025-01-01")
            category: The category (e.g., "movies", "todo_list")
            subcategory: The subcategory (e.g., "directors", "genres") - optional
            memory_type: The memory type (e.g., "preference_memory") - optional
            
        Returns:
            The session_id where the item was added, or None if not found
        """
        if not self.new_sessions:
            return None
        
        # Search through new_sessions for matching item
        for session in self.new_sessions:
            # Check if date and category match
            if session.get("date") != created_at:
                continue
            
            if session.get("category") != category:
                continue
            
            # Check memory type if provided
            if memory_type and session.get("type") != memory_type:
                continue
            
            # Only consider add/update operations
            if session.get("operation") not in ["add", "update"]:
                continue
            
            operation_details = session.get("operation_details", {})
            
            # For preference memory - check item and subcategory
            if session.get("type") == "preference_memory":
                if (operation_details.get("item") == item_value and
                    (subcategory is None or operation_details.get("subcategory") == subcategory)):
                    return session.get("id")
            
            # For activity memory - check based on category
            elif session.get("type") == "activity_memory":
                item_data = operation_details.get("item", {})
                if isinstance(item_data, dict):
                    if category == "calendar_event":
                        if item_data.get("event_name") == item_value or item_data.get("event_title") == item_value:
                            return session.get("id")
                    elif category == "todo_list":
                        if item_data.get("description") == item_value or item_data.get("task") == item_value:
                            return session.get("id")
                    elif category == "step_tracker":
                        # Step tracker uses created_at as the date field
                        item_date_in_session = item_data.get("created_at") or item_data.get("date")
                        if item_date_in_session == item_value:
                            return session.get("id")
                    elif category == "food_expenses":
                        # For food expenses, match using combined identifier
                        item_date = item_data.get("created_at") or item_data.get("date")
                        combined_id = f"{item_data.get('expense_type')}_{item_data.get('amount')}_{item_date}"
                        if combined_id == item_value:
                            return session.get("id")
        
        return None
    
    def _find_session_id_for_goal(self, goal_category: str, goal_subcategory: str, created_at: str) -> Optional[int]:
        """Find the session_id where a specific goal was created using new_sessions.json
        
        Args:
            goal_category: The goal category (e.g., "food_expenses", "step_tracker")
            goal_subcategory: The goal subcategory (e.g., "breakfast", "daily_steps")
            created_at: The creation date (e.g., "2025-01-01")
            
        Returns:
            The session_id where the goal was added, or None if not found
        """
        if not self.new_sessions:
            return None
        
        # Search through new_sessions for matching goal
        for session in self.new_sessions:
            # Check if date and type match
            if session.get("date") != created_at:
                continue
            
            if session.get("type") != "goal_memory":
                continue
            
            # Only consider add operations
            if session.get("operation") != "add":
                continue
            
            operation_details = session.get("operation_details", {})
            
            # Check if category and subcategory match
            if (operation_details.get("subcategory") == goal_subcategory and
                session.get("category") == goal_category):
                return session.get("id")
        
        return None
    
    def _enrich_preference_items_with_session_id(self, category_items: Dict, category: str) -> Dict:
        """Enrich preference memory items with session_id by looking them up in new_sessions"""
        enriched_items = {}
        
        for subcategory, subcat_data in category_items.items():
            if isinstance(subcat_data, dict):
                enriched_subcat = {}
                
                # Handle likes
                if "likes" in subcat_data:
                    enriched_likes = []
                    for item in subcat_data.get("likes", []):
                        if isinstance(item, dict):
                            enriched_item = item.copy()
                            # Look up session_id
                            session_id = self._find_session_id_for_item(
                                item.get("item"),
                                item.get("created_at"),
                                category,
                                subcategory,
                                "preference_memory"
                            )
                            if session_id is not None:
                                enriched_item["session_id"] = session_id
                            enriched_likes.append(enriched_item)
                        else:
                            enriched_likes.append(item)
                    enriched_subcat["likes"] = enriched_likes
                
                # Handle dislikes
                if "dislikes" in subcat_data:
                    enriched_dislikes = []
                    for item in subcat_data.get("dislikes", []):
                        if isinstance(item, dict):
                            enriched_item = item.copy()
                            # Look up session_id
                            session_id = self._find_session_id_for_item(
                                item.get("item"),
                                item.get("created_at"),
                                category,
                                subcategory,
                                "preference_memory"
                            )
                            if session_id is not None:
                                enriched_item["session_id"] = session_id
                            enriched_dislikes.append(enriched_item)
                        else:
                            enriched_dislikes.append(item)
                    enriched_subcat["dislikes"] = enriched_dislikes
                
                enriched_items[subcategory] = enriched_subcat
            else:
                enriched_items[subcategory] = subcat_data
        
        return enriched_items
    
    def _enrich_activity_items_with_session_id(self, activity_items: List, category: str) -> List:
        """Enrich activity memory items with session_id by looking them up in new_sessions
        
        Args:
            activity_items: List of activity memory items
            category: Activity category (e.g., food_expenses, step_tracker, calendar_event, todo_list)
            
        Returns:
            Enriched list with session_id added to each item
        """
        enriched_items = []
        
        for item in activity_items:
            if isinstance(item, dict):
                enriched_item = item.copy()
                
                # Determine the unique identifier based on category
                item_value = None
                item_date = item.get("created_at") or item.get("date")
                
                if category == "food_expenses":
                    # For food expenses, we need a unique combination
                    # Use a combination of amount, expense_type, and date
                    item_value = f"{item.get('expense_type')}_{item.get('amount')}_{item_date}"
                elif category == "step_tracker":
                    # For step tracker, date is the unique identifier
                    item_value = item_date
                elif category == "calendar_event":
                    # For calendar events, use event_name or event_title
                    item_value = item.get("event_name") or item.get("event_title")
                elif category == "todo_list":
                    # For todo list, use description or task
                    item_value = item.get("description") or item.get("task")
                
                # Look up session_id
                if item_value and item_date:
                    session_id = self._find_session_id_for_item(
                        item_value,
                        item_date,
                        category,
                        None,
                        "activity_memory"
                    )
                    if session_id is not None:
                        enriched_item["session_id"] = session_id
                
                enriched_items.append(enriched_item)
            else:
                enriched_items.append(item)
        
        return enriched_items

    def generate_preference_questions(self, snapshot: MemorySnapshot) -> List[Dict[str, Any]]:
        """Generate preference-based questions from memory snapshot"""
        questions = []
        
        memory_items = snapshot.memory_state.get("memory_items", {})
        preference_memory = memory_items.get("preference_memory", {})
        
        for category in ["movies", "books", "music", "travel"]:
            if category in preference_memory and preference_memory[category]:
                # Count actual items in this category
                category_items = preference_memory[category]
                item_count = self._count_actual_memory_items(category_items)
                
                if item_count > 0:
                    # Enrich items with session_id
                    enriched_category_items = self._enrich_preference_items_with_session_id(category_items, category)
                    
                    # Extract forgetting evidence for this category
                    forgetting_evidence = self._extract_forgetting_evidence(snapshot, category, "preference_memory")
                    
                    # Generate general recommendation question
                    general_templates = self.preference_templates[category]
                    template_index = (len(questions)) % len(general_templates)
                    question_text = general_templates[template_index]
                    
                    # Enhanced answer that includes forgetting context
                    forgotten_count = forgetting_evidence["total_forgotten_items"]
                    if forgotten_count > 0:
                        answer_text = f"Based on your {category} preferences from {item_count} items in memory, I can provide personalized {category} recommendations. Note: {forgotten_count} items were updated/deleted and should have been forgotten."
                    else:
                        answer_text = f"Based on your {category} preferences from {item_count} items in memory, I can provide personalized {category} recommendations."
                    
                    question = {
                        "question_id": f"pref_{category}_general_{snapshot.session_id}",
                        "question": question_text,
                        "answer": answer_text,
                        "question_type": "preference_based_recommendation",
                        "subcategory": "general",
                        "category": category,
                        "memory_type": "preference_memory",
                        "session_id": snapshot.session_id,
                        "session_date": snapshot.session_date,
                        "memory_evidence": {
                            "item_count": item_count,
                            "memory_items": enriched_category_items
                        },
                        "forgetting_evidence": forgetting_evidence,
                        "generation_context": {
                            "sessions_since_start": snapshot.sessions_since_start,
                            "days_since_start": snapshot.time_since_start.days
                        }
                    }
                    questions.append(question)
                    
                    # Generate subcategory-specific questions
                    subcategory_questions = self._generate_subcategory_preference_questions(
                        category, enriched_category_items, snapshot, forgetting_evidence
                    )
                    questions.extend(subcategory_questions)
        
        return questions
    
    def _generate_subcategory_preference_questions(self, category: str, category_items: Dict, 
                                                 snapshot: MemorySnapshot, forgetting_evidence: Dict) -> List[Dict[str, Any]]:
        """Generate subcategory-specific preference questions"""
        questions = []
        
        # Define subcategory mappings and templates
        subcategory_mappings = {
            "movies": {
                "actors": "Can you recommend a movie with actors I like?",
                "directors": "Can you recommend a movie by directors I prefer?", 
                "genres": "Can you recommend a movie in genres I enjoy?"
            },
            "books": {
                "authors": "Can you recommend a book by authors I like?",
                "topics": "Can you recommend a book on topics I'm interested in?"
            },
            "music": {
                "artists": "Can you recommend music by artists I like?",
                "genres": "Can you recommend music in genres I enjoy?",
                "decades": "Can you recommend music from decades I prefer?"
            },
            "travel": {
                "destination_types": "Can you recommend destinations that match my preferred types?",
                "regions": "Can you recommend places in regions I like?",
                "climates": "Can you recommend destinations with climates I prefer?"
            }
        }
        
        if category not in subcategory_mappings:
            return questions
            
        for subcategory, question_template in subcategory_mappings[category].items():
            if subcategory in category_items:
                subcategory_data = category_items[subcategory]
                
                # Count items in likes for this subcategory
                likes_count = 0
                if isinstance(subcategory_data, dict) and "likes" in subcategory_data:
                    likes_count = len(subcategory_data["likes"]) if subcategory_data["likes"] else 0
                
                if likes_count > 0:
                    # Generate subcategory-specific forgetting evidence
                    subcategory_forgetting_evidence = self._extract_subcategory_forgetting_evidence(
                        snapshot, category, subcategory
                    )
                    
                    forgotten_count = subcategory_forgetting_evidence["total_forgotten_items"]
                    if forgotten_count > 0:
                        answer_text = f"Based on your {subcategory} preferences ({likes_count} items), I can provide targeted {category} recommendations. Note: {forgotten_count} items were updated/deleted and should have been forgotten."
                    else:
                        answer_text = f"Based on your {subcategory} preferences ({likes_count} items), I can provide targeted {category} recommendations."
                    
                    question = {
                        "question_id": f"pref_{category}_{subcategory}_{snapshot.session_id}",
                        "question": question_template,
                        "answer": answer_text,
                        "question_type": "preference_based_recommendation",
                        "subcategory": subcategory,
                        "category": category,
                        "memory_type": "preference_memory",
                        "session_id": snapshot.session_id,
                        "session_date": snapshot.session_date,
                        "memory_evidence": {
                            "subcategory_data": subcategory_data,
                            "likes_count": likes_count,
                            "specific_items": subcategory_data.get("likes", [])
                        },
                        "forgetting_evidence": subcategory_forgetting_evidence,
                        "generation_context": {
                            "sessions_since_start": snapshot.sessions_since_start,
                            "days_since_start": snapshot.time_since_start.days
                        }
                    }
                    questions.append(question)
        
        return questions
    
    def _extract_subcategory_forgetting_evidence(self, snapshot: MemorySnapshot, category: str, subcategory: str) -> Dict[str, Any]:
        """Extract forgetting evidence specific to a subcategory (e.g., actors within movies)"""
        forgetting_evidence = {
            "forgotten_items": [],
            "total_forgotten_items": 0
        }
        
        # Track operations for this specific subcategory
        operations_timeline = []
        
        # Analyze all sessions up to this snapshot
        for session_id_str, session_data in self.memory_states.items():
            session_id = int(session_id_str)
            
            if session_id > snapshot.session_id:
                continue
            
            session_type = session_data.get("session_type")
            session_category = session_data.get("session_category")
            operation = session_data.get("operation_performed")
            operation_details = session_data.get("operation_details", {})
            
            # Only look at preference memory operations for this category
            if (session_type == "preference_memory" and 
                session_category == category and 
                operation in ["add", "delete", "update"]):
                
                # Check if the operation affects this specific subcategory
                operation_subcategory = operation_details.get("subcategory")
                if operation_subcategory == subcategory:
                    item = operation_details.get("item")
                    old_item = operation_details.get("old_item")
                    
                    operations_timeline.append({
                        "session_id": session_id,
                        "operation": operation,
                        "item": item,
                        "old_item": old_item,
                        "session_date": session_data.get("session_date"),
                        "subcategory": subcategory
                    })
        
        # Sort operations chronologically
        operations_timeline.sort(key=lambda x: x["session_id"])
        
        # Apply the same deletion/re-addition logic as category-level
        deleted_items = {}
        
        for op in operations_timeline:
            item = op["item"]
            old_item = op["old_item"]
            
            if op["operation"] == "delete":
                item_str = str(item) if item is not None else "None"
                deleted_items[item_str] = {
                    "item": item,
                    "session_id": op["session_id"],
                    "session_date": op["session_date"],
                    "operation_type": "delete",
                    "subcategory": subcategory
                }
                
            elif op["operation"] == "update":
                # Old item forgotten
                if old_item is not None:
                    old_item_str = str(old_item)
                    deleted_items[old_item_str] = {
                        "item": old_item,
                        "session_id": op["session_id"],
                        "session_date": op["session_date"],
                        "operation_type": "update_forgotten",
                        "replaced_by": item,
                        "subcategory": subcategory
                    }
                
                # New item effectively added (remove from forgotten if it was there)
                item_str = str(item) if item is not None else "None"
                if item_str in deleted_items:
                    del deleted_items[item_str]
                    
            elif op["operation"] == "add":
                # Remove from forgotten if re-added
                item_str = str(item) if item is not None else "None"
                if item_str in deleted_items:
                    del deleted_items[item_str]
        
        # Build final forgetting evidence
        for item_str, delete_info in deleted_items.items():
            forgotten_item = {
                "forgotten_item": delete_info["item"],
                "session_id": delete_info["session_id"],
                "session_date": delete_info["session_date"],
                "operation_type": delete_info["operation_type"],
                "subcategory": delete_info["subcategory"]
            }
            
            if delete_info["operation_type"] == "update_forgotten":
                forgotten_item["replaced_by"] = delete_info["replaced_by"]
            
            forgetting_evidence["forgotten_items"].append(forgotten_item)
            forgetting_evidence["total_forgotten_items"] += 1
        
        return forgetting_evidence
    
    def _extract_content_forgetting_evidence(self, snapshot: MemorySnapshot, category: str, item_id: str = None) -> Dict[str, Any]:
        """Extract forgetting evidence for content memory based on memory_deletes from delete operations"""
        forgetting_evidence = {
            "forgotten_items": [],
            "total_forgotten_items": 0
        }
        
        # Analyze all sessions up to this snapshot to find content delete operations
        for session_id_str, session_data in self.memory_states.items():
            session_id = int(session_id_str)
            
            # Only look at sessions up to the current snapshot
            if session_id > snapshot.session_id:
                continue
            
            session_type = session_data.get("session_type")
            session_category = session_data.get("session_category")
            operation = session_data.get("operation_performed")
            operation_details = session_data.get("operation_details", {})
            
            # Only look at content memory delete operations for this category
            if (session_type == "content_memory" and 
                session_category == category and 
                operation == "delete"):
                
                memory_states_item = operation_details.get("item")
                
                # Only include if no item_id filter or if it matches
                if item_id is None or memory_states_item == item_id:
                    
                    # Extract memory_deletes which contain the forgotten items
                    memory_deletes = operation_details.get("memory_deletes", [])
                    
                    for delete_item in memory_deletes:
                        # Get the actual deleted value based on action type
                        deleted_value = None
                        action = delete_item.get("action")
                        
                        if action == "removed_from_memory":
                            deleted_value = delete_item.get("removed_item")
                        elif action == "budget_reverted":
                            deleted_value = delete_item.get("reverted_from")
                        else:
                            # Generic fallback
                            deleted_value = delete_item.get("removed_item") or delete_item.get("value")
                        
                        forgotten_item = {
                            "forgotten_item": {
                                "field": delete_item.get("field"),
                                "value": deleted_value,
                                "item_id": memory_states_item,
                                "action": action
                            },
                            "session_id": session_id,
                            "session_date": session_data.get("session_date"),
                            "operation_type": "content_delete"
                        }
                        
                        forgetting_evidence["forgotten_items"].append(forgotten_item)
                        forgetting_evidence["total_forgotten_items"] += 1
        
        return forgetting_evidence
    
    def generate_activity_questions(self, snapshot: MemorySnapshot) -> List[Dict[str, Any]]:
        """Generate activity-based aggregation questions"""
        questions = []
        
        memory_items = snapshot.memory_state.get("memory_items", {})
        activity_memory = memory_items.get("activity_memory", {})
        
        # Food expenses - total and category-specific
        if "food_expenses" in activity_memory:
            food_expenses = activity_memory["food_expenses"]
            expense_count = self._count_actual_memory_items(food_expenses)
            if expense_count > 0:
                # Enrich food expenses with session_id
                enriched_food_expenses = self._enrich_activity_items_with_session_id(food_expenses, "food_expenses")
                
                # Total food expenses question
                total_amount = sum(expense.get("amount", 0) for expense in enriched_food_expenses if isinstance(expense, dict))
                
                # Generate relative time reference
                time_reference = self._get_relative_time_reference(snapshot.session_date, snapshot.time_since_start.days)
                
                question = {
                    "question_id": f"activity_food_total_{snapshot.session_id}",
                    "question": f"What is my total food spending {time_reference}?",
                    "answer": f"Based on {expense_count} food expense records, your total spending is ${total_amount:.2f}.",
                    "question_type": "activity_aggregation",
                    "category": "food_expenses",
                    "subcategory": "total",
                    "aggregation_type": "total_sum",
                    "memory_type": "activity_memory",
                    "session_id": snapshot.session_id,
                    "session_date": snapshot.session_date,
                    "memory_evidence": {
                        "expense_count": expense_count,
                        "total_amount": total_amount,
                        "food_expenses": enriched_food_expenses
                    }
                }
                questions.append(question)
                
                # Category-specific food expenses
                category_totals = {}
                for expense in enriched_food_expenses:
                    if isinstance(expense, dict):
                        expense_type = expense.get("expense_type", "unknown")
                        amount = expense.get("amount", 0)
                        category_totals[expense_type] = category_totals.get(expense_type, 0) + amount
                
                for expense_type, category_total in category_totals.items():
                    if category_total > 0:
                        question = {
                            "question_id": f"activity_food_{expense_type}_{snapshot.session_id}",
                            "question": f"How much have I spent on {expense_type} {time_reference}?",
                            "answer": f"Your total {expense_type} spending is ${category_total:.2f}.",
                            "question_type": "activity_aggregation",
                            "category": "food_expenses",
                            "subcategory": expense_type,
                            "aggregation_type": "category_sum",
                            "memory_type": "activity_memory",
                            "session_id": snapshot.session_id,
                            "session_date": snapshot.session_date,
                            "memory_evidence": {
                                "expense_type": expense_type,
                                "category_total": category_total,
                                "expense_items": [e for e in enriched_food_expenses if e.get("expense_type") == expense_type]
                            }
                        }
                        questions.append(question)
        
        # Step tracker - total steps
        if "step_tracker" in activity_memory:
            step_data = activity_memory["step_tracker"]
            step_count = self._count_actual_memory_items(step_data)
            if step_count > 0:
                # Enrich step data with session_id
                enriched_step_data = self._enrich_activity_items_with_session_id(step_data, "step_tracker")
                
                total_steps = sum(step.get("step_count", 0) for step in enriched_step_data if isinstance(step, dict))
                
                # Generate relative time reference
                time_reference = self._get_relative_time_reference(snapshot.session_date, snapshot.time_since_start.days)
                
                question = {
                    "question_id": f"activity_steps_total_{snapshot.session_id}",
                    "question": f"How many total steps have I taken {time_reference}?",
                    "answer": f"Based on {step_count} step records, you have taken {total_steps:,} total steps.",
                    "question_type": "activity_aggregation",
                    "category": "step_tracker",
                    "subcategory": "total",
                    "aggregation_type": "total_sum",
                    "memory_type": "activity_memory",
                    "session_id": snapshot.session_id,
                    "session_date": snapshot.session_date,
                    "memory_evidence": {
                        "step_count": step_count,
                        "total_steps": total_steps,
                        "step_data": enriched_step_data
                    }
                }
                questions.append(question)
        
        # Todo list - current items
        if "todo_list" in activity_memory:
            todo_list = activity_memory["todo_list"]
            todo_count = self._count_actual_memory_items(todo_list)
            if todo_count > 0:
                # Enrich todo list with session_id
                enriched_todo_list = self._enrich_activity_items_with_session_id(todo_list, "todo_list")
                
                # Extract forgetting evidence for todo list
                forgetting_evidence = self._extract_forgetting_evidence(snapshot, "todo_list", "activity_memory")
                forgotten_count = forgetting_evidence["total_forgotten_items"]
                
                # Generate relative time reference
                time_reference = self._get_relative_time_reference(snapshot.session_date, snapshot.time_since_start.days)
                
                if forgotten_count > 0:
                    answer_text = f"You have {todo_count} tasks remaining on your todo list. Note: {forgotten_count} tasks were deleted and should have been forgotten."
                else:
                    answer_text = f"You have {todo_count} tasks remaining on your todo list."
                
                question = {
                    "question_id": f"activity_todos_{snapshot.session_id}",
                    "question": f"What tasks remain on my todo list {time_reference}?",
                    "answer": answer_text,
                    "question_type": "activity_status",
                    "category": "todo_list",
                    "memory_type": "activity_memory",
                    "session_id": snapshot.session_id,
                    "session_date": snapshot.session_date,
                    "memory_evidence": {
                        "remaining_tasks": enriched_todo_list,
                        "task_count": todo_count
                    },
                    "forgetting_evidence": forgetting_evidence
                }
                questions.append(question)
        
        # Calendar events - future events only
        if "calendar_event" in activity_memory:
            calendar_events = activity_memory["calendar_event"]
            
            # Enrich calendar events with session_id
            enriched_calendar_events = self._enrich_activity_items_with_session_id(calendar_events, "calendar_event")
            
            # Filter to only include future events
            future_events, past_events = self._filter_future_calendar_events(enriched_calendar_events, snapshot.session_date)
            future_event_count = len(future_events)
            
            if future_event_count > 0:
                # Create forgetting evidence for past events (temporal deletion)
                calendar_forgetting_evidence = {
                    "forgotten_items": [
                        {
                            "forgotten_item": event,
                            "session_id": snapshot.session_id,
                            "session_date": snapshot.session_date
                        }
                        for event in past_events
                    ],
                    "total_forgotten_items": len(past_events)
                }
                
                question = {
                    "question_id": f"activity_calendar_{snapshot.session_id}",
                    "question": f"What upcoming events do I have in my calendar this week?",
                    "answer": f"You have {future_event_count} upcoming calendar events scheduled." + (
                        f" Note: {len(past_events)} past events were correctly filtered out." if past_events else ""
                    ),
                    "question_type": "activity_status",
                    "category": "calendar_event",
                    "memory_type": "activity_memory",
                    "session_id": snapshot.session_id,
                    "session_date": snapshot.session_date,
                    "memory_evidence": {
                        "event_count": future_event_count,
                        "calendar_events": future_events
                    },
                    "forgetting_evidence": calendar_forgetting_evidence
                }
                questions.append(question)
        
        return questions
    
    def _filter_future_calendar_events(self, calendar_events: List[Dict], current_date: str) -> Tuple[List[Dict], List[Dict]]:
        """Filter calendar events to separate future and past events
        
        Returns:
            Tuple[List[Dict], List[Dict]]: (future_events, past_events)
        """
        if not calendar_events:
            return [], []
        
        from datetime import datetime, timedelta
        
        try:
            current_datetime = datetime.strptime(current_date, "%Y-%m-%d")
        except (ValueError, TypeError):
            # If we can't parse the date, return all as future events to be safe
            return calendar_events, []
        
        future_events = []
        past_events = []
        
        for event in calendar_events:
            if not isinstance(event, dict):
                continue
            
            created_at = event.get("created_at")
            event_date_offset = event.get("date")
            
            if not created_at or not event_date_offset:
                # If we can't determine the event date, include it as future to be safe
                future_events.append(event)
                continue
            
            try:
                # Parse the creation date
                created_datetime = datetime.strptime(created_at, "%Y-%m-%d")
                
                # Parse the offset (e.g., "+3 days", "+1 week")
                if event_date_offset.startswith("+"):
                    offset_str = event_date_offset[1:].strip()
                    
                    if "day" in offset_str:
                        days = int(offset_str.split()[0])
                        event_datetime = created_datetime + timedelta(days=days)
                    elif "week" in offset_str:
                        weeks = int(offset_str.split()[0])
                        event_datetime = created_datetime + timedelta(weeks=weeks)
                    else:
                        # Unknown offset format, include as future event
                        future_events.append(event)
                        continue
                    
                    # Check if the event is in the future or past
                    if event_datetime > current_datetime:
                        # Add calculated date to future event for evidence and evaluation
                        event_with_date = event.copy()
                        event_with_date["calculated_event_date"] = event_datetime.strftime("%Y-%m-%d")
                        future_events.append(event_with_date)
                    else:
                        # Add calculated date to past event for evidence
                        event_with_date = event.copy()
                        event_with_date["calculated_event_date"] = event_datetime.strftime("%Y-%m-%d")
                        past_events.append(event_with_date)
                
                else:
                    # If it's not a relative date, include as future event to be safe
                    future_events.append(event)
                    
            except (ValueError, AttributeError, IndexError):
                # If we can't parse the dates, include as future event to be safe
                future_events.append(event)
        
        return future_events, past_events
    
    def generate_content_questions(self, snapshot: MemorySnapshot) -> List[Dict[str, Any]]:
        """Generate content regeneration questions"""
        questions = []
        
        memory_items = snapshot.memory_state.get("memory_items", {})
        content_memory = memory_items.get("content_memory", {})
        
        # Check content pools to see what's completed
        content_pools = snapshot.memory_state.get("content_pools", {})
        completed_pool = content_pools.get("completed_pool", {})
        
        content_types = ["project_proposal", "email_writeup", "social_media_post", "meeting_notes"]
        
        for content_type in content_types:
            # Only generate questions if there are completed items in this category
            completed_count = completed_pool.get(content_type, 0)
            
            if completed_count > 0:
                if content_type in content_memory:
                    content_items = content_memory[content_type]
                    
                    if content_items and isinstance(content_items, list):
                        # Take the first N items equal to completed_count
                        # These represent the completed items
                        completed_items = content_items[:completed_count]
                        
                        for i, item in enumerate(completed_items):
                            item_id = item.get("id", f"item_{i}")
                            
                            # Skip if we've already asked about this item
                            if item_id in self.asked_content_items[content_type]:
                                continue
                            
                            content_data = item.get("content_data", {})
                            # Only generate question if content_data has actual content
                            if content_data and any(content_data.values()):
                                # Extract content-specific forgetting evidence for this specific item
                                forgetting_evidence = self._extract_content_forgetting_evidence(snapshot, content_type, item_id)
                                
                                # Generate question based on content type
                                question_text = self._generate_content_question_text(content_type, content_data)
                                
                                forgotten_count = forgetting_evidence["total_forgotten_items"]
                                if forgotten_count > 0:
                                    answer_text = f"I can regenerate this {content_type.replace('_', ' ')} based on the completed content in memory. Note: {forgotten_count} content items were updated/deleted and should have been forgotten."
                                else:
                                    answer_text = f"I can regenerate this {content_type.replace('_', ' ')} based on the completed content in memory."
                                
                                question = {
                                    "question_id": f"content_{content_type}_{snapshot.session_id}_{item_id}",
                                    "question": question_text,
                                    "answer": answer_text,
                                    "question_type": "content_regeneration",
                                    "category": content_type,
                                    "memory_type": "content_memory",
                                    "session_id": snapshot.session_id,
                                    "session_date": snapshot.session_date,
                                    "memory_evidence": {
                                        "content_data": content_data,
                                        "pool_status": "completed",
                                        "item_id": item_id,
                                        "completed_pool_count": completed_count,
                                        "is_new_completion": True
                                    },
                                    "forgetting_evidence": forgetting_evidence
                                }
                                questions.append(question)
                                
                                # Mark this item as asked
                                self.asked_content_items[content_type].add(item_id)
        
        return questions
    
    def _generate_content_question_text(self, content_type: str, content_data: Dict) -> str:
        """Generate appropriate question text for content types"""
        if content_type == "project_proposal":
            title = content_data.get("project_title", "Unknown Project")
            return f"Write a project proposal for '{title}'"
        elif content_type == "email_writeup":
            purpose = content_data.get("email_purpose", "Unknown Purpose")
            return f"Write an email about: {purpose}"
        elif content_type == "social_media_post":
            platform = content_data.get("platform", "social media")
            message = content_data.get("main_message", "your topic")
            return f"Write a {platform} post about: {message}"
        elif content_type == "meeting_notes":
            title = content_data.get("meeting_title", "Unknown Meeting")
            return f"Write meeting notes for: {title}"
        else:
            return f"Generate content for {content_type.replace('_', ' ')}"
    
    def generate_goal_questions(self, snapshot: MemorySnapshot) -> List[Dict[str, Any]]:
        """Generate goal-based questions comparing budget vs actual spending
        
        Note: goal_progress_analysis questions are only generated for weekly timelines
        because goals are set weekly. For monthly/quarterly timelines, use comparative
        questions instead which properly handle weekly goal changes over time.
        """
        questions = []
        
        # Skip goal_progress_analysis questions for monthly/quarterly timelines
        # Goals are weekly, so comparing cumulative monthly/quarterly spending against
        # a single weekly goal value doesn't make sense
        if self.config.timeline_type in ["monthly", "quarterly"]:
            return questions
        
        memory_items = snapshot.memory_state.get("memory_items", {})
        goal_memory = memory_items.get("goal_memory", {})
        activity_memory = memory_items.get("activity_memory", {})
        
        # Enrich activity memory items with session_id before calculating goals
        enriched_activity_memory = {}
        if "food_expenses" in activity_memory:
            enriched_activity_memory["food_expenses"] = self._enrich_activity_items_with_session_id(
                activity_memory["food_expenses"], "food_expenses"
            )
        if "step_tracker" in activity_memory:
            enriched_activity_memory["step_tracker"] = self._enrich_activity_items_with_session_id(
                activity_memory["step_tracker"], "step_tracker"
            )
        
        if goal_memory:
            # Generate questions about goal progress and planning
            for goal_category, goals in goal_memory.items():
                goal_count = self._count_actual_memory_items(goals)
                if goal_count > 0:
                    # Process each goal for budget vs actual analysis
                    for i, goal in enumerate(goals):
                        if isinstance(goal, dict):
                            goal_subcategory = goal.get("subcategory", "unknown")
                            goal_value = goal.get("value", 0)
                            goal_created_at = goal.get("created_at")
                            
                            # Calculate actual spending/activity for this goal and collect activity sessions
                            # Use enriched activity memory so items have session_id
                            actual_value, activity_sessions = self._calculate_actual_vs_goal(
                                goal_category, goal_subcategory, enriched_activity_memory
                            )
                            
                            # Find the session where the goal was created
                            goal_session_id = None
                            if goal_created_at:
                                goal_session_id = self._find_session_id_for_goal(
                                    goal_category, goal_subcategory, goal_created_at
                                )
                            
                            # Determine if goal is met
                            goal_status = self._evaluate_goal_progress(
                                goal_category, goal_value, actual_value
                            )
                            
                            # Generate category-specific question
                            question_text, answer_text = self._generate_goal_question_text(
                                goal_category, goal_subcategory, goal_value, actual_value, goal_status
                            )
                            
                            question = {
                                "question_id": f"goal_{goal_category}_{goal_subcategory}_{snapshot.session_id}_{i}",
                                "question": question_text,
                                "answer": answer_text,
                                "question_type": "goal_progress_analysis",
                                "category": goal_category,
                                "subcategory": goal_subcategory,
                                "memory_type": "goal_memory",
                                "session_id": snapshot.session_id,
                                "session_date": snapshot.session_date,
                                "memory_evidence": {
                                    "goal_value": goal_value,
                                    "actual_value": actual_value,
                                    "goal_status": goal_status,
                                    "goal_data": goal,
                                    "goal_session": {
                                        "session_id": goal_session_id,
                                        "session_date": goal_created_at
                                    } if goal_session_id else None,
                                    "activity_sessions": activity_sessions
                                }
                            }
                            questions.append(question)
        
        return questions
    
    def _calculate_actual_vs_goal(self, goal_category: str, goal_subcategory: str, activity_memory: Dict) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculate actual spending/activity for a specific goal and collect contributing activity sessions
        
        Returns:
            Tuple of (actual_value, activity_sessions) where activity_sessions is a list of 
            dicts with session_id, session_date, and item info
        """
        actual_value = 0.0
        activity_sessions = []
        
        if goal_category == "food_expenses" and "food_expenses" in activity_memory:
            food_expenses = activity_memory["food_expenses"]
            if goal_subcategory == "total":
                # Sum all food expenses
                for expense in food_expenses:
                    if isinstance(expense, dict):
                        actual_value += expense.get("amount", 0)
                        # Collect session info
                        if "session_id" in expense:
                            activity_sessions.append({
                                "session_id": expense.get("session_id"),
                                "session_date": expense.get("created_at"),
                                "amount": expense.get("amount", 0),
                                "expense_type": expense.get("expense_type", "unknown")
                            })
            else:
                # Sum expenses for specific subcategory (e.g., coffee, grocery)
                for expense in food_expenses:
                    if isinstance(expense, dict) and expense.get("expense_type") == goal_subcategory:
                        actual_value += expense.get("amount", 0)
                        # Collect session info
                        if "session_id" in expense:
                            activity_sessions.append({
                                "session_id": expense.get("session_id"),
                                "session_date": expense.get("created_at"),
                                "amount": expense.get("amount", 0),
                                "expense_type": expense.get("expense_type", goal_subcategory)
                            })
        
        elif goal_category == "step_tracker" and "step_tracker" in activity_memory:
            step_data = activity_memory["step_tracker"]
            if goal_subcategory == "daily_steps":
                # Calculate average daily steps or total steps depending on goal type
                total_steps = 0
                for step in step_data:
                    if isinstance(step, dict):
                        step_count = step.get("step_count", 0)
                        total_steps += step_count
                        # Collect session info
                        if "session_id" in step:
                            activity_sessions.append({
                                "session_id": step.get("session_id"),
                                "session_date": step.get("created_at") or step.get("date"),
                                "step_count": step_count
                            })
                days_count = len(step_data) if step_data else 1
                actual_value = total_steps / days_count  # Average daily steps
        
        return actual_value, activity_sessions
    
    def _evaluate_goal_progress(self, goal_category: str, goal_value: float, actual_value: float) -> Dict[str, Any]:
        """Evaluate whether goal is met and calculate progress"""
        if goal_category == "food_expenses":
            # For expenses, staying under budget is good
            is_goal_met = actual_value <= goal_value
            progress_percentage = (actual_value / goal_value * 100) if goal_value > 0 else 0
            variance = actual_value - goal_value
            status = "under_budget" if is_goal_met else "over_budget"
        
        elif goal_category == "step_tracker":
            # For steps, meeting or exceeding target is good
            is_goal_met = actual_value >= goal_value
            progress_percentage = (actual_value / goal_value * 100) if goal_value > 0 else 0
            variance = actual_value - goal_value
            status = "goal_met" if is_goal_met else "below_target"
        
        else:
            # Default evaluation
            is_goal_met = actual_value >= goal_value
            progress_percentage = (actual_value / goal_value * 100) if goal_value > 0 else 0
            variance = actual_value - goal_value
            status = "goal_met" if is_goal_met else "below_target"
        
        return {
            "is_goal_met": is_goal_met,
            "progress_percentage": progress_percentage,
            "variance": variance,
            "status": status
        }
    
    def _generate_goal_question_text(self, goal_category: str, goal_subcategory: str, 
                                   goal_value: float, actual_value: float, goal_status: Dict) -> Tuple[str, str]:
        """Generate appropriate question and answer text for goal analysis"""
        
        if goal_category == "food_expenses":
            if goal_subcategory == "total":
                question = f"Am I staying within my total food budget?"
            else:
                question = f"Am I meeting my {goal_subcategory} budget goal?"
            
            if goal_status["is_goal_met"]:
                answer = f"Yes! You've spent ${actual_value:.2f} against your ${goal_value:.2f} budget, staying ${abs(goal_status['variance']):.2f} under budget ({goal_status['progress_percentage']:.1f}% of budget used)."
            else:
                answer = f"You've exceeded your budget. You've spent ${actual_value:.2f} against your ${goal_value:.2f} budget, going ${goal_status['variance']:.2f} over budget ({goal_status['progress_percentage']:.1f}% of budget used)."
        
        elif goal_category == "step_tracker":
            if goal_subcategory == "daily_steps":
                question = f"Am I meeting my daily step goal?"
            else:
                question = f"Am I meeting my {goal_subcategory} goal?"
            
            if goal_status["is_goal_met"]:
                answer = f"Great job! You're averaging {actual_value:,.0f} steps against your {goal_value:,.0f} step goal, exceeding it by {abs(goal_status['variance']):,.0f} steps ({goal_status['progress_percentage']:.1f}% of goal achieved)."
            else:
                answer = f"You're below your target. You're averaging {actual_value:,.0f} steps against your {goal_value:,.0f} step goal, falling short by {abs(goal_status['variance']):,.0f} steps ({goal_status['progress_percentage']:.1f}% of goal achieved)."
        
        else:
            question = f"How am I progressing on my {goal_category} {goal_subcategory} goal?"
            if goal_status["is_goal_met"]:
                answer = f"You're meeting your goal! Current: {actual_value:.2f}, Target: {goal_value:.2f} ({goal_status['progress_percentage']:.1f}% achieved)."
            else:
                answer = f"You're below target. Current: {actual_value:.2f}, Target: {goal_value:.2f} ({goal_status['progress_percentage']:.1f}% achieved)."
        
        return question, answer
    
    def _validate_forgetting_coverage(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that categories requiring forgetting have at least one question with forgetting evidence
        
        Returns validation report with:
        - valid: bool - whether validation passed
        - missing_categories: list of (memory_type, category) tuples that lack forgetting evidence
        - coverage_report: dict mapping (memory_type, category) to forgetting stats
        """
        # Categories that REQUIRE forgetting evidence (have delete operations)
        categories_requiring_forgetting = {
            # Preference memory - all categories support delete
            "preference_memory": ["movies", "books", "music", "travel"],
            # Content memory - all categories support delete
            "content_memory": ["project_proposal", "email_writeup", "social_media_post", "meeting_notes"],
            # Activity memory - only calendar_event and todo_list support delete
            "activity_memory": ["calendar_event", "todo_list"],
            # Goal memory - all categories support delete
            "goal_memory": ["food_expenses", "step_tracker"]
        }
        
        # Track forgetting evidence by category
        coverage = {}
        
        for question in questions:
            memory_type = question.get("memory_type")
            category = question.get("category")
            forgetting_evidence = question.get("forgetting_evidence", {})
            total_forgotten = forgetting_evidence.get("total_forgotten_items", 0)
            
            key = (memory_type, category)
            if key not in coverage:
                coverage[key] = {
                    "total_questions": 0,
                    "questions_with_forgetting": 0,
                    "total_forgotten_items": 0
                }
            
            coverage[key]["total_questions"] += 1
            if total_forgotten > 0:
                coverage[key]["questions_with_forgetting"] += 1
                coverage[key]["total_forgotten_items"] += total_forgotten
        
        # Check which required categories are missing forgetting evidence
        missing_categories = []
        
        for memory_type, categories in categories_requiring_forgetting.items():
            for category in categories:
                key = (memory_type, category)
                if key in coverage:
                    if coverage[key]["questions_with_forgetting"] == 0:
                        missing_categories.append(key)
        
        return {
            "valid": len(missing_categories) == 0,
            "missing_categories": missing_categories,
            "coverage_report": coverage
        }
    
    def _add_evaluation_questions_to_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add evaluation questions to a main question"""
        # Generate evaluation questions for this main question
        evaluation_questions = self.evaluation_generator.generate_all_evaluation_questions(question_data)
        
        # Format evaluation questions for output
        evaluation_data = self.evaluation_generator.format_evaluation_questions_for_output(evaluation_questions)
        
        # Add evaluation data to the question
        question_data["evaluation"] = evaluation_data
        
        return question_data
    
    def generate_questions_for_snapshot(self, snapshot: MemorySnapshot) -> List[Dict[str, Any]]:
        """Generate all types of questions for a memory snapshot"""
        all_questions = []
        
        print(f"🎯 Generating questions for session {snapshot.session_id} ({snapshot.session_date})")
        
        # Generate different types of questions
        preference_q = self.generate_preference_questions(snapshot)
        activity_q = self.generate_activity_questions(snapshot)
        content_q = self.generate_content_questions(snapshot)
        goal_q = self.generate_goal_questions(snapshot)
        
        # Add evaluation questions to each main question
        print(f"  🔍 Adding evaluation questions...")
        
        # Process preference questions
        for question in preference_q:
            question_with_eval = self._add_evaluation_questions_to_question(question)
            all_questions.append(question_with_eval)
        
        # Process activity questions  
        for question in activity_q:
            question_with_eval = self._add_evaluation_questions_to_question(question)
            all_questions.append(question_with_eval)
        
        # Process content questions
        for question in content_q:
            question_with_eval = self._add_evaluation_questions_to_question(question)
            all_questions.append(question_with_eval)
        
        # Process goal questions
        for question in goal_q:
            question_with_eval = self._add_evaluation_questions_to_question(question)
            all_questions.append(question_with_eval)
        
        # Count total evaluation questions
        total_eval_questions = sum(
            question.get("evaluation", {}).get("total_evaluation_questions", 0) 
            for question in all_questions
        )
        
        print(f"  📝 Generated {len(preference_q)} preference, {len(activity_q)} activity, {len(content_q)} content, {len(goal_q)} goal questions")
        print(f"  🔍 Generated {total_eval_questions} evaluation questions across all main questions")
        
        # Validate forgetting coverage
        validation_result = self._validate_forgetting_coverage(all_questions)
        if not validation_result["valid"]:
            print(f"\n  ⚠️  WARNING: Some categories lack forgetting evidence!")
            for memory_type, category in validation_result["missing_categories"]:
                coverage = validation_result["coverage_report"].get((memory_type, category), {})
                total_q = coverage.get("total_questions", 0)
                print(f"     - {memory_type}/{category}: {total_q} question(s), but ZERO have forgetting evidence")
            print(f"  💡 Suggestion: Re-run session simulation with higher delete probabilities or longer timeline")
        else:
            print(f"  ✅ Forgetting coverage validated: All categories have forgetting evidence")
        
        return all_questions
    
    def generate_all_questions(self) -> List[Dict[str, Any]]:
        """Generate questions for all memory snapshots"""
        print("🚀 Starting unified question generation...")
        
        # Extract memory snapshots at intervals
        snapshots = self.extract_memory_snapshots()
        
        if not snapshots:
            print("❌ No memory snapshots extracted")
            return []
        
        # Generate questions for each snapshot (should be only final snapshot)
        all_questions = []
        for snapshot in snapshots:
            questions = self.generate_questions_for_snapshot(snapshot)
            all_questions.extend(questions)
        
        # Generate temporal breakdown questions if monthly or quarterly
        if self.config.timeline_type in ["monthly", "quarterly"]:
            final_snapshot = snapshots[-1]  # Should be the only snapshot
            temporal_questions = self.generate_temporal_breakdown_questions(final_snapshot)
            # Add evaluation questions to temporal breakdown questions
            print(f"  🔍 Adding evaluation questions to temporal breakdown questions...")
            for question in temporal_questions:
                question_with_eval = self._add_evaluation_questions_to_question(question)
                all_questions.append(question_with_eval)
        
        self.questions = all_questions
        print(f"✅ Generated {len(all_questions)} total questions across {len(snapshots)} snapshots")
        
        return all_questions
    
    def generate_temporal_breakdown_questions(self, snapshot: MemorySnapshot) -> List[Dict[str, Any]]:
        """Generate temporal breakdown questions for monthly/quarterly timelines
        
        These questions are asked at the END but reference specific time periods:
        - Monthly: Ask about each week (week 1, week 2, etc.)
        - Quarterly: Ask about each month (month 1, month 2, etc.)
        """
        breakdown_questions = []
        
        # Calculate time periods to break down
        session_ids = sorted([int(sid) for sid in self.memory_states.keys()])
        start_session = session_ids[0]
        start_date = datetime.strptime(self.memory_states[str(start_session)]["session_date"], "%Y-%m-%d")
        total_days = snapshot.time_since_start.days
        
        if self.config.timeline_type == "monthly":
            # Generate weekly breakdown questions
            week_boundaries = [(1, 7), (8, 14), (15, 21), (22, 28)]
            periods = [(week_num, start, end) for week_num, (start, end) in enumerate(week_boundaries, 1) 
                      if start <= total_days]
            period_type = "week"
            period_label = "this month"
            print(f"🔍 Generating comparative questions across {len(periods)} weeks...")
            
        elif self.config.timeline_type == "quarterly":
            # Generate monthly breakdown questions
            month_boundaries = [(1, 30), (31, 60), (61, 90)]
            periods = [(month_num, start, end) for month_num, (start, end) in enumerate(month_boundaries, 1) 
                      if start <= total_days]
            period_type = "month"
            period_label = "the last 3 months"
            print(f"🔍 Generating comparative questions across {len(periods)} months...")
        else:
            return []
        
        # Generate comparative questions across all periods (main focus)
        comparative_questions = self._generate_comparative_period_questions(
            snapshot, periods, period_type, period_label
        )
        breakdown_questions.extend(comparative_questions)
        
        print(f"  ✅ Generated {len(breakdown_questions)} temporal comparative questions")
        return breakdown_questions
    
    def _generate_comparative_period_questions(self, snapshot: MemorySnapshot, 
                                              periods: List[Tuple[int, int, int]],
                                              period_type: str, period_label: str) -> List[Dict[str, Any]]:
        """Generate comparative questions across multiple periods
        
        Examples:
        - "Which week in last month did I have the most steps?"
        - "How many weeks was I able to meet my coffee budget?"
        """
        questions = []
        
        if len(periods) < 2:
            return questions  # Need at least 2 periods for comparison
        
        # Get start date for calculations
        session_ids = sorted([int(sid) for sid in self.memory_states.keys()])
        start_session = session_ids[0]
        start_date = datetime.strptime(self.memory_states[str(start_session)]["session_date"], "%Y-%m-%d")
        
        # Access final snapshot memory
        memory_items = snapshot.memory_state.get("memory_items", {})
        activity_memory = memory_items.get("activity_memory", {})
        goal_memory = memory_items.get("goal_memory", {})
        
        # === COMPARATIVE ACTIVITY QUESTIONS ===
        
        # Which week had the most steps?
        if "step_tracker" in activity_memory and activity_memory["step_tracker"]:
            # Enrich step data with session_id first
            enriched_step_data = self._enrich_activity_items_with_session_id(
                activity_memory["step_tracker"], "step_tracker"
            )
            
            period_step_totals = {}
            for period_num, day_start, day_end in periods:
                period_start_date = start_date + timedelta(days=day_start - 1)
                period_end_date = start_date + timedelta(days=day_end - 1)
                period_steps = self._filter_items_by_date_range(
                    enriched_step_data,
                    period_start_date.strftime("%Y-%m-%d"),
                    period_end_date.strftime("%Y-%m-%d")
                )
                total_steps = sum(s.get("step_count", 0) for s in period_steps if isinstance(s, dict))
                period_step_totals[period_num] = {
                    "total_steps": total_steps,
                    "items": period_steps,
                    "period_ref": f"{self._ordinal(period_num)} {period_type}"
                }
            
            if period_step_totals:
                max_period = max(period_step_totals.keys(), key=lambda k: period_step_totals[k]["total_steps"])
                max_steps = period_step_totals[max_period]["total_steps"]
                
                question = {
                    "question_id": f"comparative_steps_max_{period_type}_{snapshot.session_id}",
                    "question": f"Which {period_type} in {period_label} did I have the most steps?",
                    "answer": f"You had the most steps in the {period_step_totals[max_period]['period_ref']} with {max_steps:,} steps.",
                    "question_type": "temporal_comparative_analysis",
                    "category": "step_tracker",
                    "memory_type": "activity_memory",
                    "session_id": snapshot.session_id,
                    "session_date": snapshot.session_date,
                    "temporal_context": {
                        "period_type": period_type,
                        "period_label": period_label,
                        "comparison_type": "maximum",
                        "total_periods_analyzed": len(periods)
                    },
                    "memory_evidence": {
                        "max_period": max_period,
                        "max_value": max_steps,
                        "period_breakdown": period_step_totals
                    }
                }
                questions.append(question)
        
        # Which week had the most food expenses (by category)?
        if "food_expenses" in activity_memory and activity_memory["food_expenses"]:
            # Enrich food expenses with session_id first
            enriched_food_expenses = self._enrich_activity_items_with_session_id(
                activity_memory["food_expenses"], "food_expenses"
            )
            
            # Group all expense types
            all_expense_types = set()
            for expense in enriched_food_expenses:
                if isinstance(expense, dict):
                    expense_type = expense.get("expense_type")
                    if expense_type:
                        all_expense_types.add(expense_type)
            
            # Generate comparative questions for each expense type
            for expense_type in sorted(all_expense_types):
                period_expense_totals = {}
                for period_num, day_start, day_end in periods:
                    period_start_date = start_date + timedelta(days=day_start - 1)
                    period_end_date = start_date + timedelta(days=day_end - 1)
                    period_expenses = self._filter_items_by_date_range(
                        enriched_food_expenses,
                        period_start_date.strftime("%Y-%m-%d"),
                        period_end_date.strftime("%Y-%m-%d")
                    )
                    # Filter by expense type
                    type_expenses = [e for e in period_expenses if isinstance(e, dict) and e.get("expense_type") == expense_type]
                    total_amount = sum(e.get("amount", 0) for e in type_expenses)
                    period_expense_totals[period_num] = {
                        "total_amount": total_amount,
                        "items": type_expenses,
                        "period_ref": f"{self._ordinal(period_num)} {period_type}"
                    }
                
                # Only create question if there are expenses in at least one period
                if any(totals["total_amount"] > 0 for totals in period_expense_totals.values()):
                    max_period = max(period_expense_totals.keys(), key=lambda k: period_expense_totals[k]["total_amount"])
                    max_amount = period_expense_totals[max_period]["total_amount"]
                    
                    question = {
                        "question_id": f"comparative_{expense_type}_max_{period_type}_{snapshot.session_id}",
                        "question": f"Which {period_type} in {period_label} did I spend the most on {expense_type}?",
                        "answer": f"You spent the most on {expense_type} in the {period_expense_totals[max_period]['period_ref']} with ${max_amount:.2f}.",
                        "question_type": "temporal_comparative_analysis",
                        "category": "food_expenses",
                        "subcategory": expense_type,
                        "memory_type": "activity_memory",
                        "session_id": snapshot.session_id,
                        "session_date": snapshot.session_date,
                        "temporal_context": {
                            "period_type": period_type,
                            "period_label": period_label,
                            "comparison_type": "maximum",
                            "expense_type": expense_type,
                            "total_periods_analyzed": len(periods)
                        },
                        "memory_evidence": {
                            "max_period": max_period,
                            "max_value": max_amount,
                            "period_breakdown": period_expense_totals
                        }
                    }
                    questions.append(question)
        
        # === GOAL-BASED COMPARATIVE QUESTIONS ===
        
        # How many weeks was I able to meet my budget?
        if "food_expenses" in goal_memory and "food_expenses" in activity_memory:
            # Group goals by subcategory to handle multiple updates
            processed_subcategories = set()
            
            for goal in goal_memory.get("food_expenses", []):
                if isinstance(goal, dict):
                    goal_subcategory = goal.get("subcategory", "total")
                    
                    # Skip if we've already processed this subcategory
                    if goal_subcategory in processed_subcategories:
                        continue
                    processed_subcategories.add(goal_subcategory)
                    
                    # Build budget timeline for this subcategory (tracks all budget updates over time)
                    budget_timeline = self._build_budget_timeline(
                        "food_expenses", 
                        goal_subcategory, 
                        start_date, 
                        snapshot.time_since_start.days
                    )
                    
                    if not budget_timeline:
                        # No budget was ever set for this subcategory
                        continue
                    
                    periods_met = []
                    periods_exceeded = []
                    periods_no_budget = []
                    
                    # For quarterly timeline with monthly periods, convert weekly goals to monthly goals
                    # by summing weekly budgets for each week in the month
                    if period_type == "month":
                        for period_num, day_start, day_end in periods:
                            period_start_date = start_date + timedelta(days=day_start - 1)
                            period_end_date = start_date + timedelta(days=day_end - 1)
                            
                            # Break month into weeks to calculate monthly goal from weekly goals
                            month_weeks = []
                            for week_in_month in range(4):  # Approximately 4 weeks per month
                                week_start_day = day_start + (week_in_month * 7)
                                week_end_day = min(day_start + ((week_in_month + 1) * 7) - 1, day_end)
                                
                                if week_start_day <= day_end:
                                    month_weeks.append((week_start_day, week_end_day))
                            
                            # Calculate monthly goal: sum of weekly budgets for each week in the month
                            # If budget changes mid-month, use the budget active during each week
                            monthly_goal = 0.0
                            has_budget = False
                            
                            for week_start, week_end in month_weeks:
                                week_goal = self._get_applicable_budget(budget_timeline, week_end)
                                if week_goal is not None:
                                    monthly_goal += week_goal
                                    has_budget = True
                            
                            if not has_budget:
                                # No budget set for any week in this month
                                periods_no_budget.append(period_num)
                                continue
                            
                            # Get total expenses for this month
                            period_expenses = self._filter_items_by_date_range(
                                activity_memory["food_expenses"],
                                period_start_date.strftime("%Y-%m-%d"),
                                period_end_date.strftime("%Y-%m-%d")
                            )
                            
                            # Calculate actual spending for this month
                            if goal_subcategory == "total":
                                actual = sum(e.get("amount", 0) for e in period_expenses if isinstance(e, dict))
                            else:
                                actual = sum(
                                    e.get("amount", 0) for e in period_expenses 
                                    if isinstance(e, dict) and e.get("expense_type") == goal_subcategory
                                )
                            
                            # Compare monthly spending vs monthly goal (sum of weekly goals)
                            period_ref = f"{self._ordinal(period_num)} {period_type}"
                            if actual <= monthly_goal:
                                periods_met.append({
                                    "period": period_num, 
                                    "period_ref": period_ref, 
                                    "actual": actual, 
                                    "goal": monthly_goal
                                })
                            else:
                                periods_exceeded.append({
                                    "period": period_num, 
                                    "period_ref": period_ref, 
                                    "actual": actual, 
                                    "goal": monthly_goal, 
                                    "overspent": actual - monthly_goal
                                })
                    else:
                        # For monthly timeline with weekly periods, compare weekly spending vs weekly budget directly
                        for period_num, day_start, day_end in periods:
                            period_start_date = start_date + timedelta(days=day_start - 1)
                            period_end_date = start_date + timedelta(days=day_end - 1)
                            
                            # Get applicable budget for this period (carry forward from most recent update)
                            period_goal = self._get_applicable_budget(budget_timeline, day_end)
                            
                            if period_goal is None:
                                # No budget set yet for this period
                                periods_no_budget.append(period_num)
                                continue
                            
                            period_expenses = self._filter_items_by_date_range(
                                activity_memory["food_expenses"],
                                period_start_date.strftime("%Y-%m-%d"),
                                period_end_date.strftime("%Y-%m-%d")
                            )
                            
                            # Calculate actual for this period
                            if goal_subcategory == "total":
                                actual = sum(e.get("amount", 0) for e in period_expenses if isinstance(e, dict))
                            else:
                                actual = sum(
                                    e.get("amount", 0) for e in period_expenses 
                                    if isinstance(e, dict) and e.get("expense_type") == goal_subcategory
                                )
                            
                            period_ref = f"{self._ordinal(period_num)} {period_type}"
                            if actual <= period_goal:
                                periods_met.append({"period": period_num, "period_ref": period_ref, "actual": actual, "goal": period_goal})
                            else:
                                periods_exceeded.append({"period": period_num, "period_ref": period_ref, "actual": actual, "goal": period_goal, "overspent": actual - period_goal})
                    
                    # Generate "how many periods met budget" question
                    total_evaluated = len(periods_met) + len(periods_exceeded)
                    if total_evaluated > 0:
                        # Build detailed answer
                        if len(periods_no_budget) > 0:
                            answer = f"You met your {goal_subcategory} budget in {len(periods_met)} out of {total_evaluated} {period_type}s that had a budget set."
                        else:
                            answer = f"You met your {goal_subcategory} budget in {len(periods_met)} out of {len(periods)} {period_type}s."
                        
                        question_text = f"How many {period_type}s was I able to meet my {goal_subcategory} budget in {period_label}?"
                        
                        question = {
                            "question_id": f"comparative_goal_met_{goal_subcategory}_{period_type}_{snapshot.session_id}",
                            "question": question_text,
                            "answer": answer,
                            "question_type": "temporal_goal_analysis",
                            "category": "food_expenses",
                            "subcategory": goal_subcategory,
                            "memory_type": "goal_memory",
                            "session_id": snapshot.session_id,
                            "session_date": snapshot.session_date,
                            "temporal_context": {
                                "period_type": period_type,
                                "period_label": period_label,
                                "comparison_type": "budget_compliance",
                                "total_periods_analyzed": len(periods),
                                "periods_with_budget": total_evaluated,
                                "periods_without_budget": len(periods_no_budget)
                            },
                            "memory_evidence": {
                                "budget_timeline": budget_timeline,  # Shows when budgets were set/updated
                                "periods_met": periods_met,
                                "periods_exceeded": periods_exceeded,
                                "periods_no_budget": periods_no_budget,
                                "total_periods": len(periods),
                                "met_count": len(periods_met),
                                "exceeded_count": len(periods_exceeded)
                            }
                        }
                        questions.append(question)
                        
                        # Generate "how many periods overspent" question
                        if periods_exceeded:
                            if len(periods_no_budget) > 0:
                                overspent_answer = f"You exceeded your {goal_subcategory} budget in {len(periods_exceeded)} out of {total_evaluated} {period_type}s that had a budget set."
                            else:
                                overspent_answer = f"You exceeded your {goal_subcategory} budget in {len(periods_exceeded)} out of {len(periods)} {period_type}s."
                            
                            overspent_question_text = f"How many {period_type}s did I overspend on my {goal_subcategory} budget in {period_label}?"
                            
                            question = {
                                "question_id": f"comparative_goal_exceeded_{goal_subcategory}_{period_type}_{snapshot.session_id}",
                                "question": overspent_question_text,
                                "answer": overspent_answer,
                                "question_type": "temporal_goal_analysis",
                                "category": "food_expenses",
                                "subcategory": goal_subcategory,
                                "memory_type": "goal_memory",
                                "session_id": snapshot.session_id,
                                "session_date": snapshot.session_date,
                                "temporal_context": {
                                    "period_type": period_type,
                                    "period_label": period_label,
                                    "comparison_type": "budget_exceeded",
                                    "total_periods_analyzed": len(periods),
                                    "periods_with_budget": total_evaluated,
                                    "periods_without_budget": len(periods_no_budget)
                                },
                                "memory_evidence": {
                                    "budget_timeline": budget_timeline,  # Shows when budgets were set/updated
                                    "periods_exceeded": periods_exceeded,
                                    "periods_met": periods_met,
                                    "periods_no_budget": periods_no_budget,
                                    "total_periods": len(periods),
                                    "exceeded_count": len(periods_exceeded),
                                    "met_count": len(periods_met)
                                }
                            }
                            questions.append(question)
        
        return questions
    
    def _get_memory_snapshot_at_period(self, period_end_day: int) -> Optional[Dict[str, Any]]:
        """Get the memory state snapshot at the end of a specific period
        
        Args:
            period_end_day: Day number from start (1-indexed)
            
        Returns:
            Memory state at that point, or None if not found
        """
        session_ids = sorted([int(sid) for sid in self.memory_states.keys()])
        start_session = session_ids[0]
        start_date = datetime.strptime(self.memory_states[str(start_session)]["session_date"], "%Y-%m-%d")
        target_date = start_date + timedelta(days=period_end_day - 1)
        
        # Find the session closest to (but not after) the target date
        closest_session = None
        closest_date = None
        
        for sid in session_ids:
            session_date_str = self.memory_states[str(sid)]["session_date"]
            session_date = datetime.strptime(session_date_str, "%Y-%m-%d")
            
            if session_date <= target_date:
                if closest_date is None or session_date > closest_date:
                    closest_session = sid
                    closest_date = session_date
        
        if closest_session:
            return {
                "session_id": closest_session,
                "session_date": self.memory_states[str(closest_session)]["session_date"],
                "memory_state": self.memory_states[str(closest_session)]["memory_state_after_session"]
            }
        
        return None
    
    def _generate_period_specific_questions(self, snapshot: MemorySnapshot, period_num: int,
                                           day_start: int, day_end: int, period_type: str,
                                           period_label: str) -> List[Dict[str, Any]]:
        """Generate questions for a specific time period
        
        Args:
            snapshot: The final memory snapshot
            period_num: Period number (e.g., week 1, week 2)
            day_start: Start day of period (1-indexed from start_date)
            day_end: End day of period (inclusive)
            period_type: "week" or "month"
            period_label: "this month" or "this quarter"
        """
        questions = []
        
        # Get start date
        session_ids = sorted([int(sid) for sid in self.memory_states.keys()])
        start_session = session_ids[0]
        start_date = datetime.strptime(self.memory_states[str(start_session)]["session_date"], "%Y-%m-%d")
        
        # Calculate actual date range for this period
        period_start_date = start_date + timedelta(days=day_start - 1)
        period_end_date = start_date + timedelta(days=day_end - 1)
        
        # Period reference for questions
        period_ref = f"{self._ordinal(period_num)} {period_type} of {period_label}"
        periods_back_ref = f"{period_num} {period_type}{'s' if period_num > 1 else ''} back"
        
        # Get memory snapshot at the END of this period
        period_snapshot = self._get_memory_snapshot_at_period(day_end)
        
        if not period_snapshot:
            return questions
        
        period_memory_state = period_snapshot["memory_state"]
        period_session_id = period_snapshot["session_id"]
        period_session_date = period_snapshot["session_date"]
        
        # Access the final snapshot's memory for comparison
        memory_items = snapshot.memory_state.get("memory_items", {})
        activity_memory = memory_items.get("activity_memory", {})
        
        # Get period's memory items
        period_memory_items = period_memory_state.get("memory_items", {})
        period_activity_memory = period_memory_items.get("activity_memory", {})
        period_preference_memory = period_memory_items.get("preference_memory", {})
        
        # === ACTIVITY MEMORY AGGREGATION QUESTIONS ===
        # Note: We focus ONLY on aggregation questions that make sense for temporal comparison.
        # We don't generate preference/todo/calendar questions here because:
        # - Preferences might be forgotten/deleted in later sessions
        # - Todo/calendar are status queries, not aggregations suitable for comparison
        # 
        # These aggregation questions will be used in comparative analysis to answer:
        # "Which week had the most X?" type questions
        
        return questions
    
    def _filter_items_by_date_range(self, items: List, start_date: str, end_date: str) -> List:
        """Filter activity items by date range"""
        filtered = []
        for item in items:
            if isinstance(item, dict):
                item_date = item.get("created_at") or item.get("date")
                if item_date and start_date <= item_date <= end_date:
                    filtered.append(item)
        return filtered
    
    def _ordinal(self, n: int) -> str:
        """Convert number to ordinal (1st, 2nd, 3rd, etc.)"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
    
    def _build_budget_timeline(self, category: str, subcategory: str, start_date: datetime, total_days: int) -> Dict[int, float]:
        """Build a timeline of budget values by scanning through all memory states
        
        Returns a dict mapping day_offset (1-indexed) to budget value.
        Budget values carry forward until explicitly updated.
        
        Args:
            category: Goal category (e.g., "food_expenses")
            subcategory: Goal subcategory (e.g., "coffee", "total")
            start_date: Start date of the timeline
            total_days: Total days in the timeline
            
        Returns:
            Dict mapping day offset to budget value (e.g., {1: 50.0, 8: 100.0})
        """
        budget_timeline = {}
        current_budget = None
        
        # Scan through all memory states chronologically
        session_ids = sorted([int(sid) for sid in self.memory_states.keys()])
        
        for sid in session_ids:
            session_data = self.memory_states[str(sid)]
            session_date_str = session_data.get("session_date", "")
            
            if not session_date_str:
                continue
                
            session_date = datetime.strptime(session_date_str, "%Y-%m-%d")
            day_offset = (session_date - start_date).days + 1  # 1-indexed
            
            if day_offset < 1 or day_offset > total_days:
                continue
            
            # Check if this session updated the goal
            memory_state = session_data.get("memory_state_after_session", {})
            memory_items = memory_state.get("memory_items", {})
            goal_memory = memory_items.get("goal_memory", {})
            
            if category in goal_memory:
                for goal in goal_memory.get(category, []):
                    if isinstance(goal, dict):
                        if goal.get("subcategory") == subcategory:
                            new_budget = goal.get("value")
                            if new_budget != current_budget:
                                # Budget changed!
                                current_budget = new_budget
                                budget_timeline[day_offset] = current_budget
        
        return budget_timeline
    
    def _get_applicable_budget(self, budget_timeline: Dict[int, float], day_offset: int) -> Optional[float]:
        """Get the applicable budget for a given day, carrying forward from most recent update
        
        Args:
            budget_timeline: Dict mapping day offset to budget value
            day_offset: Day offset to query (1-indexed)
            
        Returns:
            Budget value, or None if no budget was set
        """
        if not budget_timeline:
            return None
        
        # Find the most recent budget update at or before this day
        applicable_budget = None
        for budget_day in sorted(budget_timeline.keys()):
            if budget_day <= day_offset:
                applicable_budget = budget_timeline[budget_day]
            else:
                break
        
        return applicable_budget
    
    def _generate_preference_answer_from_memory(self, category: str, category_items: Dict, time_reference: str) -> str:
        """Generate a detailed answer based on preference memory items
        
        Args:
            category: Preference category (movies, books, music, travel)
            category_items: Memory items for this category
            time_reference: Time reference string (e.g., "2 weeks back")
            
        Returns:
            Detailed answer with actual preferences
        """
        likes_items = []
        dislikes_items = []
        
        # Extract likes and dislikes from all subcategories
        for subcategory, subcat_data in category_items.items():
            if isinstance(subcat_data, dict):
                if "likes" in subcat_data and subcat_data["likes"]:
                    for item in subcat_data["likes"][:3]:  # Take up to 3 items per subcategory
                        if isinstance(item, dict):
                            item_value = item.get("item", "")
                            if item_value:
                                likes_items.append(f"{item_value} ({subcategory})")
                        else:
                            likes_items.append(f"{item} ({subcategory})")
                
                if "dislikes" in subcat_data and subcat_data["dislikes"]:
                    for item in subcat_data["dislikes"][:2]:  # Take up to 2 dislikes
                        if isinstance(item, dict):
                            item_value = item.get("item", "")
                            if item_value:
                                dislikes_items.append(f"{item_value} ({subcategory})")
                        else:
                            dislikes_items.append(f"{item} ({subcategory})")
        
        # Build answer
        answer_parts = [f"Based on your {category} preferences from {time_reference}:"]
        
        if likes_items:
            if len(likes_items) > 5:
                likes_str = ", ".join(likes_items[:5]) + f", and {len(likes_items) - 5} more"
            else:
                likes_str = ", ".join(likes_items)
            answer_parts.append(f"You liked: {likes_str}.")
        
        if dislikes_items:
            if len(dislikes_items) > 3:
                dislikes_str = ", ".join(dislikes_items[:3]) + f", and {len(dislikes_items) - 3} more"
            else:
                dislikes_str = ", ".join(dislikes_items)
            answer_parts.append(f"You disliked: {dislikes_str}.")
        
        answer_parts.append(f"I can provide {category[:-1] if category.endswith('s') else category} recommendations based on these preferences.")
        
        return " ".join(answer_parts)
    
    def save_questions(self, output_file: str = None) -> str:
        """Save generated questions to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.config.output_dir, f"unified_questions_{timestamp}.json")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        
        # Add task_type to each question based on the mapping
        for q in self.questions:
            question_type = q.get("question_type", "")
            q["task_type"] = self.QUESTION_TYPE_TO_TASK_TYPE.get(question_type, "Unknown")
        
        # Group questions by type for better organization
        questions_by_type = defaultdict(list)
        for q in self.questions:
            questions_by_type[q["question_type"]].append(q)
        
        # Group questions by task_type
        questions_by_task_type = defaultdict(list)
        for q in self.questions:
            questions_by_task_type[q["task_type"]].append(q)
        
        output_data = {
            "metadata": {
                "description": "Unified question generation for all memory types with evaluation questions",
                "persona": self.metadata.get("persona", "unknown"),
                "total_questions": len(self.questions),
                "total_evaluation_questions": sum(
                    q.get("evaluation", {}).get("total_evaluation_questions", 0) for q in self.questions
                ),
                "timeline_type": self.config.timeline_type,
                "memory_states_file": self.config.memory_states_file,
                "generation_timestamp": datetime.now().isoformat(),
                "question_types": {qtype: len(questions) for qtype, questions in questions_by_type.items()},
                "task_types": {task_type: len(questions) for task_type, questions in questions_by_task_type.items()},
                "question_type_to_task_type_mapping": self.QUESTION_TYPE_TO_TASK_TYPE
            },
            "questions_by_type": dict(questions_by_type),
            "questions_by_task_type": dict(questions_by_task_type),
            "all_questions": sorted(self.questions, key=lambda x: x["session_id"])
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Saved {len(self.questions)} questions to: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print a summary of generated questions"""
        if not self.questions:
            print("No questions generated")
            return
        
        print(f"\n📊 QUESTION GENERATION SUMMARY")
        print("=" * 60)
        
        # Group by type and category
        by_type = defaultdict(lambda: defaultdict(int))
        by_session = defaultdict(int)
        by_task_type = defaultdict(int)
        
        for q in self.questions:
            by_type[q["question_type"]][q.get("category", "unknown")] += 1
            by_session[q["session_id"]] += 1
            task_type = q.get("task_type") or self.QUESTION_TYPE_TO_TASK_TYPE.get(q["question_type"], "Unknown")
            by_task_type[task_type] += 1
        
        print(f"Total Questions: {len(self.questions)}")
        print(f"Timeline Type: {self.config.timeline_type.upper()}")
        print(f"Sessions with Questions: {len(by_session)}")
        
        print(f"\n🎯 Questions by Task Type:")
        for task_type in ["Remembering", "Reasoning", "Recommending"]:
            if task_type in by_task_type:
                print(f"  {task_type}: {by_task_type[task_type]}")
        
        print(f"\n📋 Questions by Type:")
        for qtype, categories in by_type.items():
            total = sum(categories.values())
            task_type = self.QUESTION_TYPE_TO_TASK_TYPE.get(qtype, "Unknown")
            print(f"  {qtype} ({task_type}): {total}")
            for category, count in categories.items():
                print(f"    └─ {category}: {count}")
        
        print(f"\n📅 Questions by Session:")
        for session_id in sorted(by_session.keys()):
            session_date = next((q["session_date"] for q in self.questions if q["session_id"] == session_id), "unknown")
            print(f"  Session {session_id} ({session_date}): {by_session[session_id]} questions")
        
        # Print forgetting coverage validation
        validation_result = self._validate_forgetting_coverage(self.questions)
        print(f"\n🔍 Forgetting Coverage Validation:")
        if validation_result["valid"]:
            print(f"  ✅ All categories requiring forgetting have evidence")
        else:
            print(f"  ⚠️  WARNING: {len(validation_result['missing_categories'])} categories lack forgetting evidence:")
            for memory_type, category in validation_result["missing_categories"]:
                print(f"     - {memory_type}/{category}")
            print(f"\n  💡 Recommendation: Re-run simulation with higher delete probabilities or longer timeline")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified Question Generator for Memora Framework')
    parser.add_argument('--session_directory', required=True,
                      help='Path to session directory containing memory_states_by_session.json')
    parser.add_argument('--timeline', type=str, default='weekly', 
                      choices=['weekly', 'monthly', 'quarterly'],
                      help='Timeline type: weekly (default), monthly, or quarterly')
    
    args = parser.parse_args()
    
    # Validate session directory
    if not os.path.exists(args.session_directory):
        print(f"❌ Session directory not found: {args.session_directory}")
        sys.exit(1)
    
    # Construct memory states file path
    memory_states_file = os.path.join(args.session_directory, "memory_states_by_session.json")
    if not os.path.exists(memory_states_file):
        print(f"❌ Memory states file not found: {memory_states_file}")
        sys.exit(1)
    
    # Generate timestamped filename in the same session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.session_directory, f"unified_questions_{args.timeline}_{timestamp}.json")
    
    # Create configuration
    config = QuestionGenerationConfig(
        memory_states_file=memory_states_file,
        output_dir=args.session_directory,
        timeline_type=args.timeline
    )
    
    try:
        # Initialize generator
        generator = UnifiedQuestionGenerator(config)
        
        # Generate questions
        questions = generator.generate_all_questions()
        
        if not questions:
            print("❌ No questions generated")
            sys.exit(1)
        
        # Save questions to the session directory
        actual_output_file = generator.save_questions(output_file)
        
        # Print summary
        generator.print_summary()
        
        print(f"\n✅ Question generation complete!")
        print(f"📁 Session directory: {args.session_directory}")
        print(f"📄 Memory states file: {memory_states_file}")
        print(f"💾 Output file: {actual_output_file}")
        
    except Exception as e:
        print(f"❌ Error during question generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
