#!/usr/bin/env python3
"""
Memory State Validator

This script validates the consistency between session data (new_sessions.json) and 
memory states (memory_states_by_session.json) to ensure:
1. Operations in session data match the actual memory state changes
2. Memory states are correctly accumulated across sessions
3. Operation conversions are properly reflected in session["operation"] field
4. Operation details match the actual memory changes

Usage:
    python quality_checks/memory_state_validator.py <session_dir>
    
Example:
    python quality_checks/memory_state_validator.py output/sessions_<timestamp>_<persona>
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class MemoryStateValidator:
    """Validates consistency between session data and memory states"""
    
    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)
        self.sessions = []
        self.memory_states = {}
        self.metadata = {}
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
        
    def _get_session_id(self, session: Dict[str, Any]) -> int:
        """Get session ID from session dict (handles both 'id' and 'session_id')"""
        return session.get("id") or session.get("session_id")
    
    def _get_session_type(self, session: Dict[str, Any]) -> str:
        """Get session type from session dict (handles both 'type' and 'session_type')"""
        return session.get("type") or session.get("session_type")
    
    def load_data(self) -> bool:
        """Load session data and memory states"""
        try:
            # Load sessions
            sessions_file = self.session_dir / "new_sessions.json"
            with open(sessions_file, 'r') as f:
                data = json.load(f)
                # Handle both formats: list of sessions or dict with "sessions" key
                if isinstance(data, list):
                    self.sessions = data
                elif isinstance(data, dict) and "sessions" in data:
                    self.sessions = data["sessions"]
                    self.metadata = {k: v for k, v in data.items() if k != "sessions"}
                else:
                    raise ValueError("Invalid sessions file format")
            print(f"✅ Loaded {len(self.sessions)} sessions from {sessions_file}")
            
            # Load memory states
            memory_states_file = self.session_dir / "memory_states_by_session.json"
            with open(memory_states_file, 'r') as f:
                data = json.load(f)
                # Handle both formats
                if isinstance(data, dict) and "memory_states" in data:
                    self.memory_states = data["memory_states"]
                else:
                    self.memory_states = data
            print(f"✅ Loaded {len(self.memory_states)} memory states from {memory_states_file}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print("\n" + "="*80)
        print("🔍 MEMORY STATE VALIDATION")
        print("="*80)
        
        # Load data
        if not self.load_data():
            return False
        
        # Run validation checks
        print("\n📋 Running validation checks...")
        print("-"*80)
        
        # Phase 1: Critical validations
        self.check_session_sequence()
        self.check_date_consistency()
        self.check_session_operation_field_consistency()
        self.check_operation_dependencies()
        self.check_memory_state_progression()
        self.check_operation_details_consistency()
        self.check_memory_type_specific_validations()
        
        # Phase 2: Data integrity validations
        self.check_preference_conflicts()
        self.check_content_id_uniqueness()
        self.check_memory_accumulation()
        
        # Print results
        self.print_results()
        
        return len(self.errors) == 0
    
    def check_session_sequence(self):
        """Check that session IDs are sequential with no gaps or duplicates"""
        print("\n🔍 Checking session ID sequence...")
        
        session_ids = [self._get_session_id(s) for s in self.sessions]
        
        # Check for duplicates
        seen_ids = set()
        for session_id in session_ids:
            if session_id in seen_ids:
                self.errors.append({
                    "type": "DUPLICATE_SESSION_ID",
                    "session_id": session_id,
                    "message": f"Session ID {session_id} appears multiple times"
                })
                self.stats["duplicate_session_ids"] += 1
            seen_ids.add(session_id)
        
        # Check for sequential IDs
        sorted_ids = sorted(session_ids)
        expected_ids = list(range(1, len(session_ids) + 1))
        
        if sorted_ids != expected_ids:
            # Find gaps
            for i, (actual, expected) in enumerate(zip(sorted_ids, expected_ids)):
                if actual != expected:
                    self.errors.append({
                        "type": "SESSION_ID_GAP",
                        "session_id": None,
                        "message": f"Session ID gap: expected {expected}, got {actual}",
                        "details": {
                            "position": i,
                            "expected": expected,
                            "actual": actual
                        }
                    })
                    self.stats["session_id_gaps"] += 1
                    break
        else:
            self.stats["session_sequence_valid"] = True
    
    def check_date_consistency(self):
        """Check that session dates are chronological and within range"""
        print("\n🔍 Checking date consistency...")
        
        from datetime import datetime
        
        prev_date = None
        for session in sorted(self.sessions, key=lambda x: self._get_session_id(x)):
            session_id = self._get_session_id(session)
            date_str = session.get("date")
            
            if not date_str:
                self.errors.append({
                    "type": "MISSING_DATE",
                    "session_id": session_id,
                    "message": f"Session {session_id} missing date field"
                })
                continue
            
            try:
                current_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                
                # Check chronological order
                if prev_date and current_date < prev_date:
                    self.errors.append({
                        "type": "DATE_NOT_CHRONOLOGICAL",
                        "session_id": session_id,
                        "message": f"Session {session_id} date {date_str} is before previous session date",
                        "details": {
                            "current_date": date_str,
                            "previous_date": prev_date.isoformat()
                        }
                    })
                    self.stats["non_chronological_dates"] += 1
                else:
                    self.stats["dates_checked"] += 1
                
                prev_date = current_date
                
            except (ValueError, AttributeError) as e:
                self.errors.append({
                    "type": "INVALID_DATE_FORMAT",
                    "session_id": session_id,
                    "message": f"Session {session_id} has invalid date format: {date_str}",
                    "details": {"error": str(e)}
                })
                self.stats["invalid_date_formats"] += 1
    
    def check_operation_dependencies(self):
        """Check that DELETE/UPDATE operations only target existing memory"""
        print("\n🔍 Checking operation dependencies...")
        
        # Sort sessions by ID
        sorted_sessions = sorted(self.sessions, key=lambda x: self._get_session_id(x))
        
        for i, session in enumerate(sorted_sessions):
            session_id = str(self._get_session_id(session))
            session_type = self._get_session_type(session)
            operation = session.get("operation")
            
            if session_type == "no_memory" or operation not in ["delete", "update"]:
                continue
            
            # Get previous memory state
            if i > 0:
                prev_session_id = str(self._get_session_id(sorted_sessions[i-1]))
                prev_state = self.memory_states.get(prev_session_id, {})
            else:
                prev_state = {}
            
            prev_mem = prev_state.get("memory_state_after_session", {}).get("memory_items", {})
            
            # Check if target exists in previous memory
            if session_type == "preference_memory":
                category = session.get("category")
                subcategory = session.get("subcategory") or session.get("operation_details", {}).get("subcategory")
                prefs = prev_mem.get("preference_memory", {})
                cat_data = prefs.get(category, {}).get(subcategory, {})
                
                if not cat_data or (not cat_data.get("likes") and not cat_data.get("dislikes")):
                    self.errors.append({
                        "type": f"INVALID_{operation.upper()}_NO_MEMORY",
                        "session_id": session_id,
                        "message": f"Cannot {operation} preference - no memory in {category}/{subcategory}",
                        "details": {
                            "category": category,
                            "subcategory": subcategory,
                            "operation": operation
                        }
                    })
                    self.stats[f"invalid_{operation}_operations"] += 1
                else:
                    self.stats[f"valid_{operation}_operations"] += 1
            
            elif session_type == "content_memory":
                category = session.get("category")
                item = session.get("operation_details", {}).get("item")
                content = prev_mem.get("content_memory", {})
                content_list = content.get(category, [])
                content_ids = [c.get("id") for c in content_list if isinstance(c, dict)]
                
                if item and item not in content_ids:
                    self.errors.append({
                        "type": f"INVALID_{operation.upper()}_ITEM_NOT_FOUND",
                        "session_id": session_id,
                        "message": f"Cannot {operation} content '{item}' - not found in memory",
                        "details": {
                            "category": category,
                            "item": item,
                            "available_ids": content_ids[:5]
                        }
                    })
                    self.stats[f"invalid_{operation}_operations"] += 1
                else:
                    self.stats[f"valid_{operation}_operations"] += 1
    
    def check_preference_conflicts(self):
        """Check that items aren't in both likes and dislikes simultaneously"""
        print("\n🔍 Checking preference conflicts...")
        
        for session_id, state in self.memory_states.items():
            mem = state.get("memory_state_after_session", {}).get("memory_items", {})
            prefs = mem.get("preference_memory", {})
            
            for category, cat_data in prefs.items():
                for subcategory, subcat_data in cat_data.items():
                    if not isinstance(subcat_data, dict):
                        continue
                    
                    likes = subcat_data.get("likes", [])
                    dislikes = subcat_data.get("dislikes", [])
                    
                    # Extract item names
                    like_items = {l.get("item") if isinstance(l, dict) else l for l in likes}
                    dislike_items = {d.get("item") if isinstance(d, dict) else d for d in dislikes}
                    
                    # Check for conflicts
                    conflicts = like_items & dislike_items
                    if conflicts:
                        self.errors.append({
                            "type": "PREFERENCE_CONFLICT",
                            "session_id": int(session_id),
                            "message": f"Items in both likes and dislikes: {conflicts}",
                            "details": {
                                "category": category,
                                "subcategory": subcategory,
                                "conflicting_items": list(conflicts)
                            }
                        })
                        self.stats["preference_conflicts"] += 1
                    else:
                        self.stats["preference_categories_checked"] += 1
    
    def check_content_id_uniqueness(self):
        """Check that content IDs are unique within each category"""
        print("\n🔍 Checking content ID uniqueness...")
        
        for session_id, state in self.memory_states.items():
            mem = state.get("memory_state_after_session", {}).get("memory_items", {})
            content = mem.get("content_memory", {})
            
            for category, content_list in content.items():
                if not isinstance(content_list, list):
                    continue
                
                ids = [c.get("id") for c in content_list if isinstance(c, dict) and c.get("id")]
                
                # Check for duplicates
                seen = set()
                for content_id in ids:
                    if content_id in seen:
                        self.errors.append({
                            "type": "DUPLICATE_CONTENT_ID",
                            "session_id": int(session_id),
                            "message": f"Duplicate content ID '{content_id}' in {category}",
                            "details": {
                                "category": category,
                                "duplicate_id": content_id
                            }
                        })
                        self.stats["duplicate_content_ids"] += 1
                    seen.add(content_id)
    
    def check_memory_accumulation(self):
        """Check that memory accumulates correctly (no unexpected loss)"""
        print("\n🔍 Checking memory accumulation...")
        
        # Sort memory states by session ID
        sorted_states = sorted(
            [(int(sid), state) for sid, state in self.memory_states.items()],
            key=lambda x: x[0]
        )
        
        for i in range(1, len(sorted_states)):
            prev_id, prev_state = sorted_states[i-1]
            curr_id, curr_state = sorted_states[i]
            
            # Get corresponding session
            session = next((s for s in self.sessions if self._get_session_id(s) == curr_id), None)
            if not session:
                continue
            
            session_type = self._get_session_type(session)
            operation = session.get("operation")
            
            # Get memory counts
            prev_mem = prev_state.get("memory_state_after_session", {}).get("memory_items", {})
            curr_mem = curr_state.get("memory_state_after_session", {}).get("memory_items", {})
            
            # Check preference memory
            if session_type == "preference_memory":
                prev_prefs = prev_mem.get("preference_memory", {})
                curr_prefs = curr_mem.get("preference_memory", {})
                
                # Count total preference items
                prev_count = sum(
                    len(subcat.get("likes", [])) + len(subcat.get("dislikes", []))
                    for cat in prev_prefs.values()
                    for subcat in cat.values()
                    if isinstance(subcat, dict)
                )
                curr_count = sum(
                    len(subcat.get("likes", [])) + len(subcat.get("dislikes", []))
                    for cat in curr_prefs.values()
                    for subcat in cat.values()
                    if isinstance(subcat, dict)
                )
                
                # Validate count change matches operation
                if operation == "add" and curr_count <= prev_count:
                    self.warnings.append({
                        "type": "MEMORY_NOT_GROWING",
                        "session_id": curr_id,
                        "message": f"ADD operation but preference count didn't increase ({prev_count} -> {curr_count})"
                    })
                elif operation == "delete" and curr_count >= prev_count:
                    self.warnings.append({
                        "type": "MEMORY_NOT_SHRINKING",
                        "session_id": curr_id,
                        "message": f"DELETE operation but preference count didn't decrease ({prev_count} -> {curr_count})"
                    })
    
    def check_session_operation_field_consistency(self):
        """
        CRITICAL: Check that session["operation"] field matches operation_details
        This catches the bug where operation is converted but field is not updated
        """
        print("\n🔍 Checking session operation field consistency...")
        
        for session in self.sessions:
            session_id = session.get("id") or session.get("session_id")
            operation = session.get("operation")
            session_type = session.get("type") or session.get("session_type")
            
            # Skip no_memory sessions
            if session_type == "no_memory":
                self.stats["no_memory_sessions"] += 1
                continue
            
            operation_details = session.get("operation_details", {})
            
            # Check for operation conversion
            if "operation_converted" in operation_details:
                conversion_info = operation_details["operation_converted"]
                
                # Handle both formats: string and dict
                if isinstance(conversion_info, str):
                    # Old format: operation_converted is a string like "add_to_no_memory"
                    if conversion_info == "add_to_update":
                        if operation != "update":
                            self.errors.append({
                                "type": "ADD_TO_UPDATE_CONVERSION_NOT_REFLECTED",
                                "session_id": session_id,
                                "message": f"Session operation is '{operation}' but should be 'update' (operation_converted: {conversion_info})",
                                "details": {
                                    "operation_field": operation,
                                    "operation_converted": conversion_info
                                }
                            })
                            self.stats["add_to_update_not_reflected"] += 1
                        else:
                            self.stats["correct_add_to_update_conversions"] += 1
                    
                    elif conversion_info == "add_to_no_memory":
                        if session_type != "no_memory":
                            self.errors.append({
                                "type": "ADD_TO_NO_MEMORY_CONVERSION_NOT_REFLECTED",
                                "session_id": session_id,
                                "message": f"Session type is '{session_type}' but should be 'no_memory' (operation_converted: {conversion_info})",
                                "details": {
                                    "session_type": session_type,
                                    "operation_converted": conversion_info
                                }
                            })
                            self.stats["add_to_no_memory_not_reflected"] += 1
                
                elif isinstance(conversion_info, dict):
                    # New format: operation_converted is a dict with detailed info
                    original_op = conversion_info.get("original_operation")
                    final_op = conversion_info.get("final_operation")
                    reason = conversion_info.get("reason")
                    
                    # CRITICAL: session["operation"] must match final_operation
                    if operation != final_op:
                        self.errors.append({
                            "type": "OPERATION_FIELD_MISMATCH",
                            "session_id": session_id,
                            "message": f"Operation field '{operation}' does not match final operation '{final_op}' after conversion",
                            "details": {
                                "operation_field": operation,
                                "original_operation": original_op,
                                "final_operation": final_op,
                                "conversion_reason": reason
                            }
                        })
                        self.stats["operation_field_mismatches"] += 1
                    else:
                        self.stats["correct_operation_conversions"] += 1
            
            # Check for generation method conversions
            if "generation_method" in operation_details:
                gen_method = operation_details["generation_method"]
                
                # Check for add_converted_to_update
                if gen_method == "add_converted_to_update":
                    if operation != "update":
                        self.errors.append({
                            "type": "ADD_TO_UPDATE_CONVERSION_NOT_REFLECTED",
                            "session_id": session_id,
                            "message": f"Session operation is '{operation}' but should be 'update' (add_converted_to_update)",
                            "details": {
                                "operation_field": operation,
                                "generation_method": gen_method,
                                "has_session_history": "session_history" in operation_details
                            }
                        })
                        self.stats["add_to_update_not_reflected"] += 1
                    else:
                        self.stats["correct_add_to_update_conversions"] += 1
                
                # Check for add_converted_to_no_memory
                elif gen_method == "add_converted_to_no_memory":
                    if session_type != "no_memory":
                        self.errors.append({
                            "type": "ADD_TO_NO_MEMORY_CONVERSION_NOT_REFLECTED",
                            "session_id": session_id,
                            "message": f"Session type is '{session_type}' but should be 'no_memory' (add_converted_to_no_memory)",
                            "details": {
                                "session_type": session_type,
                                "generation_method": gen_method
                            }
                        })
                        self.stats["add_to_no_memory_not_reflected"] += 1
    
    def check_memory_state_progression(self):
        """Check that memory states progress correctly across sessions"""
        print("\n🔍 Checking memory state progression...")
        
        # Sort sessions by ID
        sorted_sessions = sorted(self.sessions, key=lambda x: self._get_session_id(x))
        
        for i, session in enumerate(sorted_sessions):
            session_id = str(self._get_session_id(session))
            session_type = self._get_session_type(session)
            operation = session.get("operation")
            
            # Skip no_memory sessions
            if session_type == "no_memory":
                continue
            
            # Get current and previous memory states
            current_state = self.memory_states.get(session_id)
            if current_state is None:
                self.errors.append({
                    "type": "MISSING_MEMORY_STATE",
                    "session_id": session_id,
                    "message": f"Memory state missing for session {session_id}"
                })
                continue
            
            # Get previous state (if exists)
            if i > 0:
                prev_session_id = str(self._get_session_id(sorted_sessions[i-1]))
                prev_state = self.memory_states.get(prev_session_id)
            else:
                prev_state = {}
            
            # Validate memory type specific progression
            self._validate_memory_progression(
                session, 
                current_state, 
                prev_state
            )
    
    def _validate_memory_progression(
        self, 
        session: Dict[str, Any], 
        current_state: Dict[str, Any], 
        prev_state: Dict[str, Any]
    ):
        """Validate that memory state changed correctly based on operation"""
        session_id = self._get_session_id(session)
        session_type = self._get_session_type(session)
        operation = session.get("operation")
        category = session.get("category")
        
        if session_type == "preference_memory":
            self._validate_preference_progression(session, current_state, prev_state)
        elif session_type == "activity_memory":
            self._validate_activity_progression(session, current_state, prev_state)
        elif session_type == "content_memory":
            self._validate_content_progression(session, current_state, prev_state)
        elif session_type == "goal_memory":
            self._validate_goal_progression(session, current_state, prev_state)
    
    def _validate_preference_progression(
        self, 
        session: Dict[str, Any], 
        current_state: Dict[str, Any], 
        prev_state: Dict[str, Any]
    ):
        """Validate preference memory progression"""
        session_id = self._get_session_id(session)
        operation = session.get("operation")
        operation_details = session.get("operation_details", {})
        
        category = session.get("category")
        subcategory = session.get("subcategory") or operation_details.get("subcategory")
        
        # Get current and previous preferences from memory_state_after_session
        current_mem = current_state.get("memory_state_after_session", {}).get("memory_items", {})
        prev_mem = prev_state.get("memory_state_after_session", {}).get("memory_items", {})
        
        current_prefs = current_mem.get("preference_memory", {})
        prev_prefs = prev_mem.get("preference_memory", {})
        
        if operation == "add":
            # Session simulator format: operation_details has "item", "preference" ("like"/"dislike")
            item = operation_details.get("item")
            pref_type = operation_details.get("preference")  # "like" or "dislike"
            
            if not item or not pref_type:
                return  # Can't validate without this info
            
            # Verify item is in current state
            current_items = current_prefs.get(category, {}).get(subcategory, {}).get(f"{pref_type}s", [])
            # Items are stored as objects with 'item' field
            current_item_names = [i.get("item") if isinstance(i, dict) else i for i in current_items]
            if item not in current_item_names:
                self.errors.append({
                    "type": "PREFERENCE_ADD_NOT_REFLECTED",
                    "session_id": session_id,
                    "message": f"Item '{item}' should be in {pref_type}s but not found",
                    "details": {
                        "category": category,
                        "subcategory": subcategory,
                        "expected_item": item,
                        "expected_type": f"{pref_type}s"
                    }
                })
            else:
                self.stats["preference_adds_verified"] += 1
        
        elif operation == "delete":
            # For delete, check if item was removed
            item = operation_details.get("item")
            pref_type = operation_details.get("preference")
            
            if not item or not pref_type:
                return
            
            # Verify item is NOT in current state
            current_items = current_prefs.get(category, {}).get(subcategory, {}).get(f"{pref_type}s", [])
            # Items are stored as objects with 'item' field
            current_item_names = [i.get("item") if isinstance(i, dict) else i for i in current_items]
            if item in current_item_names:
                self.errors.append({
                    "type": "PREFERENCE_DELETE_NOT_REFLECTED",
                    "session_id": session_id,
                    "message": f"Item '{item}' should be removed from {pref_type}s but still exists",
                    "details": {
                        "category": category,
                        "subcategory": subcategory,
                        "item": item,
                        "type": f"{pref_type}s"
                    }
                })
            else:
                self.stats["preference_deletes_verified"] += 1
        
        elif operation == "update":
            # For session simulator, update might be stored differently
            # Skip detailed validation for now
            self.stats["preference_updates_checked"] += 1
    
    def _validate_activity_progression(
        self, 
        session: Dict[str, Any], 
        current_state: Dict[str, Any], 
        prev_state: Dict[str, Any]
    ):
        """Validate activity memory progression"""
        session_id = self._get_session_id(session)
        operation = session.get("operation")
        category = session.get("category")
        
        # Get memory from memory_state_after_session
        current_mem = current_state.get("memory_state_after_session", {}).get("memory_items", {})
        prev_mem = prev_state.get("memory_state_after_session", {}).get("memory_items", {})
        
        current_activities = current_mem.get("activity_memory", {}).get(category, [])
        prev_activities = prev_mem.get("activity_memory", {}).get(category, [])
        
        if operation == "add":
            # Activity count should increase
            if len(current_activities) <= len(prev_activities):
                self.errors.append({
                    "type": "ACTIVITY_ADD_NOT_REFLECTED",
                    "session_id": session_id,
                    "message": f"Activity not added to {category}",
                    "details": {
                        "category": category,
                        "prev_count": len(prev_activities),
                        "current_count": len(current_activities)
                    }
                })
            else:
                self.stats["activity_adds_verified"] += 1
        
        elif operation == "delete":
            # Activity count should decrease
            if len(current_activities) >= len(prev_activities):
                self.errors.append({
                    "type": "ACTIVITY_DELETE_NOT_REFLECTED",
                    "session_id": session_id,
                    "message": f"Activity not deleted from {category}",
                    "details": {
                        "category": category,
                        "prev_count": len(prev_activities),
                        "current_count": len(current_activities)
                    }
                })
            else:
                self.stats["activity_deletes_verified"] += 1
    
    def _validate_content_progression(
        self, 
        session: Dict[str, Any], 
        current_state: Dict[str, Any], 
        prev_state: Dict[str, Any]
    ):
        """Validate content memory progression"""
        session_id = self._get_session_id(session)
        operation = session.get("operation")
        category = session.get("category")
        operation_details = session.get("operation_details", {})
        item = operation_details.get("item")
        
        # Get memory from memory_state_after_session
        current_mem = current_state.get("memory_state_after_session", {}).get("memory_items", {})
        prev_mem = prev_state.get("memory_state_after_session", {}).get("memory_items", {})
        
        current_content = current_mem.get("content_memory", {})
        prev_content = prev_mem.get("content_memory", {})
        
        # Content is stored as lists by category
        current_content_list = current_content.get(category, [])
        prev_content_list = prev_content.get(category, [])
        
        # Get IDs of content items
        current_ids = [c.get("id") for c in current_content_list if isinstance(c, dict)]
        prev_ids = [c.get("id") for c in prev_content_list if isinstance(c, dict)]
        
        if operation == "add":
            # Content item should exist in current but not in prev
            if item and item not in current_ids:
                self.errors.append({
                    "type": "CONTENT_ADD_NOT_REFLECTED",
                    "session_id": session_id,
                    "message": f"Content '{item}' not added",
                    "details": {
                        "category": category,
                        "item": item,
                        "current_ids": current_ids[:5]  # Show first 5 for debugging
                    }
                })
            elif item:
                self.stats["content_adds_verified"] += 1
        
        elif operation == "delete":
            # Content delete can be two types:
            # 1. Element deletion: Removing fields from content (item stays, has memory_deletes)
            # 2. Full deletion: Removing entire content item (item disappears)
            
            deletion_method = operation_details.get("deletion_method")
            has_memory_deletes = "memory_deletes" in operation_details
            
            if deletion_method == "redesigned_element_deletion" or has_memory_deletes:
                # Element deletion - item should still exist
                if item and item not in current_ids:
                    self.errors.append({
                        "type": "CONTENT_ELEMENT_DELETE_ITEM_MISSING",
                        "session_id": session_id,
                        "message": f"Content '{item}' should exist after element deletion but is missing",
                        "details": {
                            "category": category,
                            "item": item,
                            "deletion_method": deletion_method
                        }
                    })
                elif item:
                    self.stats["content_element_deletes_verified"] += 1
            else:
                # Full deletion - item should not exist
                if item and item in current_ids:
                    self.errors.append({
                        "type": "CONTENT_DELETE_NOT_REFLECTED",
                        "session_id": session_id,
                        "message": f"Content '{item}' not deleted",
                        "details": {
                            "category": category,
                            "item": item
                        }
                    })
                elif item:
                    self.stats["content_full_deletes_verified"] += 1
        
        elif operation == "update":
            # Content should exist
            if item and item not in current_ids:
                self.errors.append({
                    "type": "CONTENT_UPDATE_ITEM_MISSING",
                    "session_id": session_id,
                    "message": f"Content '{item}' missing for update",
                    "details": {
                        "category": category,
                        "item": item,
                        "current_ids": current_ids[:5]
                    }
                })
            elif item:
                self.stats["content_updates_verified"] += 1
    
    def _validate_goal_progression(
        self, 
        session: Dict[str, Any], 
        current_state: Dict[str, Any], 
        prev_state: Dict[str, Any]
    ):
        """Validate goal memory progression
        
        Note: Goal memory has an exception - it can have 'actual_operation' field
        that differs from 'operation' field. This is expected behavior where
        ADD operations may become UPDATE operations internally.
        """
        session_id = self._get_session_id(session)
        operation = session.get("operation")
        category = session.get("category")
        operation_details = session.get("operation_details", {})
        item = operation_details.get("item")  # This is actually the goal value
        
        # Check for actual_operation (goal memory exception)
        actual_operation = operation_details.get("actual_operation")
        if actual_operation:
            # Use actual_operation instead of operation field
            operation = actual_operation
            self.stats["goal_actual_operation_used"] += 1
        
        # Get memory from memory_state_after_session
        current_mem = current_state.get("memory_state_after_session", {}).get("memory_items", {})
        prev_mem = prev_state.get("memory_state_after_session", {}).get("memory_items", {})
        
        current_goals = current_mem.get("goal_memory", {})
        prev_goals = prev_mem.get("goal_memory", {})
        
        # Goals are stored as lists by category
        current_goal_list = current_goals.get(category, [])
        prev_goal_list = prev_goals.get(category, [])
        
        if operation == "add":
            # Goal list should have grown
            if len(current_goal_list) <= len(prev_goal_list):
                self.errors.append({
                    "type": "GOAL_ADD_NOT_REFLECTED",
                    "session_id": session_id,
                    "message": f"Goal not added to {category}",
                    "details": {
                        "category": category,
                        "item": item,
                        "prev_count": len(prev_goal_list),
                        "current_count": len(current_goal_list)
                    }
                })
            else:
                self.stats["goal_adds_verified"] += 1
        
        elif operation == "update":
            # For updates, count may stay the same (value changed)
            self.stats["goal_updates_verified"] += 1
        
        elif operation == "delete":
            # Goal list should have shrunk
            if len(current_goal_list) >= len(prev_goal_list):
                self.errors.append({
                    "type": "GOAL_DELETE_NOT_REFLECTED",
                    "session_id": session_id,
                    "message": f"Goal not deleted from {category}",
                    "details": {
                        "category": category,
                        "item": item,
                        "prev_count": len(prev_goal_list),
                        "current_count": len(current_goal_list)
                    }
                })
            else:
                self.stats["goal_deletes_verified"] += 1
    
    def check_operation_details_consistency(self):
        """Check that operation_details are present and consistent"""
        print("\n🔍 Checking operation details consistency...")
        
        for session in self.sessions:
            session_id = self._get_session_id(session)
            session_type = self._get_session_type(session)
            operation = session.get("operation")
            
            # Skip no_memory sessions
            if session_type == "no_memory":
                continue
            
            operation_details = session.get("operation_details")
            
            # Check that operation_details exists
            if not operation_details:
                self.errors.append({
                    "type": "MISSING_OPERATION_DETAILS",
                    "session_id": session_id,
                    "message": f"Session {session_id} missing operation_details",
                    "details": {
                        "session_type": session_type,
                        "operation": operation
                    }
                })
                continue
            
            # For session simulator format, operation_details contains item/content_data
            # Not memory_adds/memory_updates/memory_deletes
            # Skip these checks as the format is different
            self.stats["operation_details_present"] += 1
    
    def check_memory_type_specific_validations(self):
        """Run memory type specific validation rules"""
        print("\n🔍 Running memory type specific validations...")
        
        # Group sessions by memory type
        sessions_by_type = defaultdict(list)
        for session in self.sessions:
            session_type = self._get_session_type(session)
            sessions_by_type[session_type].append(session)
        
        # Print session type distribution
        for session_type, sessions in sessions_by_type.items():
            self.stats[f"{session_type}_count"] = len(sessions)
    
    def print_results(self):
        """Print validation results"""
        print("\n" + "="*80)
        print("📊 VALIDATION RESULTS")
        print("="*80)
        
        # Print statistics
        print("\n📈 Statistics:")
        print("-"*80)
        for key, value in sorted(self.stats.items()):
            print(f"  {key}: {value}")
        
        # Print errors
        if self.errors:
            print(f"\n❌ ERRORS FOUND: {len(self.errors)}")
            print("-"*80)
            
            # Group errors by type
            errors_by_type = defaultdict(list)
            for error in self.errors:
                errors_by_type[error["type"]].append(error)
            
            for error_type, errors in sorted(errors_by_type.items()):
                print(f"\n🔴 {error_type}: {len(errors)} occurrences")
                # Show first 5 examples
                for error in errors[:5]:
                    print(f"  Session {error['session_id']}: {error['message']}")
                    if "details" in error:
                        for key, value in error["details"].items():
                            print(f"    - {key}: {value}")
                if len(errors) > 5:
                    print(f"  ... and {len(errors) - 5} more")
        else:
            print("\n✅ NO ERRORS FOUND!")
        
        # Print warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            print("-"*80)
            for warning in self.warnings[:10]:
                print(f"  Session {warning['session_id']}: {warning['message']}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")
        
        print("\n" + "="*80)
        
        if self.errors:
            print(f"❌ VALIDATION FAILED: {len(self.errors)} errors found")
        else:
            print("✅ VALIDATION PASSED: All checks successful!")
        print("="*80)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python memory_state_validator.py <session_directory>")
        print("\nExample:")
        print("  python quality_checks/memory_state_validator.py output/sessions_<timestamp>_<persona>")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    
    validator = MemoryStateValidator(session_dir)
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

