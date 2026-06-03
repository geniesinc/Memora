#!/usr/bin/env python3


import json
import random
import copy
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
import logging


# =============================================================================
# METADATA DICTIONARY - Centralized JSON file loading
# =============================================================================

# Default memory configuration file
DEFAULT_MEMORY_CONFIG = "meta_data/memory_configs/memory_config_weekly.json"

def load_metadata_files(memory_config_file: str = DEFAULT_MEMORY_CONFIG) -> Dict[str, Any]:
    """
    Load all metadata files into a centralized dictionary for easy access.
    Uses a memory configuration file to determine which files to load dynamically.

    Args:
        memory_config_file: Path to memory configuration JSON file
                           (e.g., memory_config_weekly.json, memory_config_monthly.json,
                            memory_config_quarterly.json)
    
    Returns:
        Dict containing all loaded metadata with descriptive keys
    """
    metadata = {}
    
    print(f"📚 Loading metadata files from config: {memory_config_file}")
    
    # First, load memory configuration to get file paths
    try:
        with open(memory_config_file, 'r', encoding='utf-8') as f:
            metadata['memory_categories'] = json.load(f)
        print(f"✅ Loaded memory config: {memory_config_file}")
    except Exception as e:
        print(f"⚠️  Error loading {memory_config_file}: {e}")
        metadata['memory_categories'] = {}
        return metadata
    
    # Load persona traits (always needed for persona-based generation)
    try:
        with open("meta_data/persona_traits.json", 'r', encoding='utf-8') as f:
            metadata['persona_traits'] = json.load(f)
        print("✅ Loaded persona_traits.json")
    except Exception as e:
        print(f"⚠️  Error loading persona_traits.json: {e}")
        metadata['persona_traits'] = {}
    
    # Dynamically load memory type metadata files based on memory_categories.json
    memory_types = ['preference_memory', 'content_memory', 'activity_memory', 'goal_memory']
    
    for memory_type in memory_types:
        if memory_type in metadata['memory_categories']:
            metadata_file_path = metadata['memory_categories'][memory_type].get('metadata_file')
            
            if metadata_file_path:
                try:
                    with open(metadata_file_path, 'r', encoding='utf-8') as f:
                        metadata[memory_type] = json.load(f)
                    print(f"✅ Loaded {memory_type}: {metadata_file_path}")
                except Exception as e:
                    print(f"⚠️  Error loading {memory_type} from {metadata_file_path}: {e}")
                    metadata[memory_type] = {}
            else:
                # Skip warning for content_memory as it uses redesigned approach with separate files
                if memory_type != 'content_memory':
                    print(f"⚠️  No metadata_file specified for {memory_type}")
                metadata[memory_type] = {}
            
            # Special handling for content_memory - load additional redesigned metadata files
            if memory_type == 'content_memory':
                content_config = metadata['memory_categories'][memory_type]
                
                # Load refined content fields
                refined_fields_path = content_config.get('metadata_for_content_feilds')
                if refined_fields_path:
                    try:
                        with open(refined_fields_path, 'r', encoding='utf-8') as f:
                            metadata['refined_content_fields'] = json.load(f)
                        print(f"✅ Loaded refined content fields: {refined_fields_path}")
                    except Exception as e:
                        print(f"⚠️  Error loading refined content fields from {refined_fields_path}: {e}")
                        metadata['refined_content_fields'] = {}
                
                # Load content metadata pool (JSONL format)
                content_values_path = content_config.get('metadata_for_content_values')
                if content_values_path:
                    try:
                        metadata['content_metadata_pool'] = []
                        # Handle both .json and .jsonl extensions
                        if content_values_path.endswith('.jsonl'):
                            with open(content_values_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        try:
                                            metadata_entry = json.loads(line)
                                            metadata['content_metadata_pool'].append(metadata_entry)
                                        except json.JSONDecodeError as e:
                                            print(f"⚠️  Skipping invalid JSON line: {e}")
                        else:
                            # Regular JSON file
                            with open(content_values_path, 'r', encoding='utf-8') as f:
                                metadata['content_metadata_pool'] = json.load(f)
                        
                        print(f"✅ Loaded content metadata pool: {content_values_path} ({len(metadata['content_metadata_pool'])} entries)")
                    except Exception as e:
                        print(f"⚠️  Error loading content metadata pool from {content_values_path}: {e}")
                        metadata['content_metadata_pool'] = []
        else:
            print(f"⚠️  {memory_type} not found in memory_categories.json")
            metadata[memory_type] = {}
    
    loaded_count = len([k for k, v in metadata.items() if v])
    print(f"📖 Successfully loaded {loaded_count} metadata files dynamically")
    return metadata


def reload_metadata(memory_config_file: str = DEFAULT_MEMORY_CONFIG) -> Dict[str, Any]:
    """
    Reload metadata with a different configuration file.
    This allows switching between different memory configurations at runtime.
    
    Args:
        memory_config_file: Path to memory configuration JSON file
        
    Returns:
        Dict containing all loaded metadata with descriptive keys
    """
    global METADATA
    METADATA = load_metadata_files(memory_config_file)
    return METADATA


# Global metadata dictionary - will be loaded on first use (lazy loading)
METADATA = {}



class SessionType:
    NO_MEMORY = "no_memory"
    PREFERENCE_MEMORY = "preference_memory" 
    ACTIVITY_MEMORY = "activity_memory"
    CONTENT_MEMORY = "content_memory"
    GOAL_MEMORY = "goal_memory"


class MemoryOperation:
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"
    READ = "read"


class MemoryState:
    """Track memory state across sessions for validation and read operations."""
    
    def __init__(self, memory_categories_data=None, persona: Optional[str] = None):
        # Store the persona for persona-specific loading
        self.persona = persona
        
        # Track daily meal state for chronological ordering
        self.daily_meal_state = {}  # {date: {"breakfast_done": bool, "lunch_done": bool, etc.}}
        
        # Get persona traits from centralized metadata
        self.persona_traits = None
        if persona and METADATA['persona_traits'] and "persona_profiles" in METADATA['persona_traits']:
            self.persona_traits = METADATA['persona_traits']['persona_profiles'].get(persona, {})
            if self.persona_traits:
                print(f"✅ Loaded persona traits for: {self.persona}")
            else:
                print(f"⚠️  Persona '{self.persona}' not found in persona_traits.json")
        
        # Load redesigned metadata files for content generation (loaded via memory_categories.json)
        self.refined_content_fields = METADATA.get('refined_content_fields', {})
        self.content_metadata_pool = METADATA.get('content_metadata_pool', [])
        
        # Load preference metadata (persona-specific if available)
        self.preference_metadata_file = None
        self.preference_original_metadata = None
        self.preference_available_metadata = None  # Items that can still be chosen from metadata pool
        self._load_preference_metadata(memory_categories_data)
        
        # Load content memory metadata (persona-specific if available)
        self.content_metadata = None
        self._load_content_metadata(memory_categories_data)
        
        # Load activity memory metadata (persona-specific if available)
        self.activity_metadata = None
        self.activity_original_metadata = None
        self.activity_available_metadata = None  # Items that can still be chosen from metadata pool
        self._load_activity_metadata(memory_categories_data)
        
        # Load goal memory metadata
        self.goal_metadata = None
        self._load_goal_metadata(memory_categories_data)
        
        # Memory state - initialize dynamically from metadata
        self.memory_items = {
            "preference_memory": {},  # Will be initialized from metadata
            "activity_memory": {},    # Will be initialized from metadata
            "content_memory": {},     # Will be initialized from metadata
            "goal_memory": {}         # Will be initialized from metadata
        }
        
        # Content memory three-pool system for redesigned approach
        self.content_pools = {
            "available_pool": self.content_metadata_pool.copy() if self.content_metadata_pool else [],  # All available metadata samples
            "draft_pool": {},      # Content in draft state (can be updated/deleted) - organized by category
            "completed_pool": {}   # Content moved to completed state after 3 updates - organized by category
        }
        
        # Initialize draft and completed pools by category
        if self.persona and self.persona in self.refined_content_fields:
            content_categories = list(self.refined_content_fields[self.persona].keys())
            for category in content_categories:
                self.content_pools["draft_pool"][category] = []
                self.content_pools["completed_pool"][category] = []
        
        # Initialize all memory structures from metadata
        self._initialize_memory_from_metadata(memory_categories_data)
    
    def _generate_content_data_with_redesigned_fields(self, category, session_date=None):
        """Generate content data using the redesigned approach with field definitions and sample pool."""
        # Step 1: Get field definitions from refined_content_fields.json
        if not (self.persona and self.persona in self.refined_content_fields):
            return {}
        
        persona_categories = self.refined_content_fields[self.persona]
        if category not in persona_categories:
            return {}
        
        field_definitions = persona_categories[category]
        required_fields = list(field_definitions.get("required_fields", {}).keys())
        optional_fields = list(field_definitions.get("optional_fields", {}).keys())
        
        if not required_fields and not optional_fields:
            return {}
        
        # Step 2: Select sample metadata from the pool
        sample_metadata = self._select_sample_from_pool(self.persona, category)
        if not sample_metadata:
            print(f"⚠️  No sample metadata found for {self.persona}/{category}")
            return {}
        
        # Step 3: Generate content data based on field definitions and sample
        content_data = {}
        
        # Generate data for required fields
        for field in required_fields:
            if field in sample_metadata:
                value = sample_metadata[field]
                content_data[field] = self._process_field_value_redesigned(value, session_date)
            else:
                # Fallback if field not in sample
                content_data[field] = f"Generated {field.replace('_', ' ')}"
        
        # Select and generate optional fields (exactly half of available optional fields)
        if optional_fields:
            num_optional = len(optional_fields) // 2
            if num_optional > 0:
                selected_optional = random.sample(optional_fields, num_optional)
            else:
                selected_optional = []
            
            for field in selected_optional:
                if field in sample_metadata:
                    value = sample_metadata[field]
                    content_data[field] = self._process_field_value_redesigned(value, session_date)
                else:
                    # Fallback if field not in sample
                    content_data[field] = f"Generated {field.replace('_', ' ')}"
        
        # Add metadata tracking information
        content_data["_metadata_source"] = {
            "sample_id": sample_metadata.get("_generation_info", {}).get("sample_id", "unknown"),
            "persona": self.persona,
            "category": category,
            "generation_method": "redesigned_pool_selection"
        }
        
        return content_data
    
    def _select_sample_from_pool(self, persona, content_category):
        """Select a random sample from the filtered metadata pool."""
        filtered_entries = []
        
        for entry in self.content_metadata_pool:
            # The JSONL structure is: {"persona": "software_engineer", "data": [...]}
            entry_persona = entry.get("persona")
            
            if entry_persona == persona:
                # Get the data array for this persona
                data_entries = entry.get("data", [])
                
                # Filter data entries by category
                for data_entry in data_entries:
                    # Check if data entry has generation info with category
                    gen_info = data_entry.get("_generation_info", {})
                    entry_category = gen_info.get("category")
                    
                    if entry_category == content_category:
                        filtered_entries.append(data_entry)
        
        if not filtered_entries:
            return None
        
        selected_entry = random.choice(filtered_entries)
        return selected_entry
    
    def _process_field_value_redesigned(self, value, session_date=None):
        """Process field values for redesigned approach, handling date placeholders and other transformations."""
        if not isinstance(value, str) or not session_date:
            return value
        
        # Handle session_date placeholder
        if "session_date" in value:
            return value.replace("session_date", session_date)
        
        # Handle relative date placeholders like "+7 days", "+1 week", "+1 month"
        if value.startswith("+"):
            try:
                session_dt = datetime.strptime(session_date, "%Y-%m-%d")
                
                if "days" in value:
                    days = int(value.split()[0][1:])  # Extract number after "+"
                    new_date = session_dt + timedelta(days=days)
                    return new_date.strftime("%Y-%m-%d")
                elif "week" in value:
                    weeks = int(value.split()[0][1:])
                    new_date = session_dt + timedelta(weeks=weeks)
                    return new_date.strftime("%Y-%m-%d")
                elif "month" in value:
                    months = int(value.split()[0][1:])
                    # Approximate month addition (30 days per month)
                    new_date = session_dt + timedelta(days=months * 30)
                    return new_date.strftime("%Y-%m-%d")
            except (ValueError, IndexError):
                pass
        
        return value
    
    def _generate_partial_content_data(self, category, session_date=None):
        """Generate partial content data for initial ADD operation (some list elements held back for updates).
        
        Returns:
            tuple: (clean_content_data, metadata_source, remaining_data) or (None, None, None) if failed
        """
        # Step 1: Get field definitions from refined_content_fields.json
        if not (self.persona and self.persona in self.refined_content_fields):
            return None, None, None
        
        persona_categories = self.refined_content_fields[self.persona]
        if category not in persona_categories:
            return None, None, None
        
        field_definitions = persona_categories[category]
        required_fields = list(field_definitions.get("required_fields", {}).keys())
        optional_fields = list(field_definitions.get("optional_fields", {}).keys())
        
        if not required_fields and not optional_fields:
            return None, None, None
        
        # Step 2: Select sample metadata from the available pool
        sample_metadata = self._select_sample_from_available_pool(self.persona, category)
        if not sample_metadata:
            print(f"⚠️  No sample metadata found in available pool for {self.persona}/{category}")
            return None, None, None
        
        # Step 3: Generate partial content data (hold back some list elements for updates)
        content_data = {}
        remaining_data = {}  # Data to be added in future updates
        
        # Generate data for required fields (with partial list sharing)
        for field in required_fields:
            field_type = field_definitions.get("required_fields", {}).get(field, "string")
            
            if field in sample_metadata:
                value = sample_metadata[field]
                
                # Special handling for range fields (like budget)
                if field_type == "range" and isinstance(value, list) and len(value) == 2:
                    # Generate a random value within the range and round it nicely
                    min_val, max_val = value[0], value[1]
                    random_val = random.uniform(min_val, max_val)
                    
                    # Round to nice values based on magnitude
                    if random_val >= 1000000:  # Millions
                        rounded_val = round(random_val / 50000) * 50000  # Round to nearest 50K
                    elif random_val >= 100000:  # Hundreds of thousands
                        rounded_val = round(random_val / 25000) * 25000  # Round to nearest 25K
                    else:  # Under 100K
                        rounded_val = round(random_val / 5000) * 5000   # Round to nearest 5K
                    
                    content_data[field] = int(rounded_val)
                    
                    # For ranges, preserve the ORIGINAL RANGE for unlimited future budget revisions
                    # This ensures the user can always generate new budget alternatives within the full range
                    remaining_data[field] = {
                        "type": "range",
                        "min_val": min_val,
                        "max_val": max_val,
                        "original_range": [min_val, max_val],
                        "current_value": int(rounded_val)  # Track current value for reference
                    }
                
                else:
                    # Normal processing for non-range fields
                    processed_value = self._process_field_value_redesigned(value, session_date)
                    
                    # If field is a list, share only part of it initially
                    if isinstance(processed_value, list) and len(processed_value) > 1:
                        # Share 50-70% of list elements initially, keep rest for updates
                        share_ratio = random.uniform(0.5, 0.7)
                        num_to_share = max(1, int(len(processed_value) * share_ratio))
                        
                        shared_elements = random.sample(processed_value, num_to_share)
                        remaining_elements = [item for item in processed_value if item not in shared_elements]
                        
                        content_data[field] = shared_elements
                        if remaining_elements:
                            remaining_data[field] = remaining_elements
                    else:
                        content_data[field] = processed_value
            else:
                # Fallback if field not in sample
                content_data[field] = f"Generated {field.replace('_', ' ')}"
        
        # Select and generate optional fields (exactly half)
        if optional_fields:
            num_optional = len(optional_fields) // 2
            if num_optional > 0:
                selected_optional = random.sample(optional_fields, num_optional)
                
                for field in selected_optional:
                    if field in sample_metadata:
                        value = sample_metadata[field]
                        processed_value = self._process_field_value_redesigned(value, session_date)
                        
                        # If field is a list, share only part of it initially
                        if isinstance(processed_value, list) and len(processed_value) > 1:
                            share_ratio = random.uniform(0.5, 0.7)
                            num_to_share = max(1, int(len(processed_value) * share_ratio))
                            
                            shared_elements = random.sample(processed_value, num_to_share)
                            remaining_elements = [item for item in processed_value if item not in shared_elements]
                            
                            content_data[field] = shared_elements
                            if remaining_elements:
                                remaining_data[field] = remaining_elements
                        else:
                            content_data[field] = processed_value
                    else:
                        # Fallback if field not in sample
                        content_data[field] = f"Generated {field.replace('_', ' ')}"
        
        # Create metadata tracking information (separate from content_data)
        metadata_source = {
            "sample_id": sample_metadata.get("_generation_info", {}).get("sample_id", "unknown"),
            "persona": self.persona,
            "category": category,
            "generation_method": "redesigned_partial_content",
            "original_sample": sample_metadata  # Keep reference to original sample
        }
        
        # Return clean content_data, metadata_source, and remaining_data separately
        return content_data, metadata_source, remaining_data
    
    def _select_sample_from_available_pool(self, persona, content_category):
        """Select a random sample from the available pool and remove it."""
        filtered_entries = []
        entry_indices = []
        
        for i, entry in enumerate(self.content_pools["available_pool"]):
            # The JSONL structure is: {"persona": "software_engineer", "data": [...]}
            entry_persona = entry.get("persona")
            
            if entry_persona == persona:
                # Get the data array for this persona
                data_entries = entry.get("data", [])
                
                # Filter data entries by category
                for j, data_entry in enumerate(data_entries):
                    # Check if data entry has generation info with category
                    gen_info = data_entry.get("_generation_info", {})
                    entry_category = gen_info.get("category")
                    
                    if entry_category == content_category:
                        filtered_entries.append(data_entry)
                        entry_indices.append((i, j))  # Store indices for removal
        
        if not filtered_entries:
            return None
        
        # Select random entry
        selected_index = random.randint(0, len(filtered_entries) - 1)
        selected_entry = filtered_entries[selected_index]
        pool_index, data_index = entry_indices[selected_index]
        
        # CRITICAL FIX: Create a deep copy to prevent reference sharing
        # This prevents modifications to the returned sample from affecting the original pool
        import copy
        selected_entry_copy = copy.deepcopy(selected_entry)
        
        # Remove the selected entry from available pool
        self.content_pools["available_pool"][pool_index]["data"].pop(data_index)
        
        # If the persona entry has no more data, remove the entire entry
        if not self.content_pools["available_pool"][pool_index]["data"]:
            self.content_pools["available_pool"].pop(pool_index)
        
        return selected_entry_copy
    
    def _update_content_with_remaining_data(self, content_item, session_date=None):
        """Update content item by moving 1-3 elements from remaining_data to actual fields."""
        import random  # Import at the top to avoid UnboundLocalError
        
        content_data = content_item["content_data"]
        remaining_data = content_item.get("remaining_data", {})
        
        # Track what goes into memory (for operation_details)
        memory_updates = []
        
        
        # VALIDATION: Check if there's any data to update
        if not remaining_data:
            memory_updates.append({"error": "No _remaining_data available for update"})
            return {
                "memory_updates": memory_updates,
                "remaining_data_fields": []
            }
        
        # Strategy: Move 1-3 elements from _remaining_data to actual fields
        if remaining_data:
            # Calculate total available updates (properly handle different field types)
            total_available_updates = 0
            for field, elements in remaining_data.items():
                if field == "budget" and isinstance(elements, dict) and elements.get("type") == "range":
                    # Budget/range fields always count as 1 update (unlimited revisions within range)
                    total_available_updates += 1
                elif isinstance(elements, list) and elements:
                    # List fields: each element counts as 1 update
                    total_available_updates += len(elements)
                # Ignore other field types (empty lists, non-list/non-range fields)
            
            
            if total_available_updates > 0:
                # Get category-specific operation intensity from metadata
                category = content_item.get("category", "unknown")
                min_updates, max_updates = self._get_operation_intensity(category, "update")
                
                # Choose updates based on metadata settings, but not more than available
                num_updates = min(random.randint(min_updates, max_updates), total_available_updates)
                
                # Perform multiple updates in this session
                for update_num in range(num_updates):
                    # Find fields with remaining data (refresh each iteration)
                    available_fields = [(field, elements) for field, elements in remaining_data.items() 
                                      if elements and field in content_data]
                    
                    if available_fields:
                        # Pick a random field that has remaining data
                        source_field, remaining_elements = random.choice(available_fields)
                        
                        # Check if this is a range field (budget) - handle differently
                        if (source_field == "budget" and isinstance(remaining_elements, dict) and 
                            remaining_elements.get("type") == "range" and isinstance(content_data[source_field], int)):
                            
                            # Generate a NEW budget value within the original range
                            min_val = remaining_elements["min_val"]
                            max_val = remaining_elements["max_val"]
                            current_budget = content_data[source_field]
                            
                            # Generate a different budget value (avoid current value)
                            attempts = 0
                            while attempts < 10:  # Prevent infinite loop
                                new_budget = random.uniform(min_val, max_val)
                                
                                # Apply rounding logic (same as initial generation)
                                if new_budget >= 1000000:
                                    rounded_budget = round(new_budget / 50000) * 50000
                                elif new_budget >= 100000:
                                    rounded_budget = round(new_budget / 25000) * 25000
                                else:
                                    rounded_budget = round(new_budget / 5000) * 5000
                                
                                rounded_budget = int(rounded_budget)
                                
                                # Ensure it's different from current budget (at least 5% difference)
                                if abs(rounded_budget - current_budget) >= (current_budget * 0.05):
                                    break
                                attempts += 1
                            else:
                                # Fallback: if we can't find a significantly different value, adjust by 10%
                                rounded_budget = int(current_budget * (1.1 if random.random() > 0.5 else 0.9))
                                # Apply rounding to fallback too
                                if rounded_budget >= 100000:
                                    rounded_budget = round(rounded_budget / 25000) * 25000
                                else:
                                    rounded_budget = round(rounded_budget / 5000) * 5000
                                rounded_budget = int(rounded_budget)
                            
                            # Update the budget
                            old_budget = content_data[source_field]
                            content_data[source_field] = rounded_budget
                            
                            # Update the current_value in remaining_data for future reference
                            remaining_elements["current_value"] = rounded_budget
                            
                            # Track what goes into memory (clean format for operation_details)
                            memory_updates.append({
                                "field": source_field,
                                "updated_from": old_budget,
                                "updated_to": rounded_budget,
                                "action": "budget_revised"
                            })
                            
                        else:
                            # Normal list element handling
                            moved_element = random.choice(remaining_elements)
                            remaining_elements.remove(moved_element)
                            
                            # Add to the actual field (this is what goes to memory)
                            if isinstance(content_data[source_field], list):
                                content_data[source_field].append(moved_element)
                            else:
                                # If not a list, make it a list and add
                                if not isinstance(content_data[source_field], list):
                                    content_data[source_field] = [content_data[source_field]]
                                content_data[source_field].append(moved_element)
                            
                            # Track what goes into memory (clean format for operation_details)
                            memory_updates.append({
                                "field": source_field,
                                "added_item": moved_element,
                                "action": "added"
                            })
                            
                        
                        # Clean up empty remaining data field
                        if not remaining_elements:
                            del remaining_data[source_field]
                    else:
                        # No more fields with remaining data
                        break
                
                # Update remaining data in the content_item
                content_item["remaining_data"] = remaining_data
        
        # If no remaining data available, this update fails
        if not memory_updates:
            memory_updates.append({
                "error": "No remaining data available for update"
            })
        
        # Update metadata
        if "metadata_source" in content_item:
            content_item["metadata_source"]["last_updated"] = session_date or "2024-01-01"
        content_item["updated_at"] = session_date or "2024-01-01"
        
        return {
            "memory_updates": memory_updates,  # Clean data for operation_details (what goes to memory)
            "remaining_data_fields": list(remaining_data.keys()) if remaining_data else []
        }
    
    def _move_to_completed_pool(self, content_item, category):
        """Move content item from draft pool to completed pool."""
        # Remove from draft pool
        if category in self.content_pools["draft_pool"]:
            if content_item in self.content_pools["draft_pool"][category]:
                self.content_pools["draft_pool"][category].remove(content_item)
        
        # Update status and add to completed pool
        content_item["pool_status"] = "completed"
        content_item["completed_at"] = content_item.get("updated_at", content_item["created_at"])
        
        if category not in self.content_pools["completed_pool"]:
            self.content_pools["completed_pool"][category] = []
        
        self.content_pools["completed_pool"][category].append(content_item)
        
        # Track completion in session history
        session_count = len(content_item.get("session_history", []))
        threshold = content_item.get("completion_threshold", 3)
        print(f"🎯 Moved content item {content_item['id']} to completed pool after {session_count} sessions (threshold: {threshold})")
    
    def _get_operation_intensity(self, category, operation):
        """Get min/max operations per session from metadata, with fallback to defaults."""
        # Check if metadata has operation intensity settings
        memory_categories = METADATA.get('memory_categories', {})
        content_memory = memory_categories.get('content_memory', {})
        options = content_memory.get('options', {})
        category_config = options.get(category, {})
        
        operation_intensity = category_config.get('operation_intensity', {})
        operation_config = operation_intensity.get(operation, {})
        
        # Get min/max with fallback to defaults
        min_ops = operation_config.get('min_per_session', 1)
        max_ops = operation_config.get('max_per_session', 3)  # Default fallback
        
        return min_ops, max_ops
    
    def _get_completion_threshold(self, category):
        """Get min/max completion threshold from metadata, with fallback to defaults."""
        memory_categories = METADATA.get('memory_categories', {})
        content_memory = memory_categories.get('content_memory', {})
        options = content_memory.get('options', {})
        category_config = options.get(category, {})
        
        completion_config = category_config.get('completion_threshold', {})
        
        min_sessions = completion_config.get('min_sessions', 3)  # Default fallback
        max_sessions = completion_config.get('max_sessions', 5)  # Default fallback
        
        return min_sessions, max_sessions
    
    def _restructure_content_data(self, content_data):
        """Restructure content_data to move metadata outside and keep only actual content fields.
        
        Returns:
            tuple: (clean_content_data, metadata_source, remaining_data)
        """
        if not isinstance(content_data, dict):
            return content_data, None, None
        
        # Create a copy to avoid modifying the original
        content_data_copy = content_data.copy()
        
        # Extract metadata and remaining data
        metadata_source = content_data_copy.pop('_metadata_source', None)
        remaining_data = content_data_copy.pop('_remaining_data', None)
        
        # content_data_copy now contains only the actual content fields
        clean_content_data = content_data_copy
        
        return clean_content_data, metadata_source, remaining_data

    def _delete_content_element_to_remaining_data(self, content_item, session_date=None):
        """Delete content item by moving elements from actual fields back to remaining data (metadata-driven count)."""
        import random  # Import at the top to avoid UnboundLocalError
        
        content_data = content_item["content_data"]
        remaining_data = content_item.get("remaining_data", {})
        
        # Track what goes into memory (for operation_details)
        memory_deletes = []
        
        
        # CRITICAL FIX: Create deep copies to prevent reference corruption
        import copy
        
        # CRITICAL FIX: Ensure we're working with completely independent data structures
        # The issue is that content_data might have shared references with the available pool
        content_data_snapshot = copy.deepcopy(content_data)
        
        # Strategy: Move 1-3 elements from actual fields back to remaining data (similar to UPDATE logic)
        # CRITICAL: Only consider elements that are ACTUALLY in active memory fields AT THE TIME OF DELETE
        
        # Calculate total available deletions
        total_available_deletions = 0
        list_fields_info = []
        range_fields_info = []
        
        for field, value in content_data_snapshot.items():
            if not field.startswith("_"):  # Skip metadata fields
                # Handle list fields (multiple elements can be deleted)
                if isinstance(value, list) and len(value) > 1:
                    deletable_count = len(value) - 1  # Must leave at least 1 element
                    total_available_deletions += deletable_count
                    list_fields_info.append((field, value, deletable_count))
                # Handle budget/range fields (can revert to different value within range)
                elif (field == "budget" and isinstance(value, int) and field in remaining_data and
                      isinstance(remaining_data[field], dict) and remaining_data[field].get("type") == "range"):
                    total_available_deletions += 1  # Budget can always be revised once per session
                    range_fields_info.append((field, value))
        
        if total_available_deletions == 0:
            memory_deletes.append({"error": "No elements available for deletion"})
            return {
                "memory_deletes": memory_deletes,
                "remaining_data_fields": list(remaining_data.keys())
            }
        
        # Get category-specific operation intensity from metadata
        category = content_item.get("category", "unknown")
        min_deletions, max_deletions = self._get_operation_intensity(category, "delete")
        
        # Determine number of deletions to perform (metadata-driven, but not more than available)
        num_deletions = min(random.randint(min_deletions, max_deletions), total_available_deletions)
        
        # Perform multiple deletions
        for deletion_num in range(num_deletions):
            # Rebuild available fields for each deletion (as fields may become unavailable)
            available_fields = []
            
            for field, value, deletable_count in list_fields_info:
                current_list = content_data[field]  # Get current state
                if len(current_list) > 1:  # Still has elements to delete
                    available_fields.append((field, current_list, "list"))
            
            for field, value in range_fields_info:
                if field == "budget":  # Budget can always be revised
                    available_fields.append((field, content_data[field], "range"))
            
            if not available_fields:
                break  # No more fields available for deletion
        
            # Pick a random field that can be "deleted" (reverted/removed)
            target_field, current_value, field_type = random.choice(available_fields)
            
            if field_type == "list":
                # Handle list field deletion (move one element back to remaining)
                copied_list = current_value
                # Get the ORIGINAL list from content_data to modify
                original_list = content_data[target_field]
                
                # VALIDATION: Ensure we're only deleting from active memory
                if len(original_list) <= 1:
                    memory_deletes.append({
                        "error": f"Cannot delete from {target_field} - only {len(original_list)} element(s) remaining"
                    })
                else:
                    # Move exactly ONE element from ACTIVE field to remaining data
                    # CRITICAL: Select from the copied list but verify in original
                    moved_element = random.choice(copied_list)
                    
                    # CRITICAL VALIDATION: Ensure element is actually in the ORIGINAL active list
                    if moved_element not in original_list:
                        memory_deletes.append({
                            "error": f"Element '{moved_element}' not found in active {target_field} field. Active: {original_list}"
                        })
                    else:
                        # Remove from ORIGINAL active field
                        original_list.remove(moved_element)
                        
                        # Add to remaining data
                        if target_field not in remaining_data:
                            remaining_data[target_field] = []
                        remaining_data[target_field].append(moved_element)
                        
                        # Update remaining data in content_item
                        content_item["remaining_data"] = remaining_data
                        
                        # Track what goes into memory (clean format for operation_details)
                        memory_deletes.append({
                            "field": target_field,
                            "removed_item": moved_element,
                            "action": "removed_from_memory"
                        })
                        
                        
            
            elif field_type == "range":
                # Handle budget/range field deletion (generate new value within range)
                current_budget = current_value
                range_info = remaining_data[target_field]
                
                if range_info and range_info.get("type") == "range":
                    # Generate a NEW budget value within the original range (similar to UPDATE logic)
                    min_val = range_info["min_val"]
                    max_val = range_info["max_val"]
                    
                    # Generate a different budget value (avoid current value)
                    attempts = 0
                    while attempts < 10:  # Prevent infinite loop
                        new_budget = random.uniform(min_val, max_val)
                        
                        # Apply rounding logic (same as initial generation)
                        if new_budget >= 1000000:
                            rounded_budget = round(new_budget / 50000) * 50000
                        elif new_budget >= 100000:
                            rounded_budget = round(new_budget / 25000) * 25000
                        else:
                            rounded_budget = round(new_budget / 5000) * 5000
                        
                        rounded_budget = int(rounded_budget)
                        
                        # Ensure it's different from current budget (at least 5% difference)
                        if abs(rounded_budget - current_budget) >= (current_budget * 0.05):
                            break
                        attempts += 1
                    else:
                        # Fallback: if we can't find a significantly different value, adjust by 10%
                        rounded_budget = int(current_budget * (0.9 if random.random() > 0.5 else 1.1))
                        # Apply rounding to fallback too
                        if rounded_budget >= 100000:
                            rounded_budget = round(rounded_budget / 25000) * 25000
                        else:
                            rounded_budget = round(rounded_budget / 5000) * 5000
                        rounded_budget = int(rounded_budget)
                    
                    # Update the active budget
                    content_data[target_field] = rounded_budget
                    
                    # Update the current_value in remaining_data for future reference
                    range_info["current_value"] = rounded_budget
                    content_item["remaining_data"] = remaining_data
                    
                    # Track what goes into memory (clean format for operation_details)
                    memory_deletes.append({
                        "field": target_field,
                        "reverted_from": current_budget,
                        "reverted_to": rounded_budget,
                        "action": "budget_reverted"
                    })
                    
                    # Budget can only be revised once per session, so remove from range_fields_info
                    range_fields_info = [(f, v) for f, v in range_fields_info if f != target_field]
        
        # If no list fields available for deletion, this delete fails
        if not memory_deletes:
            memory_deletes.append({
                "error": "No list fields with multiple elements available for deletion"
            })
        
        # Update metadata
        if "metadata_source" in content_item:
            content_item["metadata_source"]["last_updated"] = session_date or "2024-01-01"
        content_item["updated_at"] = session_date or "2024-01-01"
        
        return {
            "memory_deletes": memory_deletes,  # Clean data for operation_details (what was removed from memory)
            "remaining_data_fields": list(remaining_data.keys()) if remaining_data else []
        }

    def _initialize_memory_from_metadata(self, memory_categories_data):
        """Initialize all memory structures dynamically from metadata files."""
        
        # Initialize preference memory from metadata
        self._initialize_preference_memory()
        
        # Initialize activity memory from activity_metadata.json
        if METADATA['activity_memory'] and "activity_personas" in METADATA['activity_memory']:
            activity_data = METADATA['activity_memory']['activity_personas']
            if activity_data:
                # Get categories from any persona (they should all have the same structure)
                sample_persona = next(iter(activity_data.values()))
                if "activity_categories" in sample_persona:
                    activity_categories = list(sample_persona["activity_categories"].keys())
                    for category in activity_categories:
                        self.memory_items["activity_memory"][category] = []
                    print(f"✅ Initialized activity memory with categories: {activity_categories}")
        
        # Initialize content memory using redesigned approach
        if hasattr(self, 'refined_content_fields') and self.refined_content_fields:
            # Use refined fields to initialize content memory categories
            # The structure is: persona -> category -> fields, so get categories for current persona
            if self.persona and self.persona in self.refined_content_fields:
                content_categories = list(self.refined_content_fields[self.persona].keys())
                for category in content_categories:
                    self.memory_items["content_memory"][category] = []
                print(f"✅ Initialized content memory with redesigned categories: {content_categories}")
            else:
                print(f"⚠️  No refined fields found for persona: {self.persona}")
        elif METADATA['content_memory'] and "content_personas" in METADATA['content_memory']:
            # Fallback to legacy initialization
            content_data = METADATA['content_memory']['content_personas']
            if content_data:
                # Get categories from any persona (they should all have the same structure)
                sample_persona = next(iter(content_data.values()))
                if "content_categories" in sample_persona:
                    content_categories = sample_persona["content_categories"].keys()
                    for category in content_categories:
                        self.memory_items["content_memory"][category] = []
                    print(f"✅ Initialized content memory with legacy categories: {list(content_categories)}")
        
        # Initialize goal memory from goal_metadata.json
        if METADATA['goal_memory']:
            goal_categories = []
            for goal_metadata in METADATA['goal_memory']:
                if "category" in goal_metadata:
                    category = goal_metadata["category"]
                    self.memory_items["goal_memory"][category] = []
                    goal_categories.append(category)
            print(f"✅ Initialized goal memory with categories: {goal_categories}")

    def _load_content_metadata(self, memory_categories_data):
        """Load content metadata using redesigned approach with refined fields and sample pool."""
        try:
            self.content_metadata = {}
            
            # Use redesigned approach: get field definitions from refined_content_fields.json
            if self.persona and self.persona in self.refined_content_fields:
                persona_categories = self.refined_content_fields[self.persona]
                
                for category, field_definitions in persona_categories.items():
                    # Extract required and optional fields
                    required_fields = list(field_definitions.get("required_fields", {}).keys())
                    optional_fields = list(field_definitions.get("optional_fields", {}).keys())
                    
                    # Create metadata structure for this category
                    self.content_metadata[category] = {
                        "category": category,
                        "required_fields": required_fields,
                        "optional_fields": optional_fields,
                        "sample_data": {},  # Will be populated from pool when needed
                        "use_redesigned_approach": True
                    }
                
                print(f"✅ Loaded redesigned content metadata for {self.persona} with {len(self.content_metadata)} categories")
            else:
                print(f"⚠️  No redesigned content metadata found for persona: {self.persona}")
                self.content_metadata = {}
                    
        except Exception as e:
            print(f"❌ Error loading content metadata: {e}")
            self.content_metadata = {}
    
    def _generate_field_value(self, field_name, field_data):
        """Generate field value handling both regular lists and special range structures like budget."""
        if field_name == "budget" and field_data:
            # Budget is now a list of ranges - select a range and generate a rounded value
            if all(isinstance(item, list) and len(item) == 2 for item in field_data):
                # It's a list of ranges - select one range and generate a value from it
                selected_range = random.choice(field_data)
                min_val, max_val = selected_range
                raw_value = random.uniform(min_val, max_val)
                
                # Round to natural budget amounts
                return self._round_budget_value(raw_value)
            else:
                # Fallback: treat as regular list (for backward compatibility)
                return random.choice(field_data)
        else:
            # Regular field - just pick a random value from the list
            return random.choice(field_data)
    
    def _round_budget_value(self, raw_value):
        """Round budget values to natural, realistic amounts and format as currency."""
        if raw_value < 50000:
            # Small budgets: round to nearest $5,000 (e.g., $25K, $30K, $35K)
            rounded_value = round(raw_value / 5000) * 5000
        elif raw_value < 100000:
            # Medium budgets: round to nearest $10,000 (e.g., $60K, $70K, $80K)
            rounded_value = round(raw_value / 10000) * 10000
        elif raw_value < 500000:
            # Large budgets: round to nearest $25,000 (e.g., $150K, $175K, $200K)
            rounded_value = round(raw_value / 25000) * 25000
        else:
            # Very large budgets: round to nearest $100,000 (e.g., $700K, $800K, $900K)
            rounded_value = round(raw_value / 100000) * 100000
        
        # Format as currency string
        if rounded_value >= 1000000:
            # Format as millions (e.g., $1.2M, $2.5M)
            return f"${rounded_value / 1000000:.1f}M".rstrip('0').rstrip('.')
        elif rounded_value >= 1000:
            # Format as thousands (e.g., $25K, $150K)
            return f"${rounded_value // 1000}K"
        else:
            # Format as dollars (e.g., $500, $750)
            return f"${int(rounded_value)}"
    
    def _generate_coordinated_time_fields(self, content_data, selected_time_fields, sample_data, session_date):
        """Generate coordinated time-related fields to maintain logical relationships."""
        
        # Define time field pairs and their relationships
        time_field_pairs = {
            ('follow_up_date', 'deadline'): 'follow_up_before_deadline'
        }
        
        # Check if we have both fields in a coordinated pair
        if 'follow_up_date' in selected_time_fields and 'deadline' in selected_time_fields:
            # Generate coordinated follow_up_date and deadline
            self._generate_follow_up_deadline_pair(content_data, sample_data, session_date)
            
        else:
            # Generate individual time fields independently
            for field in selected_time_fields:
                if field in sample_data and sample_data[field]:
                    value = random.choice(sample_data[field])
                    content_data[field] = self._process_date_placeholder(value, session_date)
                else:
                    content_data[field] = f"Optional {field.replace('_', ' ')}"
    
    def _generate_follow_up_deadline_pair(self, content_data, sample_data, session_date):
        """Generate follow_up_date and deadline using same index - no comparison needed."""
        
        # Get the available values for both fields
        follow_up_values = sample_data.get('follow_up_date', [])
        deadline_values = sample_data.get('deadline', [])
        
        if not follow_up_values or not deadline_values:
            # Fallback to independent selection if data is missing
            if follow_up_values:
                content_data['follow_up_date'] = self._process_date_placeholder(random.choice(follow_up_values), session_date)
            if deadline_values:
                content_data['deadline'] = self._process_date_placeholder(random.choice(deadline_values), session_date)
            return
        
        # Pick same index for both fields - metadata is organized with coordinated pairs
        max_index = min(len(follow_up_values), len(deadline_values)) - 1
        selected_index = random.randint(0, max_index)
        
        # Use same index for both fields - guaranteed coordination
        content_data['follow_up_date'] = self._process_date_placeholder(follow_up_values[selected_index], session_date)
        content_data['deadline'] = self._process_date_placeholder(deadline_values[selected_index], session_date)
    
    def _process_date_placeholder(self, value, session_date):
        """Process date placeholders in metadata values."""
        if not isinstance(value, str) or not session_date:
            return value
            
        # Handle session_date placeholder
        if "session_date" in value:
            return value.replace("session_date", session_date)
        
        # Handle relative date placeholders like "+7 days", "+1 week", "+1 month"
        if value.startswith("+"):
            from datetime import datetime, timedelta
            try:
                session_dt = datetime.strptime(session_date, "%Y-%m-%d")
                
                if "days" in value:
                    days = int(value.split()[0][1:])  # Extract number after "+"
                    new_date = session_dt + timedelta(days=days)
                    return new_date.strftime("%Y-%m-%d")
                elif "week" in value:
                    weeks = int(value.split()[0][1:])
                    new_date = session_dt + timedelta(weeks=weeks)
                    # If there's a time part, include it
                    if len(value.split()) > 2:
                        time_part = value.split()[2]
                        return f"{new_date.strftime('%Y-%m-%d')} {time_part}"
                    return new_date.strftime("%Y-%m-%d")
                elif "month" in value:
                    months = int(value.split()[0][1:])
                    # Approximate month addition (30 days per month)
                    new_date = session_dt + timedelta(days=months * 30)
                    if len(value.split()) > 2:
                        time_part = value.split()[2]
                        return f"{new_date.strftime('%Y-%m-%d')} {time_part}"
                    return new_date.strftime("%Y-%m-%d")
            except (ValueError, IndexError):
                pass
        
        return value
    
    def _load_activity_metadata(self, memory_categories_data):
        """Load activity metadata using centralized metadata dictionary."""
        try:
            if METADATA['activity_memory']:
                metadata_data = METADATA['activity_memory']
                
                # Check if this is the new persona-based structure
                if "activity_personas" in metadata_data and self.persona:
                    # Use persona-specific activity data
                    persona_activity = metadata_data["activity_personas"].get(self.persona, {})
                    if persona_activity:
                        self.activity_metadata = {}
                        activity_categories = persona_activity.get("activity_categories", {})
                        
                        for category, category_data in activity_categories.items():
                            self.activity_metadata[category] = {
                                "category": category,
                                "sample_data": category_data
                            }
                        
                        # Store original and create available metadata pool
                        self.activity_original_metadata = self.activity_metadata.copy()
                        self.activity_available_metadata = copy.deepcopy(self.activity_metadata)
                        
                        print(f"✅ Loaded persona-specific activity metadata for: {self.persona}")
                        return
                    else:
                        print(f"⚠️  Persona '{self.persona}' not found in activity metadata")
                
                # Fallback to general format (list structure)
                if isinstance(metadata_data, list):
                    self.activity_metadata = {}
                    for metadata in metadata_data:
                        category = metadata.get('category')
                        if category:
                            self.activity_metadata[category] = metadata
                else:
                    # Unknown format
                    print(f"⚠️  Unexpected activity metadata format")
                    self.activity_metadata = {}
                        
                print(f"✅ Loaded activity metadata from centralized dictionary")
            else:
                print(f"⚠️  No activity metadata found in centralized dictionary")
                self.activity_metadata = {}
        except Exception as e:
            print(f"❌ Error loading activity metadata: {e}")
            self.activity_metadata = {}
    
    def _load_goal_metadata(self, memory_categories_data):
        """Load goal metadata using centralized metadata dictionary."""
        try:
            if METADATA['goal_memory']:
                metadata_list = METADATA['goal_memory']
                
                # Convert list to dict keyed by category if needed
                if isinstance(metadata_list, list):
                    self.goal_metadata = {}
                    for metadata in metadata_list:
                        category = metadata.get('category')
                        if category:
                            self.goal_metadata[category] = metadata
                else:
                    self.goal_metadata = metadata_list
                        
                print(f"✅ Loaded goal metadata from centralized dictionary")
            else:
                print(f"⚠️  No goal metadata found in centralized dictionary")
                self.goal_metadata = {}
        except Exception as e:
            print(f"❌ Error loading goal metadata: {e}")
            self.goal_metadata = {}
    
    def _generate_activity_data(self, category: str, session_date: str = None, predetermined_meal: str = None, is_weekly_session: bool = False) -> str:
        """Generate realistic activity data based on category metadata."""
        
        if self.activity_metadata and category in self.activity_metadata:
            category_metadata = self.activity_metadata[category]
            sample_data = category_metadata.get("sample_data", {})
            
            if category == "food_expenses":
                # Weekly sessions generate grocery (shopping activity)
                # Daily sessions use chronological meal selection
                if is_weekly_session:
                    meal_type = "grocery"
                elif predetermined_meal:
                    meal_type = predetermined_meal
                else:
                    meal_type = self._get_next_chronological_meal(session_date, sample_data)
                
                # Note: meal_type should never be None with "decide targets first" approach
                
                meal_data = sample_data[meal_type]
                
                # Handle amount ranges [[min, max], [min, max], ...]
                amount_ranges = meal_data.get("amount_ranges", [[10.00, 15.00]])
                amount_range = random.choice(amount_ranges)
                amount = round(random.uniform(amount_range[0], amount_range[1]), 2)
                
                return {
                    "expense_type": meal_type,
                    "amount": amount,
                    "created_at": session_date or "2024-01-01"
                }
        
        # Handle other categories
        if self.activity_metadata and category in self.activity_metadata:
            category_metadata = self.activity_metadata[category]
            sample_data = category_metadata.get("sample_data", {})
            
            if category == "step_tracker":
                # Generate daily step count
                daily_steps_data = sample_data.get("daily_steps", {})
                if "step_count_ranges" in daily_steps_data:
                    step_ranges = daily_steps_data["step_count_ranges"]
                    step_range = random.choice(step_ranges)
                    step_count = random.randint(step_range[0], step_range[1])
                else:
                    # Fallback with default range
                    step_count = random.randint(6000, 12000)
                
                return {
                    "activity_type": "daily_steps",
                    "step_count": step_count,
                    "created_at": session_date or "2024-01-01"
                }
                        
            elif category == "todo_list":
                # Select task from available pool (not original metadata)
                if self.activity_available_metadata and category in self.activity_available_metadata:
                    available_data = self.activity_available_metadata[category]["sample_data"]
                    
                    # Find subcategories with available items
                    available_subcategories = []
                    for subcat, items in available_data.items():
                        if isinstance(items, list) and len(items) > 0:
                            available_subcategories.append(subcat)
                    
                    if available_subcategories:
                        subcategory = random.choice(available_subcategories)
                        available_tasks = available_data[subcategory]
                        task = random.choice(available_tasks)
                        
                        return {
                            "task_type": subcategory,
                            "description": task,
                            "created_at": session_date or "2024-01-01"
                        }
                
                # Fallback if no available items (should not happen with proper validation)
                return {
                    "task_type": "work_tasks",
                    "description": f"Generated task {len(self.memory_items.get('activity_memory', {}).get('todo_list', []))}",
                    "created_at": session_date or "2024-01-01"
                }
                
            elif category == "calendar_event":
                # Select event from available pool (not original metadata)
                if self.activity_available_metadata and category in self.activity_available_metadata:
                    available_data = self.activity_available_metadata[category]["sample_data"]
                    
                    # Find subcategories with available events
                    available_subcategories = []
                    for subcat, subcat_data in available_data.items():
                        if isinstance(subcat_data, dict) and "events" in subcat_data:
                            if len(subcat_data["events"]) > 0:
                                available_subcategories.append(subcat)
                    
                    if available_subcategories:
                        subcategory = random.choice(available_subcategories)
                        subcategory_data = available_data[subcategory]
                        
                        event = random.choice(subcategory_data["events"])
                        date = random.choice(subcategory_data["dates"])
                        
                        return {
                            "event_type": subcategory,
                            "event_name": event,
                            "date": date,
                            "created_at": session_date or "2024-01-01"
                        }
                
                # Fallback if no available events
                return {
                    "event_type": "work_meetings",
                    "event_name": f"Generated meeting {len(self.memory_items.get('activity_memory', {}).get('calendar_event', []))}",
                    "date": "+1 day",
                    "created_at": session_date or "2024-01-01"
                }
        
        # Fallback to simple category-based sample
        return {
            "activity_type": "unknown",
            "description": f"Sample {category.replace('_', ' ')} activity",
            "created_at": session_date or "2024-01-01"
        }
    
    def _get_next_chronological_meal(self, session_date: str, sample_data: dict) -> str:
        """
        Get the next meal in chronological order for the given date.
        This ensures proper meal sequencing and respects subcategory ranges from memory_categories.json
        
        Args:
            session_date: Date string for the session
            sample_data: Available meal types from metadata
            
        Returns:
            str: Selected meal type (never None - always returns a valid meal)
        """
        # Get subcategory ranges from memory_categories.json
        subcategory_ranges = self._get_food_subcategory_ranges()
        
        # Initialize daily meal state for this date if not exists
        # Note: Targets should be pre-decided by session generation via _initialize_daily_meal_state
        if session_date not in self.daily_meal_state:
            # Fallback: if no pre-decided targets, use default empty state
            # This shouldn't happen with the new "decide targets first" approach
            self.daily_meal_state[session_date] = {
                "breakfast_done": False,
                "lunch_done": False,
                "dinner_done": False,
                "coffee_count": 0,
                "grocery_done": False,
                # No targets set - will result in no meals being selected
            }
        
        state = self.daily_meal_state[session_date]
        # Only consider daily meals for chronological selection - exclude grocery
        daily_meal_types = ["breakfast", "lunch", "dinner", "coffee"]
        available_meal_types = [meal for meal in daily_meal_types if meal in sample_data]
        
        # Build list of available meal options with weights
        import random
        meal_options = []
        
        # Coffee can happen anytime (high flexibility) - give it competitive weight
        coffee_target = state.get("coffee_target", 0)
        if ("coffee" in available_meal_types and 
            state["coffee_count"] < coffee_target and 
            coffee_target > 0):
            # Coffee gets equal weight to compete with main meals
            meal_options.append(("coffee", 40))
        
        # Breakfast (highest priority if not done and targeted)
        if (not state["breakfast_done"] and 
            "breakfast" in available_meal_types and 
            state.get("breakfast_target", 0) > 0):
            # Breakfast gets slightly higher weight but not overwhelming
            meal_options.append(("breakfast", 45))
        
        # Lunch (only if breakfast is done or not targeted - maintains chronology)
        if (not state["lunch_done"] and 
              "lunch" in available_meal_types and 
              state.get("lunch_target", 0) > 0 and
              (state["breakfast_done"] or state.get("breakfast_target", 0) == 0)):
            # Lunch gets moderate weight when chronologically appropriate
            meal_options.append(("lunch", 40))
        
        # Dinner (only if previous meals are done or not targeted - maintains chronology)
        if (not state["dinner_done"] and 
              "dinner" in available_meal_types and 
              state.get("dinner_target", 0) > 0 and
              (state["breakfast_done"] or state.get("breakfast_target", 0) == 0) and
              (state["lunch_done"] or state.get("lunch_target", 0) == 0)):
            # Dinner gets moderate weight when chronologically appropriate
            meal_options.append(("dinner", 40))
        
        # Select meal based on weighted probabilities
        if meal_options:
            meals, weights = zip(*meal_options)
            selected_meal = random.choices(meals, weights=weights)[0]
            
            # Update state based on selection
            if selected_meal == "coffee":
                state["coffee_count"] += 1
            elif selected_meal in ["breakfast", "lunch", "dinner"]:
                state[f"{selected_meal}_done"] = True
            
            return selected_meal
        
        # Fallback: If no weighted options available (shouldn't happen with "decide targets first" approach)
        # But if it does happen, return coffee as the most flexible option
        return "coffee"
    
    def _initialize_daily_meal_state(self, session_date: str, meal_targets: Dict[str, int]):
        """
        Initialize daily meal state with pre-decided meal targets.
        This implements the "decide targets first" approach.
        
        Args:
            session_date: Date string for the session
            meal_targets: Pre-decided targets (e.g., {"breakfast": 1, "lunch": 0, "coffee": 2})
        """
        if session_date not in self.daily_meal_state:
            self.daily_meal_state[session_date] = {
                "breakfast_done": False,
                "lunch_done": False,
                "dinner_done": False,
                "coffee_count": 0,
                "grocery_done": False
            }
        
        # Set the pre-decided targets
        for meal_type, target_count in meal_targets.items():
            self.daily_meal_state[session_date][f"{meal_type}_target"] = target_count
    
    def _get_food_subcategory_ranges(self) -> dict:
        """
        Get subcategory ranges for food_expenses from memory_categories.json.
        
        Returns:
            dict: Mapping of subcategory to [min, max] range
        """
        try:
            food_expenses_config = METADATA.get('memory_categories', {}).get('activity_memory', {}).get('options', {}).get('food_expenses', {})
            return food_expenses_config.get('subcategory_ranges', {})
        except:
            # Fallback ranges - all disabled
            return {
                "breakfast": [0, 0],
                "lunch": [0, 0],
                "dinner": [0, 0],
                "coffee": [0, 0],
                "grocery": [0, 0]
            }
    
    def _generate_calendar_date(self):
        """Generate a new calendar date in the format '+X days'."""
        # Generate a random number of days in the future (1 to 30 days)
        days = random.randint(1, 30)
        return f"+{days} days"
    
    def _generate_goal_value(self, category: str):
        """Generate goal value grounded in persona-specific activity patterns (7x daily ranges)."""
        if not self.persona or not self.activity_metadata:
            # Fallback to static goal metadata if no persona
            return self._generate_goal_value_from_static_metadata(category)
        
        # Generate goal based on persona's activity patterns
        persona_activity = self.activity_metadata.get(category)
        if not persona_activity or not isinstance(persona_activity, dict):
            return self._generate_goal_value_from_static_metadata(category)
        
        # Extract sample_data from the activity metadata structure
        sample_data = persona_activity.get("sample_data", {})
        if not sample_data:
            return self._generate_goal_value_from_static_metadata(category)
        
        if category == "food_expenses":
            # Select a random meal type from sample_data
            meal_types = list(sample_data.keys())
            if not meal_types:
                return self._generate_goal_value_from_static_metadata(category)
                
            meal_type = random.choice(meal_types)
            meal_data = sample_data[meal_type]
            
            # Get daily activity ranges and calculate weekly goal (7x)
            daily_ranges = meal_data.get("amount_ranges", [])
            if daily_ranges:
                daily_range = random.choice(daily_ranges)
                # Weekly goal: 7x daily range with some variation (±10%)
                weekly_min = daily_range[0] * 7 * 0.9
                weekly_max = daily_range[1] * 7 * 1.1
                return round(random.uniform(weekly_min, weekly_max), 2)
                
        elif category == "step_tracker":
            # Get daily step ranges and calculate weekly goal (7x)
            step_data = sample_data.get("daily_steps", {})
            daily_ranges = step_data.get("step_count_ranges", [])
            if daily_ranges:
                daily_range = random.choice(daily_ranges)
                # Weekly goal: 7x daily range with some variation (±10%)
                weekly_min = int(daily_range[0] * 7 * 0.9)
                weekly_max = int(daily_range[1] * 7 * 1.1)
                return random.randint(weekly_min, weekly_max)
        
        return self._generate_goal_value_from_static_metadata(category)
    
    def _generate_goal_value_with_subcategory(self, category: str, specific_subcategory: str = None):
        """Generate goal value with subcategory information for operation_details."""
        # Always use static goal metadata for consistent goal ranges
        return self._generate_goal_value_from_static_metadata_with_subcategory(category, specific_subcategory)
    
    def _generate_goal_value_from_static_metadata_with_subcategory(self, category: str, specific_subcategory: str = None):
        """Generate goal value with subcategory from static goal metadata."""
        if not METADATA['goal_memory']:
            return None
            
        # Find the category in goal metadata
        for goal_metadata in METADATA['goal_memory']:
            if goal_metadata.get("category") == category:
                subcategories = goal_metadata.get("subcategories", {})
                
                # Use specific subcategory if provided, otherwise select randomly
                if specific_subcategory and specific_subcategory in subcategories:
                    subcategory_name = specific_subcategory
                elif subcategories:
                    subcategory_name = random.choice(list(subcategories.keys()))
                else:
                    return None
                
                subcategory_data = subcategories[subcategory_name]
                
                # Get value ranges and select one
                value_ranges = subcategory_data.get("value_ranges", [])
                if value_ranges:
                    selected_range = random.choice(value_ranges)
                    
                    # Generate value based on range type
                    if isinstance(selected_range, list) and len(selected_range) == 2:
                        if isinstance(selected_range[0], int) and isinstance(selected_range[1], int):
                            # Integer range (e.g., steps) - round to nearest 500 for natural goal setting
                            raw_value = random.randint(selected_range[0], selected_range[1])
                            goal_value = self._round_step_goal(raw_value)
                        else:
                            # Float range (e.g., money) - round to natural budget amounts
                            raw_value = random.uniform(selected_range[0], selected_range[1])
                            goal_value = self._round_budget_goal(raw_value, subcategory_name)
                        return (goal_value, subcategory_name)
                
                break
        
        return None
    
    def _generate_goal_value_from_static_metadata(self, category: str):
        """Fallback: Generate goal value from static goal metadata."""
        if not METADATA['goal_memory']:
            return None
            
        # Find the category in goal metadata
        for goal_metadata in METADATA['goal_memory']:
            if goal_metadata.get("category") == category:
                subcategories = goal_metadata.get("subcategories", {})
                
                # Select a random subcategory
                if subcategories:
                    subcategory_name = random.choice(list(subcategories.keys()))
                    subcategory_data = subcategories[subcategory_name]
                    
                    # Get value ranges and select one
                    value_ranges = subcategory_data.get("value_ranges", [])
                    if value_ranges:
                        selected_range = random.choice(value_ranges)
                        
                        # Generate value based on range type
                        if isinstance(selected_range, list) and len(selected_range) == 2:
                            if isinstance(selected_range[0], int) and isinstance(selected_range[1], int):
                                # Integer range (e.g., steps)
                                return random.randint(selected_range[0], selected_range[1])
                            else:
                                # Float range (e.g., money)
                                return round(random.uniform(selected_range[0], selected_range[1]), 2)
                
                break
        
        return None
    
    def _round_step_goal(self, raw_value):
        """Round step goals to natural, realistic values."""
        if raw_value < 5000:
            # Round to nearest 250 for lower values (e.g., 4250, 4500, 4750)
            return round(raw_value / 250) * 250
        elif raw_value < 10000:
            # Round to nearest 500 for mid values (e.g., 5000, 5500, 6000)
            return round(raw_value / 500) * 500
        else:
            # Round to nearest 1000 for higher values (e.g., 10000, 11000, 12000)
            return round(raw_value / 1000) * 1000
    
    def _round_budget_goal(self, raw_value, subcategory):
        """Round budget goals to natural, realistic amounts based on subcategory."""
        if subcategory == "coffee":
            # Coffee: round to nearest $5 (e.g., $25, $30, $35)
            return round(raw_value / 5) * 5
        elif subcategory in ["breakfast", "lunch"]:
            # Meals: round to nearest $5 (e.g., $40, $45, $50)
            return round(raw_value / 5) * 5
        elif subcategory == "dinner":
            # Dinner: round to nearest $10 (e.g., $120, $130, $140)
            return round(raw_value / 10) * 10
        elif subcategory == "grocery":
            # Grocery: round to nearest $25 (e.g., $175, $200, $225)
            return round(raw_value / 25) * 25
        else:
            # Default: round to nearest $5
            return round(raw_value / 5) * 5
    
    def _load_preference_metadata(self, memory_categories_data):
        """Load preference metadata using centralized metadata dictionary."""
        try:
            # If persona is specified, try to use persona-specific preference mapping
            if self.persona and self.persona_traits:
                preference_mapping = self.persona_traits.get("category_traits", {}).get("preference_memory", {})
                if preference_mapping and METADATA['preference_memory']:
                    all_preference_data = METADATA['preference_memory']
                    
                    # Build persona-specific preference metadata
                    persona_preferences = {"domains": {}}
                    
                    # Check if data has preference_personas structure
                    if "preference_personas" in all_preference_data:
                        preference_personas = all_preference_data["preference_personas"]
                        
                        for category, persona_type in preference_mapping.items():
                            if category in preference_personas and persona_type in preference_personas[category]:
                                persona_data = preference_personas[category][persona_type]
                                
                                # Extract the actual preferences subcategories (nested under 'preferences' key)
                                if "preferences" in persona_data:
                                    categories_data = persona_data["preferences"]
                                else:
                                    # Fallback to direct structure
                                    categories_data = persona_data
                                
                                # Convert to the expected "domains" format
                                persona_preferences["domains"][category] = {
                                    "categories": categories_data
                                }
                    
                    if persona_preferences["domains"]:
                        self.preference_original_metadata = persona_preferences
                        self.preference_available_metadata = copy.deepcopy(persona_preferences)
                        print(f"✅ Loaded persona-specific preference metadata for: {self.persona}")
                        return
            
            # Fallback to general preference metadata from centralized dictionary
            if METADATA['preference_memory']:
                self.preference_original_metadata = METADATA['preference_memory']
                print(f"✅ Loaded general preference metadata")
                # Deep copy for available metadata pool
                self.preference_available_metadata = copy.deepcopy(self.preference_original_metadata)
            else:
                print(f"❌ Warning: No preference metadata found. Using empty metadata.")
                self.preference_original_metadata = {"domains": {}}
                self.preference_available_metadata = {"domains": {}}
        except Exception as e:
            print(f"❌ Error loading preference metadata: {e}")
            self.preference_original_metadata = {"domains": {}}
            self.preference_available_metadata = {"domains": {}}
    
    def _initialize_preference_memory(self):
        """Initialize preference memory structure with likes/dislikes like memory_state_manager."""
        if not self.preference_original_metadata or "domains" not in self.preference_original_metadata:
            return
            
        for domain_name, domain_data in self.preference_original_metadata["domains"].items():
            self.memory_items["preference_memory"][domain_name] = {}
            
            for category_name in domain_data["categories"].keys():
                self.memory_items["preference_memory"][domain_name][category_name] = {
                    "likes": [],
                    "dislikes": []
                }
    
    def has_memory_for_category(self, session_type: str, category: str) -> bool:
        """Check if we have any memories for a given session type and category."""
        if session_type == SessionType.NO_MEMORY:
            return False
            
        if session_type == SessionType.PREFERENCE_MEMORY:
            # For preference memory, check if any subcategory has items
            # This method is primarily used for UPDATE operations, which can work on ANY items
            # DELETE operations use the specific has_liked_items_for_category() method instead
            if category in self.memory_items["preference_memory"]:
                category_data = self.memory_items["preference_memory"][category]
                for subcategory_data in category_data.values():
                    if isinstance(subcategory_data, dict):
                        # UPDATE operations can flip likes ↔ dislikes or replace items
                        # So we check for ANY items (likes OR dislikes)
                        if len(subcategory_data.get("likes", [])) > 0 or len(subcategory_data.get("dislikes", [])) > 0:
                            return True
                return False
        elif session_type == SessionType.ACTIVITY_MEMORY:
            # For activity memory, check if category has any items
            if category in self.memory_items["activity_memory"]:
                return len(self.memory_items["activity_memory"][category]) > 0
            return False
        elif session_type == SessionType.CONTENT_MEMORY:
            # For content memory, only check draft pool (items that can be operated on)
            # Completed pool items cannot be updated/deleted, so don't count them
            if hasattr(self, 'content_pools') and self.content_pools:
                draft_items = self.content_pools["draft_pool"].get(category, [])
                return len(draft_items) > 0
            
            return False
        elif session_type == SessionType.GOAL_MEMORY:
            # For goal memory, check if category has any goals
            if category in self.memory_items["goal_memory"]:
                return len(self.memory_items["goal_memory"][category]) > 0
            return False
        else:
            return False
    
    def has_existing_calendar_events(self) -> bool:
        """Check if we have any existing calendar events that can be updated or deleted."""
        if "calendar_event" in self.memory_items["activity_memory"]:
            return len(self.memory_items["activity_memory"]["calendar_event"]) > 0
        return False
    
    def has_liked_items_for_category(self, category: str) -> bool:
        """Check if we have any LIKED items in a preference memory category that can be deleted."""
        if category not in self.memory_items["preference_memory"]:
            return False
        
        category_data = self.memory_items["preference_memory"][category]
        for subcategory_data in category_data.values():
            if isinstance(subcategory_data, dict):
                if len(subcategory_data.get("likes", [])) > 0:
                    return True
        return False
    
    def get_memory_count(self, session_type: str, category: str) -> int:
        """Get total number of memories for a category."""
        if session_type == SessionType.NO_MEMORY:
            return 0
        
        if session_type == SessionType.PREFERENCE_MEMORY:
            # For preference memory, count likes + dislikes across all subcategories
            if category in self.memory_items["preference_memory"]:
                total = 0
                category_data = self.memory_items["preference_memory"][category]
                for subcategory_data in category_data.values():
                    if isinstance(subcategory_data, dict):
                        total += len(subcategory_data.get("likes", []))
                        total += len(subcategory_data.get("dislikes", []))
                return total
        elif session_type == SessionType.ACTIVITY_MEMORY:
            # For activity memory, count items in the category list
            if category in self.memory_items["activity_memory"]:
                return len(self.memory_items["activity_memory"][category])
        elif session_type == SessionType.CONTENT_MEMORY:
            # For content memory, count documents in the category list
            if category in self.memory_items["content_memory"]:
                return len(self.memory_items["content_memory"][category])
        elif session_type == SessionType.GOAL_MEMORY:
            # For goal memory, count goals in the category list
            if category in self.memory_items["goal_memory"]:
                return len(self.memory_items["goal_memory"][category])
        
        return 0
    
    def _select_random_available_item(self, category: str, subcategory: str):
        """Select a random item from available metadata pool for preference memory with smart fallback."""
        if not self.preference_available_metadata or "domains" not in self.preference_available_metadata:
            return None
            
        if category in self.preference_available_metadata["domains"]:
            domain_data = self.preference_available_metadata["domains"][category]
            if "categories" in domain_data:
                
                # First, try the requested subcategory
                if subcategory in domain_data["categories"]:
                    subcategory_data = domain_data["categories"][subcategory]
                    
                    # Handle both old format (dict with "values") and new format (direct list)
                    if isinstance(subcategory_data, dict):
                        available_values = subcategory_data.get("values", [])
                    elif isinstance(subcategory_data, list):
                        available_values = subcategory_data
                    else:
                        available_values = []
                        
                    if available_values:
                        return random.choice(available_values)
                
                # SMART FALLBACK: If requested subcategory is empty, try other subcategories
                print(f"⚠️  Subcategory '{subcategory}' exhausted, trying fallback subcategories...")
                all_subcategories = list(domain_data["categories"].keys())
                # Remove the exhausted subcategory BEFORE shuffling for efficiency
                fallback_subcategories = [sc for sc in all_subcategories if sc != subcategory]
                random.shuffle(fallback_subcategories)  # Random order for fairness
                
                for fallback_subcat in fallback_subcategories:
                    
                    fallback_data = domain_data["categories"][fallback_subcat]
                    
                    # Handle both formats
                    if isinstance(fallback_data, dict):
                        available_values = fallback_data.get("values", [])
                    elif isinstance(fallback_data, list):
                        available_values = fallback_data
                    else:
                        available_values = []
                    
                    if available_values:
                        selected_item = random.choice(available_values)
                        print(f"✅ Using fallback: {fallback_subcat} → '{selected_item}'")
                        # Return both the item and the actual subcategory used
                        return (selected_item, fallback_subcat)
                
        return None
    
    def _select_random_active_item(self, category: str, subcategory: str):
        """Select a random item from active preference memory."""
        if category not in self.memory_items["preference_memory"]:
            return None
            
        category_data = self.memory_items["preference_memory"][category]
        if subcategory not in category_data:
            return None
            
        subcategory_data = category_data[subcategory]
        all_items = []
        
        # Collect all active items (likes and dislikes)
        for preference_type in ["likes", "dislikes"]:
            for item in subcategory_data.get(preference_type, []):
                # Return the base form (like/dislike) not the plural form
                base_preference = preference_type[:-1]  # Remove 's' from likes/dislikes
                all_items.append((item, base_preference))
        
        if all_items:
            return random.choice(all_items)
        return None
    
    def _select_random_available_activity_item(self, category: str, subcategory: str):
        """Select a random item from available activity metadata pool."""
        if not self.activity_available_metadata or category not in self.activity_available_metadata:
            return None
            
        category_data = self.activity_available_metadata[category]
        sample_data = category_data.get("sample_data", {})
        
        if subcategory in sample_data:
            available_items = sample_data[subcategory]
            if isinstance(available_items, list) and available_items:
                return random.choice(available_items)
        
        return None
    
    def _select_random_active_activity_item(self, category: str):
        """Select a random item from active activity memory."""
        if category not in self.memory_items["activity_memory"]:
            return None
            
        activity_items = self.memory_items["activity_memory"][category]
        if activity_items:
            return random.choice(activity_items)
        return None
    
    def add_memory_item(self, session_type: str, category: str, subcategory: str = None, item: str = None, preference: str = None, session_date: str = None, session_id: int = None, **kwargs):
        """Add a memory item using real metadata for preference memory."""
        import random
        import copy
        if session_type == SessionType.PREFERENCE_MEMORY:
            # Use memory_state_manager logic for preference memory
            if not subcategory:
                return {"success": False, "reason": "Subcategory required for preference memory"}
            
            # Select item from available metadata pool
            if not item:
                result = self._select_random_available_item(category, subcategory)
                if not result:
                    return {"success": False, "reason": f"No available items in {category}.{subcategory}"}
                
                # Handle both single item and tuple (item, actual_subcategory) returns
                if isinstance(result, tuple):
                    item, actual_subcategory = result
                    subcategory = actual_subcategory  # Use the fallback subcategory
                else:
                    item = result
            
            # Choose preference with bias towards likes (70% like, 30% dislike)
            if not preference:
                preference = random.choices(["like", "dislike"], weights=[70, 30])[0]
            
            # Add to active memory
            if category in self.memory_items["preference_memory"]:
                if subcategory not in self.memory_items["preference_memory"][category]:
                    self.memory_items["preference_memory"][category][subcategory] = {"likes": [], "dislikes": []}
                
                preference_list = f"{preference}s"  # "likes" or "dislikes"
                # Create item dictionary with timestamp
                item_dict = {
                    "item": item,
                    "created_at": session_date or "2024-01-01"
                }
                self.memory_items["preference_memory"][category][subcategory][preference_list].append(item_dict)
                
                # Remove from available metadata pool (use the actual subcategory where item was found)
                if (category in self.preference_available_metadata["domains"] and 
                    subcategory in self.preference_available_metadata["domains"][category]["categories"]):
                    subcategory_data = self.preference_available_metadata["domains"][category]["categories"][subcategory]
                    
                    # Handle both old format (dict with "values") and new format (direct list)
                    if isinstance(subcategory_data, dict) and "values" in subcategory_data:
                        if item in subcategory_data["values"]:
                            subcategory_data["values"].remove(item)
                    elif isinstance(subcategory_data, list):
                        if item in subcategory_data:
                            subcategory_data.remove(item)
                
                return {"success": True, "item": item, "preference": preference, "category": category, "subcategory": subcategory}
            
            return {"success": False, "reason": f"Invalid category/subcategory: {category}.{subcategory}"}
        
        elif session_type == SessionType.CONTENT_MEMORY:
            # For content memory, use redesigned three-pool system
            if category in self.content_metadata:
                # Check if we should use redesigned approach
                metadata = self.content_metadata[category]
                use_redesigned = metadata.get("use_redesigned_approach", False)
                
                if use_redesigned and hasattr(self, 'content_pools'):
                    # Use redesigned three-pool approach
                    # Generate partial content data (some list elements held back for updates)
                    content_data, metadata_source, remaining_data = self._generate_partial_content_data(category, session_date)
                    
                    if content_data is not None:
                        # Generate unique item ID using existing draft pool size + 1
                        existing_count = len(self.content_pools["draft_pool"].get(category, [])) + len(self.content_pools["completed_pool"].get(category, []))
                        item_id = f"{category}_{existing_count + 1}"
                        
                        # Generate dynamic completion threshold from metadata
                        import random
                        min_sessions, max_sessions = self._get_completion_threshold(category)
                        completion_threshold = random.randint(min_sessions, max_sessions)
                        
                        content_item = {
                            "id": item_id,
                            "category": category,
                            "created_at": session_date or "2024-01-01",
                            "content_data": content_data,
                            "metadata_source": metadata_source,
                            "remaining_data": remaining_data,
                            "pool_status": "draft",  # Track which pool this item is in
                            "completion_threshold": completion_threshold,  # Dynamic threshold for moving to completed pool
                            "session_history": [  # Track sessions when this item was accessed
                                {
                                    "session_id": session_id,
                                    "session_date": session_date or "2024-01-01",
                                    "operation": "add"
                                }
                            ]
                        }
                        
                        # Add to draft pool (this is the primary storage)
                        self.content_pools["draft_pool"][category].append(content_item)
                        
                        # Also add to regular memory for compatibility with existing code
                        self.memory_items[session_type][category].append(content_item)
                        
                        return {
                            "success": True,
                            "item": item_id, 
                            "content_data": content_data,
                            "metadata_source": metadata_source,
                            "remaining_data": remaining_data,
                            "category": category,
                            "pool_status": "draft",
                            "completion_threshold": completion_threshold,
                            "generation_method": "redesigned_three_pool"
                        }
                    else:
                        # Available pool exhausted - try to fallback to UPDATE operation on draft pool
                        draft_items = self.content_pools["draft_pool"].get(category, [])
                        
                        if draft_items:
                            # Convert ADD to UPDATE: pick item from draft pool and update with remaining data
                            item_to_update = random.choice(draft_items)
                            old_content_data = copy.deepcopy(item_to_update["content_data"])
                            
                            # Update by moving elements from _remaining_data to active fields
                            update_result = self._update_content_with_remaining_data(item_to_update, session_date)
                            
                            # Track session access for this item
                            if "session_history" not in item_to_update:
                                item_to_update["session_history"] = []
                            
                            item_to_update["session_history"].append({
                                "session_id": session_id,
                                "session_date": session_date or "2024-01-01",
                                "operation": "update"
                            })
                            
                            # Check if we've reached the dynamic session threshold
                            unique_sessions = len(item_to_update["session_history"])
                            completion_threshold = item_to_update.get("completion_threshold", 3)  # Default to 3 for legacy items
                            if unique_sessions >= completion_threshold:
                                self._move_to_completed_pool(item_to_update, category)
                            
                            return {
                                "success": True,
                                "item": item_to_update["id"],
                                "content_data": item_to_update["content_data"],
                                "old_content_data": old_content_data,
                                "operation_converted": "add_to_update",
                                "conversion_reason": "available_pool_exhausted",
                                "category": category,
                                "memory_updates": update_result["memory_updates"],
                                "pool_status": item_to_update["pool_status"],
                                "session_count": len(item_to_update.get("session_history", [])),
                                "session_history": item_to_update.get("session_history", []),
                                "completion_threshold": item_to_update.get("completion_threshold", 3),
                                "remaining_data_fields": update_result["remaining_data_fields"],
                                "generation_method": "add_converted_to_update"
                            }
                        else:
                            # Both available and draft pools exhausted - convert to NO_MEMORY session
                            completed_count = len(self.content_pools["completed_pool"].get(category, []))
                            total_added = len(self.content_pools["draft_pool"].get(category, [])) + completed_count
                            
                            return {
                                "success": True,
                                "operation_converted": "add_to_no_memory",
                                "conversion_reason": "both_pools_exhausted",
                                "fallback_session_type": "no_memory",
                                "category": category,
                                "pool_status": {
                                    "draft_pool_size": 0,
                                    "completed_pool_size": completed_count,
                                    "total_items_processed": total_added,
                                    "available_pool_exhausted": True,
                                    "draft_pool_empty": True,
                                    "all_items_completed": True
                                },
                                "generation_method": "add_converted_to_no_memory",
                                "message": f"All {category} items have been completed. Converting to general conversation."
                            }
            else:
                return {"success": False, "reason": f"No content metadata found for category: {category}"}
        
        elif session_type == SessionType.GOAL_MEMORY:
            # For goal memory, check if subcategory already exists - update if yes, add if no
            # Use subcategory hint if provided (for unique weekly goals)
            subcategory_hint = kwargs.get("subcategory_hint")
            goal_result = self._generate_goal_value_with_subcategory(category, subcategory_hint)
            if goal_result is not None:
                goal_value, subcategory = goal_result
                
                # Check if this subcategory already exists in memory
                existing_items = self.memory_items[session_type][category]
                existing_item = None
                for item in existing_items:
                    if item.get("subcategory") == subcategory:
                        existing_item = item
                        break
                
                if existing_item:
                    # Update existing goal value
                    old_value = existing_item["value"]
                    existing_item["value"] = goal_value
                    existing_item["updated_at"] = session_date or "2024-01-01"
                    
                    return {
                        "success": True,
                        "item": goal_value,
                        "subcategory": subcategory,
                        "operation_performed": "update",
                        "old_value": old_value
                    }
                else:
                    # Create new goal item (simplified structure)
                    goal_item = {
                        "category": category,
                        "subcategory": subcategory,
                        "value": goal_value,
                        "created_at": session_date or "2024-01-01"
                    }
                
                self.memory_items[session_type][category].append(goal_item)
                
                return {
                    "success": True,
                        "item": goal_value,
                        "subcategory": subcategory,
                        "operation_performed": "add"
                }
            return {"success": False, "reason": f"No goal metadata found for category: {category}"}
        
        elif session_type == SessionType.ACTIVITY_MEMORY:
            # For activity memory, use proper pool management like preference memory
            if category in self.memory_items[session_type]:
                # Simple pool check for discrete item categories
                if category in ["todo_list", "calendar_event"]:
                    # Check if any items are available in the pool
                    pool_exhausted = True
                    if self.activity_available_metadata and category in self.activity_available_metadata:
                        available_data = self.activity_available_metadata[category]["sample_data"]
                        
                        if category == "todo_list":
                            for subcat, items in available_data.items():
                                if isinstance(items, list) and len(items) > 0:
                                    pool_exhausted = False
                                    break
                        elif category == "calendar_event":
                            for subcat, subcat_data in available_data.items():
                                if isinstance(subcat_data, dict) and "events" in subcat_data:
                                    if len(subcat_data["events"]) > 0:
                                        pool_exhausted = False
                                        break
                    
                    if pool_exhausted:
                        # Both available pool exhausted - convert to NO_MEMORY session
                        return {
                            "success": True,
                            "operation_converted": "add_to_no_memory",
                            "conversion_reason": "activity_pool_exhausted",
                            "fallback_session_type": "no_memory",
                            "category": category,
                            "message": f"All {category} items have been used. Converting to general conversation."
                        }
                
                # Use provided item data or generate activity data from metadata
                if item:
                    activity_data = item
                else:
                    activity_data = self._generate_activity_data(category, session_date)
                
                # Handle pool management based on category type
                if category == "todo_list":
                    # TODO_LIST: Direct list structure
                    task_type = activity_data.get("task_type")
                    description = activity_data.get("description")
                    
                    if task_type and description:
                        # Remove the selected item from available pool
                        if (self.activity_available_metadata and 
                            category in self.activity_available_metadata and
                            task_type in self.activity_available_metadata[category]["sample_data"]):
                            
                            available_items = self.activity_available_metadata[category]["sample_data"][task_type]
                            if isinstance(available_items, list) and description in available_items:
                                available_items.remove(description)
                
                elif category == "calendar_event":
                    # CALENDAR_EVENT: Nested structure with "events" list
                    event_type = activity_data.get("event_type")
                    event_name = activity_data.get("event_name")
                    
                    if event_type and event_name:
                        # Remove the selected event from available pool
                        if (self.activity_available_metadata and 
                            category in self.activity_available_metadata and
                            event_type in self.activity_available_metadata[category]["sample_data"]):
                            
                            event_data = self.activity_available_metadata[category]["sample_data"][event_type]
                            if isinstance(event_data, dict) and "events" in event_data:
                                available_events = event_data["events"]
                                if isinstance(available_events, list) and event_name in available_events:
                                    available_events.remove(event_name)
                
                elif category in ["food_expenses", "step_tracker"]:
                    # Range-based categories: just generate and track in active memory
                    # No pool management needed since values are generated from ranges
                    pass
                
                # Add to active memory
                self.memory_items[session_type][category].append(activity_data)
                return {"success": True, "item": activity_data, "category": category}
            return {"success": False, "reason": f"Invalid category: {category}"}
        
        else:
            # Fallback for unknown session types
            return {"success": False, "reason": f"Unknown session type: {session_type}"}
    
    def _select_random_active_item_from_category(self, category: str, subcategory: str = None):
        """Select a random item from active preference memory for a specific category."""
        if category not in self.memory_items["preference_memory"]:
            return None
        
        category_data = self.memory_items["preference_memory"][category]
        
        # If subcategory specified, only search in that subcategory
        if subcategory and subcategory in category_data:
            subcategories_to_search = [subcategory]
        else:
            # Search all subcategories in the category, but exclude already_* subcategories
            # since those should only support ADD operations
            all_subcategories = list(category_data.keys())
            subcategories_to_search = [
                subcat for subcat in all_subcategories 
                if not (subcat.startswith("already_") and subcat.endswith("_list"))
            ]
        
        all_items = []
        
        # Collect all active items from the specified subcategories
        for subcat in subcategories_to_search:
            if subcat in category_data:
                subcategory_data = category_data[subcat]
                for preference_type in ["likes", "dislikes"]:
                    for item_dict in subcategory_data.get(preference_type, []):
                        # Handle both old format (string) and new format (dict)
                        if isinstance(item_dict, dict):
                            item = item_dict["item"]
                        else:
                            item = item_dict  # Legacy support
                        # Return the base form (like/dislike) not the plural form
                        base_preference = preference_type[:-1]  # Remove 's' from likes/dislikes
                        all_items.append((item, base_preference, subcat))
        
        if all_items:
            return random.choice(all_items)
        return None

    def _select_random_liked_item_from_category(self, category: str):
        """Select a random LIKED item from active preference memory for value updates."""
        if category not in self.memory_items["preference_memory"]:
            return None
        
        category_data = self.memory_items["preference_memory"][category]
        all_liked_items = []
        
        # Collect only liked items from all subcategories, but exclude already_* subcategories
        # since those should only support ADD operations
        for subcategory, subcategory_data in category_data.items():
            # Skip already_* subcategories for UPDATE operations
            if subcategory.startswith("already_") and subcategory.endswith("_list"):
                continue
            for item_dict in subcategory_data.get("likes", []):
                # Handle both old format (string) and new format (dict)
                if isinstance(item_dict, dict):
                    item = item_dict["item"]
                else:
                    item = item_dict  # Legacy support
                all_liked_items.append((item, subcategory))
        
        if all_liked_items:
            return random.choice(all_liked_items)
        return None

    def _is_calendar_event_valid_for_modification(self, event_item: dict, current_session_date: str) -> bool:
        """
        Check if a calendar event is still valid for UPDATE/DELETE operations.
        
        Args:
            event_item: Calendar event dictionary with 'date' and 'created_at' fields
            current_session_date: Current session date (YYYY-MM-DD)
            
        Returns:
            bool: True if event can still be modified, False if it's expired
        """
        if not isinstance(event_item, dict):
            return False
        
        event_date_str = event_item.get("date")
        created_at = event_item.get("created_at")
        
        if not event_date_str or not created_at:
            return False
        
        try:
            from datetime import datetime, timedelta
            
            # Parse current session date
            current_date = datetime.strptime(current_session_date, "%Y-%m-%d")
            
            # Parse event date - handle relative dates like "+3 days"
            if event_date_str.startswith("+"):
                # Relative date format: "+X days"
                created_date = datetime.strptime(created_at, "%Y-%m-%d")
                
                if "days" in event_date_str:
                    days = int(event_date_str.split()[0][1:])  # Extract number after "+"
                    event_date = created_date + timedelta(days=days)
                elif "week" in event_date_str:
                    weeks = int(event_date_str.split()[0][1:])
                    event_date = created_date + timedelta(weeks=weeks)
                elif "month" in event_date_str:
                    months = int(event_date_str.split()[0][1:])
                    event_date = created_date + timedelta(days=months * 30)  # Approximate
                else:
                    # Unknown relative format, assume it's still valid
                    return True
            else:
                # Absolute date format: "YYYY-MM-DD"
                event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
            
            # Event is valid for modification if it hasn't passed yet
            # Allow modification on the same day as the event
            return current_date <= event_date
            
        except (ValueError, IndexError, AttributeError):
            # If we can't parse the dates, assume it's still valid to be safe
            return True
    
    def _filter_valid_calendar_events(self, activity_items: list, current_session_date: str) -> list:
        """
        Filter calendar events to only include those still valid for modification.
        
        Args:
            activity_items: List of calendar event items
            current_session_date: Current session date (YYYY-MM-DD)
            
        Returns:
            list: Filtered list of events that can still be modified
        """
        valid_events = []
        
        for event_item in activity_items:
            if self._is_calendar_event_valid_for_modification(event_item, current_session_date):
                valid_events.append(event_item)
        
        return valid_events
    
    def remove_memory_item(self, session_type: str, category: str, subcategory: str = None, session_date: str = None, session_id: int = None):
        """Remove a memory item (for delete operations)."""
        if session_type == SessionType.PREFERENCE_MEMORY:
            # For DELETE operations, only select from LIKED items (not dislikes)
            # since disliked items are harder to generate conversations for
            item_data = self._select_random_liked_item_from_category(category)
            if not item_data:
                return {"success": False, "reason": f"No liked items available for deletion in {category}"}
            
            item, actual_subcategory = item_data
            preference = "like"  # We only select from liked items for DELETE
            preference_list = "likes"  # Always "likes" since we only select from liked items
            
            # Remove from active memory
            if (category in self.memory_items["preference_memory"] and 
                actual_subcategory in self.memory_items["preference_memory"][category] and
                preference_list in self.memory_items["preference_memory"][category][actual_subcategory]):
                
                # Find and remove the item (handle both dict and string formats)
                items_list = self.memory_items["preference_memory"][category][actual_subcategory][preference_list]
                item_to_remove = None
                
                for item_dict in items_list:
                    if isinstance(item_dict, dict):
                        if item_dict["item"] == item:
                            item_to_remove = item_dict
                            break
                    else:
                        if item_dict == item:  # Legacy format
                            item_to_remove = item_dict
                            break
                
                if item_to_remove is not None:
                    items_list.remove(item_to_remove)
                else:
                    return {"success": False, "reason": f"Item '{item}' not found in active memory"}
                
                # Add back to available metadata pool
                if (category in self.preference_available_metadata["domains"] and 
                    actual_subcategory in self.preference_available_metadata["domains"][category]["categories"]):
                    subcategory_data = self.preference_available_metadata["domains"][category]["categories"][actual_subcategory]
                    
                    # Handle both old format (dict with "values") and new format (direct list)
                    if isinstance(subcategory_data, dict) and "values" in subcategory_data:
                        subcategory_data["values"].append(item)
                    elif isinstance(subcategory_data, list):
                        subcategory_data.append(item)
                
                return {"success": True, "item": item, "preference": preference, "category": category, "subcategory": actual_subcategory}
            
            return {"success": False, "reason": f"Item not found in active memory"}
        
        elif session_type == SessionType.CONTENT_MEMORY:
            # For content memory, check redesigned approach first
            if category in self.content_metadata:
                metadata = self.content_metadata[category]
                use_redesigned = metadata.get("use_redesigned_approach", False)
                
                if use_redesigned and hasattr(self, 'content_pools'):
                        # Use redesigned three-pool approach
                        # Select item from draft pool for element deletion (ONLY look in draft_pool)
                        draft_items = self.content_pools["draft_pool"].get(category, [])
                        
                        if draft_items:
                            # Pick a draft item and delete ONE element from it
                            selected_item = random.choice(draft_items)
                            old_content_data = copy.deepcopy(selected_item["content_data"])
                            
                            # Delete one element by moving it back to remaining data
                            delete_result = self._delete_content_element_to_remaining_data(selected_item, session_date)
                            
                            # Check if DELETE actually succeeded (has actual deletes, not just errors)
                            successful_deletes = [d for d in delete_result.get("memory_deletes", []) if "error" not in d]
                            
                            if successful_deletes:
                                # DELETE succeeded - proceed normally
                                
                                # Track session access for this item
                                if "session_history" not in selected_item:
                                    selected_item["session_history"] = []
                                
                                selected_item["session_history"].append({
                                    "session_id": session_id,
                                    "session_date": session_date or "2024-01-01",
                                    "operation": "delete"
                                })
                                
                                # Check if we've reached the dynamic session threshold after DELETE
                                unique_sessions = len(selected_item["session_history"])
                                completion_threshold = selected_item.get("completion_threshold", 3)  # Default to 3 for legacy items
                                if unique_sessions >= completion_threshold:
                                    self._move_to_completed_pool(selected_item, category)
                                
                                return {
                                    "success": True, 
                                    "item": selected_item["id"], 
                                    "content_data": selected_item["content_data"],
                                    "metadata_source": selected_item.get("metadata_source"),
                                    "remaining_data": selected_item.get("remaining_data"),
                                    "category": category,
                                    "pool_status": selected_item["pool_status"],
                                    "session_count": len(selected_item.get("session_history", [])),
                                    "session_history": selected_item.get("session_history", []),
                                    "completion_threshold": selected_item.get("completion_threshold", 3),
                                    "deletion_method": "redesigned_element_deletion",
                                    "memory_deletes": delete_result["memory_deletes"],  # What was removed from memory
                                    "remaining_data_fields": delete_result["remaining_data_fields"]
                                }
                            else:
                                # DELETE failed due to insufficient active elements -> try UPDATE instead
                                print(f"🔄 DELETE failed (no deletable elements), trying UPDATE on {selected_item['id']}")
                                
                                # Reset the item to pre-delete state (since delete failed)
                                selected_item["content_data"] = old_content_data
                                
                                # Try UPDATE operation on the same item
                                update_result = self._update_content_with_remaining_data(selected_item, session_date)
                                
                                # Check if UPDATE succeeded
                                successful_updates = [u for u in update_result.get("memory_updates", []) if "error" not in u]
                                
                                if successful_updates:
                                    # UPDATE succeeded as fallback
                                    print(f"✅ Fallback UPDATE succeeded on {selected_item['id']}")
                                    
                                    # Track session access for this item
                                    if "session_history" not in selected_item:
                                        selected_item["session_history"] = []
                                    
                                    selected_item["session_history"].append({
                                        "session_id": session_id,
                                        "session_date": session_date or "2024-01-01",
                                        "operation": "update"
                                    })
                                    
                                    # Check if we've reached the dynamic session threshold
                                    unique_sessions = len(selected_item["session_history"])
                                    completion_threshold = selected_item.get("completion_threshold", 3)
                                    if unique_sessions >= completion_threshold:
                                        self._move_to_completed_pool(selected_item, category)
                                    
                                    return {
                                        "success": True,
                                        "item": selected_item["id"],
                                        "content_data": selected_item["content_data"],
                                        "update_type": "delete_fallback_to_update",
                                        "category": category,
                                        "memory_updates": update_result["memory_updates"],
                                        "pool_status": selected_item["pool_status"],
                                        "session_count": len(selected_item.get("session_history", [])),
                                        "session_history": selected_item.get("session_history", []),
                                        "completion_threshold": selected_item.get("completion_threshold", 3),
                                        "remaining_data_fields": update_result["remaining_data_fields"],
                                        "operation_converted": "delete_to_update",
                                        "conversion_reason": "Insufficient active elements for delete operation"
                                    }
                                else:
                                    # Both DELETE and UPDATE failed -> convert to NO_MEMORY session
                                    print(f"🔄 Both DELETE and UPDATE failed on {selected_item['id']}, converting to NO_MEMORY session")
                                    return {
                                        "success": True,
                                        "operation_converted": "delete_to_no_memory",
                                        "conversion_reason": f"Both DELETE and UPDATE operations failed for {selected_item['id']} - item in unrecoverable state",
                                        "fallback_session_type": "no_memory",
                                        "category": category,
                                        "item_id": selected_item['id'],
                                        "message": f"Content item {selected_item['id']} reached unrecoverable state. Converting to general conversation.",
                                        "delete_error": delete_result.get("memory_deletes", []),
                                        "update_error": update_result.get("memory_updates", [])
                                    }
                        else:
                            # No draft items available for DELETE - convert to ADD operation
                            print(f"🔄 Converting DELETE to ADD: No draft items in {category}")
                            add_result = self.add_memory_item(session_type, category, session_date=session_date, session_id=session_id)
                            if add_result and add_result.get("success"):
                                # Check if ADD was converted to NO_MEMORY due to exhausted pools
                                if add_result.get("operation_converted") == "add_to_no_memory":
                                    # Both available and draft pools exhausted - convert DELETE to NO_MEMORY
                                    add_result["operation_converted"] = "delete_to_no_memory"
                                    add_result["conversion_reason"] = "No draft items for delete and all pools exhausted"
                                    add_result["original_operation"] = "delete"
                                    return add_result
                                else:
                                    # Normal ADD conversion
                                    add_result["operation_converted"] = "delete_to_add"
                                    add_result["conversion_reason"] = f"No draft items available for deletion in {category}"
                                    add_result["original_operation"] = "delete"
                                    return add_result
                            else:
                                return {"success": False, "reason": f"DELETE conversion to ADD failed in {category}"}
            else:
                return {"success": False, "reason": f"No content metadata found for category: {category}"}
        
        elif session_type == SessionType.GOAL_MEMORY:
            # Goal memory does not support DELETE operations - goals are set and tracked, not deleted
            return {"success": False, "reason": f"DELETE operation not supported for goal memory"}
        
        elif session_type == SessionType.ACTIVITY_MEMORY:
            # For activity memory, select from actual items in memory and return to pool
            if category in self.memory_items[session_type] and self.memory_items[session_type][category]:
                activity_items = self.memory_items[session_type][category]
                
                # For calendar events, filter to only include events that are still valid for modification
                if category == "calendar_event":
                    valid_items = self._filter_valid_calendar_events(activity_items, session_date)
                    if not valid_items:
                        # No valid events to delete - convert to NO_MEMORY session
                        expired_count = len(activity_items)
                        return {
                            "success": True,
                            "operation_converted": "delete_to_no_memory",
                            "conversion_reason": "all_calendar_events_expired",
                            "fallback_session_type": "no_memory",
                            "category": category,
                            "message": f"All {expired_count} calendar events have expired. Converting to general conversation.",
                            "expired_events_count": expired_count
                        }
                    # Use filtered valid events for selection
                    selectable_items = valid_items
                else:
                    # For other activity categories, use all items
                    selectable_items = activity_items
                
                # Select a random item from selectable items
                removed_item = random.choice(selectable_items)
                activity_items.remove(removed_item)
                
                # Handle pool management based on category type
                if category == "todo_list" and isinstance(removed_item, dict):
                    # TODO_LIST: Return to direct list structure
                    task_type = removed_item.get("task_type")
                    description = removed_item.get("description")
                    
                    if task_type and description:
                        # Add back to available pool
                        if (self.activity_available_metadata and 
                            category in self.activity_available_metadata and
                            task_type in self.activity_available_metadata[category]["sample_data"]):
                            
                            available_items = self.activity_available_metadata[category]["sample_data"][task_type]
                            if isinstance(available_items, list) and description not in available_items:
                                available_items.append(description)
                
                elif category == "calendar_event" and isinstance(removed_item, dict):
                    # CALENDAR_EVENT: Return to nested "events" list structure
                    event_type = removed_item.get("event_type")
                    event_name = removed_item.get("event_name")
                    
                    if event_type and event_name:
                        # Add back to available pool
                        if (self.activity_available_metadata and 
                            category in self.activity_available_metadata and
                            event_type in self.activity_available_metadata[category]["sample_data"]):
                            
                            event_data = self.activity_available_metadata[category]["sample_data"][event_type]
                            if isinstance(event_data, dict) and "events" in event_data:
                                available_events = event_data["events"]
                                if isinstance(available_events, list) and event_name not in available_events:
                                    available_events.append(event_name)
                
                elif category in ["food_expenses", "step_tracker"]:
                    # Range-based categories: just remove from active memory
                    # No pool to return to since values are generated from ranges
                    pass
                
                return {"success": True, "item": removed_item, "category": category}
            # No items to remove - convert to NO_MEMORY session
            return {
                "success": True,
                "operation_converted": "delete_to_no_memory",
                "conversion_reason": "no_activity_items_to_delete",
                "fallback_session_type": "no_memory",
                "category": category,
                "message": f"No {category} items to delete. Converting to general conversation."
            }
        
        else:
            # Fallback for unknown session types
            return {"success": False, "reason": f"Unknown session type: {session_type}"}
    
    def update_memory_item(self, session_type: str, category: str, subcategory: str = None, session_date: str = None, session_id: int = None):
        """Update a memory item (for update operations) - supports both preference and value updates."""
        if session_type == SessionType.PREFERENCE_MEMORY:
            # Choose update strategy: 30% preference flip, 70% item replacement (like memory_state_manager)
            update_type = random.choice(["preference_flip"] * 3 + ["item_replacement"] * 7)
            
            if update_type == "preference_flip":
                # Strategy 1: Simple preference flip (like ↔ dislike)
                item_data = self._select_random_active_item_from_category(category, subcategory)
                
                if not item_data:
                    return {"success": False, "reason": f"No active items in {category}"}
                
                item, old_preference, actual_subcategory = item_data
                
                # Flip preference
                new_preference = "dislike" if old_preference == "like" else "like"
                
                # Update in memory
                old_preference_list = f"{old_preference}s"
                new_preference_list = f"{new_preference}s"
                
                # Find and remove the item from old preference list (handle both dict and string formats)
                old_items_list = self.memory_items["preference_memory"][category][actual_subcategory][old_preference_list]
                item_to_move = None
                
                for item_dict in old_items_list:
                    if isinstance(item_dict, dict):
                        if item_dict["item"] == item:
                            item_to_move = item_dict
                            break
                    else:
                        if item_dict == item:  # Legacy format
                            item_to_move = item_dict
                            break
                
                if item_to_move is not None:
                    old_items_list.remove(item_to_move)
                    
                    # Add to new preference list with updated_at timestamp
                    if isinstance(item_to_move, dict):
                        # Update existing dictionary with new timestamp
                        item_to_move["updated_at"] = session_date or "2024-01-01"
                        self.memory_items["preference_memory"][category][actual_subcategory][new_preference_list].append(item_to_move)
                    else:
                        # Convert legacy format to new format
                        new_item_dict = {
                            "item": item_to_move,
                            "created_at": session_date or "2024-01-01",  # Use session_date since we don't know original
                            "updated_at": session_date or "2024-01-01"
                        }
                        self.memory_items["preference_memory"][category][actual_subcategory][new_preference_list].append(new_item_dict)
                
                return {
                    "success": True, 
                    "item": item, 
                    "preference": new_preference, 
                    "old_preference": old_preference,
                    "category": category, 
                    "subcategory": actual_subcategory,
                    "update_type": "preference_update"
                }
                
            else:  # item_replacement
                # Strategy 2: Replace item with new one from same domain/category
                # ONLY work on items they currently LIKE
                item_data = self._select_random_liked_item_from_category(category)
                
                if not item_data:
                    # No liked items to replace - fallback to preference flip
                    fallback_data = self._select_random_active_item_from_category(category, subcategory)
                    if not fallback_data:
                        return {"success": False, "reason": f"No active items in {category}"}
                    
                    item, old_preference, actual_subcategory = fallback_data
                    new_preference = "dislike" if old_preference == "like" else "like"
                    
                    # Update in memory (same logic as preference flip above)
                    old_preference_list = f"{old_preference}s"
                    new_preference_list = f"{new_preference}s"
                    
                    # Find and remove the item from old preference list
                    old_items_list = self.memory_items["preference_memory"][category][actual_subcategory][old_preference_list]
                    item_to_move = None
                    
                    for item_dict in old_items_list:
                        if isinstance(item_dict, dict):
                            if item_dict["item"] == item:
                                item_to_move = item_dict
                                break
                        else:
                            if item_dict == item:  # Legacy format
                                item_to_move = item_dict
                                break
                    
                    if item_to_move is not None:
                        old_items_list.remove(item_to_move)
                        
                        # Add to new preference list with updated_at timestamp
                        if isinstance(item_to_move, dict):
                            item_to_move["updated_at"] = session_date or "2024-01-01"
                            self.memory_items["preference_memory"][category][actual_subcategory][new_preference_list].append(item_to_move)
                        else:
                            new_item_dict = {
                                "item": item_to_move,
                                "created_at": session_date or "2024-01-01",
                                "updated_at": session_date or "2024-01-01"
                            }
                            self.memory_items["preference_memory"][category][actual_subcategory][new_preference_list].append(new_item_dict)
                    
                    return {
                        "success": True, 
                        "item": item, 
                        "preference": new_preference, 
                        "old_preference": old_preference,
                        "category": category, 
                        "subcategory": actual_subcategory,
                        "update_type": "preference_update"
                    }
                
                old_item, actual_subcategory = item_data
                old_preference = "like"  # We know it's a like since we selected from likes only
                
                # Check if there are available items in the same category for replacement
                if (category in self.preference_available_metadata["domains"] and 
                    actual_subcategory in self.preference_available_metadata["domains"][category]["categories"]):
                    subcategory_data = self.preference_available_metadata["domains"][category]["categories"][actual_subcategory]
                    
                    # Get available items list
                    if isinstance(subcategory_data, dict) and "values" in subcategory_data:
                        available_items = subcategory_data["values"]
                    elif isinstance(subcategory_data, list):
                        available_items = subcategory_data
                    else:
                        available_items = []
                    
                    if len(available_items) > 0:
                        # Select new item from available metadata in same domain/category FIRST
                        new_item = random.choice(available_items)
                        available_items.remove(new_item)
                        
                        # Find and remove old liked item from active memory and return to available pool
                        likes_list = self.memory_items["preference_memory"][category][actual_subcategory]["likes"]
                        old_item_dict = None
                        
                        for item_dict in likes_list:
                            if isinstance(item_dict, dict):
                                if item_dict["item"] == old_item:
                                    old_item_dict = item_dict
                                    break
                            else:
                                if item_dict == old_item:  # Legacy format
                                    old_item_dict = item_dict
                                    break
                        
                        if old_item_dict is not None:
                            likes_list.remove(old_item_dict)
                            available_items.append(old_item)
                    else:
                        # No available items for replacement - fallback to preference flip
                        new_preference = "dislike"  # Flip like to dislike
                        
                        # Update in memory (find and move item with timestamp)
                        likes_list = self.memory_items["preference_memory"][category][actual_subcategory]["likes"]
                        dislikes_list = self.memory_items["preference_memory"][category][actual_subcategory]["dislikes"]
                        item_to_move = None
                        
                        for item_dict in likes_list:
                            if isinstance(item_dict, dict):
                                if item_dict["item"] == old_item:
                                    item_to_move = item_dict
                                    break
                            else:
                                if item_dict == old_item:  # Legacy format
                                    item_to_move = item_dict
                                    break
                        
                        if item_to_move is not None:
                            likes_list.remove(item_to_move)
                            
                            # Add to dislikes with updated_at timestamp
                            if isinstance(item_to_move, dict):
                                item_to_move["updated_at"] = session_date or "2024-01-01"
                                dislikes_list.append(item_to_move)
                            else:
                                new_item_dict = {
                                    "item": item_to_move,
                                    "created_at": session_date or "2024-01-01",
                                    "updated_at": session_date or "2024-01-01"
                                }
                                dislikes_list.append(new_item_dict)
                        
                        return {
                            "success": True, 
                            "item": old_item, 
                            "preference": new_preference, 
                            "old_preference": old_preference,
                            "category": category, 
                            "subcategory": actual_subcategory,
                            "update_type": "preference_update"
                        }
                    
                    # Keep as like (replacing one like with another like)
                    new_preference = "like"
                    
                    # Add new item to active memory as like with timestamp
                    new_item_dict = {
                        "item": new_item,
                        "created_at": session_date or "2024-01-01"
                    }
                    self.memory_items["preference_memory"][category][actual_subcategory]["likes"].append(new_item_dict)
                    
                    return {
                        "success": True, 
                        "item": new_item, 
                        "old_item": old_item,
                        "preference": new_preference, 
                        "old_preference": old_preference,
                        "category": category, 
                        "subcategory": actual_subcategory,
                        "update_type": "value_update"
                    }
                    
                else:
                    # No available items for replacement in same category - fallback to preference flip
                    new_preference = "dislike"  # Flip like to dislike
                    
                    # Update in memory (find and move item with timestamp)
                    likes_list = self.memory_items["preference_memory"][category][actual_subcategory]["likes"]
                    dislikes_list = self.memory_items["preference_memory"][category][actual_subcategory]["dislikes"]
                    item_to_move = None
                    
                    for item_dict in likes_list:
                        if isinstance(item_dict, dict):
                            if item_dict["item"] == old_item:
                                item_to_move = item_dict
                                break
                        else:
                            if item_dict == old_item:  # Legacy format
                                item_to_move = item_dict
                                break
                    
                    if item_to_move is not None:
                        likes_list.remove(item_to_move)
                        
                        # Add to dislikes with updated_at timestamp
                        if isinstance(item_to_move, dict):
                            item_to_move["updated_at"] = session_date or "2024-01-01"
                            dislikes_list.append(item_to_move)
                        else:
                            new_item_dict = {
                                "item": item_to_move,
                                "created_at": session_date or "2024-01-01",
                                "updated_at": session_date or "2024-01-01"
                            }
                            dislikes_list.append(new_item_dict)
                    
                    return {
                        "success": True, 
                        "item": old_item, 
                        "preference": new_preference, 
                        "old_preference": old_preference,
                        "category": category, 
                        "subcategory": actual_subcategory,
                        "update_type": "preference_update"
                    }
        
        elif session_type == SessionType.CONTENT_MEMORY:
            # For content memory, check redesigned approach first
            if category in self.content_metadata:
                metadata = self.content_metadata[category]
                use_redesigned = metadata.get("use_redesigned_approach", False)
                
                if use_redesigned and hasattr(self, 'content_pools'):
                    # Use redesigned three-pool approach
                    # Select item from draft pool for update (ONLY look in draft_pool)
                    draft_items = self.content_pools["draft_pool"].get(category, [])
                    
                    if draft_items:
                        item_to_update = random.choice(draft_items)
                        old_content_data = copy.deepcopy(item_to_update["content_data"])
                        
                        # Update by moving elements from _remaining_data to active fields
                        update_result = self._update_content_with_remaining_data(item_to_update, session_date)
                        
                        # Check if UPDATE actually succeeded (has actual updates, not just errors)
                        successful_updates = [u for u in update_result.get("memory_updates", []) if "error" not in u]
                        
                        if successful_updates:
                            # UPDATE succeeded - proceed normally
                            
                            # Track session access for this item
                            if "session_history" not in item_to_update:
                                item_to_update["session_history"] = []
                            
                            item_to_update["session_history"].append({
                                "session_id": session_id,
                                "session_date": session_date or "2024-01-01",
                                "operation": "update"
                            })
                            
                            # Check if we've reached the dynamic session threshold
                            unique_sessions = len(item_to_update["session_history"])
                            completion_threshold = item_to_update.get("completion_threshold", 3)  # Default to 3 for legacy items
                            if unique_sessions >= completion_threshold:
                                self._move_to_completed_pool(item_to_update, category)
                            
                            return {
                                "success": True,
                                "item": item_to_update["id"],
                                "content_data": item_to_update["content_data"],
                                "metadata_source": item_to_update.get("metadata_source"),
                                "remaining_data": item_to_update.get("remaining_data"),
                                "old_content_data": old_content_data,
                                "update_type": "redesigned_list_update",
                                "category": category,
                                "memory_updates": update_result["memory_updates"],  # What goes to memory (clean)
                                "pool_status": item_to_update["pool_status"],
                                "session_count": len(item_to_update.get("session_history", [])),
                                "session_history": item_to_update.get("session_history", []),
                                "completion_threshold": item_to_update.get("completion_threshold", 3),
                                "remaining_data_fields": update_result["remaining_data_fields"]
                            }
                        else:
                            # UPDATE failed due to insufficient remaining data -> try DELETE instead
                            print(f"🔄 UPDATE failed (no remaining data), trying DELETE on {item_to_update['id']}")
                            
                            # Reset the item to pre-update state (since update failed)
                            item_to_update["content_data"] = old_content_data
                            
                            # Try DELETE operation on the same item
                            delete_result = self._delete_content_element_to_remaining_data(item_to_update, session_date)
                            
                            # Check if DELETE succeeded
                            successful_deletes = [d for d in delete_result.get("memory_deletes", []) if "error" not in d]
                            
                            if successful_deletes:
                                # DELETE succeeded as fallback
                                print(f"✅ Fallback DELETE succeeded on {item_to_update['id']}")
                                
                                # Track session access for this item
                                if "session_history" not in item_to_update:
                                    item_to_update["session_history"] = []
                                
                                item_to_update["session_history"].append({
                                    "session_id": session_id,
                                    "session_date": session_date or "2024-01-01",
                                    "operation": "delete"
                                })
                                
                                # Check if we've reached the dynamic session threshold
                                unique_sessions = len(item_to_update["session_history"])
                                completion_threshold = item_to_update.get("completion_threshold", 3)
                                if unique_sessions >= completion_threshold:
                                    self._move_to_completed_pool(item_to_update, category)
                                
                                return {
                                    "success": True,
                                    "item": item_to_update["id"],
                                    "content_data": item_to_update["content_data"],
                                    "category": category,
                                    "pool_status": item_to_update["pool_status"],
                                    "session_count": len(item_to_update.get("session_history", [])),
                                    "session_history": item_to_update.get("session_history", []),
                                    "completion_threshold": item_to_update.get("completion_threshold", 3),
                                    "deletion_method": "update_fallback_to_delete",
                                    "memory_deletes": delete_result["memory_deletes"],
                                    "remaining_data_fields": delete_result["remaining_data_fields"],
                                    "operation_converted": "update_to_delete",
                                    "conversion_reason": "Insufficient remaining data for update operation"
                                }
                            else:
                                # Both UPDATE and DELETE failed -> convert to NO_MEMORY session
                                print(f"🔄 Both UPDATE and DELETE failed on {item_to_update['id']}, converting to NO_MEMORY session")
                                return {
                                    "success": True,
                                    "operation_converted": "update_to_no_memory",
                                    "conversion_reason": f"Both UPDATE and DELETE operations failed for {item_to_update['id']} - item in unrecoverable state",
                                    "fallback_session_type": "no_memory",
                                    "category": category,
                                    "item_id": item_to_update['id'],
                                    "message": f"Content item {item_to_update['id']} reached unrecoverable state. Converting to general conversation.",
                                    "update_error": update_result.get("memory_updates", []),
                                    "delete_error": delete_result.get("memory_deletes", [])
                                }
                    else:
                        # No draft items available for UPDATE - convert to ADD operation
                        print(f"🔄 Converting UPDATE to ADD: No draft items in {category}")
                        add_result = self.add_memory_item(session_type, category, session_date=session_date, session_id=session_id)
                        if add_result and add_result.get("success"):
                            # Check if ADD was converted to NO_MEMORY due to exhausted pools
                            if add_result.get("operation_converted") == "add_to_no_memory":
                                # Both available and draft pools exhausted - convert UPDATE to NO_MEMORY
                                add_result["operation_converted"] = "update_to_no_memory"
                                add_result["conversion_reason"] = "No draft items for update and all pools exhausted"
                                add_result["original_operation"] = "update"
                                return add_result
                            else:
                                # Normal ADD conversion
                                add_result["operation_converted"] = "update_to_add"
                                add_result["conversion_reason"] = f"No draft items available for update in {category}"
                                add_result["original_operation"] = "update"
                                return add_result
                        else:
                            return {"success": False, "reason": f"UPDATE conversion to ADD failed in {category}"}
            else:
                return {"success": False, "reason": f"No content metadata found for category: {category}"}
        
        elif session_type == SessionType.GOAL_MEMORY:
            # Goal memory does not support UPDATE operations - goals are set and tracked, not updated
            return {"success": False, "reason": f"UPDATE operation not supported for goal memory"}
        
        elif session_type == SessionType.ACTIVITY_MEMORY:
            # For activity memory, update existing item with pool management
            if category in self.memory_items[session_type] and self.memory_items[session_type][category]:
                activity_items = self.memory_items[session_type][category]
                
                # For calendar events, filter to only include events that are still valid for modification
                if category == "calendar_event":
                    valid_items = self._filter_valid_calendar_events(activity_items, session_date)
                    if not valid_items:
                        # No valid events to update - convert to NO_MEMORY session
                        expired_count = len(activity_items)
                        return {
                            "success": True,
                            "operation_converted": "update_to_no_memory",
                            "conversion_reason": "all_calendar_events_expired",
                            "fallback_session_type": "no_memory",
                            "category": category,
                            "message": f"All {expired_count} calendar events have expired. Converting to general conversation.",
                            "expired_events_count": expired_count
                        }
                    # Use filtered valid events for selection
                    selectable_items = valid_items
                else:
                    # For other activity categories, use all items
                    selectable_items = activity_items
                
                # Select a random item from selectable items to update
                item_to_update = random.choice(selectable_items)
                old_item = item_to_update.copy() if isinstance(item_to_update, dict) else item_to_update
                
                # Generate new activity data
                new_activity_data = self._generate_activity_data(category, session_date)
                
                # Handle pool management based on category type
                if category == "todo_list" and isinstance(item_to_update, dict):
                    # TODO_LIST: Handle direct list structure
                    # Return old item to pool
                    old_task_type = item_to_update.get("task_type")
                    old_description = item_to_update.get("description")
                    
                    if old_task_type and old_description:
                        # Add old item back to available pool
                        if (self.activity_available_metadata and 
                            category in self.activity_available_metadata and
                            old_task_type in self.activity_available_metadata[category]["sample_data"]):
                            
                            available_items = self.activity_available_metadata[category]["sample_data"][old_task_type]
                            if isinstance(available_items, list) and old_description not in available_items:
                                available_items.append(old_description)
                    
                    # Remove new item from pool
                    new_task_type = new_activity_data.get("task_type")
                    new_description = new_activity_data.get("description")
                    
                    if new_task_type and new_description:
                        if (self.activity_available_metadata and 
                            category in self.activity_available_metadata and
                            new_task_type in self.activity_available_metadata[category]["sample_data"]):
                            
                            available_items = self.activity_available_metadata[category]["sample_data"][new_task_type]
                            if isinstance(available_items, list) and new_description in available_items:
                                available_items.remove(new_description)
                
                elif category == "calendar_event" and isinstance(item_to_update, dict):
                    # CALENDAR_EVENT: For updates, only change the date, keep the same event
                    # No pool management needed since we're not changing the event itself
                    
                    # Keep the same event_type and event_name, only update the date
                    new_activity_data = item_to_update.copy()
                    
                    # Generate a new date for the same event
                    new_date = self._generate_calendar_date()
                    new_activity_data["date"] = new_date
                    
                    # Don't change event_type or event_name - this is just a date update
                
                elif category in ["food_expenses", "step_tracker"]:
                    # Range-based categories: just replace with new generated value
                    # No pool management needed since values are generated from ranges
                    pass
                
                # Add updated_at timestamp to the new activity data
                if isinstance(new_activity_data, dict):
                    new_activity_data["updated_at"] = session_date or "2024-01-01"
                
                # Update the item in active memory
                activity_items[activity_items.index(item_to_update)] = new_activity_data
                
                return {
                    "success": True,
                    "item": new_activity_data,
                    "old_item": old_item,
                    "update_type": "activity_update",
                    "category": category
                }
            # No items to update - convert to NO_MEMORY session
            return {
                "success": True,
                "operation_converted": "update_to_no_memory", 
                "conversion_reason": "no_activity_items_to_update",
                "fallback_session_type": "no_memory",
                "category": category,
                "message": f"No {category} items to update. Converting to general conversation."
            }
        
        else:
            # Fallback for unknown session types
            return {"success": False, "reason": f"Unknown session type: {session_type}"}
    
    def get_snapshot(self) -> dict:
        """Get current memory state snapshot."""
        snapshot = {
            "memory_items": copy.deepcopy(self.memory_items),
            "summary": {
                session_type: {
                    category: self.get_memory_count(session_type, category)
                    for category in categories.keys()
                }
                for session_type, categories in self.memory_items.items()
            }
        }
        
        # Add content pools information if using redesigned approach
        if hasattr(self, 'content_pools') and self.content_pools:
            # Count actual samples in available pool (not just persona entries)
            available_count = 0
            for persona_entry in self.content_pools["available_pool"]:
                if isinstance(persona_entry, dict) and "data" in persona_entry:
                    available_count += len(persona_entry["data"])
                else:
                    available_count += 1  # Fallback for non-dict entries
            
            snapshot["content_pools"] = {
                "available_pool_count": available_count,
                "draft_pool": {
                    category: len(items) for category, items in self.content_pools["draft_pool"].items()
                },
                "completed_pool": {
                    category: len(items) for category, items in self.content_pools["completed_pool"].items()
                }
            }
        
        return snapshot


class CategoryConstraints:
    """Defines frequency constraints for memory categories loaded from memory_categories.json."""
    
    def _parse_memory_categories(self, categories_data: dict, memory_type: str, default_frequency: str = "weekly"):
        """
        Parse memory categories and assign them to appropriate frequency dictionaries.
        
        Args:
            categories_data: Dictionary of category configurations
            memory_type: Type of memory (e.g., "activity", "content", "preference", "goal")
            default_frequency: Default frequency for unknown frequencies
        """
        daily_dict = getattr(self, f"{memory_type}_daily")
        weekly_dict = getattr(self, f"{memory_type}_weekly") 
        monthly_dict = getattr(self, f"{memory_type}_monthly")
        
        for category, info in categories_data.items():
            # Handle subcategory-based categories first (like food_expenses)
            if "subcategory_ranges" in info:
                # Special handling for subcategory-based categories
                subcategory_ranges = info["subcategory_ranges"]
                
                # Handle weekly subcategories
                for subcat, (min_val, max_val) in subcategory_ranges.items():
                    subcat_frequency = info.get("subcategory_frequencies", {}).get(subcat, "daily")
                    if subcat_frequency == "weekly":
                        # Weekly subcategories are handled within the main category
                        # Add to weekly constraints for the main category
                        if category not in weekly_dict:
                            weekly_dict[category] = (0, 0)
                        current_min, current_max = weekly_dict[category]
                        weekly_dict[category] = (current_min + min_val, current_max + max_val)
                
                # For daily subcategories, DON'T sum ranges - let session generation handle dynamically
                # This allows "decide targets first, then generate sessions" approach
                # Mark this category as subcategory-based for special handling in session generation
                daily_dict[f"{category}_subcategory_based"] = True
            
            # Handle traditional frequency-based categories
            elif "frequency" in info:
                frequency = info.get("frequency")
                range_tuple = tuple(info.get("range", [1, 1]))
                
                if frequency == "daily":
                    daily_dict[category] = range_tuple
                elif frequency == "weekly":
                    weekly_dict[category] = range_tuple
                elif frequency == "monthly":
                    monthly_dict[category] = range_tuple
                else:
                    # Future-proof: handle unknown frequencies
                    print(f"Warning: Unknown frequency '{frequency}' for {memory_type} category '{category}', defaulting to {default_frequency}")
                    if default_frequency == "daily":
                        daily_dict[category] = range_tuple
                    elif default_frequency == "weekly":
                        weekly_dict[category] = range_tuple
                    else:
                        monthly_dict[category] = range_tuple
            
            else:
                # Category has neither subcategory_ranges nor frequency
                print(f"Warning: {memory_type.title()} category '{category}' has no frequency or subcategory_ranges, skipping")
                continue

    def __init__(self, memory_categories_data=None):
        """
        Initialize constraints from memory_categories.json.
        
        Args:
            memory_categories_data: Data loaded from memory_categories.json
        """
        if memory_categories_data:
            # Load constraints from memory_categories.json
            self.activity_daily = {}
            self.activity_weekly = {}
            self.activity_monthly = {}
            self.content_daily = {}
            self.content_weekly = {}
            self.content_monthly = {}
            self.preference_daily = {}
            self.preference_weekly = {}
            self.preference_monthly = {}
            self.goal_daily = {}
            self.goal_weekly = {}
            self.goal_monthly = {}
            self.no_memory_per_day = (3, 5)  # Default
            
            # Parse activity memory categories
            activity_categories = memory_categories_data.get("activity_memory", {}).get("options", {})
            self._parse_memory_categories(activity_categories, "activity", default_frequency="daily")
            
            # Parse content memory categories
            content_categories = memory_categories_data.get("content_memory", {}).get("options", {})
            self._parse_memory_categories(content_categories, "content", default_frequency="weekly")
            
            # Parse preference memory categories
            preference_categories = memory_categories_data.get("preference_memory", {}).get("options", {})
            self._parse_memory_categories(preference_categories, "preference", default_frequency="weekly")
            
            # Parse goal memory categories
            goal_categories = memory_categories_data.get("goal_memory", {}).get("options", {})
            self._parse_memory_categories(goal_categories, "goal", default_frequency="weekly")
            
            # Parse no_memory configuration
            no_memory_info = memory_categories_data.get("no_memory", {}).get("options", {}).get("no_category", {})
            if no_memory_info.get("frequency") == "daily":
                self.no_memory_per_day = tuple(no_memory_info.get("range", [3, 5]))
        else:
            # Fallback to hardcoded defaults
            self.activity_daily = {
                "food_expenses": (2, 3),
                "step_tracker": (1, 1),
                "todo_list": (2, 5)
            }
            self.activity_weekly = {"calendar_event": (2, 3)}
            self.content_weekly = {
                "project_proposal": (2, 3),
                "email_writeup": (2, 3),
                "social_media_post": (2, 3),
                "meeting_notes": (2, 3)
            }
            self.preference_weekly = {
                "movies": (2, 4),
                "books": (1, 3),
                "music": (3, 5)
            }
            self.preference_monthly = {"travel": (2, 3)}
            self.no_memory_per_day = (3, 5)


class SimpleSessionSimulator:
    """Session simulator with category-level frequency control and persona support."""
    
    def __init__(self, memory_config_file: str = DEFAULT_MEMORY_CONFIG, use_constraints: bool = True, persona: Optional[str] = None):
        """
        Initialize session simulator with category constraints and persona.
        
        Args:
            memory_config_file: Path to memory configuration JSON file 
                              (e.g., "meta_data/memory_categories.json", 
                                    "meta_data/memory_config_weekly.json",
                                    "meta_data/memory_config_monthly.json")
            use_constraints: Whether to apply category frequency constraints
            persona: Optional persona name for persona-specific behavior (e.g., "software_engineer")
        """
        self.persona = persona
        self.memory_config_file = memory_config_file
        
        # Load metadata only if not already loaded (avoid redundant loading)
        if not METADATA:
            reload_metadata(memory_config_file)
        
        # Use centralized metadata for memory categories
        self.memory_categories = METADATA['memory_categories']
        
        # Load constraints from file
        if use_constraints:
            self.constraints = CategoryConstraints(self.memory_categories)
        else:
            self.constraints = None
        
        # Initialize memory state tracking with memory categories data and persona
        self.memory_state = MemoryState(self.memory_categories, persona=self.persona)
        
        print(f"🎭 Session simulator initialized")
        print(f"📋 Memory config: {memory_config_file}")
        if self.persona:
            print(f"👤 Persona: {self.persona}")
        

    
    def get_dates(self, start_date: str, end_date: str) -> List[str]:
        """Get list of dates between start and end."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        return dates
    
    def get_memory_operation(self, session_type: str, category: str) -> dict:
        """Determine memory operation based on session type and category-specific constraints."""
        if session_type == SessionType.NO_MEMORY:
            return {"operation": None}
        
        # Get category-specific operation constraints from memory_categories.json
        category_info = None
        if session_type == SessionType.ACTIVITY_MEMORY:
            category_info = self.memory_categories.get("activity_memory", {}).get("options", {}).get(category, {})
        elif session_type == SessionType.PREFERENCE_MEMORY:
            category_info = self.memory_categories.get("preference_memory", {}).get("options", {}).get(category, {})
        elif session_type == SessionType.CONTENT_MEMORY:
            category_info = self.memory_categories.get("content_memory", {}).get("options", {}).get(category, {})
        elif session_type == SessionType.GOAL_MEMORY:
            category_info = self.memory_categories.get("goal_memory", {}).get("options", {}).get(category, {})
        
        if category_info:
            # Check if operation_weights are defined
            operation_weights = category_info.get("operation_weights", {})
            allowed_ops = category_info.get("allowed_operations", ["add"])
            
            if operation_weights:
                # Use configured weights for allowed operations only
                weights = {}
                for op in allowed_ops:
                    if op in operation_weights:
                        weights[op] = operation_weights[op]
                
                # Normalize weights to ensure they sum to 1.0
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {op: w/total_weight for op, w in weights.items()}
                else:
                    # Fallback to equal distribution if weights are invalid
                    weights = {op: 1.0/len(allowed_ops) for op in allowed_ops}
            else:
                # Fallback to equal distribution if no weights specified
                weights = {op: 1.0/len(allowed_ops) for op in allowed_ops}
        else:
            # Fallback for any unconfigured categories
            weights = {MemoryOperation.ADD: 1.0}
        
        if not weights:
            return {"operation": MemoryOperation.ADD}
        
        # Select operation based on weights (no memory validation during generation)
        operations = list(weights.keys())
        probabilities = list(weights.values())
        selected_operation = random.choices(operations, weights=probabilities)[0]
        
        return {
            "operation": selected_operation
        }
    
    def validate_and_adjust_operation(self, session: dict) -> str:
        """Validate and adjust operation during execution based on current memory state."""
        session_type = session["type"]
        category = session["category"]
        operation = session["operation"]
        
        if session_type == SessionType.NO_MEMORY or operation is None:
            return operation
        
        # Get category configuration to check requires_existing_for
        category_info = None
        if session_type == SessionType.ACTIVITY_MEMORY:
            category_info = self.memory_categories.get("activity_memory", {}).get("options", {}).get(category, {})
        elif session_type == SessionType.PREFERENCE_MEMORY:
            category_info = self.memory_categories.get("preference_memory", {}).get("options", {}).get(category, {})
        elif session_type == SessionType.CONTENT_MEMORY:
            category_info = self.memory_categories.get("content_memory", {}).get("options", {}).get(category, {})
        elif session_type == SessionType.GOAL_MEMORY:
            category_info = self.memory_categories.get("goal_memory", {}).get("options", {}).get(category, {})
        
        if category_info:
            requires_existing_for = category_info.get("requires_existing_for", [])
            
            # Check if this operation requires existing memory
            if operation in requires_existing_for:
                # For calendar events, we need to check if there are existing events to update/delete
                if category == "calendar_event":
                    has_existing_events = self.memory_state.has_existing_calendar_events()
                    if not has_existing_events:
                        return MemoryOperation.ADD
                # Special handling for preference memory DELETE operations
                # DELETE only works on LIKED items, so check specifically for those
                elif session_type == SessionType.PREFERENCE_MEMORY and operation == MemoryOperation.DELETE:
                    has_liked_items = self.memory_state.has_liked_items_for_category(category)
                    if not has_liked_items:
                        # No liked items available for deletion, convert to ADD
                        return MemoryOperation.ADD
                else:
                    has_memory = self.memory_state.has_memory_for_category(session_type, category)
                    if not has_memory:
                        return MemoryOperation.ADD
        
        return operation
    
    
    
    def generate_weekly_category_requirements(self, week_dates: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Generate required category sessions for one week."""
        weekly_requirements = {date: [] for date in week_dates}
        
        if not self.constraints:
            return weekly_requirements
        
        # Activity weekly requirements
        for category, (min_count, max_count) in self.constraints.activity_weekly.items():
            count = random.randint(min_count, max_count)
            # Distribute sessions across the week (allow multiple sessions per day)
            for _ in range(count):
                selected_date = random.choice(week_dates)
                session_type = SessionType.ACTIVITY_MEMORY
                op_info = self.get_memory_operation(session_type, category)
                session_data = {
                    "type": session_type,
                    "category": category,
                    "operation": op_info["operation"],
                    "_weekly_session": True  # Mark as weekly session
                }
                
                # Weekly sessions will be handled by meal selection logic based on frequency
                
                weekly_requirements[selected_date].append(session_data)
        
        # Content weekly requirements
        for category, (min_count, max_count) in self.constraints.content_weekly.items():
            count = random.randint(min_count, max_count)
            # Distribute sessions across the week (allow multiple sessions per day)
            for _ in range(count):
                selected_date = random.choice(week_dates)
                session_type = SessionType.CONTENT_MEMORY
                op_info = self.get_memory_operation(session_type, category)
                weekly_requirements[selected_date].append({
                    "type": session_type,
                    "category": category,
                    "operation": op_info["operation"]
                })
        
        # Preference weekly requirements
        for category, (min_count, max_count) in self.constraints.preference_weekly.items():
            count = random.randint(min_count, max_count)
            # Distribute sessions across the week (allow multiple sessions per day)
            for _ in range(count):
                selected_date = random.choice(week_dates)
                session_type = SessionType.PREFERENCE_MEMORY
                op_info = self.get_memory_operation(session_type, category)
                weekly_requirements[selected_date].append({
                    "type": session_type,
                    "category": category,
                    "operation": op_info["operation"]
                })
        
        # Goal weekly requirements - SPECIAL HANDLING for evenly spaced weekly goals
        for category, (min_count, max_count) in self.constraints.goal_weekly.items():
            count = random.randint(min_count, max_count)
            
            # For goal memory, create multiple sessions based on count
            # Goals are set on the first day of each week for even spacing
            if count > 0 and week_dates:
                selected_date = week_dates[0]  # First day of this week
                session_type = SessionType.GOAL_MEMORY
                
                # Track used subcategories for this week to avoid duplicates
                used_subcategories = set()
                
                # Create the specified number of goal sessions for this category
                for i in range(count):
                    # Generate a goal with subcategory information
                    goal_result = self._generate_goal_with_unique_subcategory(category, used_subcategories)
                    if goal_result:
                        subcategory = goal_result
                        used_subcategories.add(subcategory)
                        
                        op_info = self.get_memory_operation(session_type, category)
                        weekly_requirements[selected_date].append({
                            "type": session_type,
                            "category": category,
                            "operation": op_info["operation"],
                            "_subcategory_hint": subcategory  # Hint for goal generation
                        })
                    else:
                        # If we can't generate unique subcategories, stop adding more goals
                        print(f"⚠️  Warning: Could only generate {i} unique goals for {category} (requested {count})")
                        break
        
        # SHUFFLE each day's weekly requirements to avoid grouping by type
        for date in week_dates:
            random.shuffle(weekly_requirements[date])
        
        return weekly_requirements
    
    def _generate_goal_with_unique_subcategory(self, category: str, used_subcategories: set):
        """Generate a goal subcategory that hasn't been used this week."""
        if not METADATA['goal_memory']:
            return None
            
        # Find the category in goal metadata
        for goal_metadata in METADATA['goal_memory']:
            if goal_metadata.get("category") == category:
                subcategories = goal_metadata.get("subcategories", {})
                
                # Get available subcategories that haven't been used
                available_subcategories = [sc for sc in subcategories.keys() if sc not in used_subcategories]
                
                if available_subcategories:
                    return random.choice(available_subcategories)
                else:
                    return None  # All subcategories have been used
                
        return None
    
    def generate_monthly_category_requirements(self, all_dates: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Generate required category sessions for monthly constraints across all dates."""
        monthly_requirements = {date: [] for date in all_dates}
        
        if not self.constraints:
            return monthly_requirements
        
        # Preference monthly requirements (like travel)
        for category, (min_count, max_count) in self.constraints.preference_monthly.items():
            count = random.randint(min_count, max_count)
            # Distribute monthly sessions across the entire date range
            selected_dates = random.sample(all_dates, min(count, len(all_dates)))
            for date in selected_dates:
                session_type = SessionType.PREFERENCE_MEMORY
                op_info = self.get_memory_operation(session_type, category)
                monthly_requirements[date].append({
                    "type": session_type,
                    "category": category,
                    "operation": op_info["operation"]
                })
        
        # Goal monthly requirements
        for category, (min_count, max_count) in self.constraints.goal_monthly.items():
            count = random.randint(min_count, max_count)
            # Distribute monthly sessions across the entire date range
            selected_dates = random.sample(all_dates, min(count, len(all_dates)))
            for date in selected_dates:
                session_type = SessionType.GOAL_MEMORY
                op_info = self.get_memory_operation(session_type, category)
                monthly_requirements[date].append({
                    "type": session_type,
                    "category": category,
                    "operation": op_info["operation"]
                })
        
        return monthly_requirements
    
    def execute_memory_operation(self, session: dict):
        """Execute the memory operation and update memory state."""
        # Import copy at the top to avoid UnboundLocalError
        import copy
        
        if session["operation"] is None:
            return
        
        session_type = session["type"]
        category = session["category"]
        session_date = session.get("date")
        
        # Validate and adjust operation based on current memory state
        operation = self.validate_and_adjust_operation(session)
        
        # Update session record with the adjusted operation (important for tracking)
        session["operation"] = operation
        
        # Only execute memory operations for preference_memory type
        # Activity and content memory will be handled by separate components later
        if session_type == SessionType.PREFERENCE_MEMORY:
            # For preference memory, we need to select a random subcategory
            # Use persona-specific subcategories from the actual loaded metadata
            valid_subcategories = []
            
            # Get subcategories from the persona-specific preference metadata that was actually loaded
            if (self.memory_state.preference_original_metadata and 
                "domains" in self.memory_state.preference_original_metadata and
                self.memory_state.preference_original_metadata["domains"] is not None and
                category in self.memory_state.preference_original_metadata["domains"]):
                
                domain_data = self.memory_state.preference_original_metadata["domains"][category]
                if "categories" in domain_data:
                    valid_subcategories = list(domain_data["categories"].keys())
            
            #this part actually never gets called because we are using the preference_metadata_test.json
            #it only gets called if the metadata is courrapted or not loaded properly
            subcategory = None
            if valid_subcategories:
                subcategory = random.choice(valid_subcategories)
            else:
                # Metadata corruption detected - this should never happen in normal operation
                print(f"METADATA CORRUPTION: No valid subcategories for category '{category}' with persona '{self.memory_state.persona}'")
                session["operation_details"] = {
                    "error": "metadata_corruption",
                    "error_type": "no_valid_subcategories",
                    "category": category,
                    "persona": self.memory_state.persona,
                    "available_domains": list(self.memory_state.preference_original_metadata.get("domains", {}).keys()) if (self.memory_state.preference_original_metadata and self.memory_state.preference_original_metadata.get("domains")) else [],
                    "message": "Preference metadata appears to be corrupted. No valid subcategories found for this category."
                }
                return  # Skip this operation entirely
            
            if operation == MemoryOperation.ADD:
                result = self.memory_state.add_memory_item(session_type, category, subcategory, session_date=session_date)
                if result and result.get("success"):
                    # Add operation details to session for tracking
                    session["operation_details"] = {
                        "item": result.get("item"),
                        "preference": result.get("preference"),
                        "subcategory": result.get("subcategory")
                    }
                else:
                    # ADD failed - check if it's due to pool exhaustion
                    failure_reason = result.get("reason", "") if result else "unknown"
                    if "exhausted" in failure_reason.lower() or "no available items" in failure_reason.lower():
                        # Pool exhausted - fallback to DELETE to recycle items
                        print(f"🔄 Pool exhausted for {category}, falling back to DELETE operation")
                        delete_result = self.memory_state.remove_memory_item(session_type, category, None)
                        if delete_result and delete_result.get("success"):
                            # Successfully deleted - update session to reflect actual operation
                            session["operation"] = "delete"  # Update the operation type
                            session["operation_details"] = {
                                "item": delete_result.get("item"),
                                "preference": delete_result.get("preference"),
                                "subcategory": delete_result.get("subcategory"),
                                "fallback_reason": "pool_exhausted_add_to_delete",
                                "original_operation": "add"
                            }
                            print(f"✅ Fallback successful: Deleted '{delete_result.get('item')}' from {category}")
                        else:
                            # Even DELETE failed (no items to delete) - this is a true failure
                            print(f"❌ Fallback failed: No items to delete from {category}")
                            session["operation_details"] = {
                                "error": "fallback_failed",
                                "reason": "Pool exhausted and DELETE fallback failed - no items to delete",
                                "category": category,
                                "original_operation": "add",
                                "fallback_operation": "delete"
                            }
                    else:
                        # ADD failed for non-pool-exhaustion reasons
                        session["operation_details"] = {
                            "error": "add_failed",
                            "reason": failure_reason,
                            "category": category
                        }
            elif operation == MemoryOperation.DELETE:
                # For DELETE, don't specify subcategory - search all subcategories in the category
                result = self.memory_state.remove_memory_item(session_type, category, None)
                if result and result.get("success"):
                    # Add operation details to session for tracking
                    session["operation_details"] = {
                        "item": result.get("item"),
                        "preference": result.get("preference"),
                        "subcategory": result.get("subcategory")
                    }
                else:
                    # DELETE failed - fallback to ADD operation
                    print(f"🔄 DELETE failed for {category}, falling back to ADD operation")
                    add_result = self.memory_state.add_memory_item(session_type, category, subcategory, session_date=session_date)
                    if add_result and add_result.get("success"):
                        session["operation"] = MemoryOperation.ADD  # Update the operation type
                        session["operation_details"] = {
                            "item": add_result.get("item"),
                            "preference": add_result.get("preference"),
                            "subcategory": add_result.get("subcategory"),
                            "fallback_reason": "delete_failed_no_valid_items"
                        }
            elif operation == MemoryOperation.UPDATE:
                # For UPDATE, use the selected subcategory to update within the same subcategory (e.g., actor → actor)
                result = self.memory_state.update_memory_item(session_type, category, None, session_date=session_date)
                if result and result.get("success"):
                    # Add operation details to session for tracking
                    operation_details = {
                        "item": result.get("item"),
                        "preference": result.get("preference"),
                        "subcategory": result.get("subcategory"),
                        "update_type": result.get("update_type")
                    }
                    
                    # Add old values for update operations
                    if result.get("old_preference"):
                        operation_details["old_preference"] = result.get("old_preference")
                    if result.get("old_item"):
                        operation_details["old_item"] = result.get("old_item")
                    
                    session["operation_details"] = operation_details
                else:
                    # UPDATE failed - fallback to ADD operation
                    print(f"🔄 UPDATE failed for {category}, falling back to ADD operation")
                    add_result = self.memory_state.add_memory_item(session_type, category, subcategory, session_date=session_date)
                    if add_result and add_result.get("success"):
                        session["operation"] = MemoryOperation.ADD  # Update the operation type
                        session["operation_details"] = {
                            "item": add_result.get("item"),
                            "preference": add_result.get("preference"),
                            "subcategory": add_result.get("subcategory"),
                            "fallback_reason": "update_failed_no_valid_items"
                        }
        
        elif session_type == SessionType.CONTENT_MEMORY:
            # For content memory, execute three-pool system operations
            if operation == MemoryOperation.ADD:
                # ADD: Move from available pool → draft pool with partial content
                result = self.memory_state.add_memory_item(session_type, category, session_date=session_date, session_id=session.get("id"))
                if result and result.get("success"):
                    # Add operation details to session for tracking  
                    # Content data is already clean (no embedded metadata)
                    # Make deep copies to prevent reference sharing with memory state
                    session["operation_details"] = {
                        "item": result.get("item"),
                        "content_data": copy.deepcopy(result.get("content_data")),
                        "pool_status": result.get("pool_status", "unknown"),
                        "generation_method": result.get("generation_method", "unknown")
                    }
                    
                    # Handle ADD operations converted to UPDATE
                    if result.get("generation_method") == "add_converted_to_update":
                        session["operation_details"]["operation_converted"] = result.get("operation_converted")
                        session["operation_details"]["conversion_reason"] = result.get("conversion_reason")
                        session["operation_details"]["memory_updates"] = result.get("memory_updates", [])
                        session["operation_details"]["session_count"] = result.get("session_count", 0)
                        
                        # CRITICAL: Update the operation field to "update"
                        session["operation"] = "update"
                        
                        # Check if item moved to completed pool
                        if result.get("pool_status") == "completed":
                            session["operation_details"]["moved_to_completed"] = True
                            session["operation_details"]["total_sessions_before_completion"] = result.get("session_count", 0)
                            session["operation_details"]["completion_threshold"] = result.get("completion_threshold", 3)
                            session["operation_details"]["session_history"] = result.get("session_history", [])
                    
                    # Handle ADD operations converted to NO_MEMORY
                    elif result.get("generation_method") == "add_converted_to_no_memory":
                        session["operation_details"]["operation_converted"] = result.get("operation_converted")
                        session["operation_details"]["conversion_reason"] = result.get("conversion_reason")
                        session["operation_details"]["fallback_session_type"] = result.get("fallback_session_type")
                        session["operation_details"]["message"] = result.get("message")
                        session["operation_details"]["pool_status"] = result.get("pool_status")
                        
                        # Convert the session type to NO_MEMORY
                        session["type"] = "no_memory"
                        session["category"] = None
                        session["operation"] = None
                        
                    
                    # Add three-pool specific details for regular ADD operations
                    elif result.get("generation_method") == "redesigned_three_pool":
                        session["operation_details"]["partial_content_generated"] = True
                else:
                    # Handle ADD failure (no available samples)
                    session["operation_details"] = {
                        "error": result.get("reason", "Unknown error") if result else "No result returned"
                    }
                    
                    # Include pool status for debugging when ADD fails
                    if result and result.get("pool_status"):
                        session["operation_details"]["pool_status"] = result.get("pool_status")
                    
            elif operation == MemoryOperation.UPDATE:
                # UPDATE: Modify draft pool items by adding remaining data or modifying lists
                result = self.memory_state.update_memory_item(session_type, category, session_date=session_date, session_id=session.get("id"))
                if result and result.get("success"):
                    # Add operation details to session for tracking
                    # Content data is already clean (no embedded metadata)
                    # Make deep copies to prevent reference sharing with memory state
                    session["operation_details"] = {
                        "item": result.get("item"),
                        "content_data": copy.deepcopy(result.get("content_data")),
                        "update_type": result.get("update_type", "unknown"),
                        "pool_status": result.get("pool_status", "unknown")
                    }
                    
                    # Add three-pool specific details for successful UPDATE operations
                    if result.get("update_type") == "redesigned_list_update":
                        # Clean memory updates for operation_details (what actually goes to memory)
                        session["operation_details"]["memory_updates"] = result.get("memory_updates", [])
                        
                        # Check if item moved to completed pool  
                        if result.get("pool_status") == "completed":
                            session["operation_details"]["moved_to_completed"] = True
                            session["operation_details"]["total_sessions_before_completion"] = result.get("session_count", 0)
                            session["operation_details"]["completion_threshold"] = result.get("completion_threshold", 3)
                            session["operation_details"]["session_history"] = result.get("session_history", [])
                    
                    # Handle UPDATE operations that were converted to DELETE due to insufficient remaining data
                    elif result.get("operation_converted") == "update_to_delete":
                        # UPDATE failed but DELETE succeeded as fallback
                        session["operation_details"]["operation_converted"] = result.get("operation_converted")
                        session["operation_details"]["conversion_reason"] = result.get("conversion_reason")
                        session["operation_details"]["memory_deletes"] = result.get("memory_deletes", [])
                        session["operation_details"]["deletion_method"] = result.get("deletion_method")
                        
                        # Update session to reflect what actually happened (DELETE)
                        session["operation"] = "delete"  # Change operation type to match what was performed
                    
                    # Handle UPDATE operations that were converted to NO_MEMORY due to unrecoverable state
                    elif result.get("operation_converted") == "update_to_no_memory":
                        # Both UPDATE and DELETE failed - convert to NO_MEMORY session
                        session["operation_details"]["operation_converted"] = result.get("operation_converted")
                        session["operation_details"]["conversion_reason"] = result.get("conversion_reason")
                        session["operation_details"]["fallback_session_type"] = result.get("fallback_session_type")
                        session["operation_details"]["message"] = result.get("message")
                        session["operation_details"]["item_id"] = result.get("item_id")
                        session["operation_details"]["update_error"] = result.get("update_error", [])
                        session["operation_details"]["delete_error"] = result.get("delete_error", [])
                        
                        # Convert the session type to NO_MEMORY
                        session["type"] = "no_memory"
                        session["category"] = None
                        session["operation"] = "no_memory"
                        
                        # Check if item moved to completed pool during DELETE fallback
                        if result.get("pool_status") == "completed":
                            session["operation_details"]["moved_to_completed"] = True
                            session["operation_details"]["total_sessions_before_completion"] = result.get("session_count", 0)
                            session["operation_details"]["completion_threshold"] = result.get("completion_threshold", 3)
                            session["operation_details"]["session_history"] = result.get("session_history", [])
                        
                else:
                    # Check if this was a converted UPDATE operation
                    if result and result.get("operation_converted") == "update_to_add":
                        # Handle UPDATE converted to ADD
                        session["operation_details"] = {
                            "item": result.get("item"),
                            "content_data": copy.deepcopy(result.get("content_data")),
                            "pool_status": result.get("pool_status", "unknown"),
                            "generation_method": result.get("generation_method", "unknown"),
                            "operation_converted": result.get("operation_converted"),
                            "conversion_reason": result.get("conversion_reason"),
                            "original_operation": result.get("original_operation")
                        }
                        
                        # Update session to reflect the conversion
                        session["operation"] = "add"  # Change operation type
                    elif result and result.get("operation_converted") == "update_to_no_memory":
                        # Handle UPDATE converted to NO_MEMORY due to exhausted pools
                        session["operation_details"] = {
                            "operation_converted": result.get("operation_converted"),
                            "conversion_reason": result.get("conversion_reason"),
                            "fallback_session_type": result.get("fallback_session_type"),
                            "message": result.get("message"),
                            "pool_status": result.get("pool_status"),
                            "original_operation": result.get("original_operation")
                        }
                        
                        # Convert the session type to NO_MEMORY
                        session["type"] = "no_memory"
                        session["category"] = None
                        session["operation"] = None
                    else:
                        # Handle UPDATE failure (no draft items available and conversion failed)
                        session["operation_details"] = {
                            "error": result.get("reason", "Unknown error") if result else "No result returned"
                        }
                    
            elif operation == MemoryOperation.DELETE:
                # DELETE: Move element from actual field back to remaining data
                result = self.memory_state.remove_memory_item(session_type, category, session_date=session_date, session_id=session.get("id"))
                if result and result.get("success"):
                    # Add operation details to session for tracking
                    # Content data is already clean (no embedded metadata)
                    # Make deep copies to prevent reference sharing with memory state
                    session["operation_details"] = {
                        "item": result.get("item"),
                        "content_data": copy.deepcopy(result.get("content_data")),
                        "pool_status": result.get("pool_status", "unknown"),
                        "deletion_method": result.get("deletion_method", "unknown")
                    }
                    
                    # Add three-pool specific details for successful DELETE operations
                    if result.get("deletion_method") == "redesigned_element_deletion":
                        # Clean memory deletes for operation_details (what was actually removed from memory)
                        session["operation_details"]["memory_deletes"] = result.get("memory_deletes", [])
                        
                        # Check if item moved to completed pool during DELETE operation
                        if result.get("pool_status") == "completed":
                            session["operation_details"]["moved_to_completed"] = True
                            session["operation_details"]["total_sessions_before_completion"] = result.get("session_count", 0)
                            session["operation_details"]["completion_threshold"] = result.get("completion_threshold", 3)
                            session["operation_details"]["session_history"] = result.get("session_history", [])
                    
                    # Handle DELETE operations that were converted to UPDATE due to insufficient active elements
                    elif result.get("operation_converted") == "delete_to_update":
                        # DELETE failed but UPDATE succeeded as fallback
                        session["operation_details"]["operation_converted"] = result.get("operation_converted")
                        session["operation_details"]["conversion_reason"] = result.get("conversion_reason")
                        session["operation_details"]["memory_updates"] = result.get("memory_updates", [])
                        session["operation_details"]["update_type"] = result.get("update_type")
                        
                        # Update session to reflect what actually happened (UPDATE)
                        session["operation"] = "update"  # Change operation type to match what was performed
                    
                    # Handle DELETE operations that were converted to NO_MEMORY due to unrecoverable state
                    elif result.get("operation_converted") == "delete_to_no_memory":
                        # Both DELETE and UPDATE failed - convert to NO_MEMORY session
                        session["operation_details"]["operation_converted"] = result.get("operation_converted")
                        session["operation_details"]["conversion_reason"] = result.get("conversion_reason")
                        session["operation_details"]["fallback_session_type"] = result.get("fallback_session_type")
                        session["operation_details"]["message"] = result.get("message")
                        session["operation_details"]["item_id"] = result.get("item_id")
                        session["operation_details"]["delete_error"] = result.get("delete_error", [])
                        session["operation_details"]["update_error"] = result.get("update_error", [])
                        
                        # Convert the session type to NO_MEMORY
                        session["type"] = "no_memory"
                        session["category"] = None
                        session["operation"] = "no_memory"
                        
                        # Check if item moved to completed pool during UPDATE fallback
                        if result.get("pool_status") == "completed":
                            session["operation_details"]["moved_to_completed"] = True
                            session["operation_details"]["total_sessions_before_completion"] = result.get("session_count", 0)
                            session["operation_details"]["completion_threshold"] = result.get("completion_threshold", 3)
                            session["operation_details"]["session_history"] = result.get("session_history", [])
                        
                else:
                    # Check if this was a converted DELETE operation
                    if result and result.get("operation_converted") == "delete_to_add":
                        # Handle DELETE converted to ADD
                        # Restructure content_data to move metadata outside
                        original_content_data = copy.deepcopy(result.get("content_data"))
                        clean_content_data, metadata_source, remaining_data = self.memory_state._restructure_content_data(original_content_data)
                        
                        session["operation_details"] = {
                            "item": result.get("item"),
                            "content_data": clean_content_data,
                            "pool_status": result.get("pool_status", "unknown"),
                            "generation_method": result.get("generation_method", "unknown"),
                            "operation_converted": result.get("operation_converted"),
                            "conversion_reason": result.get("conversion_reason"),
                            "original_operation": result.get("original_operation")
                        }
                        
                        # Update session to reflect the conversion
                        session["operation"] = "add"  # Change operation type
                    elif result and result.get("operation_converted") == "delete_to_no_memory":
                        # Handle DELETE converted to NO_MEMORY due to exhausted pools
                        session["operation_details"] = {
                            "operation_converted": result.get("operation_converted"),
                            "conversion_reason": result.get("conversion_reason"),
                            "fallback_session_type": result.get("fallback_session_type"),
                            "message": result.get("message"),
                            "pool_status": result.get("pool_status"),
                            "original_operation": result.get("original_operation")
                        }
                        
                        # Convert the session type to NO_MEMORY
                        session["type"] = "no_memory"
                        session["category"] = None
                        session["operation"] = None
                    else:
                        # Handle DELETE failure (no draft items to delete and conversion failed)
                        session["operation_details"] = {
                            "error": result.get("reason", "Unknown error") if result else "No result returned"
                        }
        
        elif session_type == SessionType.ACTIVITY_MEMORY:
            # For activity memory, execute simple list-based operations
            if operation == MemoryOperation.ADD:
                # Generate simple activity data based on category
                # For food_expenses, check if this is a weekly session
                predetermined_meal = session.get("predetermined_meal")
                is_weekly_session = session.get("_weekly_session", False)
                meal_targets = session.get("_meal_targets")  # Pre-decided targets from session generation
                
                # If we have pre-decided meal targets, initialize the daily meal state with them
                if meal_targets and category == "food_expenses":
                    self.memory_state._initialize_daily_meal_state(session_date, meal_targets)
                
                activity_data = self.memory_state._generate_activity_data(category, session_date, predetermined_meal, is_weekly_session)
                
                # Note: activity_data should never be None with "decide targets first" approach
                
                result = self.memory_state.add_memory_item(session_type, category, item=activity_data, session_date=session_date)
                if result and result.get("success"):
                    # Check if ADD was converted to NO_MEMORY due to exhausted pools
                    if result.get("operation_converted") == "add_to_no_memory":
                        # Activity pool exhausted - convert to NO_MEMORY session
                        session["operation_details"] = {
                            "operation_converted": result.get("operation_converted"),
                            "conversion_reason": result.get("conversion_reason"),
                            "fallback_session_type": result.get("fallback_session_type"),
                            "message": result.get("message"),
                            "category": category
                        }
                        
                        # Convert the session type to NO_MEMORY
                        session["type"] = "no_memory"
                        session["category"] = None
                        session["operation"] = None
                    else:
                        # Normal ADD operation
                        session["operation_details"] = {
                            "item": result.get("item"),
                            "category": category
                        }
                else:
                    # ADD failed - check if it's due to pool exhaustion
                    failure_reason = result.get("reason", "") if result else "unknown"
                    if "exhausted" in failure_reason.lower() or "no available items" in failure_reason.lower():
                        # Pool exhausted - fallback to DELETE to recycle items
                        print(f"🔄 Pool exhausted for {category}, falling back to DELETE operation")
                        delete_result = self.memory_state.remove_memory_item(session_type, category, session_date=session_date)
                        if delete_result and delete_result.get("success"):
                            # Successfully deleted - update session to reflect actual operation
                            session["operation"] = "delete"  # Update the operation type
                            session["operation_details"] = {
                                "item": delete_result.get("item"),
                                "category": category,
                                "fallback_reason": "pool_exhausted_add_to_delete",
                                "original_operation": "add"
                            }
                            print(f"✅ Fallback successful: Deleted '{delete_result.get('item')}' from {category}")
                        else:
                            # Even DELETE failed (no items to delete) - this is a true failure
                            print(f"❌ Fallback failed: No items to delete from {category}")
                            session["operation_details"] = {
                                "error": "fallback_failed",
                                "reason": "Pool exhausted and DELETE fallback failed - no items to delete",
                                "category": category,
                                "original_operation": "add",
                                "fallback_operation": "delete"
                            }
                    else:
                        # ADD failed for non-pool-exhaustion reasons
                        session["operation_details"] = {
                            "error": "add_failed",
                            "reason": failure_reason,
                            "category": category
                        }
            elif operation == MemoryOperation.DELETE:
                result = self.memory_state.remove_memory_item(session_type, category, session_date=session_date)
                if result and result.get("success"):
                    # Check if DELETE was converted to NO_MEMORY due to no items
                    if result.get("operation_converted") == "delete_to_no_memory":
                        # No activity items to delete - convert to NO_MEMORY session
                        session["operation_details"] = {
                            "operation_converted": result.get("operation_converted"),
                            "conversion_reason": result.get("conversion_reason"),
                            "fallback_session_type": result.get("fallback_session_type"),
                            "message": result.get("message"),
                            "category": category
                        }
                        
                        # Add expired events count if available
                        if result.get("expired_events_count"):
                            session["operation_details"]["expired_events_count"] = result.get("expired_events_count")
                        
                        # Convert the session type to NO_MEMORY
                        session["type"] = "no_memory"
                        session["category"] = None
                        session["operation"] = None
                    else:
                        # Normal DELETE operation
                        session["operation_details"] = {
                            "item": result.get("item"),
                            "category": category,
                        }
            elif operation == MemoryOperation.UPDATE:
                result = self.memory_state.update_memory_item(session_type, category, session_date=session_date)
                if result and result.get("success"):
                    # Check if UPDATE was converted to NO_MEMORY due to no items
                    if result.get("operation_converted") == "update_to_no_memory":
                        # No activity items to update - convert to NO_MEMORY session
                        session["operation_details"] = {
                            "operation_converted": result.get("operation_converted"),
                            "conversion_reason": result.get("conversion_reason"),
                            "fallback_session_type": result.get("fallback_session_type"),
                            "message": result.get("message"),
                            "category": category
                        }
                        
                        # Add expired events count if available
                        if result.get("expired_events_count"):
                            session["operation_details"]["expired_events_count"] = result.get("expired_events_count")
                        
                        # Convert the session type to NO_MEMORY
                        session["type"] = "no_memory"
                        session["category"] = None
                        session["operation"] = None
                    else:
                        # Normal UPDATE operation
                        session["operation_details"] = {
                            "item": result.get("item", "updated_item"),
                            "category": category
                        }
            
            # Clean up internal activity memory fields from session
            session.pop("_meal_targets", None)
            session.pop("_weekly_session", None)
        
        elif session_type == SessionType.GOAL_MEMORY:
            # For goal memory, only ADD operations are allowed (but internally may be ADD or UPDATE)
            if operation == MemoryOperation.ADD:
                # Check if there's a subcategory hint from weekly scheduling
                subcategory_hint = session.get("_subcategory_hint")
                result = self.memory_state.add_memory_item(session_type, category, session_date=session_date, subcategory_hint=subcategory_hint)
                if result and result.get("success"):
                    # Add operation details to session for tracking
                    session["operation_details"] = {
                        "subcategory": result.get("subcategory"),
                        "item": result.get("item"),
                        "actual_operation": result.get("operation_performed", "add")
                    }
                    
                    # If it was an update, add old value for reference
                    if result.get("operation_performed") == "update":
                        session["operation_details"]["old_value"] = result.get("old_value")
                    
                    # Clean up the hint from the session
                    session.pop("_subcategory_hint", None)
        
        # For no_memory: no operations needed
    
    def generate(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Generate session types with categories for date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict with dates and their sessions (type + category)
        """
        dates = self.get_dates(start_date, end_date)
        
        result = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": len(dates),
            "persona": self.persona,
            "sessions": []
        }
        
        # Generate sessions and execute memory operations in ID order
        all_sessions = []
        session_id_counter = 1
        i = 0
        
        # Generate monthly requirements for the entire date range first
        monthly_requirements = self.generate_monthly_category_requirements(dates)
        
        # First, generate all sessions with IDs
        while i < len(dates):
            # Get current week (up to 7 days)
            week_end = min(i + 7, len(dates))
            week_dates = dates[i:week_end]
            
            # Get weekly requirements for categories
            weekly_requirements = self.generate_weekly_category_requirements(week_dates)
            
            # Generate sessions for each day in the week
            for date_str in week_dates:
                # Combine weekly and monthly requirements
                weekly_sessions = weekly_requirements.get(date_str, [])
                monthly_sessions = monthly_requirements.get(date_str, [])
                required_sessions = weekly_sessions + monthly_sessions
                
                sessions = self.generate_daily_sessions(required_sessions)
                
                # Assign session IDs and dates
                for session in sessions:
                    session["id"] = session_id_counter
                    session["date"] = date_str
                    session_id_counter += 1
                    all_sessions.append(session)
            
            i = week_end
        
        # Sort all sessions by ID to ensure proper execution order (ID 1 first)
        all_sessions.sort(key=lambda x: x["id"])
        
        # Execute memory operations in strict ID order
        for session in all_sessions:
            # Get memory state before operation
            memory_before = self.memory_state.get_snapshot()
            session["memory_state_before"] = memory_before
            
            # Execute memory operation to update state
            self.execute_memory_operation(session)
            
            # Get memory state after operation
            memory_after = self.memory_state.get_snapshot()
            session["memory_state_after"] = memory_after
            
            result["sessions"].append(session)
        
        # Add summary statistics
        result["summary"] = self.generate_summary(result["sessions"])
        
        # Create timestamped output folder to prevent data loss from multiple runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder_name = f"sessions_{timestamp}_{self.persona}" if self.persona else f"sessions_{timestamp}"
        
        # Ensure main output directory exists
        main_output_dir = Path("output")
        main_output_dir.mkdir(exist_ok=True)
        
        # Create timestamped subfolder
        output_folder = main_output_dir / output_folder_name
        output_folder.mkdir(exist_ok=True)
        
        # Add output folder information to result metadata
        result["output_folder"] = str(output_folder_name)
        result["generation_timestamp"] = timestamp
        
        # Extract memory states for separate file
        self.save_memory_states_separately(result["sessions"], output_folder)
        
        # Clean up memory states from main sessions file to keep it lightweight
        result = self.clean_sessions_data(result)
        
        # Save main sessions file to timestamped folder
        sessions_file = output_folder / "new_sessions.json"
        with open(sessions_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"📁 All session data saved to folder: {output_folder}")
        print(f"📄 Main sessions file: {sessions_file}")
        
        # Store the output folder path for return
        result["_output_folder_path"] = str(output_folder)
        
        return result
    
    def clean_sessions_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove memory states from sessions data to keep it lightweight."""
        # Create a clean copy without memory states
        clean_result = result.copy()
        clean_sessions = []
        
        for session in result["sessions"]:
            clean_session = {k: v for k, v in session.items() 
                           if k not in ["memory_state_before", "memory_state_after"]}
            clean_sessions.append(clean_session)
        
        clean_result["sessions"] = clean_sessions
        return clean_result
    
    def save_memory_states_separately(self, sessions: List[Dict[str, Any]], output_folder: Path):
        """Save memory states in a separate file organized by session ID."""
        memory_states = {
            "metadata": {
                "description": "Memory states after each session",
                "total_sessions": len(sessions),
                "persona": self.persona,
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "output_folder": str(output_folder.name)
            },
            "memory_states": {}
        }
        
        for session in sessions:
            session_id = session["id"]
            memory_states["memory_states"][str(session_id)] = {
                "session_id": session_id,
                "session_date": session["date"],
                "session_type": session["type"],
                "session_category": session["category"],
                "operation_performed": session["operation"],
                "operation_details": session.get("operation_details"),
                "memory_state_after_session": session["memory_state_after"]
            }
        
        # Save to separate file in timestamped output directory
        memory_states_file = output_folder / "memory_states_by_session.json"
        with open(memory_states_file, 'w') as f:
            json.dump(memory_states, f, indent=2)
        
        print(f"💾 Memory states saved to: {memory_states_file}")
        
        # Also create a summary file with memory counts and operation statistics
        memory_summary = {
            "metadata": {
                "description": "Summary of memory counts and operation statistics",
                "total_sessions": len(sessions),
                "persona": self.persona,
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "output_folder": str(output_folder.name)
            },
            "operation_statistics": self._generate_operation_statistics(sessions),
            "memory_evolution": {}
        }
        
        for session in sessions:
            session_id = session["id"]
            memory_after = session["memory_state_after"]
            memory_summary["memory_evolution"][str(session_id)] = {
                "session_id": session_id,
                "session_date": session["date"],
                "session_type": session["type"],
                "session_category": session["category"],
                "operation": session["operation"],
                "memory_counts": memory_after["summary"]
            }
        
        summary_file = output_folder / "memory_evolution_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(memory_summary, f, indent=2)
        
        print(f"📊 Memory evolution summary saved to: {summary_file}")
    
    def _generate_operation_statistics(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive operation statistics by memory type and category.
        
        Returns:
            Dict containing operation counts (add, update, delete) for each memory type and category
        """
        stats = {
            "overview": {
                "total_operations": 0,
                "add_operations": 0,
                "update_operations": 0,
                "delete_operations": 0,
                "no_memory_sessions": 0
            },
            "by_memory_type": {
                "preference_memory": {"total": 0, "add": 0, "update": 0, "delete": 0, "by_category": {}},
                "activity_memory": {"total": 0, "add": 0, "update": 0, "delete": 0, "by_category": {}},
                "content_memory": {"total": 0, "add": 0, "update": 0, "delete": 0, "by_category": {}},
                "goal_memory": {"total": 0, "add": 0, "update": 0, "delete": 0, "by_category": {}}
            }
        }
        
        # Track operations including conversions
        for session in sessions:
            session_type = session.get("type")
            category = session.get("category")
            operation = session.get("operation")
            operation_details = session.get("operation_details", {})
            
            # Skip no_memory sessions
            if session_type == "no_memory" or operation is None:
                stats["overview"]["no_memory_sessions"] += 1
                continue
            
            # Check if operation was converted
            converted_operation = operation_details.get("operation_converted")
            if converted_operation:
                # Handle converted operations (e.g., "add_to_update", "delete_to_add", etc.)
                if "to_no_memory" in converted_operation:
                    # Operation converted to no_memory, don't count as operation
                    stats["overview"]["no_memory_sessions"] += 1
                    continue
                elif converted_operation in ["add_to_update", "delete_fallback_to_update", "update_fallback_to_delete"]:
                    # Operation was converted but still executed - use the final operation
                    pass  # operation variable already reflects the final operation
            
            # Count total operations
            stats["overview"]["total_operations"] += 1
            
            # Count by operation type
            if operation == "add":
                stats["overview"]["add_operations"] += 1
            elif operation == "update":
                stats["overview"]["update_operations"] += 1
            elif operation == "delete":
                stats["overview"]["delete_operations"] += 1
            
            # Count by memory type
            if session_type in stats["by_memory_type"]:
                memory_type_stats = stats["by_memory_type"][session_type]
                memory_type_stats["total"] += 1
                
                if operation == "add":
                    memory_type_stats["add"] += 1
                elif operation == "update":
                    memory_type_stats["update"] += 1
                elif operation == "delete":
                    memory_type_stats["delete"] += 1
                
                # Count by category within memory type
                if category:
                    if category not in memory_type_stats["by_category"]:
                        memory_type_stats["by_category"][category] = {
                            "total": 0,
                            "add": 0,
                            "update": 0,
                            "delete": 0
                        }
                    
                    category_stats = memory_type_stats["by_category"][category]
                    category_stats["total"] += 1
                    
                    if operation == "add":
                        category_stats["add"] += 1
                    elif operation == "update":
                        category_stats["update"] += 1
                    elif operation == "delete":
                        category_stats["delete"] += 1
        
        return stats
    
    def generate_summary(self, sessions_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate summary statistics for all sessions."""
        summary = {
            "total_sessions": len(sessions_list),
            "session_types": {}
        }
        
        # Count session types
        for session in sessions_list:
            session_type = session["type"]
            summary["session_types"][session_type] = summary["session_types"].get(session_type, 0) + 1
        
        return summary
    
    def _determine_daily_meal_sequence(self, count: int) -> List[str]:
        """
        Determine the meal sequence for food_expenses sessions in chronological order.
        
        Args:
            count: Number of food expense sessions to generate
            
        Returns:
            List[str]: Meal types in chronological order
        """
        # Get subcategory ranges to see what's enabled
        if hasattr(self.memory_state, '_get_food_subcategory_ranges'):
            ranges = self.memory_state._get_food_subcategory_ranges()
        else:
            ranges = {}
        
        # Build list of enabled meals in chronological order
        enabled_meals = []
        chronological_order = ["breakfast", "lunch", "dinner", "coffee", "grocery"]
        
        for meal in chronological_order:
            if ranges.get(meal, [0, 0])[1] > 0:
                enabled_meals.append(meal)
        
        # Generate sequence based on count
        meal_sequence = []
        
        # First, add main meals in order
        main_meals = [m for m in enabled_meals if m in ["breakfast", "lunch", "dinner"]]
        for meal in main_meals:
            if len(meal_sequence) < count:
                meal_sequence.append(meal)
        
        # Fill remaining slots with flexible items (coffee, grocery)
        flexible_meals = [m for m in enabled_meals if m in ["coffee", "grocery"]]
        while len(meal_sequence) < count and flexible_meals:
            # Add coffee up to reasonable limit, then grocery
            if "coffee" in flexible_meals and meal_sequence.count("coffee") < 2:
                meal_sequence.append("coffee")
            elif "grocery" in flexible_meals and "grocery" not in meal_sequence:
                meal_sequence.append("grocery")
            elif flexible_meals:
                # Fallback: add any flexible meal
                meal_sequence.append(flexible_meals[0])
            else:
                break
        
        # If still need more, repeat main meals (shouldn't happen with proper constraints)
        while len(meal_sequence) < count and enabled_meals:
            meal_sequence.append(enabled_meals[0])
        
        return meal_sequence
    
    def _decide_daily_meal_targets(self, category: str) -> Dict[str, int]:
        """
        Decide daily meal targets for subcategory-based categories like food_expenses.
        This implements the "decide targets first, then generate sessions" approach.
        
        Args:
            category: Category name (e.g., "food_expenses")
            
        Returns:
            Dict mapping meal type to target count (e.g., {"breakfast": 1, "lunch": 0, "coffee": 2})
        """
        import random
        
        # Get subcategory ranges and frequencies from memory_categories.json
        if not hasattr(self.memory_state, '_get_food_subcategory_ranges'):
            return {}
        
        subcategory_ranges = self.memory_state._get_food_subcategory_ranges()
        
        # Get frequencies from metadata
        memory_categories_data = METADATA.get('memory_categories', {})
        category_config = memory_categories_data.get('activity_memory', {}).get('options', {}).get(category, {})
        subcategory_frequencies = category_config.get('subcategory_frequencies', {})
        
        # Decide targets for each daily subcategory
        meal_targets = {}
        
        for meal_type, (min_val, max_val) in subcategory_ranges.items():
            frequency = subcategory_frequencies.get(meal_type, 'daily')
            
            if frequency == 'daily' and max_val > 0:
                # Randomly decide target within range
                meal_targets[meal_type] = random.randint(min_val, max_val)
            else:
                # Weekly or disabled meals get 0 for daily targets
                meal_targets[meal_type] = 0
        
        return meal_targets
    
    def generate_daily_sessions(self, required_category_sessions=None) -> List[Dict[str, str]]:
        """
        Generate session types with categories for one day.
        Combines pre-calculated constraint-based requirements with daily no_memory sessions.
        
        Args:
            required_category_sessions: List of pre-calculated sessions from weekly/monthly constraints
        """
        sessions = []
        
        # Add DAILY activity requirements
        if self.constraints and hasattr(self.constraints, 'activity_daily'):
            for category_key, constraint_value in self.constraints.activity_daily.items():
                
                # Check if this is a subcategory-based category marker
                if category_key.endswith("_subcategory_based"):
                    category = category_key.replace("_subcategory_based", "")
                    
                    # For subcategory-based categories: decide meal targets first, then generate sessions
                    meal_targets = self._decide_daily_meal_targets(category)
                    total_sessions = sum(meal_targets.values())
                    
                    # Store the meal targets for this date (will be used by meal selection)
                    # This ensures the session generation and meal selection are coordinated
                    session_date = "2024-01-01"  # This will be set properly during execution
                    
                    # Generate exactly the number of sessions needed
                    for _ in range(total_sessions):
                        session_type = SessionType.ACTIVITY_MEMORY
                        op_info = self.get_memory_operation(session_type, category)
                        sessions.append({
                            "type": session_type,
                            "category": category,
                            "operation": op_info["operation"],
                            "_meal_targets": meal_targets  # Pass targets to execution phase
                        })
                
                else:
                    # Traditional constraint-based categories
                    category = category_key  # Use the category_key directly for traditional categories
                    min_count, max_count = constraint_value
                    count = random.randint(min_count, max_count)
                    
                    for _ in range(count):
                        session_type = SessionType.ACTIVITY_MEMORY
                        op_info = self.get_memory_operation(session_type, category)
                        sessions.append({
                            "type": session_type,
                            "category": category,
                            "operation": op_info["operation"]
                        })
        
        # Add no_memory sessions (3-5 per day) - these are truly daily requirements
        if self.constraints and hasattr(self.constraints, 'no_memory_per_day'):
            no_memory_count = random.randint(*self.constraints.no_memory_per_day)
        else:
            no_memory_count = random.randint(3, 5)  # Default
        
        for _ in range(no_memory_count):
            sessions.append({
                "type": SessionType.NO_MEMORY,
                "category": None,
                "operation": None
            })
        
        # Add pre-calculated weekly/monthly requirements
        if required_category_sessions:
            sessions.extend(required_category_sessions)
        
        # SHUFFLE sessions to create realistic mixed interactions, but preserve chronological order for food expenses
        # Separate food expense sessions from others
        food_sessions = [s for s in sessions if s.get('category') == 'food_expenses']
        other_sessions = [s for s in sessions if s.get('category') != 'food_expenses']
        
        # Shuffle non-food sessions for natural conversation flow
        random.shuffle(other_sessions)
        
        # Keep food sessions in generation order (chronological) and insert them naturally
        # This preserves breakfast → lunch → dinner order while maintaining mixed interactions
        result_sessions = []
        food_index = 0
        
        # Interleave food sessions with other sessions, maintaining food order
        for i, other_session in enumerate(other_sessions):
            # Insert food session occasionally, but in order
            if food_index < len(food_sessions) and random.random() < 0.3:  # 30% chance
                result_sessions.append(food_sessions[food_index])
                food_index += 1
            result_sessions.append(other_session)
        
        # Add any remaining food sessions at the end (in order)
        while food_index < len(food_sessions):
            result_sessions.append(food_sessions[food_index])
            food_index += 1
        
        return result_sessions


def main():
    """Generate sessions using configurable memory config file with persona support."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate memory sessions with configurable memory configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python session_simulator.py

  # Use weekly configuration
  python session_simulator.py --config meta_data/memory_configs/memory_config_weekly.json

  # Use monthly configuration with specific persona
  python session_simulator.py --config meta_data/memory_configs/memory_config_monthly.json --persona software_engineer

  # Full customization
  python session_simulator.py --config meta_data/memory_configs/memory_config_quarterly.json --persona software_engineer --start-date 2025-01-01 --end-date 2025-12-31
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="meta_data/memory_configs/memory_config_weekly.json",
        help="Path to memory configuration file (default: meta_data/memory_configs/memory_config_weekly.json)"
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="Persona to use (default: random selection from available personas)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-09-09",
        help="Start date in YYYY-MM-DD format (default: 2025-09-09)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-10-19",
        help="End date in YYYY-MM-DD format (default: 2025-10-19)"
    )
    
    args = parser.parse_args()
    
    # Load metadata once with the specified config
    loaded_metadata = load_metadata_files(args.config)
    
    # Determine persona from loaded metadata
    if args.persona:
        persona = args.persona
        print(f"👤 Using specified persona: {persona}")
    else:
        available_personas = []
        if loaded_metadata.get('persona_traits') and "persona_profiles" in loaded_metadata['persona_traits']:
            available_personas = list(loaded_metadata['persona_traits']['persona_profiles'].keys())
        
        if available_personas:
            persona = random.choice(available_personas)
            print(f"👤 Randomly selected persona: {persona}")
        else:
            print("⚠️  No personas found in metadata. Using fallback persona.")
            persona = "software_engineer"
    
    # Set global METADATA so simulator doesn't reload
    global METADATA
    METADATA = loaded_metadata
    
    # Start generation
    print(f"\n{'='*70}")
    print(f"🚀 Starting session generation...")
    print(f"{'='*70}\n")
    
    # Create simulator (no reload needed - metadata already loaded)
    simulator = SimpleSessionSimulator(
        memory_config_file=args.config,
        use_constraints=True,
        persona=persona
    )
    
    result = simulator.generate(args.start_date, args.end_date)
    
    # Extract output folder path
    output_folder_path = result.get("_output_folder_path", "output")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"✅ SESSION GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Total Sessions: {result['summary']['total_sessions']}")
    print(f"👤 Persona: {persona}")
    print(f"📋 Memory Config: {args.config}")
    print(f"📅 Date Range: {args.start_date} to {args.end_date}")
    print(f"📁 Output Folder: {output_folder_path}")
    print(f"💾 All files saved in timestamped folder to prevent data loss")
    print(f"🔄 Multiple runs will create separate folders automatically")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
