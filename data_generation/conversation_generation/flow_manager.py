import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class SessionMetadata:
    memory_type: str
    operation: str
    category: str
    # Core session data - always present
    session_id: int = None
    date: str = None
    operation_details: Dict[str, Any] = None
    
    # Memory-type specific fields - may be None for some memory types
    subcategory: str = None
    preference_type: str = None
    item: str = None
    
    # Update-specific fields for preference memory
    update_type: str = None  # "preference_update" or "value_update"
    old_preference: str = None  # Previous preference value
    old_item: str = None  # Previous item value (for value_update)
    
    def __post_init__(self):
        """Initialize operation_details if not provided"""
        if self.operation_details is None:
            self.operation_details = {}
    
    def get_memory_context_string(self) -> str:
        """Get memory context string that works for any memory type"""
        if self.memory_type == "preference_memory":
            if self.update_type == 'preference_update':
                return f"Memory Context - Category: {self.category}, Subcategory: {self.subcategory}, Preference Type: {self.preference_type} (was: {self.old_preference}), Item: '{self.item}'"
            elif self.update_type == 'value_update':
                return f"Memory Context - Category: {self.category}, Subcategory: {self.subcategory}, Preference Type: {self.preference_type}, Item: '{self.item}' (was: '{self.old_item}')"
            else:
                return f"Memory Context - Category: {self.category}, Subcategory: {self.subcategory}, Preference Type: {self.preference_type}, Item: '{self.item}'"
        
        elif self.memory_type == "activity_memory":
            # For activity memory, extract specific details from operation_details
            if self.operation_details and 'item' in self.operation_details:
                item_details = self.operation_details['item']
                if isinstance(item_details, dict):
                    # Food expenses
                    if 'expense_type' in item_details:
                        expense_type = item_details['expense_type']
                        amount = item_details.get('amount', '')
                        item_desc = f"${amount} {expense_type}"
                    # Todo list
                    elif 'task_type' in item_details:
                        task_type = item_details.get('task_type', 'task')
                        description = item_details.get('description', 'task')
                        item_desc = f"{description} ({task_type})"
                    # Calendar events
                    elif 'event_type' in item_details:
                        event_name = item_details.get('event_name', 'event')
                        event_type = item_details.get('event_type', 'event')
                        item_desc = f"{event_name} ({event_type})"
                    # Step tracker
                    elif 'step_count' in item_details:
                        step_count = item_details.get('step_count', 0)
                        item_desc = f"{step_count} steps"
                    else:
                        item_desc = str(item_details)
                else:
                    item_desc = str(item_details)
            else:
                item_desc = self.item if self.item else 'activity'
            return f"Memory Context - Category: {self.category}, Activity: {item_desc}, Operation: {self.operation}"
        
        elif self.memory_type == "content_memory":
            return f"Memory Context - Category: {self.category}, Operation: {self.operation}"
        
        elif self.memory_type == "goal_memory":
            item_desc = self.item if self.item else str(self.operation_details.get('item', 'goal'))
            subcategory = self.subcategory if self.subcategory else self.operation_details.get('subcategory', '')
            
            # All goals are weekly (not monthly or quarterly)
            if self.category == "food_expenses":
                return f"Memory Context - Category: {self.category}, Subcategory: {subcategory}, Goal: ${item_desc} per week, Operation: {self.operation}"
            elif self.category == "step_tracker":
                return f"Memory Context - Category: {self.category}, Subcategory: {subcategory}, Goal: {item_desc} steps per day, Operation: {self.operation}"
            else:
                return f"Memory Context - Category: {self.category}, Subcategory: {subcategory}, Goal: {item_desc} per week, Operation: {self.operation}"
        
        else:
            # Generic fallback
            item_desc = self.item if self.item else str(self.operation_details.get('item', 'memory'))
            return f"Memory Context - Category: {self.category}, Item: {item_desc}, Operation: {self.operation}"

class FlowManager:
    """
    Flow manager for conversation generation using the conversations_templates directory structure.
    Supports random selection from categorized flows with filtering by memory type and operation.
    """
    
    def __init__(self, templates_dir: str = None):
        import os
        
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if templates_dir is None:
            templates_dir = os.path.join(script_dir, "conversations_templates")
            
        self.templates_dir = templates_dir
        self.flattened_intents = {}
        self.flows = {}
        
        # Initialize the instruction generator
        from instruction_generator import InstructionGenerator
        self.instruction_generator = InstructionGenerator()
        
        self.load_data()
    
    def load_data(self):
        """Load and process intent and flow data from conversations_templates directory"""
        self._load_intents_from_templates()
        self._load_flows_from_templates()
    
    def _load_intents_from_templates(self):
        """Load and flatten intents from conversations_templates directory"""
        import os
        
        if not os.path.exists(self.templates_dir):
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")
            
        print(f"🔄 Loading intents from template structure: {self.templates_dir}")
        self._load_intents_from_directory()
    
    def _load_intents_from_directory(self):
        """Load intents from conversations_templates directory structure"""
        import os
        import glob
        
        # Load intents from all template files
        general_dir = os.path.join(self.templates_dir, "general_conversations")
        memory_dir = os.path.join(self.templates_dir, "memory_conversations")
        
        # Load from general conversations
        if os.path.exists(general_dir):
            for file_path in glob.glob(os.path.join(general_dir, "*.json")):
                self._load_intents_from_file(file_path, "general")
        
        # Load from memory conversations  
        if os.path.exists(memory_dir):
            for file_path in glob.glob(os.path.join(memory_dir, "*.json")):
                self._load_intents_from_file(file_path, "memory")
        
        print(f"✅ Loaded {len(self.flattened_intents)} intents from template files")
    
    def _load_intents_from_file(self, file_path: str, category: str):
        """Load intents from a single template file"""
        import os
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            filename = os.path.basename(file_path)
            intent_descriptions = data.get('intent_description', {})
            
            if intent_descriptions:
                for intent_name, intent_data in intent_descriptions.items():
                    # Add category info to intent data
                    intent_data['template_file'] = filename
                    intent_data['template_category'] = category
                    self.flattened_intents[intent_name] = intent_data
                
                print(f"   📄 {filename}: {len(intent_descriptions)} intents")
            else:
                print(f"   📄 {filename}: No intents found")
                
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {e}")
    
    
    def _load_flows_from_templates(self):
        """Load conversation flows from conversations_templates directory"""
        import os
        
        if not os.path.exists(self.templates_dir):
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")
            
        print(f"🔄 Loading flows from template structure: {self.templates_dir}")
        self._load_flows_from_directory()
    
    def _load_flows_from_directory(self):
        """Load flows from conversations_templates directory structure"""
        import os
        import glob
        
        self.flows = {
            'opening_flows': [],
            'exploration_flows': [],
            'memory_flows': [],
            'closing_flows': []
        }
        
        # Load general conversation flows
        general_dir = os.path.join(self.templates_dir, "general_conversations")
        if os.path.exists(general_dir):
            # Load opening phase flows
            opening_file = os.path.join(general_dir, "opening_phase.json")
            if os.path.exists(opening_file):
                self._load_flows_from_file(opening_file, 'opening_flows')
            
            # Load exploration phase flows  
            exploration_file = os.path.join(general_dir, "exploration_phase.json")
            if os.path.exists(exploration_file):
                self._load_flows_from_file(exploration_file, 'exploration_flows')
            
            # Load closing phase flows
            closing_file = os.path.join(general_dir, "closing_phase.json")
            if os.path.exists(closing_file):
                self._load_flows_from_file(closing_file, 'closing_flows')
        
        # Load memory conversation flows
        memory_dir = os.path.join(self.templates_dir, "memory_conversations")
        if os.path.exists(memory_dir):
            for file_path in glob.glob(os.path.join(memory_dir, "*.json")):
                self._load_flows_from_file(file_path, 'memory_flows')
        
        # Count total flows
        total_flows = sum(len(flows) for flows in self.flows.values())
        print(f"✅ Loaded {total_flows} flows from template files")
        print(f"   - Opening flows: {len(self.flows.get('opening_flows', []))}")
        print(f"   - Exploration flows: {len(self.flows.get('exploration_flows', []))}")
        print(f"   - Memory flows: {len(self.flows.get('memory_flows', []))}")
        print(f"   - Closing flows: {len(self.flows.get('closing_flows', []))}")
    
    def _load_flows_from_file(self, file_path: str, flow_category: str):
        """Load flows from a single template file"""
        import os
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            filename = os.path.basename(file_path)
            flows_list = data.get('conversation_flows_list', [])
            
            if flows_list:
                # Add source file info to each flow
                for flow in flows_list:
                    flow['source_file'] = filename
                    flow['template_category'] = flow_category
                
                self.flows[flow_category].extend(flows_list)
                print(f"   📄 {filename}: {len(flows_list)} flows → {flow_category}")
            else:
                print(f"   📄 {filename}: No flows found")
                
        except Exception as e:
            print(f"   ❌ Error loading flows from {file_path}: {e}")
    
    
    def get_conversation_flow(self, session: SessionMetadata, flow_ids: Optional[Dict[str, int]] = None) -> tuple[List[str], Dict[str, Any]]:
        """
        Get complete conversation flow by combining flows from different phases
        
        Args:
            session: Session metadata containing memory_type and operation
            flow_ids: Optional dict specifying exact flow IDs to use for each phase
            
        Returns:
            Tuple of (intent_list, flow_metadata) where flow_metadata contains selected flow IDs
        """
        complete_flow = []
        flow_metadata = {
            'selected_flows': {},
            'flow_structure': {},
            'total_intents': 0
        }
        
        # 1. Select opening flow
        # For content_memory, we need opening to end with AI since all content flows are user-initiated
        if flow_ids and 'opening' in flow_ids:
            opening_flow, opening_id = self._select_specific_opening_flow(flow_ids['opening'])
        else:
            # Content memory always starts with user turns, so opening must end with AI
            required_last_speaker = 'ai' if session.memory_type == "content_memory" else None
            opening_flow, opening_id = self._select_opening_flow_with_id(required_last_speaker)
        
        if opening_flow:
            complete_flow.extend(opening_flow)
            flow_metadata['selected_flows']['opening'] = opening_id
            flow_metadata['selected_flows']['opening_flow'] = opening_flow
            flow_metadata['flow_structure']['opening'] = len(opening_flow)
            print(f"🎬 Added opening flow {opening_id}: {len(opening_flow)} intents")
        
        # 2. Add exploration flow (skip for content memory and no_memory has different logic)
        if session.memory_type == "no_memory":
            # For no_memory sessions, always include exploration for a fuller conversation
            # Determine last speaker from opening flow for coherence
            last_speaker = self._get_last_speaker_from_flow(opening_flow) if opening_flow else None
            
            if flow_ids and 'exploration' in flow_ids:
                exploration_flow, exploration_id = self._select_specific_exploration_flow(flow_ids['exploration'])
            else:
                exploration_flow, exploration_id = self._select_exploration_flow_with_id(last_speaker)
            
            if exploration_flow:
                complete_flow.extend(exploration_flow)
                flow_metadata['selected_flows']['exploration'] = exploration_id
                flow_metadata['selected_flows']['exploration_flow'] = exploration_flow
                flow_metadata['flow_structure']['exploration'] = len(exploration_flow)
                print(f"🔍 Added exploration flow {exploration_id}: {len(exploration_flow)} intents (no_memory)")
        elif session.memory_type != "content_memory":
            # Skip exploration flow for content memory to avoid topic interference
            # Determine last speaker from opening flow for coherence
            last_speaker = self._get_last_speaker_from_flow(opening_flow) if opening_flow else None
            
            if flow_ids and 'exploration' in flow_ids:
                exploration_flow, exploration_id = self._select_specific_exploration_flow(flow_ids['exploration'])
            else:
                exploration_flow, exploration_id = self._select_exploration_flow_with_id(last_speaker)
            
            if exploration_flow:
                complete_flow.extend(exploration_flow)
                flow_metadata['selected_flows']['exploration'] = exploration_id
                flow_metadata['selected_flows']['exploration_flow'] = exploration_flow
                flow_metadata['flow_structure']['exploration'] = len(exploration_flow)
                print(f"🔍 Added exploration flow {exploration_id}: {len(exploration_flow)} intents")
        else:
            print(f"⏭️  Skipped exploration phase for content_memory to maintain focus")
        
        # 3. Select memory flow (skip for no_memory sessions)
        if session.memory_type != "no_memory":
            # Determine last speaker for memory flow coherence
            if 'exploration_flow' in flow_metadata['selected_flows']:
                last_speaker_for_memory = self._get_last_speaker_from_flow(flow_metadata['selected_flows']['exploration_flow'])
            elif opening_flow:
                last_speaker_for_memory = self._get_last_speaker_from_flow(opening_flow)
            else:
                last_speaker_for_memory = None
            
            if flow_ids and 'memory' in flow_ids:
                memory_flow, memory_id, memory_approach = self._select_specific_memory_flow(flow_ids['memory'], session)
            else:
                memory_flow, memory_id, memory_approach = self._select_memory_flow_with_id(session, last_speaker_for_memory)
            
            if memory_flow:
                complete_flow.extend(memory_flow)
                flow_metadata['selected_flows']['memory'] = memory_id
                flow_metadata['selected_flows']['memory_flow'] = memory_flow
                flow_metadata['selected_flows']['memory_approach'] = memory_approach
                flow_metadata['flow_structure']['memory'] = len(memory_flow)
                print(f"💭 Added memory flow {memory_id} ({memory_approach}): {len(memory_flow)} intents")
        else:
            print(f"⏭️  Skipped memory phase for no_memory session - pure general conversation")
        
        # 4. Select closing flow
        # Determine last speaker from previous phase for coherence
        last_speaker = None
        if session.memory_type != "no_memory" and 'memory_flow' in flow_metadata['selected_flows']:
            # If we have a memory flow, use its last speaker
            last_speaker = self._get_last_speaker_from_flow(flow_metadata['selected_flows']['memory_flow'])
        elif 'exploration_flow' in flow_metadata['selected_flows']:
            # If we have an exploration flow, use its last speaker
            last_speaker = self._get_last_speaker_from_flow(flow_metadata['selected_flows']['exploration_flow'])
        elif opening_flow:
            # Fallback to opening flow
            last_speaker = self._get_last_speaker_from_flow(opening_flow)
        
        if flow_ids and 'closing' in flow_ids:
            closing_flow, closing_id = self._select_specific_closing_flow(flow_ids['closing'])
        else:
            closing_flow, closing_id = self._select_closing_flow_with_id(last_speaker)
        
        if closing_flow:
            complete_flow.extend(closing_flow)
            flow_metadata['selected_flows']['closing'] = closing_id
            flow_metadata['selected_flows']['closing_flow'] = closing_flow
            flow_metadata['flow_structure']['closing'] = len(closing_flow)
            print(f"🎬 Added closing flow {closing_id}: {len(closing_flow)} intents")
        
        flow_metadata['total_intents'] = len(complete_flow)
        print(f"🎯 Complete conversation flow: {len(complete_flow)} total intents")
        return complete_flow, flow_metadata
    
    def _select_opening_flow_with_id(self, required_last_speaker: str = None) -> tuple[List[str], int]:
        """
        Select an opening flow and return flow with ID
        
        Args:
            required_last_speaker: If specified, only select flows that end with this speaker ('ai' or 'user')
                                   This is used for content_memory which always starts with user turns
        """
        opening_flows = self.flows.get('opening_flows', [])
        if not opening_flows:
            return [], None
        
        # If required_last_speaker is specified, filter flows
        if required_last_speaker:
            filtered_flows = []
            for flow in opening_flows:
                conversation_flow = flow.get('conversation_flow', [])
                last_speaker = self._get_last_speaker_from_flow(conversation_flow)
                if last_speaker == required_last_speaker:
                    filtered_flows.append(flow)
            
            if filtered_flows:
                selected_flow = random.choice(filtered_flows)
                flow_id = selected_flow.get('flow_id')
                conversation_flow = selected_flow.get('conversation_flow', [])
                print(f"🎯 Selected opening flow {flow_id} ending with {required_last_speaker} (filtered from {len(opening_flows)} to {len(filtered_flows)} flows)")
                return conversation_flow, flow_id
            else:
                print(f"⚠️  No opening flows found ending with {required_last_speaker}, using random selection")
        
        # Fallback to random selection
        selected_flow = random.choice(opening_flows)
        flow_id = selected_flow.get('flow_id')
        conversation_flow = selected_flow.get('conversation_flow', [])
        return conversation_flow, flow_id
    
    def _get_last_speaker_from_flow(self, conversation_flow: List[str]) -> str:
        """Determine who spoke last in a conversation flow"""
        if not conversation_flow:
            return None
        
        last_intent = conversation_flow[-1]
        # Extract agent from intent name (e.g., "user_ask_question" -> "user")
        if last_intent.startswith('user_'):
            return 'user'
        elif last_intent.startswith('ai_'):
            return 'ai'
        else:
            return None
    
    def _select_exploration_flow_with_id(self, last_speaker: str = None) -> tuple[List[str], int]:
        """Select an exploration flow based on conversation coherence and return flow with ID"""
        exploration_flows = self.flows.get('exploration_flows', [])
        if not exploration_flows:
            return [], None
        
        # If we know who spoke last, select a coherent flow
        if last_speaker:
            coherent_flows = []
            for flow in exploration_flows:
                flow_type = flow.get('flow_type', '')
                # If AI spoke last, select user_initiated flows for coherence
                # If user spoke last, select ai_initiated flows for coherence
                if (last_speaker == 'ai' and flow_type == 'user_initiated') or \
                   (last_speaker == 'user' and flow_type == 'ai_initiated'):
                    coherent_flows.append(flow)
            
            if coherent_flows:
                selected_flow = random.choice(coherent_flows)
                flow_id = selected_flow.get('flow_id')
                conversation_flow = selected_flow.get('conversation_flow', [])
                print(f"🎯 Selected coherent exploration flow {flow_id} ({selected_flow.get('flow_type')}) after {last_speaker} spoke last")
                return conversation_flow, flow_id
        
        # Fallback to random selection if no coherent flows or no last_speaker info
        selected_flow = random.choice(exploration_flows)
        flow_id = selected_flow.get('flow_id')
        conversation_flow = selected_flow.get('conversation_flow', [])
        print(f"🎯 Selected random exploration flow {flow_id} ({selected_flow.get('flow_type')})")
        return conversation_flow, flow_id
    
    def _select_memory_flow_with_id(self, session: SessionMetadata, last_speaker: str = None) -> tuple[List[str], int, str]:
        """
        Select memory flow filtered by memory_type and operation, return flow with ID and approach
        
        Args:
            session: Session metadata for filtering
            last_speaker: Who spoke last in the previous phase (for speaker coherence)
            
        Returns:
            Tuple of (intent_list, flow_id, approach)
        """
        # Special handling for content memory - generate dynamic flow
        # Content memory flows are always user-initiated, so opening must end with AI
        if session.memory_type == "content_memory":
            return self._generate_content_memory_flow(session)
        
        memory_flows = self.flows.get('memory_flows', [])
        if not memory_flows:
            print(f"❌ No memory flows available")
            return [], None, None
        
        # Filter flows by memory_type and operation
        filtered_flows = []
        for flow in memory_flows:
            flow_memory_type = flow.get('memory_type')
            flow_operation = flow.get('operation')
            
            # Match both memory_type and operation
            if (flow_memory_type == session.memory_type and 
                flow_operation == session.operation):
                filtered_flows.append(flow)
        
        if not filtered_flows:
            print(f"❌ No memory flows found for {session.memory_type}:{session.operation}")
            return [], None, None
        
        # Apply speaker coherence logic (same as exploration/closing flows)
        if last_speaker:
            coherent_flows = []
            for flow in filtered_flows:
                flow_type = flow.get('flow_type', '')
                # If AI spoke last, select user_initiated flows for coherence
                # If user spoke last, select ai_initiated flows for coherence
                if (last_speaker == 'ai' and flow_type == 'user_initiated') or \
                   (last_speaker == 'user' and flow_type == 'ai_initiated'):
                    coherent_flows.append(flow)
            
            if coherent_flows:
                selected_flow = random.choice(coherent_flows)
                flow_id = selected_flow.get('flow_id')
                approach = selected_flow.get('approach', 'unknown')
                conversation_flow = selected_flow.get('conversation_flow', [])
                print(f"🎯 Selected coherent memory flow {flow_id} ({selected_flow.get('flow_type')}) after {last_speaker} spoke last")
                return conversation_flow, flow_id, approach
            else:
                print(f"⚠️  No coherent memory flows found for {session.memory_type}:{session.operation} after {last_speaker}, using random")
        
        # Fallback to random selection from filtered flows
        selected_flow = random.choice(filtered_flows)
        flow_id = selected_flow.get('flow_id')
        approach = selected_flow.get('approach', 'unknown')
        conversation_flow = selected_flow.get('conversation_flow', [])
        
        print(f"🎯 Selected memory flow {flow_id} ({approach}) for {session.memory_type}:{session.operation}")
        return conversation_flow, flow_id, approach
    
    def _generate_content_memory_flow(self, session: SessionMetadata) -> tuple[List[str], int, str]:
        """
        Generate conversation flow for content memory based on operation type
        
        Args:
            session: Session metadata containing content information
            
        Returns:
            Tuple of (intent_list, flow_id, approach)
        """
        print(f"🎨 Generating content memory flow for {session.category} ({session.operation})")
        
        if session.operation == "add":
            return self._generate_content_add_flow(session)
        elif session.operation == "update":
            return self._generate_content_update_flow(session)
        elif session.operation == "delete":
            return self._generate_content_delete_flow(session)
        else:
            print(f"⚠️  Unknown operation {session.operation}, defaulting to ADD flow")
            return self._generate_content_add_flow(session)
    
    def _generate_content_add_flow(self, session: SessionMetadata) -> tuple[List[str], int, str]:
        """Generate field-by-field flow for ADD operations"""
        # Get content data to determine number of fields
        content_data = session.operation_details.get('content_data', {})
        field_names = list(content_data.keys()) if content_data else []
        num_fields = len(field_names)
        
        if num_fields == 0:
            print(f"⚠️  No content data found, using minimal flow")
            num_fields = 1
        
        # Base flow for content memory ADD
        flow = []
        
        # 1. User asks for help with content writeup
        flow.append("user_initiate_content_conversation")
        
        # 2. AI agrees to help with the writeup
        flow.append("ai_agree_to_help_with_content_conversation")
        
        # 3. Field-by-field content sharing (AI asks, user shares)
        for i in range(num_fields):
            flow.append("ai_ask_about_content_fields")
            flow.append("user_share_content_fields")
        
        # 4. AI confirms and wraps up
        flow.append("ai_confirm_content_writeup_complete")
        
        # Generate metadata
        flow_id = f"content_field_by_field_add_{num_fields}fields"
        approach = f"field_by_field_{num_fields}_fields"
        
        print(f"🎯 Generated ADD flow: {len(flow)} intents for {num_fields} fields")
        print(f"   Fields: {', '.join(field_names[:5])}{'...' if len(field_names) > 5 else ''}")
        print(f"   Flow pattern: user_initiate_content → ai_agree_help → field_pairs × {num_fields} → confirm")
        
        return flow, flow_id, approach
    
    def _generate_content_update_flow(self, session: SessionMetadata) -> tuple[List[str], int, str]:
        """Generate user-driven flow for UPDATE operations"""
        # Get memory updates to determine number of changes
        memory_updates = session.operation_details.get('memory_updates', [])
        num_updates = len(memory_updates)
        
        if num_updates == 0:
            print(f"⚠️  No memory updates found, using minimal flow")
            num_updates = 1
        
        # Flow for content memory UPDATE
        flow = []
        
        # 1. User initiates update conversation
        flow.append("user_initiate_content_update_conversation")
        
        # 2. AI agrees to help with updates
        flow.append("ai_agree_to_help_with_content_update")
        
        # 3. User shares updates one by one (user drives the conversation)
        for i in range(num_updates):
            flow.append("user_share_content_update")
            flow.append("ai_acknowledge_content_update")
        
        # 4. User confirms all updates are complete
        flow.append("user_confirm_content_updates_complete")
        
        # 5. AI confirms and wraps up
        flow.append("ai_confirm_content_writeup_complete")
        
        # Generate metadata
        flow_id = f"content_update_flow_{num_updates}updates"
        approach = f"user_driven_{num_updates}_updates"
        
        print(f"🎯 Generated UPDATE flow: {len(flow)} intents for {num_updates} updates")
        update_fields = [u.get('field', '?') for u in memory_updates]
        print(f"   Updates: {', '.join(update_fields[:5])}{'...' if len(update_fields) > 5 else ''}")
        print(f"   Flow pattern: user_initiate_update → ai_agree → update_pairs × {num_updates} → user_confirm → ai_confirm")
        
        return flow, flow_id, approach
    
    def _generate_content_delete_flow(self, session: SessionMetadata) -> tuple[List[str], int, str]:
        """Generate user-driven flow for DELETE operations"""
        # Get memory deletes to determine number of deletions
        memory_deletes = session.operation_details.get('memory_deletes', [])
        num_deletes = len(memory_deletes)
        
        if num_deletes == 0:
            print(f"⚠️  No memory deletes found, using minimal flow")
            num_deletes = 1
        
        # Flow for content memory DELETE
        flow = []
        
        # 1. User initiates delete conversation
        flow.append("user_initiate_content_delete_conversation")
        
        # 2. AI agrees to help with deletions
        flow.append("ai_agree_to_help_with_content_delete")
        
        # 3. User shares deletions one by one (user drives the conversation)
        for i in range(num_deletes):
            flow.append("user_share_content_delete")
            flow.append("ai_acknowledge_content_delete")
        
        # 4. User confirms all deletions are complete
        flow.append("user_confirm_content_deletes_complete")
        
        # 5. AI confirms and wraps up
        flow.append("ai_confirm_content_writeup_complete")
        
        # Generate metadata
        flow_id = f"content_delete_flow_{num_deletes}deletes"
        approach = f"user_driven_{num_deletes}_deletes"
        
        print(f"🎯 Generated DELETE flow: {len(flow)} intents for {num_deletes} deletions")
        delete_fields = [d.get('field', '?') for d in memory_deletes]
        print(f"   Deletions: {', '.join(delete_fields[:5])}{'...' if len(delete_fields) > 5 else ''}")
        print(f"   Flow pattern: user_initiate_delete → ai_agree → delete_pairs × {num_deletes} → user_confirm → ai_confirm")
        
        return flow, flow_id, approach
    
    def _select_closing_flow_with_id(self, last_speaker: str = None) -> tuple[List[str], int]:
        """Select a closing flow based on conversation coherence and return flow with ID"""
        closing_flows = self.flows.get('closing_flows', [])
        if not closing_flows:
            return [], None
        
        # If we know who spoke last, select a coherent flow
        if last_speaker:
            coherent_flows = []
            for flow in closing_flows:
                flow_type = flow.get('flow_type', '')
                # If AI spoke last, select user_initiated flows for coherence
                # If user spoke last, select ai_initiated flows for coherence
                if (last_speaker == 'ai' and flow_type == 'user_initiated') or \
                   (last_speaker == 'user' and flow_type == 'ai_initiated'):
                    coherent_flows.append(flow)
            
            if coherent_flows:
                selected_flow = random.choice(coherent_flows)
                flow_id = selected_flow.get('flow_id')
                conversation_flow = selected_flow.get('conversation_flow', [])
                print(f"🎯 Selected coherent closing flow {flow_id} ({selected_flow.get('flow_type')}) after {last_speaker} spoke last")
                return conversation_flow, flow_id
        
        # Fallback to random selection if no coherent flows or no last_speaker info
        selected_flow = random.choice(closing_flows)
        flow_id = selected_flow.get('flow_id')
        conversation_flow = selected_flow.get('conversation_flow', [])
        print(f"🎯 Selected random closing flow {flow_id} ({selected_flow.get('flow_type')})")
        return conversation_flow, flow_id
    
    def _select_specific_opening_flow(self, flow_id: int) -> tuple[List[str], int]:
        """Select specific opening flow by ID"""
        opening_flows = self.flows.get('opening_flows', [])
        for flow in opening_flows:
            if flow.get('flow_id') == flow_id:
                conversation_flow = flow.get('conversation_flow', [])
                print(f"🎯 Found specific opening flow {flow_id}")
                return conversation_flow, flow_id
        
        print(f"❌ Opening flow {flow_id} not found, using random selection")
        return self._select_opening_flow_with_id()
    
    def _select_specific_exploration_flow(self, flow_id: int) -> tuple[List[str], int]:
        """Select specific exploration flow by ID"""
        exploration_flows = self.flows.get('exploration_flows', [])
        for flow in exploration_flows:
            if flow.get('flow_id') == flow_id:
                conversation_flow = flow.get('conversation_flow', [])
                print(f"🎯 Found specific exploration flow {flow_id}")
                return conversation_flow, flow_id
        
        print(f"❌ Exploration flow {flow_id} not found, using random selection")
        return self._select_exploration_flow_with_id()
    
    def _select_specific_memory_flow(self, flow_id: int, session: SessionMetadata) -> tuple[List[str], int, str]:
        """Select specific memory flow by ID (still filtered by memory type and operation)"""
        memory_flows = self.flows.get('memory_flows', [])
        
        # First try to find the exact flow ID with matching memory type and operation
        for flow in memory_flows:
            if (flow.get('flow_id') == flow_id and 
                flow.get('memory_type') == session.memory_type and 
                flow.get('operation') == session.operation):
                conversation_flow = flow.get('conversation_flow', [])
                approach = flow.get('approach', 'unknown')
                print(f"🎯 Found specific memory flow {flow_id} for {session.memory_type}:{session.operation}")
                return conversation_flow, flow_id, approach
        
        print(f"❌ Memory flow {flow_id} not found for {session.memory_type}:{session.operation}, using random selection")
        return self._select_memory_flow_with_id(session)
    
    def _select_specific_closing_flow(self, flow_id: int) -> tuple[List[str], int]:
        """Select specific closing flow by ID"""
        closing_flows = self.flows.get('closing_flows', [])
        for flow in closing_flows:
            if flow.get('flow_id') == flow_id:
                conversation_flow = flow.get('conversation_flow', [])
                print(f"🎯 Found specific closing flow {flow_id}")
                return conversation_flow, flow_id
        
        print(f"❌ Closing flow {flow_id} not found, using random selection")
        return self._select_closing_flow_with_id()
    
    def generate_instruction(self, intent_id: str, session: SessionMetadata, current_field: str = None, feedback: str = None) -> str:
        """
        Generate contextual instruction for a specific intent
        
        Args:
            intent_id: The intent identifier
            session: Session metadata for context
            current_field: Current content field being discussed (for content memory)
            
        Returns:
            Contextual instruction string for the intent
        """
        # Check if intent exists
        if intent_id not in self.flattened_intents:
            return f"[ERROR: Intent '{intent_id}' not found]"
        
        intent_data = self.flattened_intents[intent_id]
        requires_instruction = intent_data.get('requires_instruction', False)
        
        if not requires_instruction:
            # Use the intent description as the instruction
            description = intent_data.get('description', f"Execute intent: {intent_id}")
            return description
        else:
            # Use the new InstructionGenerator for all memory types
            return self.instruction_generator.generate_custom_instruction(intent_id, intent_data, session, current_field, feedback)
    
    # Note: _generate_contextual_instruction method removed
    # All custom instruction generation now handled by InstructionGenerator class
    
    # Note: All instruction generation methods removed
    # All custom instruction generation now handled by InstructionGenerator class
    
    # All instruction generation now handled by InstructionGenerator class

    def generate_flow_instructions(self, session: SessionMetadata, flow_ids: Optional[Dict[str, int]] = None, feedback: str = None) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate complete flow with instructions for each intent
        
        Args:
            session: Session metadata for context
            flow_ids: Optional dict specifying exact flow IDs to use for each phase
            
        Returns:
            Tuple of (flow_instructions, flow_metadata) where flow_metadata contains selected flow IDs
        """
        # Get the conversation flow with metadata
        flow_intents, flow_metadata = self.get_conversation_flow(session, flow_ids=flow_ids)
        
        if not flow_intents:
            print(f"❌ No conversation flow generated for {session.memory_type}:{session.operation}")
            return [], {}
        
        # Generate instructions for each intent with field tracking for content memory
        flow_instructions = []
        content_field_index = 0
        content_fields = []
        content_update_index = 0
        content_delete_index = 0
        
        # For content memory, get the field names in order for ADD operations
        if session.memory_type == "content_memory" and session.operation == "add":
            content_data = session.operation_details.get('content_data', {})
            content_fields = list(content_data.keys()) if content_data else []
        
        for intent_id in flow_intents:
            intent_info = self.flattened_intents.get(intent_id, {})
            
            # For content field intents, track which field we're discussing
            current_field = None
            if session.memory_type == "content_memory" and intent_id in ["ai_ask_about_content_fields", "user_share_content_fields"]:
                if content_field_index < len(content_fields):
                    current_field = content_fields[content_field_index]
                    # Increment field index after user shares (on user_share_content_fields)
                    if intent_id == "user_share_content_fields":
                        content_field_index += 1
            
            # For content update intents, track which update we're discussing
            if session.memory_type == "content_memory" and intent_id == "user_share_content_update":
                # Set current update index on session for instruction generation
                session._current_update_index = content_update_index
                content_update_index += 1
            
            # For content delete intents, track which deletion we're discussing
            if session.memory_type == "content_memory" and intent_id == "user_share_content_delete":
                # Set current delete index on session for instruction generation
                session._current_delete_index = content_delete_index
                content_delete_index += 1
            
            instruction = self.generate_instruction(intent_id, session, current_field, feedback)
            flow_instructions.append({
                'intent_id': intent_id,
                'instruction': instruction,
                'agent': intent_info.get('agent', 'unknown'),
                'phase': intent_info.get('phase', 'unknown'),
                'share_memory': intent_info.get('share_memory', False),
                'conversation_type': intent_info.get('conversation_type', 'general'),
                'requires_instruction': intent_info.get('requires_instruction', False),
                'content_field': current_field  # Add field info for content memory
            })
        
        return flow_instructions, flow_metadata

    def get_flow_statistics(self) -> Dict[str, Any]:
        """Get statistics about available flows"""
        stats = {}
        
        for flow_type, flows in self.flows.items():
            if isinstance(flows, list):
                stats[flow_type] = {
                    'total_flows': len(flows),
                    'flow_types': {},
                    'memory_types': {},
                    'operations': {}
                }
                
                for flow in flows:
                    # Count by flow_type (ai_initiated vs user_initiated)
                    flow_init_type = flow.get('flow_type', 'unknown')
                    stats[flow_type]['flow_types'][flow_init_type] = stats[flow_type]['flow_types'].get(flow_init_type, 0) + 1
                    
                    # Count by memory_type (for memory flows)
                    memory_type = flow.get('memory_type')
                    if memory_type:
                        stats[flow_type]['memory_types'][memory_type] = stats[flow_type]['memory_types'].get(memory_type, 0) + 1
                    
                    # Count by operation (for memory flows)
                    operation = flow.get('operation')
                    if operation:
                        stats[flow_type]['operations'][operation] = stats[flow_type]['operations'].get(operation, 0) + 1
        
        return stats
