"""
Prompt Manager

Main prompt management system that combines FlowManager with LLM generation 
for complete turn-by-turn memory-grounded conversation generation.
"""

import json
import os
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from flow_manager import FlowManager, SessionMetadata
from session_processor import SessionInfo

load_dotenv()


class PromptManager:
    """
    Main prompt manager that orchestrates complete conversation generation.
    Combines FlowManager with LLM generation for memory-grounded conversations.
    """
    
    def __init__(self, model_name: str = "google/gemini-2.0-flash-exp"):
        self.flow_manager = FlowManager()
        self.conversation_history = []
        self.model_name = model_name
        self.client = self._initialize_client()
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        self._prompt_cache = {}
    
    def _initialize_client(self) -> OpenAI:
        """Initialize OpenRouter client for LLM generation"""
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPEN_ROUTER_API_KEY environment variable not set")
        
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
    
    def _load_prompt_from_file(self, agent_type: str, prompt_type: str) -> str:
        """Load system prompt from external file with caching"""
        cache_key = f"{agent_type}_{prompt_type}"
        
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        
        file_path = os.path.join(self.prompts_dir, agent_type, f"{prompt_type}.md")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read().strip()
                self._prompt_cache[cache_key] = prompt_content
                return prompt_content
        except FileNotFoundError:
            print(f"⚠️  Prompt file not found: {file_path}")
            # Fallback to default behavior
            return self._get_fallback_prompt(agent_type, prompt_type)
        except Exception as e:
            print(f"❌ Error loading prompt from {file_path}: {e}")
            return self._get_fallback_prompt(agent_type, prompt_type)
    
    def _get_fallback_prompt(self, agent_type: str, prompt_type: str) -> str:
        """Fallback prompts in case file loading fails"""
        if agent_type == "ai_agent":
            if prompt_type == "general":
                return "You are an AI assistant engaging in natural conversation."
            else:
                return f"You are an AI assistant helping with {prompt_type.replace('_', ' ')}."
        else:  # user_agent
            if prompt_type == "general":
                return "You are simulating a natural user in conversation with an AI assistant."
            else:
                return f"You are simulating a natural user sharing {prompt_type.replace('_', ' ')} information."
    

    def get_ai_general_system_prompt(self) -> str:
        """AI Agent system prompt for GENERAL conversation - strict no personal questions"""
        return self._load_prompt_from_file("ai_agent", "general")

    def get_ai_memory_system_prompt(self, memory_type: str = "preference_memory") -> str:
        """AI Agent system prompt for MEMORY conversation - memory-type specific"""
        return self._load_prompt_from_file("ai_agent", memory_type)

    def get_user_general_system_prompt(self) -> str:
        """User Agent system prompt for GENERAL conversation - strict no oversharing"""
        return self._load_prompt_from_file("user_agent", "general")

    def get_user_memory_system_prompt(self, memory_type: str = "preference_memory") -> str:
        """User Agent system prompt for MEMORY conversation - memory-type specific"""
        return self._load_prompt_from_file("user_agent", memory_type)

    
    def get_memory_sharing_rules(self, share_memory: bool, session: SessionMetadata, agent_type: str) -> str:
        """Memory sharing rules are now handled by instruction generation - return empty string"""
        # Memory sharing behavior is now handled by the instruction generator
        # which provides context-specific instructions for each turn
        return ""

    def _build_conversation_history(self, conversation_history: List[Dict]) -> str:
        """Build conversation history string from turn data"""
        if conversation_history:
            return "\n".join([
                f"{turn['agent']}: {turn['content']}" 
                for turn in conversation_history
            ])
        return "[No previous conversation]"
    
    def _build_session_context(self, session: SessionMetadata, share_memory: bool, requires_instruction: bool, intent_phase: str) -> str:
        """Build session context based on memory type and intent requirements"""
        # Only share session context if this intent explicitly deals with memory
        if not (share_memory or (requires_instruction and 'memory' in intent_phase)):
            return ""
        
        # Special handling for bridge intents - don't share specific details
        if requires_instruction and intent_phase == "memory_share" and not share_memory:
            # For bridge intents, only share generic category information
            return f"""
SESSION CONTEXT:
- Category: {session.category}
- Operation: {session.operation}
"""
        
        if session.memory_type == "content_memory":
            # For content memory, don't include generic item name
            return f"""
SESSION CONTEXT:
- Category: {session.category}
- Operation: {session.operation}
"""
        else:
            return f"""
SESSION CONTEXT:
- Category: {session.category}
- Subcategory: {session.subcategory} 
- Preference Type: {session.preference_type}
- Item: {session.item}
"""
    
    def _get_system_prompt(self, agent_type: str, conversation_type: str, session: SessionMetadata) -> str:
        """Get appropriate system prompt based on agent type and conversation type"""
        if agent_type == "ai":
            if conversation_type == "memory":
                return self.get_ai_memory_system_prompt(session.memory_type)
            else:
                return self.get_ai_general_system_prompt()
        else:  # user agent
            if conversation_type == "memory":
                return self.get_user_memory_system_prompt(session.memory_type)
            else:
                return self.get_user_general_system_prompt()
    
    def _build_user_content(self, history_str: str, instruction: str, memory_rules: str, session_context: str) -> str:
        """Build the user content portion of the prompt"""
        return f"""
CONVERSATION HISTORY:
{history_str}

CURRENT TURN INSTRUCTION:
{instruction}

{memory_rules}{session_context}

Generate a natural, conversational response that follows the instruction above.

CRITICAL LENGTH REQUIREMENT: Your response MUST be 1-2 sentences maximum. NO LONG RESPONSES. BE EXTREMELY BRIEF AND CONCISE.
"""
    
    def build_turn_prompt(self, intent_data: Dict, session: SessionMetadata, conversation_history: List[Dict]) -> Dict[str, str]:
        """Build complete prompt for a single conversation turn"""
        # Extract intent data
        agent_type = intent_data.get('agent', 'ai')
        share_memory = intent_data.get('share_memory', False)
        requires_instruction = intent_data.get('requires_instruction', False)
        conversation_type = intent_data.get('conversation_type', 'general')
        intent_phase = intent_data.get('phase', '')
        
        # Build prompt components
        history_str = self._build_conversation_history(conversation_history)
        memory_rules = self.get_memory_sharing_rules(share_memory, session, agent_type)
        session_context = self._build_session_context(session, share_memory, requires_instruction, intent_phase)
        system_prompt = self._get_system_prompt(agent_type, conversation_type, session)
        user_content = self._build_user_content(history_str, intent_data['instruction'], memory_rules, session_context)
        
        return {
            "system": system_prompt,
            "user": user_content
        }
    
    def generate_llm_response(self, prompt: Dict[str, str], max_tokens: int = None, max_retries: int = 3) -> str:
        """
        Generate response using LLM with retry logic
        
        Args:
            prompt: Dictionary with 'system' and 'user' content
            max_tokens: Maximum tokens for response (None = no limit)
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            Generated response string
        """
        import time
        
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
        
        # Build API call parameters
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7
        }
        
        # Only add max_tokens if specified
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**api_params)
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                attempt_num = attempt + 1
                if attempt_num < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"⚠️  LLM call failed (attempt {attempt_num}/{max_retries}): {e}")
                    print(f"🔄 Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ LLM generation failed after {max_retries} attempts: {e}")
                    return f"[Error generating response after {max_retries} attempts: {str(e)}]"

    def _session_info_to_metadata(self, session: SessionInfo) -> SessionMetadata:
        """Convert SessionInfo to SessionMetadata with flexible structure"""
        # Extract update-specific fields from operation_details
        operation_details = getattr(session, 'operation_details', {}) or {}
        update_type = operation_details.get('update_type')
        old_preference = operation_details.get('old_preference')
        old_item = operation_details.get('old_item')
        
        return SessionMetadata(
            memory_type=session.memory_type,
            operation=session.operation,
            category=session.category,
            session_id=session.session_id,
            date=getattr(session, 'date', None),
            operation_details=operation_details,
            subcategory=session.subcategory,
            preference_type=session.preference_type,
            item=session.item,
            update_type=update_type,
            old_preference=old_preference,
            old_item=old_item
        )
    
    def generate_conversation(self, session: SessionInfo, use_llm: bool = True, flow_ids: Optional[Dict[str, int]] = None, feedback: str = None) -> Dict[str, Any]:
        """
        Generate complete conversation for a session
        
        Args:
            session: Session information
            use_llm: Whether to use actual LLM generation or demo responses
            flow_ids: Optional dict specifying exact flow IDs to use for each phase
            
        Returns:
            Complete conversation with metadata
        """
        # Convert SessionInfo to SessionMetadata for FlowManager
        session_metadata = self._session_info_to_metadata(session)
        
        print(f"\n🎭 Generating conversation for: {session_metadata.memory_type}:{session_metadata.operation}")
        print(f"Context: {session_metadata.category} {session_metadata.subcategory} {session_metadata.preference_type} '{session_metadata.item}'")
        
        if flow_ids:
            print(f"🎯 Using specified flow IDs: {flow_ids}")
        
        # Get conversation flow instructions with metadata
        flow_instructions, flow_metadata = self.flow_manager.generate_flow_instructions(session_metadata, flow_ids=flow_ids, feedback=feedback)
        
        if not flow_instructions:
            return {
                "error": "No flow instructions generated",
                "session": session.__dict__,
                "success": False
            }
        
        print(f"🎯 Selected flow {len(flow_instructions)} intents")
        print(f"📋 Generated {len(flow_instructions)} instructions for {session_metadata.memory_type}:{session_metadata.operation}")
        
        conversation_turns = []
        conversation_history = []
        
        print(f"\n📋 Generating {len(flow_instructions)} turns...")
        print("-" * 80)
        
        for i, intent_data in enumerate(flow_instructions, 1):
            print(f"\nTurn {i}: {intent_data['intent_id']} ({intent_data['agent']})")
            
            # Build prompt for this turn
            prompt = self.build_turn_prompt(intent_data, session_metadata, conversation_history)
            
            # Generate response using LLM or demo
            if use_llm:
                response = self.generate_llm_response(prompt)
                print(f"  🤖 LLM Generated: {response}")
            
            # Create turn record with full prompt details
            turn = {
                "turn_number": i,
                "intent_id": intent_data['intent_id'],
                "agent": intent_data['agent'],
                "phase": intent_data['phase'],
                "share_memory": intent_data['share_memory'],
                "conversation_type": intent_data['conversation_type'],
                "requires_instruction": intent_data['requires_instruction'],
                "instruction": intent_data['instruction'],
                "content": response,
                "prompt": {
                    "system_prompt": prompt["system"],
                    "user_content": prompt["user"],
                    "full_messages": [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                    ]
                }
            }
            
            conversation_turns.append(turn)
            
            # Add to history for next turn
            conversation_history.append({
                "agent": intent_data['agent'],
                "content": response
            })
            

        
        result = {
            "session": session.__dict__,
            "flow_length": len(flow_instructions),
            "flow_metadata": flow_metadata,
            "conversation_turns": conversation_turns,
            "conversation_history": conversation_history,
            "success": True
        }
        
        print(f"\n✅ Generated complete conversation with {len(conversation_turns)} turns")
        return result


def main():
    """Test the prompt manager"""
    
    print("🎯 Prompt Manager Test")
    print("=" * 60)
    
    # Create test session
    from session_processor import SessionInfo
    
    test_session = SessionInfo(
        session_id=999,
        memory_type="preference_memory",
        category="movies",
        subcategory="actors",
        preference_type="dislike",
        item="Meryl Streep",
        operation="add",
        operation_details={},
        date="2024-01-01"
    )
    
    # Initialize prompt manager
    prompt_manager = PromptManager()
    
    # Generate complete conversation using LLM
    print("🤖 Using actual LLM generation...")
    result = prompt_manager.generate_conversation(test_session, use_llm=True)
    
    if result['success']:
        print(f"\n🎬 Complete Conversation Generated")
        print("=" * 60)
        
        for turn in result['conversation_turns']:
            icon = "🤖" if turn['agent'] == "ai" else "👤"
            memory_icon = "💾" if turn['share_memory'] else "💬"
            
            print(f"{turn['turn_number']:2d}. {icon} {memory_icon} {turn['intent_id']}")
            print(f"    {turn['content']}")
            print()
        
        print(f"📊 Conversation Statistics:")
        print(f"  - Total turns: {len(result['conversation_turns'])}")
        print(f"  - Memory sharing turns: {sum(1 for t in result['conversation_turns'] if t['share_memory'])}")
        print(f"  - Session: {test_session.category} {test_session.subcategory} {test_session.preference_type}")
    
    else:
        print(f"❌ Failed to generate conversation: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
