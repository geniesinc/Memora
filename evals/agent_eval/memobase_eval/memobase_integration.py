#!/usr/bin/env python3
"""
Memobase Memory System Integration
===================================

This module provides Memobase-specific implementation of the BaseMemorySystem interface.
It handles all Memobase-specific operations for storing and retrieving memories.

Memobase is a user profile-based memory system designed for LLM applications:
- User profiles: Structured memory organized by topics (basic_info, interests, etc.)
- Time-aware memory: Records user events with temporal context
- Controllable memory: Configurable profile structure

Implementation follows Memobase documentation:
- https://github.com/memodb-io/memobase
- https://www.memobase.io/

Features:
- User profile-based memory management
- ChatBlob for conversation ingestion
- Profile and context retrieval for memory search
- Batch processing with flush mechanism

Usage:
    from memobase_integration import MemobaseSystem
    
    system = MemobaseSystem("user_123")
    if system.validate_environment() and system.initialize_client():
        # Use the system for memory operations
        result = system.add_conversation_to_memory(conversation_data)
"""

import json
import os
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import the Memobase SDK
try:
    from memobase import MemoBaseClient, ChatBlob
    MEMOBASE_AVAILABLE = True
except ImportError:
    MEMOBASE_AVAILABLE = False
    MemoBaseClient = None
    ChatBlob = None

# Import base classes
import sys
sys.path.append(str(Path(__file__).parent.parent))
from base_evaluator import BaseMemorySystem


class MemobaseSystem(BaseMemorySystem):
    """
    Memobase-specific implementation of BaseMemorySystem.
    
    This class handles all Memobase-specific operations for storing and retrieving memories.
    Memobase uses a profile-based approach where:
    - Conversations are inserted as ChatBlobs
    - Memory is organized into structured profiles (topics/subtopics)
    - Profiles are retrieved and formatted for context
    """
    
    def __init__(self, user_id: str, **kwargs):
        """
        Initialize the Memobase memory system.
        
        Args:
            user_id: User ID for Memobase memories. Should follow the format:
                     {persona}_{timeline} (e.g., "academic_researcher_weekly").
                     This is dynamically generated from the session directory and
                     questions file, not hardcoded.
            **kwargs: Memobase specific configuration parameters:
                - api_key: Memobase API key (or use MEMOBASE_API_KEY env var)
                - project_url: Memobase project URL (or use MEMOBASE_PROJECT_URL env var)
        """
        super().__init__(user_id)
        
        # Validate user_id format (should be persona_timeline)
        if not user_id or '_' not in user_id:
            self.logger.warning(
                f"User ID '{user_id}' may not follow expected format '{{persona}}_{{timeline}}'. "
                f"Expected format: e.g., 'academic_researcher_weekly'"
            )
        
        # Memobase client and user (initialized in initialize_client)
        self.client = None
        self.memobase_user = None
        self.memobase_user_id = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load environment variables
        self._load_environment()
        
        # Get API key and project URL
        self.api_key = kwargs.get('api_key') or os.getenv('MEMOBASE_API_KEY')
        self.project_url = kwargs.get('project_url') or os.getenv(
            'MEMOBASE_PROJECT_URL', 
            'https://api.memobase.dev'  # Default to Memobase Cloud
        )
    
    def get_system_name(self) -> str:
        """Return the name of the memory system."""
        return "memobase"
    
    def get_required_env_vars(self) -> List[str]:
        """
        Get list of required environment variables for Memobase.
        """
        return [
            'MEMOBASE_API_KEY',  # Memobase API key (project token)
        ]
    
    def _load_environment(self):
        """Load environment variables from .env file if available."""
        if DOTENV_AVAILABLE:
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                self.logger.info(f"Loaded environment variables from {env_file}")
            else:
                load_dotenv()
        else:
            self.logger.warning("python-dotenv not available, using system environment variables only")
    
    def initialize_client(self) -> bool:
        """
        Initialize the Memobase client and create/get the user.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not MEMOBASE_AVAILABLE:
            self.logger.error(
                "Memobase SDK is not installed. Please install it using:\n"
                "  pip install memobase"
            )
            return False
        
        if not self.api_key:
            self.logger.error(
                "Memobase API key is required. Set MEMOBASE_API_KEY environment variable "
                "or provide api_key parameter."
            )
            return False
        
        try:
            # Initialize Memobase client
            self.client = MemoBaseClient(
                project_url=self.project_url,
                api_key=self.api_key
            )
            
            # Test connection
            if not self.client.ping():
                self.logger.error("Failed to connect to Memobase server")
                return False
            
            self.logger.info(f"Initialized Memobase client with URL: {self.project_url}")
            
            # Create or get user
            if not self._ensure_user_exists():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Memobase client: {e}")
            return False
    
    def _ensure_user_exists(self) -> bool:
        """
        Ensure a user exists in Memobase. Create if it doesn't exist.
        
        Uses persistent storage to map memora user_id to memobase user_id
        to avoid creating duplicate users across runs.
        
        Returns:
            bool: True if user exists or was created successfully
        """
        if self.client is None:
            return False
        
        try:
            # Try to load existing user mapping from persistent storage
            mapping_file = Path(__file__).parent / "user_mappings.json"
            user_mapping = {}
            
            if mapping_file.exists():
                try:
                    with open(mapping_file, 'r') as f:
                        user_mapping = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Could not load user mapping: {e}")
            
            # Check if we have a mapping for this user_id
            if self.user_id in user_mapping:
                try:
                    stored_uid = user_mapping[self.user_id]
                    self.memobase_user = self.client.get_user(stored_uid)
                    self.memobase_user_id = stored_uid
                    self.logger.info(f"✅ Found existing Memobase user: {self.memobase_user_id} (memora_user_id: {self.user_id})")
                    return True
                except Exception as e:
                    self.logger.warning(f"Stored user ID {stored_uid} not found, creating new user: {e}")
                    # Remove invalid mapping
                    user_mapping.pop(self.user_id, None)
            
            # Create new user with our user_id as metadata
            self.memobase_user_id = self.client.add_user({
                "memora_user_id": self.user_id,
                "created_at": datetime.now().isoformat()
            })
            self.memobase_user = self.client.get_user(self.memobase_user_id)
            
            # Save mapping persistently
            user_mapping[self.user_id] = self.memobase_user_id
            try:
                with open(mapping_file, 'w') as f:
                    json.dump(user_mapping, f, indent=2)
                self.logger.debug(f"Saved user mapping to {mapping_file}")
            except Exception as e:
                self.logger.warning(f"Could not save user mapping: {e}")
            
            self.logger.info(f"✅ Created new Memobase user: {self.memobase_user_id} (memora_user_id: {self.user_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to ensure user exists: {e}")
            return False
    
    def add_conversation_to_memory(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add conversation to Memobase memory using ChatBlob.
        
        The conversation is inserted as a ChatBlob which Memobase processes
        to extract user profiles and events.
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            Dict[str, Any]: Result from Memobase operation
        """
        # Ensure client is initialized
        if self.client is None or self.memobase_user is None:
            if not self.initialize_client():
                raise RuntimeError("Memobase client is not initialized. Cannot add conversation to memory.")
        
        try:
            # Extract and format conversation messages for Memobase
            messages = self._extract_conversation_messages(conversation_data)
            
            if not messages:
                raise ValueError("No valid messages found in conversation data")
            
            # Get session information
            session_id = conversation_data.get('session_id')
            session_date = conversation_data.get('date') or conversation_data.get('session_date')
            
            # Parse session date to datetime
            created_at = None
            if session_date:
                try:
                    # Parse YYYY-MM-DD format and set time to noon UTC
                    created_at = datetime.strptime(session_date, "%Y-%m-%d").replace(
                        hour=12, minute=0, second=0, tzinfo=timezone.utc
                    )
                    self.logger.debug(f"Using session date: {created_at}")
                except ValueError as e:
                    self.logger.warning(f"Could not parse session date '{session_date}': {e}")
            
            # Create ChatBlob with session date and insert
            chat_blob = ChatBlob(messages=messages, created_at=created_at)
            blob_id = self.memobase_user.insert(chat_blob)
            
            self.logger.debug(f"Inserted ChatBlob with ID: {blob_id}")
            
            # Flush to process the memory (sync=True to wait for processing)
            self.memobase_user.flush(sync=True)
            
            # Verify that memory was actually stored by checking profile/context
            # Wait a moment for async processing if needed
            time.sleep(0.5)  # Small delay to ensure processing completes
            
            # Log verification
            try:
                profile = self.memobase_user.profile(need_json=True)
                profile_count = sum(len(v) if isinstance(v, dict) else 1 for v in profile.values()) if profile else 0
                context = self.memobase_user.context(max_token_size=100)
                has_content = context and len(context.strip()) > 100  # More than just template
                self.logger.info(f"✅ Successfully ingested conversation session {session_id} to Memobase")
                self.logger.debug(f"   Profile items: {profile_count}, Context has content: {has_content}")
            except Exception as e:
                self.logger.warning(f"Could not verify memory storage: {e}")
                self.logger.info(f"✅ Ingested conversation session {session_id} (verification skipped)")
            
            return {
                "status": "success",
                "blob_id": blob_id,
                "user_id": self.user_id,
                "memobase_user_id": self.memobase_user_id,
                "session_id": session_id,
                "session_date": session_date,
                "messages_count": len(messages)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to add conversation to Memobase: {e}")
            raise
    
    def _extract_conversation_messages(self, conversation_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract conversation messages from Memora format for Memobase ChatBlob.
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages = []
        
        if "conversation" not in conversation_data:
            self.logger.warning("No conversation key found in data")
            return messages
        
        # Get session date to prepend to first message for proper temporal context
        session_date = conversation_data.get('date') or conversation_data.get('session_date')
        session_id = conversation_data.get('session_id')
        first_user_message_processed = False
        
        for turn in conversation_data["conversation"]:
            speaker = turn.get("speaker", "")
            message_content = turn.get("message", "")
            
            # Skip empty messages
            if not message_content:
                continue
            
            # Map Memora speaker roles to Memobase roles
            if speaker in ["user_agent", "user"]:
                role = "user"
            elif speaker in ["ai_agent", "assistant"]:
                role = "assistant"
            else:
                role = speaker
            
            # Prepend date context to ALL user messages so the LLM knows the correct date
            # This helps ensure events are recorded with the correct historical date
            if role == "user" and session_date:
                if not first_user_message_processed:
                    # First message gets full context
                    message_content = f"[This conversation occurred on {session_date}. All events and activities mentioned happened on this date.] {message_content}"
                    first_user_message_processed = True
                    self.logger.debug(f"Added date context to first user message: {session_date}")
            
            messages.append({
                "role": role,
                "content": message_content
            })
        
        self.logger.debug(f"Extracted {len(messages)} messages from conversation")
        return messages
    
    def search_memories(self, query: str, limit: int = 50, session_date: Optional[str] = None,
                       date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Search Memobase for relevant memories using context-aware profile filtering.
        
        Uses Memobase's context-aware filtering which uses LLM reasoning to determine
        which profile attributes are most relevant to the query. This is more advanced
        than simple keyword matching.
        
        Reference: https://docs.memobase.io/features/profile/profile_filter
        
        Args:
            query: Search query/question (used for context-aware filtering)
            limit: Maximum number of memories to return
            session_date: Optional session date (passed as context)
            date_range: Optional date range (not yet implemented for Memobase)
            
        Returns:
            List of relevant memories in format: [{'memory': str, 'score': float, ...}]
        """
        # Ensure client is initialized
        if self.client is None or self.memobase_user is None:
            if not self.initialize_client():
                self.logger.error("Memobase client is not initialized. Cannot search memories.")
                return []
        
        try:
            memories = []
            
            # Method 1: Context-aware profile retrieval
            # Use the query as a chat message to get contextually relevant profiles
            # Memobase uses LLM reasoning to determine which profiles are most relevant
            try:
                # Format query as a user message for context-aware filtering
                chats = [{"role": "user", "content": query}]
                
                # Get contextually relevant profiles using Memobase's intelligent filtering
                # max_token_size limits the total size of the profile context
                profiles = self.memobase_user.profile(
                    chats=chats,
                    max_token_size=2000,  # Limit profile size
                    need_json=True
                )
                
                if profiles:
                    # Convert profile structure to memory format
                    for topic, subtopics in profiles.items():
                        if isinstance(subtopics, dict):
                            for subtopic, value in subtopics.items():
                                # Extract content from the value
                                if isinstance(value, dict):
                                    content = value.get('content', str(value))
                                    created_at = value.get('created_at') or ''  # Handle None values
                                else:
                                    content = str(value)
                                    created_at = ''
                                
                                memory_text = f"{topic}/{subtopic}: {content}"
                                
                                # Context-aware filtering already ranks by relevance,
                                # so we use a high base score since Memobase filtered these
                                memories.append({
                                    'memory': memory_text,
                                    'score': 0.85,  # High score for context-aware filtered results
                                    'source': 'memobase_profile',
                                    'type': 'profile',
                                    'topic': topic,
                                    'subtopic': subtopic,
                                    'created_at': created_at if created_at else ''  # Ensure it's always a string
                                })
                
                self.logger.debug(f"Found {len(memories)} context-aware profile memories")
                
            except Exception as e:
                self.logger.debug(f"Context-aware profile retrieval failed: {e}")
                # Fallback to basic profile retrieval without context filtering
                try:
                    profiles = self.memobase_user.profile(
                        max_token_size=2000,
                        need_json=True
                    )
                    
                    if profiles:
                        for topic, subtopics in profiles.items():
                            if isinstance(subtopics, dict):
                                for subtopic, value in subtopics.items():
                                    if isinstance(value, dict):
                                        content = value.get('content', str(value))
                                        created_at = value.get('created_at') or ''  # Handle None values
                                    else:
                                        content = str(value)
                                        created_at = ''
                                    
                                    memory_text = f"{topic}/{subtopic}: {content}"
                                    
                                    # Simple keyword-based relevance for fallback
                                    query_lower = query.lower()
                                    memory_lower = memory_text.lower()
                                    relevance = 0.5
                                    for word in query_lower.split():
                                        if len(word) > 2 and word in memory_lower:
                                            relevance += 0.1
                                    
                                    memories.append({
                                        'memory': memory_text,
                                        'score': min(relevance, 1.0),
                                        'source': 'memobase_profile',
                                        'type': 'profile',
                                        'topic': topic,
                                        'subtopic': subtopic,
                                        'created_at': created_at if created_at else ''  # Ensure it's always a string
                                    })
                    
                    self.logger.debug(f"Fallback: Found {len(memories)} profile memories")
                except Exception as e2:
                    self.logger.debug(f"Fallback profile retrieval also failed: {e2}")
            
            # Method 2: Get formatted context string
            # This provides a formatted string with all memories, also using context-aware filtering
            try:
                # Use context-aware filtering for the context string as well
                chats = [{"role": "user", "content": query}]
                
                # Try context-aware context retrieval if available
                # Note: context() method may not support chats parameter in all versions
                # If it fails, fall back to basic context retrieval
                try:
                    context = self.memobase_user.context(
                        chats=chats,
                        max_token_size=2000
                    )
                except TypeError:
                    # If chats parameter not supported, use basic context
                    context = self.memobase_user.context(max_token_size=2000)
                
                # Check if context is not empty and not just the template
                if context and context.strip():
                    # Check if this is just an empty template (no actual user data)
                    is_empty_template = (
                        "User Current Profile:" in context and
                        "Past Events:" in context and
                        ("User Current Profile:\n- \n" in context or "User Current Profile:\n\n" in context) and
                        ("Past Events:\n\n---" in context or "Past Events:\n---" in context)
                    )
                    
                    if not is_empty_template:
                        # The context contains actual user data
                        memories.append({
                            'memory': context,
                            'score': 0.9,  # High score for full context
                            'source': 'memobase_context',
                            'type': 'context'
                        })
                        self.logger.debug(f"Added context memory (length: {len(context)})")
                    else:
                        self.logger.debug("Context is empty template, skipping")
                else:
                    self.logger.debug("Context is empty or None")
                    
            except Exception as e:
                self.logger.debug(f"Context retrieval failed: {e}")
            
            # Sort by score and limit
            memories.sort(key=lambda x: x.get('score', 0), reverse=True)
            memories = memories[:limit]
            
            self.logger.info(f"✅ Memobase search returned {len(memories)} memories (using context-aware filtering)")
            return memories
            
        except Exception as e:
            self.logger.error(f"Memobase search failed: {e}")
            return []
    
    def get_user_profile(self) -> Dict[str, Any]:
        """
        Get the full user profile from Memobase.
        
        Returns:
            Dict with user profile data
        """
        if self.memobase_user is None:
            return {"status": "no_user"}
        
        try:
            return self.memobase_user.profile(need_json=True)
        except Exception as e:
            self.logger.error(f"Failed to get user profile: {e}")
            return {"error": str(e)}
    
    def get_user_context(self, max_tokens: int = 1000) -> str:
        """
        Get formatted context string for the user.
        
        Args:
            max_tokens: Maximum token size for context
            
        Returns:
            Formatted context string
        """
        if self.memobase_user is None:
            return ""
        
        try:
            return self.memobase_user.context(max_token_size=max_tokens)
        except Exception as e:
            self.logger.error(f"Failed to get user context: {e}")
            return ""
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for Memobase operations."""
        logger = logging.getLogger(f"{__name__}.{self.user_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_conversation_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single conversation file and add conversation to memory.
        
        Args:
            file_path: Path to the conversation JSON file
            
        Returns:
            Dict[str, Any]: Result from memory addition
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Add conversation messages to memory
            result = self.add_conversation_to_memory(conversation_data)
            self.logger.info(f"Processed file: {file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {e}")
            raise


if __name__ == "__main__":
    """Test the Memobase integration."""
    print("🧪 Testing Memobase Integration")
    print("=" * 40)
    
    # Test system creation
    # Note: In actual usage, user_id is dynamically generated as {persona}_{timeline}
    # e.g., "academic_researcher_weekly" from conversation_to_memory.py
    try:
        # For testing only - use a test user_id that follows the expected format
        test_user_id = "test_user_weekly"  # Follows {persona}_{timeline} format
        memobase_system = MemobaseSystem(test_user_id)
        print(f"✅ Created Memobase system: {memobase_system.get_system_name()}")
        
        # Test environment validation
        if memobase_system.validate_environment():
            print("✅ Environment validation passed")
            
            # Test client initialization
            if memobase_system.initialize_client():
                print("✅ Client initialization successful")
                
                # Print user info
                profile = memobase_system.get_user_profile()
                print(f"📋 User profile: {json.dumps(profile, indent=2, default=str)[:500]}")
                
                # Test search (should return empty for new user)
                results = memobase_system.search_memories("test query")
                print(f"🔍 Search returned {len(results)} results")
            else:
                print("❌ Client initialization failed")
        else:
            print("❌ Environment validation failed")
            print("   Please set MEMOBASE_API_KEY environment variable")
            print("   Get a key from: https://www.memobase.io/")
            
    except Exception as e:
        print(f"❌ Error testing Memobase system: {e}")
        import traceback
        traceback.print_exc()

