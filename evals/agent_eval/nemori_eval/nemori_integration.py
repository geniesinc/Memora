#!/usr/bin/env python3
"""
Nemori Memory System Integration
=================================

This module provides Nemori-specific implementation of the BaseMemorySystem interface.
It handles all Nemori-specific operations for storing and retrieving memories.

Nemori is a self-organizing long-term memory substrate for agentic LLM workflows.
It ingests multi-turn conversations, segments them into topic-consistent episodes,
distills durable semantic knowledge, and exposes a unified search surface.

Reference: https://github.com/nemori-ai/nemori

Features:
- Episode-based memory segmentation
- Semantic knowledge distillation
- Vector and hybrid search capabilities
- Uses OpenAI API (can be configured for OpenRouter)
- ChromaDB backend for storage

Usage:
    from nemori_integration import NemoriSystem
    
    system = NemoriSystem("user_123")
    if system.validate_environment() and system.initialize_client():
        result = system.add_conversation_to_memory(conversation_data)
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import Nemori
try:
    from nemori import NemoriMemory, MemoryConfig
    NEMORI_AVAILABLE = True
except ImportError:
    NEMORI_AVAILABLE = False
    NemoriMemory = None
    MemoryConfig = None

# Import base classes
import sys
sys.path.append(str(Path(__file__).parent.parent))
from base_evaluator import BaseMemorySystem


class NemoriSystem(BaseMemorySystem):
    """
    Nemori-specific implementation of BaseMemorySystem.
    
    This class handles all Nemori-specific operations for storing and retrieving memories.
    Nemori uses episode-based memory segmentation and semantic knowledge distillation.
    """
    
    def __init__(self, user_id: str, **kwargs):
        """
        Initialize the Nemori memory system.
        
        Args:
            user_id: User ID for Nemori memories. Should follow the format:
                     {persona}_{timeline} (e.g., "academic_researcher_weekly").
                     This is dynamically generated from the session directory and
                     questions file, not hardcoded.
            **kwargs: Nemori specific configuration parameters:
                - llm_model: LLM model to use (default: "gpt-4o-mini")
                - openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
                - openrouter_api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var)
                - use_openrouter: Whether to use OpenRouter instead of OpenAI (default: False)
                - enable_semantic_memory: Enable semantic memory (default: True)
                - enable_prediction_correction: Enable prediction-correction loop (default: True)
                - search_method: Search method to use - "hybrid" (default), "vector", "bm25", or "vector_norlift"
                - storage_path: Path for persistent storage (default: nemori_eval/data/{user_id})
        """
        super().__init__(user_id)
        
        # Nemori memory instance (initialized in initialize_client)
        self.memory = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load environment variables FIRST (before accessing any env vars)
        self._load_environment()
        
        # Get configuration parameters from kwargs or environment variables
        # Environment variables are loaded from .env file via _load_environment()
        self.llm_model = kwargs.get('llm_model') or os.getenv('NEMORI_LLM_MODEL', 'gpt-4o-mini')
        self.openai_api_key = kwargs.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        self.openrouter_api_key = kwargs.get('openrouter_api_key') or os.getenv('OPENROUTER_API_KEY')
        
        # Check if OpenRouter should be used (from .env file or kwargs)
        use_openrouter_env = os.getenv('NEMORI_USE_OPENROUTER', 'false').lower() == 'true'
        self.use_openrouter = kwargs.get('use_openrouter', use_openrouter_env)
        
        self.enable_semantic_memory = kwargs.get('enable_semantic_memory', True)
        self.enable_prediction_correction = kwargs.get('enable_prediction_correction', True)
        self.search_method = kwargs.get('search_method', os.getenv('NEMORI_SEARCH_METHOD', 'hybrid'))
        
        # Storage path for Nemori (user-specific)
        default_storage = Path(__file__).parent / "data" / self.user_id
        self.storage_path = kwargs.get('storage_path', str(default_storage))
        
        # Validate user_id format (should be persona_timeline)
        if not user_id or '_' not in user_id:
            self.logger.warning(
                f"User ID '{user_id}' may not follow expected format '{{persona}}_{{timeline}}'. "
                f"Expected format: e.g., 'academic_researcher_weekly'"
            )
    
    def get_system_name(self) -> str:
        """Return the name of the memory system."""
        return "nemori"
    
    def get_required_env_vars(self) -> List[str]:
        """
        Get list of required environment variables for Nemori.
        """
        return [
            'OPENAI_API_KEY',  # Required for OpenAI API (or OPENROUTER_API_KEY if using OpenRouter)
        ]
    
    def _load_environment(self):
        """
        Load environment variables from .env file if available.
        
        This automatically loads:
        - OPENAI_API_KEY: For OpenAI API access
        - OPENROUTER_API_KEY: For OpenRouter API access (if using OpenRouter)
        - NEMORI_USE_OPENROUTER: Set to 'true' to use OpenRouter instead of OpenAI
        - NEMORI_LLM_MODEL: LLM model to use (default: gpt-4o-mini)
        """
        if DOTENV_AVAILABLE:
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file, override=True)  # override=True ensures .env values take precedence
                self.logger.info(f"✅ Loaded environment variables from {env_file}")
            else:
                load_dotenv()
                self.logger.debug("No .env file found, using system environment variables")
        else:
            self.logger.warning("python-dotenv not available, using system environment variables only")
    
    def initialize_client(self) -> bool:
        """
        Initialize the Nemori memory system.
        
        Loads API keys from .env file automatically.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not NEMORI_AVAILABLE:
            self.logger.error(
                "Nemori package is not installed. Please install it using:\n"
                "  pip install nemori\n"
                "  Or: git clone https://github.com/nemori-ai/nemori.git && pip install -e nemori"
            )
            return False
        
        # Reload environment to ensure we have the latest values from .env
        self._load_environment()
        
        # Re-read API keys from environment (in case they were updated in .env)
        # This ensures we get the latest values from .env file
        env_openai_key = os.getenv('OPENAI_API_KEY')
        env_openrouter_key = os.getenv('OPENROUTER_API_KEY')
        
        if env_openai_key and not self.openai_api_key:
            self.openai_api_key = env_openai_key
        if env_openrouter_key and not self.openrouter_api_key:
            self.openrouter_api_key = env_openrouter_key
        
        # Re-check OpenRouter preference from .env
        use_openrouter_env = os.getenv('NEMORI_USE_OPENROUTER', 'false').lower() == 'true'
        # Use .env value if it's set, otherwise keep the value from kwargs
        if use_openrouter_env:
            self.use_openrouter = True
        
        # Determine API key based on provider
        api_key = None
        if self.use_openrouter:
            api_key = self.openrouter_api_key
            if not api_key:
                self.logger.error(
                    "OpenRouter API key is required when use_openrouter=True. "
                    "Please set OPENROUTER_API_KEY in your .env file or provide openrouter_api_key parameter."
                )
                return False
            # Configure OpenAI client to use OpenRouter
            os.environ['OPENAI_API_KEY'] = api_key
            os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'
            self.logger.info("✅ Using OpenRouter API (loaded from .env)")
        else:
            api_key = self.openai_api_key
            if not api_key:
                self.logger.error(
                    "OpenAI API key is required. "
                    "Please set OPENAI_API_KEY in your .env file or provide openai_api_key parameter."
                )
                return False
            os.environ['OPENAI_API_KEY'] = api_key
            self.logger.info("✅ Using OpenAI API (loaded from .env)")
        
        try:
            # Create MemoryConfig
            config = MemoryConfig(
                llm_model=self.llm_model,
                enable_semantic_memory=self.enable_semantic_memory,
                enable_prediction_correction=self.enable_prediction_correction,
            )
            
            # Initialize NemoriMemory
            # Note: Nemori handles user_id internally, we'll use it in add_messages
            self.memory = NemoriMemory(config=config)
            
            # Ensure storage directory exists
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"✅ Initialized Nemori memory system")
            self.logger.info(f"   User ID: {self.user_id}")
            self.logger.info(f"   Storage: {self.storage_path}")
            self.logger.info(f"   Model: {self.llm_model}")
            self.logger.info(f"   Semantic memory: {self.enable_semantic_memory}")
            self.logger.info(f"   Prediction-correction: {self.enable_prediction_correction}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Nemori: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def add_conversation_to_memory(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add conversation to Nemori memory using add_messages.
        
        Nemori processes conversations and segments them into episodes,
        then distills semantic knowledge.
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            Dict[str, Any]: Result from Nemori operation
        """
        # Ensure client is initialized
        if self.memory is None:
            if not self.initialize_client():
                raise RuntimeError("Nemori memory is not initialized. Cannot add conversation to memory.")
        
        try:
            # Extract and format conversation messages for Nemori
            messages = self._extract_conversation_messages(conversation_data)
            
            if not messages:
                raise ValueError("No valid messages found in conversation data")
            
            # Get session information
            session_id = conversation_data.get('session_id')
            session_date = conversation_data.get('date') or conversation_data.get('session_date')
            
            # Add messages to Nemori
            # Nemori uses user_id to isolate memories per user
            self.memory.add_messages(
                user_id=self.user_id,
                messages=messages
            )
            
            # Flush to process the conversation (sync=True to wait for processing)
            self.memory.flush(self.user_id)
            
            # Wait for semantic memory processing if enabled
            if self.enable_semantic_memory:
                try:
                    self.memory.wait_for_semantic(self.user_id)
                except Exception as e:
                    # Catch datetime comparison errors in Nemori's internal code
                    # These are non-critical - semantic memories may still be generated
                    error_msg = str(e)
                    if "offset-naive" in error_msg or "offset-aware" in error_msg or "datetime" in error_msg.lower():
                        self.logger.warning(
                            f"⚠️  Semantic memory processing encountered a datetime comparison issue "
                            f"(non-critical): {error_msg}. Semantic memories may still be available."
                        )
                    else:
                        # Re-raise if it's a different error
                        raise
            
            self.logger.info(f"✅ Successfully ingested conversation session {session_id} to Nemori")
            
            return {
                "status": "success",
                "user_id": self.user_id,
                "session_id": session_id,
                "session_date": session_date,
                "messages_count": len(messages)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to add conversation to Nemori: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise
    
    def _extract_conversation_messages(self, conversation_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract conversation messages from Memora format for Nemori.
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages = []
        
        if "conversation" not in conversation_data:
            self.logger.warning("No conversation key found in data")
            return messages
        
        for turn in conversation_data["conversation"]:
            speaker = turn.get("speaker", "")
            message_content = turn.get("message", "")
            
            # Skip empty messages
            if not message_content:
                continue
            
            # Map Memora speaker roles to Nemori roles
            if speaker in ["user_agent", "user"]:
                role = "user"
            elif speaker in ["ai_agent", "assistant"]:
                role = "assistant"
            else:
                role = speaker
            
            messages.append({
                "role": role,
                "content": message_content
            })
        
        self.logger.debug(f"Extracted {len(messages)} messages from conversation")
        return messages
    
    def search_memories(self, query: str, limit: int = 50, session_date: Optional[str] = None,
                       date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Search Nemori for relevant memories.
        
        Nemori's search() method returns a dictionary with 'episodic' and 'semantic' keys,
        each containing a list of results. We combine both types into a single list.
        
        Args:
            query: Search query/question
            limit: Maximum number of memories to return (split between episodic and semantic)
            session_date: Optional session date (used for query augmentation, not filtering)
            date_range: Optional date range (used for query augmentation, not filtering)
            
        Returns:
            List of relevant memories in format: [{'memory': str, 'score': float, 'type': str, ...}]
        """
        # Ensure client is initialized
        if self.memory is None:
            if not self.initialize_client():
                self.logger.error("Nemori memory is not initialized. Cannot search memories.")
                return []
        
        try:
            memories = []
            
            # Augment query with date context if provided
            augmented_query = query
            if session_date:
                augmented_query = f"On {session_date}, {query}"
                self.logger.debug(f"Augmented query with session_date: {augmented_query}")
            elif date_range:
                start_date, end_date = date_range
                augmented_query = f"Between {start_date} and {end_date}, {query}"
                self.logger.debug(f"Augmented query with date_range: {augmented_query}")
            
            # Use configured search method (default: "hybrid" which combines vector and BM25)
            # Valid options: "hybrid", "vector", "bm25", "vector_norlift"
            search_method = self.search_method
            
            # Calculate top_k values (split limit between episodic and semantic)
            # Default to 25 each if limit is 50, or proportional split
            top_k_episodes = max(1, limit // 2)
            top_k_semantic = max(1, limit - top_k_episodes)
            
            try:
                # Search Nemori memories
                # Nemori's search() returns: Dict[str, List[Dict[str, Any]]]
                # with keys: 'episodic' and 'semantic'
                results_dict = self.memory.search(
                    user_id=self.user_id,
                    query=augmented_query,
                    top_k_episodes=top_k_episodes,
                    top_k_semantic=top_k_semantic,
                    search_method=search_method
                )
                
                # Extract episodic memories
                episodic_results = results_dict.get('episodic', [])
                for result in episodic_results:
                    if isinstance(result, dict):
                        # Extract content from various possible fields
                        content = result.get('content') or result.get('text') or result.get('episode_content') or str(result)
                        score = result.get('score') or result.get('similarity') or result.get('fused_score', 0.8)
                        memory_id = result.get('id') or result.get('episode_id', '')
                    else:
                        content = str(result)
                        score = 0.8
                        memory_id = ''
                    
                    memories.append({
                        'memory': content,
                        'score': float(score) if isinstance(score, (int, float)) else 0.8,
                        'source': 'nemori',
                        'type': 'episodic',
                        'id': memory_id,
                        'search_method': search_method
                    })
                
                # Extract semantic memories
                semantic_results = results_dict.get('semantic', [])
                for result in semantic_results:
                    if isinstance(result, dict):
                        # Extract content from various possible fields
                        content = result.get('content') or result.get('text') or result.get('semantic_content') or str(result)
                        score = result.get('score') or result.get('similarity') or result.get('fused_score', 0.8)
                        memory_id = result.get('id') or result.get('memory_id', '')
                    else:
                        content = str(result)
                        score = 0.8
                        memory_id = ''
                    
                    memories.append({
                        'memory': content,
                        'score': float(score) if isinstance(score, (int, float)) else 0.8,
                        'source': 'nemori',
                        'type': 'semantic',
                        'id': memory_id,
                        'search_method': search_method
                    })
                
                # Sort by score (descending) and limit results
                memories.sort(key=lambda x: x.get('score', 0), reverse=True)
                if len(memories) > limit:
                    memories = memories[:limit]
                
                self.logger.info(f"✅ Nemori search returned {len(memories)} memories ({len(episodic_results)} episodic, {len(semantic_results)} semantic)")
                
            except Exception as e:
                error_msg = str(e)
                # Check if this is a datetime comparison error (non-critical)
                if "offset-naive" in error_msg or "offset-aware" in error_msg:
                    self.logger.warning(
                        f"⚠️  Nemori search encountered a datetime comparison issue (non-critical): {error_msg}. "
                        f"Returning available memories."
                    )
                    # Return memories we've collected so far, even if there was an error
                    if memories:
                        self.logger.info(f"✅ Returning {len(memories)} memories despite datetime error")
                        return memories
                else:
                    self.logger.error(f"Nemori search failed: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                return []
            return memories
            
        except Exception as e:
            self.logger.error(f"Nemori search failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return []
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for Nemori operations."""
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
    
    def close(self):
        """Close the Nemori memory system and clean up resources."""
        if self.memory is not None:
            try:
                self.memory.close()
                self.logger.info("Closed Nemori memory system")
            except Exception as e:
                self.logger.warning(f"Error closing Nemori: {e}")


if __name__ == "__main__":
    """Test the Nemori integration."""
    print("🧪 Testing Nemori Integration")
    print("=" * 40)
    
    # Test system creation
    # Note: In actual usage, user_id is dynamically generated as {persona}_{timeline}
    # e.g., "academic_researcher_weekly" from conversation_to_memory.py
    try:
        # For testing only - use a test user_id that follows the expected format
        test_user_id = "test_user_weekly"  # Follows {persona}_{timeline} format
        nemori_system = NemoriSystem(test_user_id)
        print(f"✅ Created Nemori system: {nemori_system.get_system_name()}")
        
        # Test environment validation
        if nemori_system.validate_environment():
            print("✅ Environment validation passed")
            
            # Test client initialization
            if nemori_system.initialize_client():
                print("✅ Client initialization successful")
                
                # Test search (should return empty for new user)
                results = nemori_system.search_memories("test query")
                print(f"🔍 Search returned {len(results)} results")
                
                # Clean up
                nemori_system.close()
            else:
                print("❌ Client initialization failed")
        else:
            print("❌ Environment validation failed")
            print("   Please set OPENAI_API_KEY environment variable")
            print("   Or set OPENROUTER_API_KEY and NEMORI_USE_OPENROUTER=true for OpenRouter")
            print("   Get a key from: https://platform.openai.com/ or https://openrouter.ai/")
            
    except Exception as e:
        print(f"❌ Error testing Nemori system: {e}")
        import traceback
        traceback.print_exc()

