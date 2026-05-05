#!/usr/bin/env python3
"""
LangMem Memory System Integration
==================================

This module provides LangMem-specific implementation of the BaseMemorySystem interface.
It handles all LangMem-specific operations for storing and retrieving memories.

LangMem is a LangChain/LangGraph memory system that:
- Extracts important information from conversations
- Optimizes agent behavior through prompt refinement
- Maintains long-term memory with semantic search

Implementation follows LangMem documentation:
- https://langchain-ai.github.io/langmem/
- https://github.com/langchain-ai/langmem

Features:
- Core memory API with semantic search
- Background memory manager for automatic extraction
- Integration with LangGraph's storage layer

Usage:
    from langmem_integration import LangMemSystem
    
    system = LangMemSystem("user_123")
    if system.validate_environment() and system.initialize_client():
        # Use the system for memory operations
        result = system.add_conversation_to_memory(conversation_data)
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import hashlib

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import LangMem and LangGraph components
LANGMEM_AVAILABLE = False
LANGGRAPH_AVAILABLE = False

try:
    from langmem import create_manage_memory_tool, create_search_memory_tool
    from langmem import create_memory_store_manager
    LANGMEM_AVAILABLE = True
except ImportError:
    create_manage_memory_tool = None
    create_search_memory_tool = None
    create_memory_store_manager = None

try:
    from langgraph.store.memory import InMemoryStore
    LANGGRAPH_AVAILABLE = True
except ImportError:
    InMemoryStore = None

# Import base classes
import sys
sys.path.append(str(Path(__file__).parent.parent))
from base_evaluator import BaseMemorySystem


class LangMemSystem(BaseMemorySystem):
    """
    LangMem-specific implementation of BaseMemorySystem.
    
    This class handles all LangMem-specific operations for storing and retrieving memories.
    LangMem uses a store-based approach with semantic search capabilities.
    """
    
    def __init__(self, user_id: str, **kwargs):
        """
        Initialize the LangMem memory system.
        
        Args:
            user_id: User ID for LangMem memories
            **kwargs: LangMem specific configuration parameters:
                - openai_api_key: OpenAI API key for embeddings
                - anthropic_api_key: Anthropic API key for LLM
                - embedding_model: Embedding model to use (default: openai:text-embedding-3-small)
                - embedding_dims: Embedding dimensions (default: 1536)
        """
        super().__init__(user_id)
        
        # LangMem store and tools (initialized in initialize_client)
        self.store = None
        self.memory_manager = None
        self.namespace = None
        
        # Configuration
        self.embedding_model = kwargs.get('embedding_model', 'openai:text-embedding-3-small')
        self.embedding_dims = kwargs.get('embedding_dims', 1536)
        
        # Persistence: file path for storing memories
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.persistence_file = self.data_dir / f"langmem_{user_id}.json"
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load environment variables
        self._load_environment()
        
        # Get API keys
        self.openai_api_key = kwargs.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = kwargs.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY')
    
    def get_system_name(self) -> str:
        """Return the name of the memory system."""
        return "langmem"
    
    def get_required_env_vars(self) -> List[str]:
        """
        Get list of required environment variables for LangMem.
        """
        return [
            'OPENAI_API_KEY',  # For embeddings
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
    
    def _save_to_file(self):
        """Save the current store data to a JSON file for persistence."""
        if self.store is None:
            return
        
        try:
            # Get all items from the store for this user's namespace
            memories = []
            namespace_prefix = self.namespace
            
            # Search with empty query to get all items
            try:
                results = self.store.search(namespace_prefix, query="", limit=10000)
                for item in results:
                    memories.append({
                        'key': item.key,
                        'value': item.value,
                        'namespace': list(item.namespace) if hasattr(item, 'namespace') else list(namespace_prefix),
                    })
            except Exception as e:
                self.logger.warning(f"Could not retrieve memories for persistence: {e}")
                return
            
            # Save to JSON file
            data = {
                'user_id': self.user_id,
                'namespace': list(self.namespace),
                'memories': memories,
                'saved_at': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Saved {len(memories)} memories to {self.persistence_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save memories to file: {e}")
    
    def _load_from_file(self):
        """Load store data from JSON file if it exists."""
        if not self.persistence_file.exists():
            self.logger.info(f"No persistence file found at {self.persistence_file}")
            return
        
        try:
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)
            
            memories = data.get('memories', [])
            if not memories:
                self.logger.info("No memories found in persistence file")
                return
            
            # Restore memories to the store
            loaded_count = 0
            for mem in memories:
                try:
                    namespace = tuple(mem.get('namespace', self.namespace))
                    key = mem.get('key')
                    value = mem.get('value')
                    
                    if key and value:
                        self.store.put(namespace, key, value)
                        loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to restore memory {mem.get('key')}: {e}")
            
            self.logger.info(f"Loaded {loaded_count} memories from {self.persistence_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load memories from file: {e}")
    
    def initialize_client(self) -> bool:
        """
        Initialize the LangMem store and memory tools.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not LANGMEM_AVAILABLE:
            self.logger.error(
                "LangMem SDK is not installed. Please install it using:\n"
                "  pip install langmem"
            )
            return False
        
        if not LANGGRAPH_AVAILABLE:
            self.logger.error(
                "LangGraph is not installed. Please install it using:\n"
                "  pip install langgraph"
            )
            return False
        
        if not self.openai_api_key:
            self.logger.error(
                "OpenAI API key is required for embeddings. Set OPENAI_API_KEY environment variable."
            )
            return False
        
        try:
            # Set the API key in environment for LangMem to use
            os.environ['OPENAI_API_KEY'] = self.openai_api_key
            
            # Create InMemoryStore with embedding configuration
            self.store = InMemoryStore(
                index={
                    "dims": self.embedding_dims,
                    "embed": self.embedding_model,
                }
            )
            
            # Set up namespace for this user's memories
            self.namespace = ("memories", self.user_id)
            
            # Load persisted memories if available
            self._load_from_file()
            
            self.logger.info(f"Initialized LangMem with InMemoryStore (dims={self.embedding_dims})")
            self.logger.info(f"User namespace: {self.namespace}")
            self.logger.info(f"Persistence file: {self.persistence_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangMem: {e}")
            return False
    
    def add_conversation_to_memory(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add conversation to LangMem memory store.
        
        Extracts memories from the conversation and stores them with semantic embeddings.
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            Dict[str, Any]: Result from LangMem operation
        """
        # Ensure client is initialized
        if self.store is None:
            if not self.initialize_client():
                raise RuntimeError("LangMem store is not initialized. Cannot add conversation to memory.")
        
        try:
            # Get session information
            session_id = conversation_data.get('session_id')
            session_date = conversation_data.get('date') or conversation_data.get('session_date')
            
            # Extract memories from conversation
            memories = self._extract_memories_from_conversation(conversation_data)
            
            if not memories:
                self.logger.warning(f"No memories extracted from session {session_id}")
                return {
                    "status": "success",
                    "memories_stored": 0,
                    "session_id": session_id
                }
            
            # Store each memory in the store
            stored_count = 0
            for memory in memories:
                try:
                    # Create a unique key for this memory
                    memory_key = self._generate_memory_key(memory['content'], session_id)
                    
                    # Store the memory with metadata
                    self.store.put(
                        namespace=self.namespace,
                        key=memory_key,
                        value={
                            "content": memory['content'],
                            "type": memory.get('type', 'general'),
                            "session_id": session_id,
                            "session_date": session_date,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "user_id": self.user_id
                        }
                    )
                    stored_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to store memory: {e}")
            
            self.logger.info(f"✅ Stored {stored_count} memories from session {session_id}")
            
            # Persist to file for cross-session access
            self._save_to_file()
            
            return {
                "status": "success",
                "memories_stored": stored_count,
                "user_id": self.user_id,
                "session_id": session_id,
                "session_date": session_date,
                "namespace": self.namespace
            }
            
        except Exception as e:
            self.logger.error(f"Failed to add conversation to LangMem: {e}")
            raise
    
    def _extract_memories_from_conversation(self, conversation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract memorable information from a conversation.
        
        This extracts key facts, preferences, and information from the conversation
        to store as individual memories.
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            List of memory dicts with 'content' and 'type' keys
        """
        memories = []
        
        if "conversation" not in conversation_data:
            self.logger.warning("No conversation key found in data")
            return memories
        
        # Get session context
        session_date = conversation_data.get('date') or conversation_data.get('session_date')
        session_id = conversation_data.get('session_id')
        memory_type = conversation_data.get('memory_type', 'general')
        
        # Build conversation text for context
        conversation_text = []
        user_messages = []
        
        for turn in conversation_data["conversation"]:
            speaker = turn.get("speaker", "")
            message = turn.get("message", "")
            
            if not message:
                continue
            
            if speaker in ["user_agent", "user"]:
                user_messages.append(message)
                conversation_text.append(f"User: {message}")
            elif speaker in ["ai_agent", "assistant"]:
                conversation_text.append(f"Assistant: {message}")
        
        # Store the full conversation as a memory with date context
        if conversation_text:
            full_conversation = "\n".join(conversation_text)
            date_prefix = f"[Conversation from {session_date}] " if session_date else ""
            memories.append({
                "content": f"{date_prefix}{full_conversation}",
                "type": "conversation"
            })
        
        # Extract individual user statements as separate memories
        for msg in user_messages:
            if len(msg) > 20:  # Only store substantial messages
                date_prefix = f"[{session_date}] " if session_date else ""
                memories.append({
                    "content": f"{date_prefix}User said: {msg}",
                    "type": "user_statement"
                })
        
        # Add a summary memory if there's content
        if conversation_text and session_date:
            summary = f"On {session_date}, the user had a conversation about: "
            # Extract topics from the conversation
            topics = self._extract_topics(user_messages)
            if topics:
                summary += ", ".join(topics[:3])
                memories.append({
                    "content": summary,
                    "type": "summary"
                })
        
        self.logger.debug(f"Extracted {len(memories)} memories from conversation")
        return memories
    
    def _extract_topics(self, messages: List[str]) -> List[str]:
        """Extract main topics from user messages."""
        # Simple topic extraction based on keywords
        topics = []
        keywords = {
            "work": ["work", "job", "career", "office", "meeting", "project"],
            "food": ["food", "eat", "restaurant", "coffee", "lunch", "dinner", "breakfast"],
            "health": ["health", "exercise", "gym", "doctor", "sleep", "tired"],
            "travel": ["travel", "trip", "vacation", "flight", "hotel"],
            "family": ["family", "kids", "children", "parents", "spouse", "wife", "husband"],
            "hobbies": ["hobby", "reading", "music", "movie", "game", "sport"],
            "learning": ["learn", "study", "course", "book", "research"],
        }
        
        text = " ".join(messages).lower()
        for topic, words in keywords.items():
            if any(word in text for word in words):
                topics.append(topic)
        
        return topics
    
    def _generate_memory_key(self, content: str, session_id: Any) -> str:
        """Generate a unique key for a memory."""
        # Create a hash of the content and session for uniqueness
        hash_input = f"{content}_{session_id}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def search_memories(self, query: str, limit: int = 50, session_date: Optional[str] = None,
                       date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Search LangMem store for relevant memories using semantic search.
        
        Args:
            query: Search query/question
            limit: Maximum number of memories to return
            session_date: Optional session date (not yet implemented for LangMem)
            date_range: Optional date range (not yet implemented for LangMem)
            
        Returns:
            List of relevant memories in format: [{'memory': str, 'score': float, ...}]
        """
        # Ensure client is initialized
        if self.store is None:
            if not self.initialize_client():
                self.logger.error("LangMem store is not initialized. Cannot search memories.")
                return []
        
        try:
            memories = []
            
            # Search the store using semantic search
            # LangGraph store.search takes namespace_prefix as positional arg
            try:
                results = self.store.search(
                    self.namespace,  # namespace_prefix as positional argument
                    query=query,
                    limit=limit
                )
                
                for result in results:
                    # Extract memory content and metadata
                    value = result.value if hasattr(result, 'value') else result.get('value', {})
                    score = result.score if hasattr(result, 'score') else result.get('score', 0.5)
                    
                    if isinstance(value, dict):
                        content = value.get('content', str(value))
                        memory_type = value.get('type', 'general')
                        session_date = value.get('session_date', '')
                    else:
                        content = str(value)
                        memory_type = 'general'
                        session_date = ''
                    
                    memories.append({
                        'memory': content,
                        'score': float(score) if score else 0.5,
                        'source': 'langmem',
                        'type': memory_type,
                        'session_date': session_date
                    })
                
            except Exception as e:
                self.logger.warning(f"Semantic search failed: {e}")
                
                # Fallback: search without query to get all items
                try:
                    all_results = self.store.search(self.namespace, limit=limit)
                    for result in all_results:
                        value = result.value if hasattr(result, 'value') else {}
                        if isinstance(value, dict):
                            content = value.get('content', str(value))
                        else:
                            content = str(value)
                        
                        # Simple keyword matching for relevance
                        query_words = set(query.lower().split())
                        content_words = set(content.lower().split())
                        overlap = len(query_words & content_words)
                        score = min(0.5 + (overlap * 0.1), 1.0)
                        
                        memories.append({
                            'memory': content,
                            'score': score,
                            'source': 'langmem_fallback',
                            'type': 'general'
                        })
                except Exception as e2:
                    self.logger.warning(f"Fallback search also failed: {e2}")
            
            # Sort by score
            memories.sort(key=lambda x: x.get('score', 0), reverse=True)
            memories = memories[:limit]
            
            self.logger.info(f"✅ LangMem search returned {len(memories)} memories")
            return memories
            
        except Exception as e:
            self.logger.error(f"LangMem search failed: {e}")
            return []
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Get all memories for this user from the store.
        
        Returns:
            List of all memories
        """
        if self.store is None:
            return []
        
        try:
            all_items = list(self.store.list(namespace=self.namespace))
            memories = []
            for item in all_items:
                value = item.value if hasattr(item, 'value') else item.get('value', {})
                memories.append(value)
            return memories
        except Exception as e:
            self.logger.error(f"Failed to get all memories: {e}")
            return []
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for LangMem operations."""
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
    """Test the LangMem integration."""
    print("🧪 Testing LangMem Integration")
    print("=" * 40)
    
    # Test system creation
    try:
        langmem_system = LangMemSystem("test_user")
        print(f"✅ Created LangMem system: {langmem_system.get_system_name()}")
        
        # Test environment validation
        if langmem_system.validate_environment():
            print("✅ Environment validation passed")
            
            # Test client initialization
            if langmem_system.initialize_client():
                print("✅ Client initialization successful")
                
                # Test storing a memory
                test_conversation = {
                    "session_id": "test_001",
                    "date": "2025-06-01",
                    "conversation": [
                        {"speaker": "user", "message": "I love coffee, especially dark roast."},
                        {"speaker": "assistant", "message": "That's great! Do you have a favorite coffee shop?"},
                        {"speaker": "user", "message": "Yes, I usually go to Blue Bottle."}
                    ]
                }
                
                result = langmem_system.add_conversation_to_memory(test_conversation)
                print(f"📝 Stored memories: {result.get('memories_stored')}")
                
                # Test search
                results = langmem_system.search_memories("coffee preferences")
                print(f"🔍 Search returned {len(results)} results")
                for i, mem in enumerate(results[:3]):
                    print(f"   [{i+1}] {mem.get('memory', '')[:100]}...")
            else:
                print("❌ Client initialization failed")
        else:
            print("❌ Environment validation failed")
            print("   Please set OPENAI_API_KEY environment variable")
            
    except Exception as e:
        print(f"❌ Error testing LangMem system: {e}")
        import traceback
        traceback.print_exc()

