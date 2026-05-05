#!/usr/bin/env python3
"""
Mem0 Memory System Integration
==============================

This module provides Mem0-specific implementation of the BaseMemorySystem interface.
It handles all Mem0-specific operations for storing and retrieving memories.

Features:
- Mem0 API integration
- Conversation data extraction and formatting
- User-specific memory storage
- Smart retry logic with exponential backoff
- Semantic search capabilities
"""

import json
import os
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    MemoryClient = None

# Import base classes
import sys
sys.path.append(str(Path(__file__).parent.parent))
from base_evaluator import BaseMemorySystem


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is transient and should be retried.
    
    Retryable errors (transient):
    - Network errors (timeouts, connection errors)
    - Server errors (502, 503, 504)
    - Rate limiting (429)
    
    Non-retryable errors (permanent):
    - Client errors (400, 401, 403, 404, 422)
    - Validation errors
    - Programming errors (ValueError, TypeError, etc.)
    
    Args:
        error: The exception to check
        
    Returns:
        bool: True if error should be retried, False otherwise
    """
    error_msg = str(error).lower()
    
    # Retryable: Server errors (5xx)
    retryable_patterns = [
        '502', 'bad gateway',
        '503', 'service unavailable',
        '504', 'gateway timeout',
        'timeout', 'timed out',
        'connection', 'connect',
        '429', 'too many requests', 'rate limit',
        'temporarily unavailable',
        'try again',
        'internal server error', '500'
    ]
    
    # Check if error matches retryable patterns
    for pattern in retryable_patterns:
        if pattern in error_msg:
            return True
    
    # Non-retryable: Client errors (4xx except 429)
    non_retryable_patterns = [
        '400', 'bad request',
        '401', 'unauthorized', 'authentication',
        '403', 'forbidden', 'permission denied',
        '404', 'not found',
        '422', 'unprocessable entity', 'validation',
        'invalid api key',
        'invalid request',
        'malformed',
    ]
    
    # Check if error matches non-retryable patterns
    for pattern in non_retryable_patterns:
        if pattern in error_msg:
            return False
    
    # Check for Python programming errors (shouldn't retry)
    if isinstance(error, (ValueError, TypeError, KeyError, AttributeError)):
        return False
    
    # Default: Retry unknown errors (conservative approach)
    return True


class Mem0System(BaseMemorySystem):
    """
    Mem0-specific implementation of BaseMemorySystem.
    
    This class handles all Mem0-specific operations for storing and retrieving memories.
    """
    
    def __init__(self, 
                 user_id: str,
                 api_key: Optional[str] = None,
                 max_retries: int = -1,
                 initial_retry_delay: float = 2.0,
                 max_retry_delay: float = 120.0):
        """
        Initialize the Mem0 memory system.
        
        Args:
            user_id: User ID for Mem0 memories
            api_key: Mem0 API key (optional, will use env var if not provided)
            max_retries: Maximum retry attempts (-1 for infinite retries)
            initial_retry_delay: Initial delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds
        """
        super().__init__(user_id)
        
        # Retry configuration
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        
        # Mem0 client (initialized in initialize_client)
        self.memory_client = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load environment variables
        self._load_environment()
        
        # Get API key
        self.api_key = api_key or os.getenv('MEM0_API_KEY')
    
    def get_system_name(self) -> str:
        """Return the name of the memory system."""
        return "mem_0"
    
    def get_required_env_vars(self) -> List[str]:
        """Get list of required environment variables."""
        return ['MEM0_API_KEY']
    
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
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for Mem0 operations."""
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
    
    def initialize_client(self) -> bool:
        """
        Initialize the Mem0 memory client.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not MEM0_AVAILABLE:
            self.logger.error("mem0ai package is not installed. Please install it using: pip install mem0ai")
            return False
        
        if not self.api_key:
            self.logger.error("Mem0 API key is required. Set MEM0_API_KEY environment variable or provide api_key parameter.")
            return False
        
        try:
            self.memory_client = MemoryClient(api_key=self.api_key)
            self.logger.info("Initialized Mem0 client with API key")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Mem0: {e}")
            return False
    
    def extract_conversation_messages(self, conversation_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract conversation messages from Memora conversation format.
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            List of messages formatted for Mem0
        """
        messages = []
        
        if "conversation" not in conversation_data:
            self.logger.warning("No conversation key found in data")
            return messages
        
        for turn in conversation_data["conversation"]:
            # Map Memora speaker roles to standard roles
            role_mapping = {
                "user_agent": "user",
                "ai_agent": "assistant",
                "user": "user",
                "assistant": "assistant"
            }
            
            speaker = turn.get("speaker", "")
            role = role_mapping.get(speaker, speaker)
            message = turn.get("message", "")
            
            if message and role:
                messages.append({
                    "role": role,
                    "content": message
                })
        
        self.logger.info(f"Extracted {len(messages)} messages from conversation")
        return messages
    
    def add_conversation_to_memory(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add conversation messages to Mem0 memory with session date from conversation file.
        Includes automatic retry with exponential backoff for transient errors.
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            Dict[str, Any]: Result from Mem0 add operation
        """
        # Extract the conversation messages
        messages = self.extract_conversation_messages(conversation_data)
        
        if not messages:
            raise ValueError("No valid messages found in conversation data")
        
        # Get session date and ID from conversation data
        session_id = conversation_data.get('session_id')
        session_date = conversation_data.get('date') or conversation_data.get('session_date')
        
        # Prepare metadata with session date
        metadata = {}
        if session_date:
            metadata['session_date'] = session_date
            metadata['session_id'] = session_id
        
        # Send conversation with metadata to Mem0 with retry logic
        return self._add_with_retry(messages, metadata, session_id)
    
    def _add_with_retry(self, messages: List[Dict[str, str]], metadata: Dict[str, Any], session_id: Any) -> Dict[str, Any]:
        """
        Internal method to add messages to Mem0 with retry logic.
        
        Args:
            messages: List of conversation messages
            metadata: Metadata dictionary
            session_id: Session identifier for logging
            
        Returns:
            Dict[str, Any]: Result from Mem0 add operation
        """
        retry_count = 0
        delay = self.initial_retry_delay
        
        # Extract session_date for timestamp if available
        session_date = metadata.get('session_date') or metadata.get('date') if metadata else None
        timestamp = None
        
        if session_date:
            try:
                # Convert session_date (YYYY-MM-DD) to Unix timestamp
                dt = datetime.strptime(session_date, "%Y-%m-%d")
                timestamp = int(dt.timestamp())
                self.logger.info(f"🕐 Converting session_date '{session_date}' to timestamp {timestamp} (UTC: {dt})")
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to convert session_date to timestamp: {e}")
                timestamp = None
        
        while True:
            try:
                # Attempt to add to Mem0 with custom timestamp
                if timestamp:
                    # Use custom timestamp for accurate chronological ordering
                    result = self.memory_client.add(messages, user_id=self.user_id, metadata=metadata, timestamp=timestamp)
                    self.logger.info(f"✅ Successfully added conversation (session {session_id}) with custom timestamp {session_date}")
                elif metadata:
                    result = self.memory_client.add(messages, user_id=self.user_id, metadata=metadata)
                    self.logger.info(f"✅ Successfully added conversation (session {session_id}) to memory")
                else:
                    result = self.memory_client.add(messages, user_id=self.user_id)
                    self.logger.info(f"✅ Successfully added conversation (session {session_id}) to memory")
                
                # If we had retries, log success after retries
                if retry_count > 0:
                    self.logger.info(f"🎉 Success after {retry_count} retry attempts for session {session_id}")
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if this is a retryable error
                if not is_retryable_error(e):
                    self.logger.error(
                        f"❌ Non-retryable error for session {session_id}: {error_msg}. "
                        f"This is likely a permanent error (auth, validation, or data issue)."
                    )
                    raise
                
                retry_count += 1
                
                # Check if we should stop retrying
                if self.max_retries != -1 and retry_count > self.max_retries:
                    self.logger.error(
                        f"❌ Max retries ({self.max_retries}) exceeded for session {session_id}. "
                        f"Last error: {error_msg}"
                    )
                    raise
                
                # Log the retry attempt
                self.logger.warning(
                    f"⚠️  Attempt {retry_count} failed for session {session_id}: {error_msg[:200]}. "
                    f"Retrying in {delay:.1f}s..."
                )
                
                # Wait before retrying
                time.sleep(delay)
                
                # Calculate next delay with exponential backoff
                delay = min(delay * 2.0, self.max_retry_delay)
    
    def search_memories(self, query: str, limit: int = 50, session_date: Optional[str] = None, 
                       date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Search Mem0 for relevant memories using semantic search.
        
        Note: session_date is passed as context but we do NOT filter by date.
        Mem0's semantic search handles temporal queries (like "this week", "last month")
        naturally through the query text itself. The session_date can help provide
        temporal context but doesn't restrict the search results.
        
        Args:
            query: Question to search for (may include temporal references like "this week")
            limit: Maximum number of memories to return
            session_date: Optional session date passed as context (not used for filtering)
            date_range: Optional date range passed as context (not used for filtering)
            
        Returns:
            List of relevant memories from semantic search
        """
        try:
            # Mem0 v2 API requires explicit filters parameter
            # Use filters to specify user_id only (no date filtering - let semantic search handle temporal queries)
            filters = {"user_id": self.user_id}
            
            # Note: session_date is available but not used for filtering
            # Mem0's semantic search handles temporal queries like "this week" or "last month"
            # naturally through the query text itself. The session_date can be logged for context.
            if session_date:
                self.logger.debug(f"🕐 Question context date: {session_date} (not used for filtering)")
            
            memories = []
            
            # Try search with filters
            try:
                response = self.memory_client.search(
                    query=query,
                    filters=filters,
                    limit=limit
                )
                # Mem0 API returns {"results": [list of memories]}
                if isinstance(response, dict) and 'results' in response:
                    memories = response['results']
                else:
                    memories = response if isinstance(response, list) else []
                
                self.logger.info(f"🔍 Search with filters found {len(memories)} memories")
            except Exception as e1:
                self.logger.warning(f"Search with filters failed: {e1}")
                
                # Fallback: Try old API format (for compatibility)
                try:
                    response = self.memory_client.search(query, user_id=self.user_id, limit=limit)
                    if isinstance(response, dict) and 'results' in response:
                        memories = response['results']
                    else:
                        memories = response if isinstance(response, list) else []
                    self.logger.info(f"🔍 Fallback search found {len(memories)} memories")
                except Exception as e2:
                    self.logger.error(f"All search methods failed: {e2}")
            
            self.logger.info(f"✅ Semantic search returned {len(memories)} memories")
            return memories if memories else []
            
        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            return []
    
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
    """Test the Mem0 integration."""
    print("🧪 Testing Mem0 Integration")
    print("=" * 40)
    
    # Test system creation
    try:
        mem0_system = Mem0System("test_user")
        print(f"✅ Created Mem0 system: {mem0_system.get_system_name()}")
        
        # Test environment validation
        if mem0_system.validate_environment():
            print("✅ Environment validation passed")
        else:
            print("❌ Environment validation failed")
        
        # Test client initialization
        if mem0_system.initialize_client():
            print("✅ Client initialization successful")
        else:
            print("❌ Client initialization failed")
            
    except Exception as e:
        print(f"❌ Error testing Mem0 system: {e}")
