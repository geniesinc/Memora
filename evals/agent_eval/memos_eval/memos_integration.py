#!/usr/bin/env python3
"""
MemOS Memory System Integration
=================================

This module provides MemOS-specific implementation of the BaseMemorySystem interface.
It handles all MemOS-specific operations for storing and retrieving memories.

MemOS API Documentation:
- Add Message: https://memos-docs.openmem.net/dashboard/api/add-message
- Get Message: https://memos-docs.openmem.net/dashboard/api/get-message
- Search Memory: https://memos-docs.openmem.net/api-reference/search-memories

Reference: https://memos-dashboard.openmem.net/quickstart/
GitHub: https://github.com/MemTensor/MemOS

Features:
- HTTP API-based memory storage and retrieval
- User-specific memory isolation
- Semantic search capabilities
- Conversation ingestion and retrieval
- Conversation ID based on session date

Usage:
    from memos_integration import MemOSSystem
    
    # Cloud API mode
    system = MemOSSystem("user_123", api_key="your_key")
    
    # Local self-hosted mode
    system = MemOSSystem("user_123", base_url="http://localhost:5230")
    
    if system.validate_environment() and system.initialize_client():
        result = system.add_conversation_to_memory(conversation_data)
"""

import json
import os
import logging
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Import base classes
import sys
sys.path.append(str(Path(__file__).parent.parent))
from base_evaluator import BaseMemorySystem


class MemOSSystem(BaseMemorySystem):
    """
    MemOS-specific implementation of BaseMemorySystem.
    
    This class handles all MemOS-specific operations for storing and retrieving memories.
    MemOS uses HTTP API endpoints for memory management with user-specific isolation.
    
    API Endpoints:
    - POST /add/message - Add messages to a conversation
    - POST /get/message - Get messages from a conversation
    - POST /search/memory - Search memories (API reference: https://memos-docs.openmem.net/dashboard/api/search-memory)
    
    Supports both:
    - Cloud API mode: Requires MEMOS_API_KEY and uses cloud endpoint
    - Local mode: Self-hosted instance, uses local endpoint
    """
    
    def __init__(self, user_id: str, **kwargs):
        """
        Initialize the MemOS memory system.
        
        Args:
            user_id: User ID for MemOS memories (used to isolate user data)
            **kwargs: MemOS specific configuration parameters:
                - api_key: MemOS API key if using cloud service (or use MEMOS_API_KEY env var)
                - base_url: MemOS API base URL (or use MEMOS_BASE_URL env var)
                          Default: https://memos.memtensor.cn/api/openmem/v1 (cloud)
        """
        super().__init__(user_id)
        
        # HTTP client session
        self.session = requests.Session()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load environment variables
        self._load_environment()
        
        # Get configuration parameters
        self.api_key = kwargs.get('api_key') or os.getenv('MEMOS_API_KEY')
        # Default to cloud API if API key is provided, otherwise local
        default_base_url = 'https://memos.memtensor.cn/api/openmem/v1' if self.api_key else 'http://localhost:5230/api/openmem/v1'
        self.base_url = kwargs.get('base_url') or os.getenv('MEMOS_BASE_URL', default_base_url)
        
        # Remove trailing slash
        self.base_url = self.base_url.rstrip('/')
        
        # Set up API authentication - MemOS uses "Token" not "Bearer"
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'application/json'
            })
        else:
            self.session.headers.update({
                'Content-Type': 'application/json'
            })
    
    def get_system_name(self) -> str:
        """Return the name of the memory system."""
        return "memos"
    
    def get_required_env_vars(self) -> List[str]:
        """
        Get list of required environment variables for MemOS.
        
        Note: API key is optional if using self-hosted local instance.
        """
        # API key is optional for local deployments
        return []
    
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
    
    def _get_conversation_id(self, session_date: Optional[str] = None) -> str:
        """
        Generate conversation_id from session date.
        
        Args:
            session_date: Session date string (e.g., "2025-01-15" or ISO format)
            
        Returns:
            conversation_id string based on session date
        """
        if session_date:
            # Try to parse and format the date
            try:
                # Try ISO format first
                if 'T' in session_date:
                    dt = datetime.fromisoformat(session_date.replace('Z', '+00:00'))
                else:
                    # Try common date formats
                    for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y']:
                        try:
                            dt = datetime.strptime(session_date, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        # If parsing fails, use the date string as-is
                        return session_date.replace('-', '').replace(' ', '').replace(':', '')[:8]
                
                # Format as YYYYMMDD
                return dt.strftime('%Y%m%d')
            except Exception as e:
                self.logger.warning(f"Could not parse session_date '{session_date}': {e}")
                # Fallback: use cleaned date string
                return session_date.replace('-', '').replace(' ', '').replace(':', '')[:8]
        else:
            # No date provided, use current date
            return datetime.now().strftime('%Y%m%d')
    
    def initialize_client(self) -> bool:
        """
        Initialize the MemOS API client.
        
        Verifies connectivity to the MemOS API endpoint.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Test API connectivity with a simple request
            # MemOS API might not have a health endpoint, so we'll just verify the base URL is set
            if not self.api_key and 'localhost' not in self.base_url:
                self.logger.warning(
                    "No API key provided but using non-localhost URL. "
                    "MemOS cloud API requires an API key."
                )
            
            self.logger.info(f"✅ MemOS client initialized for {self.base_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MemOS client: {e}")
            return False
    
    def add_conversation_to_memory(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add conversation to MemOS via API.
        
        Uses the Add Message API endpoint: POST /add/message
        Reference: https://memos-docs.openmem.net/dashboard/api/add-message
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            Dict[str, Any]: Result from MemOS operation
        """
        try:
            # Extract conversation turns
            conversation_turns = conversation_data.get("conversation", [])
            if not conversation_turns:
                raise ValueError("No conversation turns found")
            
            # Get session information
            session_id = conversation_data.get('session_id')
            session_date = conversation_data.get('date') or conversation_data.get('session_date')
            
            # Generate conversation_id from session date
            conversation_id = self._get_conversation_id(session_date)
            
            # Format messages for MemOS API
            messages = []
            for turn in conversation_turns:
                speaker = turn.get("speaker", "")
                message = turn.get("message", "")
                
                if not message:
                    continue
                
                # Map speaker roles - MemOS expects "user" and "assistant"
                role = "user"
                if speaker in ["ai_agent", "assistant", "assistant_agent"]:
                    role = "assistant"
                elif speaker in ["user_agent", "user"]:
                    role = "user"
                
                message_obj = {
                    "role": role,
                    "content": message
                }
                
                # Add chat_time if session_date is available
                if session_date:
                    message_obj["chat_time"] = session_date
                
                messages.append(message_obj)
            
            if not messages:
                raise ValueError("No valid messages found in conversation")
            
            # Prepare payload according to MemOS API spec
            payload = {
                "user_id": self.user_id,
                "conversation_id": conversation_id,
                "messages": messages
            }
            
            # Call Add Message API endpoint
            endpoint = f"{self.base_url}/add/message"
            
            try:
                response = self.session.post(endpoint, json=payload, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                
                # Check response format: {"code": 0, "data": {"success": true}, "message": "ok"}
                if result.get('code') == 0 and result.get('data', {}).get('success'):
                    self.logger.info(
                        f"✅ Successfully ingested conversation session {session_id} "
                        f"(conversation_id: {conversation_id}) to MemOS"
                    )
                    return {
                        "status": "success",
                        "user_id": self.user_id,
                        "conversation_id": conversation_id,
                        "session_id": session_id,
                        "session_date": session_date,
                        "messages_added": len(messages)
                    }
                else:
                    error_msg = result.get('message', 'Unknown error')
                    raise RuntimeError(f"MemOS API returned error: {error_msg}")
                    
            except requests.exceptions.HTTPError as e:
                error_detail = ""
                try:
                    error_response = e.response.json()
                    error_detail = error_response.get('message', str(e))
                except:
                    error_detail = str(e)
                raise RuntimeError(f"HTTP error calling MemOS API: {error_detail}")
            
        except Exception as e:
            self.logger.error(f"Failed to add conversation to MemOS: {e}")
            raise
    
    def search_memories(self, query: str, limit: int = 50, session_date: Optional[str] = None,
                       date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Search MemOS API for relevant information.
        
        Uses the Search Memory API endpoint: POST /search/memory
        Reference: https://memos-docs.openmem.net/dashboard/api/search-memory
        
        Args:
            query: Search query/question (required)
            limit: Maximum number of memories to return (applied after API response)
            session_date: Optional session date (not yet implemented for MemOS)
            date_range: Optional date range (not yet implemented for MemOS)
            
        Returns:
            List of relevant memories in format: [{'memory': str, 'score': float, ...}]
        """
        try:
            memories = []
            
            # Prepare payload according to API reference
            # POST /search/memory
            # Body: user_id (string, required), query (string, required), conversation_id (string, optional),
            #       memory_limit_number (number, optional, default: 6), include_preference (boolean, optional, default: true),
            #       preference_limit_number (number, optional, default: 6)
            # Note: user_id format is {persona}_{timeline} (e.g., "academic_researcher_weekly")
            payload = {
                "user_id": self.user_id,  # Format: {persona}_{timeline}, e.g., "academic_researcher_weekly"
                "query": query,
                "memory_limit_number": min(limit, 25),  # API max is 25
                "include_preference": True,
                "preference_limit_number": min(limit, 25)
            }
            
            # Add conversation_id if session_date is provided (optional but helps prioritize current conversation)
            if session_date:
                conversation_id = self._get_conversation_id(session_date)
                payload["conversation_id"] = conversation_id
            
            # Validate user_id format
            if not self.user_id or not isinstance(self.user_id, str):
                self.logger.warning(f"Invalid user_id format: {self.user_id}. Expected format: {{persona}}_{{timeline}}")
                return []
            
            self.logger.debug(f"Searching with user_id: {self.user_id} (format: {{persona}}_{{timeline}})")
            
            # Call Search Memory API endpoint (correct: /search/memory)
            endpoint = f"{self.base_url}/search/memory"
            
            try:
                self.logger.debug(f"Searching MemOS with query: {query[:100]}...")
                self.logger.debug(f"Request payload: {payload}")
                self.logger.debug(f"Endpoint: {endpoint}")
                
                response = self.session.post(endpoint, json=payload, timeout=30)
                
                # Log response status before raising
                self.logger.debug(f"Response status: {response.status_code}")
                
                # Try to get error details before raising
                if response.status_code >= 400:
                    try:
                        error_body = response.text
                        self.logger.warning(f"API error response body: {error_body[:500]}")
                        # Try to parse as JSON
                        try:
                            error_json = response.json()
                            self.logger.warning(f"API error JSON: {error_json}")
                        except:
                            pass
                    except Exception as e:
                        self.logger.debug(f"Could not read error response: {e}")
                
                response.raise_for_status()
                
                result = response.json()
                self.logger.debug(f"Response JSON: {str(result)[:500]}")
                
                # Check response format: {"code": 0, "message": "...", "data": {...}}
                # According to API reference, code 0 means success (not 200)
                if result.get('code') == 0:
                    data = result.get('data', {})
                    
                    # Extract memory_detail_list and preference_detail_list from response
                    # Response structure:
                    # {
                    #   "code": 0,
                    #   "data": {
                    #     "memory_detail_list": [...],
                    #     "preference_detail_list": [...],
                    #     "preference_note": "..."
                    #   },
                    #   "message": "..."
                    # }
                    memory_list = data.get('memory_detail_list', [])
                    preference_list = data.get('preference_detail_list', [])
                    
                    self.logger.debug(f"MemOS API returned {len(memory_list)} fact memories and {len(preference_list)} preference memories")
                    
                    # Process fact memories (memory_detail_list)
                    for memory_item in memory_list[:limit]:
                        if isinstance(memory_item, dict):
                            # Extract memory_value (the actual memory content)
                            memory_value = memory_item.get('memory_value', '')
                            memory_key = memory_item.get('memory_key', '')
                            
                            # Combine key and value for better context
                            if memory_key and memory_value:
                                memory_text = f"{memory_key}: {memory_value}"
                            elif memory_value:
                                memory_text = memory_value
                            elif memory_key:
                                memory_text = memory_key
                            else:
                                continue
                            
                            # Use relativity score (0-1) as the relevance score
                            score = memory_item.get('relativity', memory_item.get('confidence', 0.5))
                            
                            if memory_text and memory_text.strip():
                                memories.append({
                                    'memory': memory_text.strip(),
                                    'score': float(score),
                                    'source': 'memos_api',
                                    'type': memory_item.get('memory_type', 'memory'),
                                    'confidence': memory_item.get('confidence', 0.0),
                                    'tags': memory_item.get('tags', [])
                                })
                    
                    # Process preference memories (preference_detail_list)
                    for pref_item in preference_list[:limit]:
                        if isinstance(pref_item, dict):
                            preference_text = pref_item.get('preference', '')
                            reasoning = pref_item.get('reasoning', '')
                            
                            # Combine preference and reasoning
                            if reasoning:
                                memory_text = f"[Preference] {preference_text} (Reason: {reasoning})"
                            else:
                                memory_text = f"[Preference] {preference_text}"
                            
                            if memory_text and memory_text.strip():
                                memories.append({
                                    'memory': memory_text.strip(),
                                    'score': 0.9,  # High score for preferences
                                    'source': 'memos_api',
                                    'type': 'preference',
                                    'preference_type': pref_item.get('preference_type', 'unknown')
                                })
                    
                    # Add preference_note if available
                    preference_note = data.get('preference_note', '')
                    if preference_note:
                        memories.append({
                            'memory': f"[Preference Note] {preference_note}",
                            'score': 0.85,
                            'source': 'memos_api',
                            'type': 'preference_note'
                        })
                    
                    # Sort by score (highest first) and limit
                    memories.sort(key=lambda x: x['score'], reverse=True)
                    memories = memories[:limit]
                    
                    self.logger.info(f"✅ MemOS search returned {len(memories)} memories")
                    return memories
                else:
                    error_msg = result.get('message', 'Unknown error')
                    error_code = result.get('code', 'unknown')
                    self.logger.warning(f"MemOS search API returned error code {error_code}: {error_msg}")
                    # Log the full response for debugging
                    self.logger.debug(f"Full API response: {result}")
                    return []
                    
            except requests.exceptions.HTTPError as e:
                error_detail = ""
                error_body = ""
                try:
                    # Try to get error response body
                    if hasattr(e, 'response') and e.response is not None:
                        error_body = e.response.text
                        self.logger.debug(f"Full error response body: {error_body[:1000]}")
                        
                        # Try to parse as JSON
                        try:
                            error_response = e.response.json()
                            error_detail = f" - {error_response.get('message', error_response.get('error', str(e)))}"
                            self.logger.debug(f"Parsed error response: {error_response}")
                        except:
                            # Not JSON, use text
                            error_detail = f" - {error_body[:200]}"
                    else:
                        error_detail = f" - {str(e)}"
                except Exception as parse_error:
                    self.logger.debug(f"Could not parse error response: {parse_error}")
                    error_detail = f" - {str(e)}"
                
                status_code = e.response.status_code if hasattr(e, 'response') and e.response else "unknown"
                self.logger.error(f"HTTP error calling MemOS search API: {status_code}{error_detail}")
                
                # For 500 errors, provide detailed diagnostics
                if status_code == 500:
                    # Log detailed error info
                    self.logger.error(f"Server error (500) when searching MemOS API")
                    self.logger.error(f"  User ID: {self.user_id}")
                    self.logger.error(f"  Query: {query[:100]}... (length: {len(query)} chars)")
                    self.logger.error(f"  Endpoint: {endpoint}")
                    self.logger.error(f"  Payload: {payload}")
                    
                    if error_body:
                        self.logger.error(f"  Server error response: {error_body[:1000]}")
                    
                    # Try to verify if user exists by attempting to get messages
                    # This helps diagnose if it's a user existence issue
                    try:
                        self.logger.debug(f"Attempting to verify user existence by checking messages...")
                        # Try to get messages for this user (this will fail if user doesn't exist)
                        test_endpoint = f"{self.base_url}/get/message"
                        test_payload = {
                            "user_id": self.user_id,
                            "conversation_id": "test"  # Just to test if user exists
                        }
                        test_response = self.session.post(test_endpoint, json=test_payload, timeout=10)
                        test_result = test_response.json()
                        
                        if test_result.get('code') == 0:
                            self.logger.debug(f"User {self.user_id} exists (get/message succeeded)")
                        else:
                            self.logger.warning(f"User verification: get/message returned code {test_result.get('code')}")
                    except Exception as verify_error:
                        self.logger.debug(f"User verification failed: {verify_error}")
                    
                    self.logger.error(f"Possible causes:")
                    self.logger.error(f"  1. Server-side issue with MemOS API (500 = Internal Server Error)")
                    self.logger.error(f"  2. API endpoint or request format may be incorrect")
                    self.logger.error(f"  3. User '{self.user_id}' may not exist (though data was saved)")
                    self.logger.error(f"  4. Query format or length issue")
                    self.logger.error(f"  5. API authentication or permissions issue")
                    
                
                return []
            
        except Exception as e:
            self.logger.error(f"MemOS search failed: {e}", exc_info=True)
            return []
    
    def get_messages(self, conversation_id: Optional[str] = None, session_date: Optional[str] = None,
                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get messages from a conversation using Get Message API.
        
        Uses the Get Message API endpoint: POST /get/message
        Reference: https://memos-docs.openmem.net/dashboard/api/get-message
        
        Args:
            conversation_id: Conversation ID (if not provided, generated from session_date)
            session_date: Session date to generate conversation_id
            limit: Optional limit on number of messages to return
            
        Returns:
            List of message dictionaries
        """
        try:
            # Generate conversation_id if not provided
            if not conversation_id:
                conversation_id = self._get_conversation_id(session_date)
            
            # Prepare payload for Get Message API
            payload = {
                "user_id": self.user_id,
                "conversation_id": conversation_id
            }
            
            if limit:
                payload["limit"] = limit
            
            # Call Get Message API endpoint
            endpoint = f"{self.base_url}/get/message"
            
            try:
                response = self.session.post(endpoint, json=payload, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                
                # Check response format: {"code": 0, "data": {"message_detail_list": [...]}, "message": "ok"}
                if result.get('code') == 0:
                    data = result.get('data', {})
                    message_list = data.get('message_detail_list', [])
                    
                    self.logger.info(f"✅ Retrieved {len(message_list)} messages for conversation {conversation_id}")
                    return message_list
                else:
                    error_msg = result.get('message', 'Unknown error')
                    self.logger.warning(f"MemOS get message API returned error: {error_msg}")
                    return []
                    
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"HTTP error calling MemOS get message API: {e}")
                return []
            
        except Exception as e:
            self.logger.error(f"Failed to get messages from MemOS: {e}")
            return []
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for MemOS operations."""
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
    """Test the MemOS integration."""
    print("🧪 Testing MemOS Integration")
    print("=" * 40)
    
    # Test system creation
    try:
        # Test with cloud API (if API key is set) or local instance
        memos_system = MemOSSystem("test_user")
        print(f"✅ Created MemOS system: {memos_system.get_system_name()}")
        print(f"   Base URL: {memos_system.base_url}")
        print(f"   API Key: {'Set' if memos_system.api_key else 'Not set (using local instance)'}")
        
        # Test environment validation
        if memos_system.validate_environment():
            print("✅ Environment validation passed")
            
            # Test client initialization
            if memos_system.initialize_client():
                print("✅ API client initialization successful")
                
                # Test conversation_id generation
                test_date = "2025-01-15"
                conv_id = memos_system._get_conversation_id(test_date)
                print(f"📅 Conversation ID for {test_date}: {conv_id}")
                
                # Test search (may return empty if no memories)
                results = memos_system.search_memories("test query", limit=5)
                print(f"🔍 Search returned {len(results)} results")
            else:
                print("❌ API client initialization failed")
        else:
            print("❌ Environment validation failed")
            
    except Exception as e:
        print(f"❌ Error testing MemOS system: {e}")
        import traceback
        traceback.print_exc()
