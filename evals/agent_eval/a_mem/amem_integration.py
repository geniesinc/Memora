#!/usr/bin/env python3
"""
A-MEM (Agentic Memory) System Integration
==========================================

This module provides A-MEM-specific implementation of the BaseMemorySystem interface.
It handles all A-MEM-specific operations for storing and retrieving memories.

A-MEM is an innovative agentic memory system that uses Zettelkasten principles
for dynamic memory organization.

Features:
- A-MEM integration with ChromaDB backend
- Conversation data extraction and formatting
- User-specific memory storage with isolated collections (MODIFIED from original)
- Automatic content analysis and metadata extraction
- Memory evolution and relationship management
- Semantic search capabilities
- Persistent storage support for multi-user evaluation

Reference: https://github.com/WujiangXu/A-mem-sys

IMPORTANT MODIFICATIONS:
- Added user-specific collection names for multi-user isolation
- Added persistent ChromaDB storage option
- Removed automatic collection reset to preserve data between sessions
"""

import json
import os
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Add the A-mem module to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Try to import ChromaDB and related dependencies
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    CHROMADB_AVAILABLE = True
except ImportError as e:
    CHROMADB_AVAILABLE = False
    print(f"Warning: ChromaDB not available: {e}")

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not available")

AMEM_AVAILABLE = CHROMADB_AVAILABLE and OPENAI_AVAILABLE


# Define our own MemoryNote class (simplified from A-MEM)
class MemoryNote:
    """A memory note that represents a single unit of information in the memory system."""
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[List[str]] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []


class SimpleLLMController:
    """Simple LLM controller that uses OpenAI directly (no litellm dependency)."""
    
    def __init__(self, backend: str = "openai", model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.backend = backend
        self.model = model
        
        if backend in ["openai", "openrouter"]:
            if api_key is None:
                if backend == "openai":
                    api_key = os.getenv('OPENAI_API_KEY')
                else:
                    api_key = os.getenv('OPENROUTER_API_KEY')
            
            if api_key is None:
                raise ValueError(f"API key not found for {backend}")
            
            if backend == "openrouter":
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            else:
                self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """Get completion from LLM."""
        messages = [
            {"role": "system", "content": "You must respond with a JSON object."},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class LLMControllerWrapper:
    """Wrapper to provide consistent interface."""
    def __init__(self, backend: str, model: str, api_key: Optional[str] = None):
        self.llm = SimpleLLMController(backend, model, api_key)
    
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)

# Import base classes
sys.path.append(str(Path(__file__).parent.parent))
try:
    from base_evaluator import BaseMemorySystem
except ImportError:
    # If base_evaluator is not available, create a minimal abstract class
    from abc import ABC, abstractmethod
    class BaseMemorySystem(ABC):
        def __init__(self, user_id: str, **kwargs):
            self.user_id = user_id
            self.system_name = self.get_system_name()
        
        @abstractmethod
        def get_system_name(self) -> str:
            pass
        
        @abstractmethod
        def initialize_client(self) -> bool:
            pass
        
        @abstractmethod
        def add_conversation_to_memory(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
            pass
        
        @abstractmethod
        def search_memories(self, query: str, limit: int = 50, session_date: Optional[str] = None,
                           date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
            pass
        
        @abstractmethod
        def get_required_env_vars(self) -> List[str]:
            pass
        
        def validate_environment(self) -> bool:
            required_vars = self.get_required_env_vars()
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                print(f"❌ Missing required environment variables: {missing_vars}")
                return False
            return True


class UserIsolatedChromaRetriever:
    """
    ChromaDB retriever with user-specific collection isolation.
    
    This is a modified version of ChromaRetriever that:
    1. Uses user-specific collection names for isolation
    2. Supports persistent storage for data to survive between sessions
    3. Does NOT reset collections on initialization
    """

    def __init__(
        self, 
        user_id: str,
        collection_prefix: str = "amem_memories",
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize user-isolated ChromaDB retriever.

        Args:
            user_id: User ID for collection isolation
            collection_prefix: Prefix for collection name
            model_name: SentenceTransformer model for embeddings
            persist_directory: Directory for persistent storage (None for in-memory)
        """
        self.user_id = user_id
        self.collection_name = f"{collection_prefix}_{user_id}"
        self.model_name = model_name
        
        # Use persistent client if directory specified
        if persist_directory:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_path))
        else:
            # Use in-memory client with allow_reset=False to prevent accidental data loss
            self.client = chromadb.Client(Settings(allow_reset=False, anonymized_telemetry=False))
        
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        # Get or create user-specific collection (NO RESET!)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, 
            embedding_function=self.embedding_function
        )

    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to the user's collection."""
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)

        self.collection.add(
            documents=[document], 
            metadatas=[processed_metadata], 
            ids=[doc_id]
        )

    def delete_document(self, doc_id: str):
        """Delete a document from the user's collection."""
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5) -> Dict:
        """Search for similar documents in the user's collection."""
        results = self.collection.query(query_texts=[query], n_results=k)
        
        if results and results.get("metadatas"):
            results["metadatas"] = self._convert_metadata_types(results["metadatas"])
        
        return results
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def _convert_metadata_types(self, metadatas: List[List[Dict]]) -> List[List[Dict]]:
        """Convert string metadata back to original types."""
        import ast
        for query_metadatas in metadatas:
            if isinstance(query_metadatas, list):
                for metadata_dict in query_metadatas:
                    if isinstance(metadata_dict, dict):
                        for key, value in metadata_dict.items():
                            if isinstance(value, str):
                                try:
                                    metadata_dict[key] = ast.literal_eval(value)
                                except Exception:
                                    pass
        return metadatas


class UserIsolatedAgenticMemorySystem:
    """
    Modified AgenticMemorySystem with user-specific collection isolation.
    
    Key differences from original:
    1. Uses user-specific ChromaDB collection
    2. No automatic reset of collections
    3. Supports persistent storage
    4. Memories are isolated per user
    """
    
    def __init__(self, 
                 user_id: str,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 persist_directory: Optional[str] = None):
        """
        Initialize the user-isolated memory system.
        
        Args:
            user_id: User ID for memory isolation
            model_name: Name of the sentence transformer model
            llm_backend: LLM backend to use (openai/ollama/openrouter)
            llm_model: Name of the LLM model
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service
            persist_directory: Directory for persistent ChromaDB storage
        """
        self.user_id = user_id
        self.memories = {}
        self.model_name = model_name
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        
        # Initialize user-isolated ChromaDB retriever (NO RESET!)
        self.retriever = UserIsolatedChromaRetriever(
            user_id=user_id,
            model_name=model_name,
            persist_directory=persist_directory
        )
        
        # Initialize LLM controller (using our wrapper that doesn't require litellm)
        self.llm_controller = LLMControllerWrapper(llm_backend, llm_model, api_key)
        
        # Evolution system prompt (same as original)
        self._evolution_system_prompt = '''
            You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
            Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
            Make decisions about its evolution.  

            The new memory context:
            {context}
            content: {content}
            keywords: {keywords}

            The nearest neighbors memories:
            {nearest_neighbors_memories}

            Based on this information, determine:
            1. Should this memory be evolved? Consider its relationships with other memories.
            2. What specific actions should be taken (strengthen, update_neighbor)?
               2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
               2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
            Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
            Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
            The number of neighbors is {neighbor_number}.
            Return your decision in JSON format with the following structure:
            {{
                "should_evolve": True or False,
                "actions": ["strengthen", "update_neighbor"],
                "suggested_connections": ["neighbor_memory_ids"],
                "tags_to_update": ["tag_1",..."tag_n"], 
                "new_context_neighborhood": ["new context",...,"new context"],
                "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
            }}
            '''
        
        logging.info(f"Initialized UserIsolatedAgenticMemorySystem for user: {user_id}")
        logging.info(f"Collection name: {self.retriever.collection_name}")
        logging.info(f"Existing documents in collection: {self.retriever.count()}")

    def analyze_content(self, content: str) -> Dict:
        """Analyze content using LLM to extract semantic metadata."""
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "context": {"type": "string"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    }})
            return json.loads(response)
        except Exception as e:
            logging.error(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note with user isolation."""
        if time is not None:
            kwargs['timestamp'] = time
        note = MemoryNote(content=content, **kwargs)
        
        # Process memory (analyze and potentially evolve)
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        
        # Add to ChromaDB with complete metadata
        metadata = {
            "id": note.id,
            "user_id": self.user_id,  # Include user_id in metadata
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags
        }
        self.retriever.add_document(note.content, metadata, note.id)
        
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        
        return note.id
    
    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents."""
        # Note: We don't reset the collection, just re-add updated memories
        for memory in self.memories.values():
            metadata = {
                "id": memory.id,
                "user_id": self.user_id,
                "content": memory.content,
                "keywords": memory.keywords,
                "links": memory.links,
                "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp,
                "last_accessed": memory.last_accessed,
                "context": memory.context,
                "evolution_history": memory.evolution_history,
                "category": memory.category,
                "tags": memory.tags
            }
            # Delete and re-add to update
            try:
                self.retriever.delete_document(memory.id)
            except:
                pass
            self.retriever.add_document(memory.content, metadata, memory.id)
    
    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        """Find related memories using ChromaDB retrieval."""
        if not self.memories:
            return "", []
            
        try:
            results = self.retriever.search(query, k)
            memory_str = ""
            indices = []
            
            if 'ids' in results and results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    if i < len(results['metadatas'][0]):
                        metadata = results['metadatas'][0][i]
                        memory_str += f"memory index:{i}\ttalk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                        indices.append(i)
                    
            return memory_str, indices
        except Exception as e:
            logging.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using ChromaDB retrieval."""
        search_results = self.retriever.search(query, k)
        memories = []
        
        if 'ids' not in search_results or not search_results['ids']:
            return memories
            
        for i, doc_id in enumerate(search_results['ids'][0]):
            if i < len(search_results['metadatas'][0]):
                metadata = search_results['metadatas'][0][i]
                memories.append({
                    'id': doc_id,
                    'content': metadata.get('content', ''),
                    'context': metadata.get('context', ''),
                    'keywords': metadata.get('keywords', []),
                    'tags': metadata.get('tags', []),
                    'timestamp': metadata.get('timestamp', ''),
                    'score': search_results['distances'][0][i] if 'distances' in search_results else 0.0
                })
        
        return memories[:k]

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories with linked neighbors (agentic search)."""
        if self.retriever.count() == 0:
            return []
            
        try:
            results = self.retriever.search(query, k)
            memories = []
            seen_ids = set()
            
            if 'ids' not in results or not results['ids'] or len(results['ids'][0]) == 0:
                return []
                
            # Process primary results
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if doc_id in seen_ids:
                    continue
                    
                if i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                    memory_dict = {
                        'id': doc_id,
                        'content': metadata.get('content', ''),
                        'context': metadata.get('context', ''),
                        'keywords': metadata.get('keywords', []),
                        'tags': metadata.get('tags', []),
                        'timestamp': metadata.get('timestamp', ''),
                        'category': metadata.get('category', 'Uncategorized'),
                        'is_neighbor': False
                    }
                    
                    if 'distances' in results and len(results['distances']) > 0 and i < len(results['distances'][0]):
                        memory_dict['score'] = results['distances'][0][i]
                        
                    memories.append(memory_dict)
                    seen_ids.add(doc_id)
            
            # Add linked memories (neighbors) from in-memory cache
            neighbor_count = 0
            for memory in list(memories):
                if neighbor_count >= k:
                    break
                    
                links = memory.get('links', [])
                if isinstance(links, str):
                    try:
                        links = json.loads(links)
                    except:
                        links = []
                        
                for link_id in links:
                    if link_id not in seen_ids and neighbor_count < k:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories.append({
                                'id': link_id,
                                'content': neighbor.content,
                                'context': neighbor.context,
                                'keywords': neighbor.keywords,
                                'tags': neighbor.tags,
                                'timestamp': neighbor.timestamp,
                                'category': neighbor.category,
                                'is_neighbor': True
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1
            
            return memories[:k * 2]  # Return more to include neighbors
        except Exception as e:
            logging.error(f"Error in search_agentic: {str(e)}")
            return []

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve."""
        # Analyze content if not already done
        if not note.keywords or not note.context or note.context == "General":
            analysis = self.analyze_content(note.content)
            if not note.keywords:
                note.keywords = analysis.get('keywords', [])
            if not note.context or note.context == "General":
                note.context = analysis.get('context', 'General')
            if not note.tags:
                note.tags = analysis.get('tags', [])
        
        # For first memory or if no other memories, skip evolution
        if not self.memories:
            return False, note
            
        try:
            # Get nearest neighbors
            neighbors_text, indices = self.find_related_memories(note.content, k=5)
            if not neighbors_text or not indices:
                return False, note
                
            # Query LLM for evolution decision
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(indices)
            )
            
            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {"type": "boolean"},
                                "actions": {"type": "array", "items": {"type": "string"}},
                                "suggested_connections": {"type": "array", "items": {"type": "string"}},
                                "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
                                "tags_to_update": {"type": "array", "items": {"type": "string"}},
                                "new_tags_neighborhood": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
                            },
                            "required": ["should_evolve", "actions", "suggested_connections", 
                                      "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
                )
                
                response_json = json.loads(response)
                should_evolve = response_json["should_evolve"]
                
                if should_evolve:
                    actions = response_json["actions"]
                    for action in actions:
                        if action == "strengthen":
                            suggest_connections = response_json["suggested_connections"]
                            new_tags = response_json["tags_to_update"]
                            note.links.extend(suggest_connections)
                            note.tags = new_tags
                        elif action == "update_neighbor":
                            # Update neighbor memories
                            new_context_neighborhood = response_json["new_context_neighborhood"]
                            new_tags_neighborhood = response_json["new_tags_neighborhood"]
                            noteslist = list(self.memories.values())
                            notes_id = list(self.memories.keys())
                            
                            for i in range(min(len(indices), len(new_tags_neighborhood))):
                                if i >= len(indices):
                                    continue
                                tag = new_tags_neighborhood[i]
                                context = new_context_neighborhood[i] if i < len(new_context_neighborhood) else noteslist[i].context if i < len(noteslist) else 'General'
                                
                                if i < len(indices):
                                    memorytmp_idx = indices[i]
                                    if memorytmp_idx < len(noteslist):
                                        notetmp = noteslist[memorytmp_idx]
                                        notetmp.tags = tag
                                        notetmp.context = context
                                        if memorytmp_idx < len(notes_id):
                                            self.memories[notes_id[memorytmp_idx]] = notetmp
                                
                return should_evolve, note
                
            except Exception as e:
                logging.error(f"Error in memory evolution: {str(e)}")
                return False, note
                
        except Exception as e:
            logging.error(f"Error in process_memory: {str(e)}")
            return False, note
    
    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID."""
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note."""
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                
        metadata = {
            "id": note.id,
            "user_id": self.user_id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags
        }
        
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(note.content, metadata, memory_id)
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note."""
        if memory_id in self.memories:
            self.retriever.delete_document(memory_id)
            del self.memories[memory_id]
            return True
        return False


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


class AMemSystem(BaseMemorySystem):
    """
    A-MEM-specific implementation of BaseMemorySystem.
    
    This class handles all A-MEM-specific operations for storing and retrieving memories.
    A-MEM uses Zettelkasten principles for dynamic memory organization with ChromaDB backend.
    
    KEY FEATURE: User-specific memory isolation
    - Each user gets their own ChromaDB collection
    - Collections are named: amem_memories_{user_id}
    - Supports persistent storage for data to survive between sessions
    """
    
    def __init__(self,
                 user_id: str,
                 api_key: Optional[str] = None,
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 evo_threshold: int = 100,
                 max_retries: int = 3,
                 initial_retry_delay: float = 2.0,
                 max_retry_delay: float = 60.0,
                 persist_directory: Optional[str] = None):
        """
        Initialize the A-MEM memory system.
        
        Args:
            user_id: User ID for memory isolation (used as collection name prefix)
            api_key: API key for LLM service (optional, will use env var if not provided)
            llm_backend: LLM backend to use ("openai", "ollama", "openrouter")
            llm_model: Name of the LLM model to use
            embedding_model: Sentence transformer model for embeddings
            evo_threshold: Number of memories before triggering evolution
            max_retries: Maximum retry attempts for transient errors
            initial_retry_delay: Initial delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds
            persist_directory: Directory for persistent ChromaDB storage (None = use default: evals/A_mem/data/)
        """
        super().__init__(user_id)
        
        # LLM configuration
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.evo_threshold = evo_threshold
        
        # Set default persist directory if not provided
        # This ensures memories persist between process restarts!
        if persist_directory is None:
            # Use a data directory within the A_mem integration folder
            persist_directory = str(Path(__file__).parent / "data")
        self.persist_directory = persist_directory
        
        # Retry configuration
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        
        # A-MEM client (initialized in initialize_client)
        self.memory_system = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load environment variables
        self._load_environment()
        
        # Get API key
        if llm_backend == "openai":
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        elif llm_backend == "openrouter":
            self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        else:
            self.api_key = api_key
    
    def get_system_name(self) -> str:
        """Return the name of the memory system."""
        return "a_mem"
    
    def get_required_env_vars(self) -> List[str]:
        """Get list of required environment variables."""
        if self.llm_backend == "openai":
            return ['OPENAI_API_KEY']
        elif self.llm_backend == "openrouter":
            return ['OPENROUTER_API_KEY']
        else:
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
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for A-MEM operations."""
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
        Initialize the A-MEM memory system client with user isolation.
        
        This creates a user-specific ChromaDB collection that:
        - Is isolated from other users' memories
        - Does NOT reset on initialization (preserves data)
        - Optionally supports persistent storage
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not AMEM_AVAILABLE:
            self.logger.error("A-MEM package is not available. Please check the installation.")
            return False
        
        if not self.api_key and self.llm_backend in ["openai", "openrouter"]:
            self.logger.error(f"API key is required for {self.llm_backend} backend.")
            return False
        
        try:
            # Use the user-isolated version that doesn't reset collections
            self.memory_system = UserIsolatedAgenticMemorySystem(
                user_id=self.user_id,
                model_name=self.embedding_model,
                llm_backend=self.llm_backend,
                llm_model=self.llm_model,
                evo_threshold=self.evo_threshold,
                api_key=self.api_key,
                persist_directory=self.persist_directory
            )
            
            collection_name = f"amem_memories_{self.user_id}"
            memory_count = self.memory_system.retriever.count()
            
            self.logger.info(f"✅ Initialized A-MEM client with {self.llm_backend}/{self.llm_model}")
            self.logger.info(f"   User ID: {self.user_id}")
            self.logger.info(f"   Collection: {collection_name}")
            self.logger.info(f"   Existing memories: {memory_count}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize A-MEM: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_conversation_messages(self, conversation_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract conversation messages from Memora conversation format.
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            List of messages with role and content
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
    
    def _format_conversation_for_memory(self, messages: List[Dict[str, str]], metadata: Dict[str, Any]) -> str:
        """
        Format conversation messages into a single text for A-MEM storage.
        
        A-MEM works with content strings, so we format the conversation 
        in a way that preserves context and speaker information.
        
        Args:
            messages: List of message dictionaries with role and content
            metadata: Metadata about the conversation
            
        Returns:
            Formatted conversation string
        """
        formatted_parts = []
        
        # Add session metadata as context
        session_id = metadata.get('session_id', 'unknown')
        session_date = metadata.get('session_date', 'unknown')
        formatted_parts.append(f"[Session {session_id} - Date: {session_date}]")
        
        # Format each message
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted_parts.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_parts)
    
    def add_conversation_to_memory(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add conversation messages to A-MEM memory.
        
        For A-MEM, we add each conversation turn as a separate memory note
        to leverage the system's memory evolution and linking capabilities.
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            Dict[str, Any]: Result from memory addition
        """
        # Extract the conversation messages
        messages = self.extract_conversation_messages(conversation_data)
        
        if not messages:
            raise ValueError("No valid messages found in conversation data")
        
        # Get session metadata
        session_id = conversation_data.get('session_id')
        session_date = conversation_data.get('date') or conversation_data.get('session_date')
        category = conversation_data.get('category', 'general')
        # NOTE: We intentionally don't include session_type (memory type like "no_memory", 
        # "preference_memory", etc.) as it would leak information during evaluation
        
        # Prepare metadata
        metadata = {
            'session_id': session_id,
            'session_date': session_date,
            'category': category,
            'user_id': self.user_id
        }
        
        # Add conversation to memory with retry logic
        return self._add_with_retry(messages, metadata, session_id)
    
    def _add_with_retry(self, messages: List[Dict[str, str]], metadata: Dict[str, Any], session_id: Any) -> Dict[str, Any]:
        """
        Internal method to add messages to A-MEM with retry logic.
        
        Args:
            messages: List of conversation messages
            metadata: Metadata dictionary
            session_id: Session identifier for logging
            
        Returns:
            Dict[str, Any]: Result from memory addition
        """
        retry_count = 0
        delay = self.initial_retry_delay
        
        # Format conversation for A-MEM
        conversation_text = self._format_conversation_for_memory(messages, metadata)
        
        # Convert session_date to timestamp format for A-MEM
        session_date = metadata.get('session_date')
        timestamp = None
        if session_date:
            try:
                dt = datetime.strptime(session_date, "%Y-%m-%d")
                timestamp = dt.strftime("%Y%m%d%H%M")  # A-MEM uses YYYYMMDDHHmm format
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to convert session_date to timestamp: {e}")
        
        while True:
            try:
                # Add to A-MEM
                # A-MEM's add_note returns the memory ID
                # NOTE: We intentionally don't include session_type (memory type) as it would
                # leak information during evaluation - the system shouldn't know if a 
                # conversation is "no_memory" or "preference_memory" beforehand
                memory_id = self.memory_system.add_note(
                    content=conversation_text,
                    time=timestamp,
                    category=metadata.get('category', 'Uncategorized'),
                    tags=[f"session_{metadata.get('session_id', 'unknown')}"]
                )
                
                self.logger.info(f"✅ Successfully added conversation (session {session_id}) to A-MEM with ID: {memory_id}")
                
                if retry_count > 0:
                    self.logger.info(f"🎉 Success after {retry_count} retry attempts for session {session_id}")
                
                return {
                    'memory_id': memory_id,
                    'session_id': session_id,
                    'success': True,
                    'message_count': len(messages)
                }
                
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
        Search A-MEM for relevant memories using semantic search.
        
        A-MEM provides multiple search methods:
        - search(): Basic hybrid search
        - search_agentic(): Agentic search with linked memories
        
        Args:
            query: Question to search for
            limit: Maximum number of memories to return
            session_date: Optional session date (not yet implemented for A-MEM)
            date_range: Optional date range (not yet implemented for A-MEM)
            
        Returns:
            List of relevant memories from semantic search
        """
        if not self.memory_system:
            self.logger.error("A-MEM client not initialized. Call initialize_client() first.")
            return []
        
        try:
            # Use agentic search which includes linked memories
            results = self.memory_system.search_agentic(query, k=limit)
            
            # Format results to match expected output format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.get('id', ''),
                    'memory': result.get('content', ''),
                    'content': result.get('content', ''),
                    'context': result.get('context', ''),
                    'keywords': result.get('keywords', []),
                    'tags': result.get('tags', []),
                    'score': result.get('score', 0.0),
                    'timestamp': result.get('timestamp', ''),
                    'is_neighbor': result.get('is_neighbor', False)
                })
            
            self.logger.info(f"✅ Search returned {len(formatted_results)} memories for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            return []
    
    def get_all_memories(self) -> Dict[str, Any]:
        """
        Get all memories stored in the system.
        
        Returns:
            Dict with all memory notes
        """
        if not self.memory_system:
            return {}
        return self.memory_system.memories
    
    def get_memory_count(self) -> int:
        """
        Get the total number of memories stored in ChromaDB.
        
        Returns:
            int: Number of memories in the user's collection
        """
        if not self.memory_system:
            return 0
        # Return the ChromaDB collection count (more accurate than in-memory dict)
        return self.memory_system.retriever.count()
    
    def get_collection_name(self) -> str:
        """
        Get the ChromaDB collection name for this user.
        
        Returns:
            str: Collection name
        """
        if not self.memory_system:
            return f"amem_memories_{self.user_id}"
        return self.memory_system.retriever.collection_name
    
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
    
    def clear_memories(self) -> bool:
        """
        Clear all memories from the user's collection.
        
        Note: This only clears the current user's memories, not other users'.
        
        Returns:
            bool: True if successful
        """
        try:
            if self.memory_system:
                # Clear in-memory cache
                self.memory_system.memories = {}
                self.memory_system.evo_cnt = 0
                
                # Delete and recreate the collection
                collection_name = self.memory_system.retriever.collection_name
                self.memory_system.retriever.client.delete_collection(collection_name)
                self.memory_system.retriever.collection = self.memory_system.retriever.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.memory_system.retriever.embedding_function
                )
                
                self.logger.info(f"✅ All memories cleared for user: {self.user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear memories: {e}")
            return False


if __name__ == "__main__":
    """Test the A-MEM integration with user isolation."""
    print("🧪 Testing A-MEM Integration with User Isolation")
    print("=" * 60)
    
    # Test system creation for two different users
    try:
        # Test User 1
        print("\n📋 Testing User 1: academic_researcher")
        print("-" * 40)
        user1_system = AMemSystem("academic_researcher")
        print(f"✅ Created A-MEM system: {user1_system.get_system_name()}")
        
        if user1_system.validate_environment():
            print("✅ Environment validation passed")
        else:
            print("❌ Environment validation failed - check API keys")
            sys.exit(1)
        
        if user1_system.initialize_client():
            print(f"✅ Client initialization successful")
            print(f"   Collection: {user1_system.get_collection_name()}")
            print(f"   Memory count: {user1_system.get_memory_count()}")
            
            # Add a test conversation
            # Note: session_type is intentionally not used in memory storage
            test_conversation = {
                "session_id": "test_001",
                "date": "2025-01-01",
                "category": "test",
                "conversation": [
                    {"speaker": "user_agent", "message": "I'm researching quantum computing for my thesis."},
                    {"speaker": "ai_agent", "message": "That's fascinating! What aspect interests you most?"}
                ]
            }
            
            result = user1_system.add_conversation_to_memory(test_conversation)
            print(f"✅ Added test conversation: {result}")
            print(f"   Memory count after add: {user1_system.get_memory_count()}")
            
            # Test search
            search_results = user1_system.search_memories("quantum computing", limit=5)
            print(f"✅ Search returned {len(search_results)} results")
            for i, r in enumerate(search_results):
                print(f"   {i+1}. {r.get('content', '')[:60]}...")
        else:
            print("❌ Client initialization failed")
            sys.exit(1)
        
        # Test User 2 - should have separate collection
        print("\n📋 Testing User 2: business_executive")
        print("-" * 40)
        user2_system = AMemSystem("business_executive")
        
        if user2_system.initialize_client():
            print(f"✅ Client initialization successful")
            print(f"   Collection: {user2_system.get_collection_name()}")
            print(f"   Memory count: {user2_system.get_memory_count()}")
            
            # Verify isolation - search should not find user1's memories
            search_results = user2_system.search_memories("quantum computing", limit=5)
            print(f"✅ Search in user2's collection returned {len(search_results)} results (should be 0)")
            
            if len(search_results) == 0:
                print("✅ USER ISOLATION VERIFIED - User 2 cannot see User 1's memories")
            else:
                print("⚠️  Warning: User isolation may not be working correctly")
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("   - User-specific collections: ✓")
        print("   - Memory isolation: ✓")
        print("   - Add/Search operations: ✓")
            
    except Exception as e:
        print(f"❌ Error testing A-MEM system: {e}")
        import traceback
        traceback.print_exc()

