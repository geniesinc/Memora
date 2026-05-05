#!/usr/bin/env python3
"""
Base Evaluator Framework for Memory Systems
===========================================

This module provides a common framework for evaluating different memory systems
using a standardized pipeline approach.

COMMON PIPELINE:
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Store Conversations (conversation_to_memory.py)   │
│  ├─ Input: Conversation JSON files                         │
│  ├─ Process: Send to memory system with user_id            │
│  └─ Output: Memories stored in memory system               │
├─────────────────────────────────────────────────────────────┤
│  STEP 2: Answer Questions (memory_to_answer.py)            │
│  ├─ Input: Questions JSON file + user_id                   │
│  ├─ Process: Retrieve memories & generate answers           │
│  └─ Output: Answers with evaluation metrics                │
└─────────────────────────────────────────────────────────────┘

Each released memory system (a_mem, langmem, mem_0, memobase, memos, nemori)
implements the BaseMemorySystem interface to provide system-specific
functionality.

Usage:
    # Released layout: data/<period>/<persona>/conversations/, user_id="<persona>_<period>"
    python conversation_to_memory.py --system mem_0 --user-id academic_researcher_weekly \\
        --conversation-directory data/weekly/academic_researcher/conversations
    python memory_to_answer.py data/weekly/academic_researcher/evaluation_questions_academic_researcher.json \\
        --system mem_0 --user-id academic_researcher_weekly
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json


class BaseMemorySystem(ABC):
    """
    Abstract base class for memory system implementations.
    
    Each memory system (a_mem, langmem, mem_0, memobase, memos, nemori)
    inherits from this class and implements the required methods.
    """
    
    def __init__(self, user_id: str, **kwargs):
        """
        Initialize the memory system.
        
        Args:
            user_id: User ID for memory operations
            **kwargs: System-specific configuration parameters
        """
        self.user_id = user_id
        self.system_name = self.get_system_name()
        
    @abstractmethod
    def get_system_name(self) -> str:
        """Return the name of the memory system (e.g., 'mem_0', 'a_mem')."""
        pass
    
    @abstractmethod
    def initialize_client(self) -> bool:
        """
        Initialize the memory system client.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def add_conversation_to_memory(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add conversation messages to the memory system.
        
        Args:
            conversation_data: Raw conversation data from Memora
            
        Returns:
            Dict[str, Any]: Result from memory addition
        """
        pass
    
    @abstractmethod
    def search_memories(self, query: str, limit: int = 50, session_date: Optional[str] = None,
                       date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant memories using semantic search.
        
        Args:
            query: Search query/question
            limit: Maximum number of memories to return
            session_date: Optional session date (YYYY-MM-DD) to filter memories by exact date
            date_range: Optional tuple (start_date, end_date) to filter memories by date range
            
        Returns:
            List of relevant memories
        """
        pass
    
    @abstractmethod
    def get_required_env_vars(self) -> List[str]:
        """
        Get list of required environment variables for this system.
        
        Returns:
            List of environment variable names
        """
        pass
    
    def validate_environment(self) -> bool:
        """
        Validate that all required environment variables are set.
        
        Returns:
            bool: True if all required env vars are present
        """
        required_vars = self.get_required_env_vars()
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing required environment variables for {self.system_name}:")
            for var in missing_vars:
                print(f"   - {var}")
            return False
        
        return True
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get system configuration information.
        
        Returns:
            Dict with configuration details
        """
        return {
            'system_name': self.system_name,
            'user_id': self.user_id,
            'required_env_vars': self.get_required_env_vars(),
            'env_vars_set': {var: bool(os.getenv(var)) for var in self.get_required_env_vars()}
        }


class BaseEvaluator:
    """
    Base evaluator class that provides common evaluation functionality.
    
    This class handles the common evaluation pipeline and delegates
    memory system-specific operations to the provided memory system.
    """
    
    def __init__(self, memory_system: BaseMemorySystem):
        """
        Initialize the evaluator with a memory system.
        
        Args:
            memory_system: Instance of BaseMemorySystem implementation
        """
        self.memory_system = memory_system
        self.stats = {
            'total_questions': 0,
            'answered': 0,
            'failed': 0,
            'no_memories_found': 0,
            'with_memories_found': 0,
            'total_evaluations': 0,
            'passed_evaluations': 0,
            'questions_with_evaluations': 0,
            'memory_presence_total': 0,
            'memory_presence_passed': 0,
            'forgetting_absence_total': 0,
            'forgetting_absence_passed': 0
        }
    
    def validate_setup(self) -> bool:
        """
        Validate that the memory system is properly configured.
        
        Returns:
            bool: True if setup is valid
        """
        print(f"🔍 Validating {self.memory_system.system_name} setup...")
        
        # Check environment variables
        if not self.memory_system.validate_environment():
            return False
        
        # Initialize client
        if not self.memory_system.initialize_client():
            print(f"❌ Failed to initialize {self.memory_system.system_name} client")
            return False
        
        print(f"✅ {self.memory_system.system_name} setup validated successfully")
        return True
    
    def print_system_info(self):
        """Print information about the configured memory system."""
        config = self.memory_system.get_config_info()
        
        print(f"\n📋 Memory System Configuration")
        print("=" * 50)
        print(f"System: {config['system_name']}")
        print(f"User ID: {config['user_id']}")
        print(f"Environment Variables:")
        for var, is_set in config['env_vars_set'].items():
            status = "✅" if is_set else "❌"
            print(f"   {status} {var}")
        print("=" * 50)


def get_memory_system(system_name: str, user_id: str, **kwargs) -> BaseMemorySystem:
    """
    Factory function to create memory system instances.
    
    Args:
        system_name: Name of the memory system ('mem_0', 'a_mem', etc.)
        user_id: User ID for memory operations
        **kwargs: System-specific configuration parameters
        
    Returns:
        Instance of the appropriate memory system
        
    Raises:
        ImportError: If the memory system module is not available
        ValueError: If the system name is not recognized
    """
    system_name = system_name.lower()

    # Six memory agents reported in the paper (Table 3).
    try:
        if system_name == 'mem_0':
            from mem_0.mem0_integration import Mem0System
            return Mem0System(user_id, **kwargs)
        elif system_name == 'a_mem':
            from a_mem.amem_integration import AMemSystem
            return AMemSystem(user_id, **kwargs)
        elif system_name == 'memobase':
            from memobase_eval.memobase_integration import MemobaseSystem
            return MemobaseSystem(user_id, **kwargs)
        elif system_name == 'langmem':
            from langmem_eval.langmem_integration import LangMemSystem
            return LangMemSystem(user_id, **kwargs)
        elif system_name == 'nemori':
            from nemori_eval.nemori_integration import NemoriSystem
            return NemoriSystem(user_id, **kwargs)
        elif system_name == 'memos':
            from memos_eval.memos_integration import MemOSSystem
            return MemOSSystem(user_id, **kwargs)
        else:
            raise ValueError(
                f"Unknown memory system: {system_name}. "
                f"Released systems: a_mem, langmem, mem_0, memobase, memos, nemori"
            )

    except ImportError as e:
        raise ImportError(f"Failed to import {system_name} memory system: {e}")


def list_available_systems() -> List[str]:
    """
    List all available memory systems.
    
    Returns:
        List of available system names (normalized to lowercase)
    """
    available_systems = []
    
    # Check which system directories exist
    evals_dir = Path(__file__).parent
    for system_dir in evals_dir.iterdir():
        if system_dir.is_dir() and not system_dir.name.startswith('.') and not system_dir.name.startswith('__'):
            # Look for any *_integration.py or *integration.py file in the directory
            integration_files = list(system_dir.glob("*integration.py"))
            if integration_files:
                # Map directory name to system name
                dir_name = system_dir.name
                # For directories ending in _eval, use the base name
                if dir_name.endswith('_eval'):
                    system_name = dir_name[:-5]  # Remove '_eval' suffix
                else:
                    system_name = dir_name
                available_systems.append(system_name.lower())
    
    return sorted(available_systems)


if __name__ == "__main__":
    """Test the base evaluator framework."""
    print("🧪 Testing Base Evaluator Framework")
    print("=" * 50)
    
    # List available systems
    systems = list_available_systems()
    print(f"Available memory systems: {systems}")
    
    # Test system creation (if any systems are available)
    if systems:
        test_system = systems[0]
        print(f"\n🔍 Testing {test_system} system...")
        
        try:
            memory_system = get_memory_system(test_system, "test_user")
            evaluator = BaseEvaluator(memory_system)
            evaluator.print_system_info()
            
            if evaluator.validate_setup():
                print(f"✅ {test_system} system test passed")
            else:
                print(f"❌ {test_system} system test failed")
                
        except Exception as e:
            print(f"❌ Error testing {test_system}: {e}")
    else:
        print("No memory systems available for testing")
