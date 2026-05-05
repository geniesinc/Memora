#!/usr/bin/env python3
"""
Common Conversation to Memory Pipeline
======================================

This script provides a unified interface for storing conversations into different
memory systems using a standardized pipeline approach.

SUPPORTED MEMORY SYSTEMS (six released agents):
- a_mem      A-Mem (local ChromaDB)
- langmem    LangMem (local LangGraph store)
- mem_0      Mem0 (cloud)
- memobase   Memobase (cloud)
- memos      MemOS (cloud or self-hosted)
- nemori     Nemori (local)

COMMON PIPELINE:
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Store Conversations (THIS SCRIPT)                 │
│  ├─ Input: Conversation JSON files                         │
│  ├─ Process: Send to memory system with user_id            │
│  └─ Output: Memories stored in memory system               │
├─────────────────────────────────────────────────────────────┤
│  STEP 2: Answer Questions (memory_to_answer.py)            │
│  ├─ Input: Questions JSON file + SAME user_id              │
│  ├─ Process: Retrieve memories & generate answers           │
│  └─ Output: Answers with evaluation metrics                │
└─────────────────────────────────────────────────────────────┘

IMPORTANT: Use the SAME user_id for both steps!

Features:
- Auto-detects conversations/ subdirectory within session output directory
- Processes conversations in session order with embedded session dates
- Robust error tracking for failed sessions
- Automatic progress saving (every 5 sessions)
- Resume capability if process is interrupted (Ctrl+C or crash)
- Number-based and range-based session selection
- Skip already processed sessions (resume-safe)
- Support for multiple memory systems

Usage Examples:
    # Process with any of the six released agents (a_mem, langmem, mem_0, memobase, memos, nemori).
    # The released layout is data/<period>/<persona>/conversations/session_*.json,
    # and the convention is user_id="<persona>_<period>".
    python conversation_to_memory.py \
        --system mem_0 \
        --user-id academic_researcher_weekly \
        --conversation-directory data/weekly/academic_researcher/conversations

    # Process first 10 sessions only
    python conversation_to_memory.py \
        --system a_mem --user-id academic_researcher_weekly \
        --conversation-directory data/weekly/academic_researcher/conversations \
        --num-sessions 10
    
    # Process sessions in range (e.g., 10-50)
    python conversation_to_memory.py --system mem_0 --user-id test_user --range 10-50
    
    # Resume from interrupted run
    python conversation_to_memory.py --system mem_0 --user-id test_user --resume --progress-file mem0_progress_20251015_120000.json
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from base_evaluator import get_memory_system, list_available_systems
except ImportError as e:
    print(f"Error importing base_evaluator: {e}")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: simple progress without tqdm
    def tqdm(iterable, **kwargs):
        return iterable


class ConversationToMemoryProcessor:
    """
    Processes conversations in bulk from a directory to any supported memory system.
    
    Features:
    - Track successful and unsuccessful sessions
    - Resume capability if process is disrupted
    - Range-based and number-based session selection
    - Periodic progress saving
    - Support for multiple memory systems
    """
    
    def __init__(self, 
                 memory_system_name: str,
                 user_id: Optional[str] = None, 
                 progress_file: Optional[str] = None,
                 max_retries: int = -1,
                 initial_retry_delay: float = 2.0,
                 max_retry_delay: float = 120.0):
        """
        Initialize the conversation processor.
        
        Args:
            memory_system_name: Name of the memory system to use
            user_id: User ID for memories (falls back to MEMORA_USER_ID env var, then "memora_user")
            progress_file: Path to progress tracking file (auto-generated if None)
            max_retries: Maximum number of retry attempts (-1 for infinite retries, default: -1)
            initial_retry_delay: Initial delay between retries in seconds (default: 2.0)
            max_retry_delay: Maximum delay between retries in seconds (default: 120.0)
        """
        # Set user_id with priority: parameter > env var > default
        self.user_id = user_id or os.getenv('MEMORA_USER_ID') or "memora_user"
        self.memory_system_name = memory_system_name
        self.progress_file = progress_file
        self.progress_data = {
            'memory_system': memory_system_name,
            'user_id': self.user_id,
            'successful_sessions': [],
            'failed_sessions': [],
            'skipped_sessions': [],
            'processing_started': None,
            'last_updated': None,
            'total_processed': 0,
            'total_failed': 0
        }
        
        # Initialize memory system
        try:
            self.memory_system = get_memory_system(
                memory_system_name, 
                self.user_id,
                max_retries=max_retries,
                initial_retry_delay=initial_retry_delay,
                max_retry_delay=max_retry_delay
            )
            
            # Validate environment and initialize client
            if not self.memory_system.validate_environment():
                raise RuntimeError(f"Environment validation failed for {memory_system_name}. Check required environment variables.")
            
            if not self.memory_system.initialize_client():
                raise RuntimeError(f"Failed to initialize {memory_system_name} client. Check API keys and configuration.")
            
            print(f"✅ Connected to {memory_system_name} with user ID: {self.user_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {memory_system_name} connection: {e}")
    
    def extract_persona_and_timeline(self, directory_path: str = None, conversation_directory: str = None) -> Tuple[str, str]:
        """
        Extract (persona, timeline) from a path under the released layout
        ``data/<timeline>/<persona>/...``.

        Recognised inputs:
          - data/<timeline>/<persona>/                          (sessions dir)
          - data/<timeline>/<persona>/conversations/            (conv dir)
          - data/<timeline>/<persona>/conversations/successful/ (legacy gen layout)
        """
        TIMELINES = {"weekly", "monthly", "quarterly"}

        if conversation_directory:
            conv_path = Path(conversation_directory).resolve()
            if conv_path.name == "successful" and conv_path.parent.name == "conversations":
                base_dir = conv_path.parent.parent
            elif conv_path.name == "conversations":
                base_dir = conv_path.parent
            else:
                base_dir = conv_path
        elif directory_path:
            base_dir = Path(directory_path).resolve()
        else:
            return "unknown_persona", "weekly"

        # Released layout: parent is the timeline (weekly|monthly|quarterly),
        # current dir is the persona name.
        persona = base_dir.name
        timeline = base_dir.parent.name if base_dir.parent.name in TIMELINES else "weekly"
        return persona, timeline
    
    def find_conversation_directory(self, base_path: str = None, conversation_directory: str = None) -> Path:
        """
        Find the conversation directory within the provided path.
        
        Args:
            base_path: Path to a persona directory (e.g., data/weekly/academic_researcher).
                       This function will look for a `conversations/` subdirectory inside it.
            conversation_directory: Explicit path to the conversations directory
                       (takes priority over base_path).
        
        Returns:
            Path to the conversations directory, or None if not found
        """
        # If explicit conversation directory is provided, use it directly
        if conversation_directory:
            conv_dir = Path(conversation_directory)
            if conv_dir.exists():
                # Verify it contains session files
                session_files = list(conv_dir.glob("session_*.json"))
                session_files.extend(list(conv_dir.glob("**/session_*.json")))
                if session_files:
                    return conv_dir
                else:
                    print(f"⚠️  Warning: Conversation directory {conversation_directory} exists but contains no session_*.json files")
                    return None
            else:
                print(f"❌ Conversation directory does not exist: {conversation_directory}")
                return None
        
        if not base_path:
            return None
        base_path = Path(base_path)
        if not base_path.exists():
            return None
        
        # Check if base_path itself is a conversation directory or subdirectory
        if base_path.name in ["conversations", "conversations/", "successful"]:
            # Check for session files recursively
            session_files = list(base_path.glob("session_*.json"))
            session_files.extend(list(base_path.glob("**/session_*.json")))
            if session_files:
                return base_path
        
        # Check if there's a 'conversations' subdirectory
        conversations_dir = base_path / "conversations"
        if conversations_dir.exists() and conversations_dir.is_dir():
            # Check for session files directly in conversations/
            session_files = list(conversations_dir.glob("session_*.json"))
            
            # Also check in subdirectories like 'successful/'
            session_files.extend(list(conversations_dir.glob("*/session_*.json")))
            session_files.extend(list(conversations_dir.glob("**/session_*.json")))
            
            if session_files:
                # Return the conversations_dir (files might be in subdirectories)
                return conversations_dir
        
        # Check if base_path itself contains session files
        session_files = list(base_path.glob("session_*.json"))
        if session_files:
            return base_path
        
        return None
    
    def load_progress(self, progress_file: str) -> bool:
        """
        Load progress from a previous run.
        
        Args:
            progress_file: Path to progress file
            
        Returns:
            bool: True if progress was loaded successfully
        """
        progress_path = Path(progress_file)
        if not progress_path.exists():
            print(f"📝 No previous progress file found: {progress_file}")
            return False
        
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                self.progress_data = json.load(f)
            
            print(f"📂 Loaded progress from: {progress_file}")
            print(f"   Previously processed: {len(self.progress_data.get('successful_sessions', []))}")
            print(f"   Previously failed: {len(self.progress_data.get('failed_sessions', []))}")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load progress file: {e}")
            return False
    
    def save_progress(self, force: bool = False):
        """
        Save current progress to file.
        
        Args:
            force: Force save even if no progress file is set
        """
        if not self.progress_file and not force:
            return
        
        # Generate progress file name if not set
        if not self.progress_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.progress_file = f"{self.memory_system_name}_progress_{timestamp}.json"
        
        self.progress_data['last_updated'] = datetime.now().isoformat()
        
        try:
            progress_path = Path(self.progress_file)
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Progress saved to: {self.progress_file}")
            
        except Exception as e:
            print(f"⚠️  Failed to save progress: {e}")
    
    def extract_session_id(self, file_path: Path) -> Optional[int]:
        """
        Extract session ID from filename.
        
        Args:
            file_path: Path to session file
            
        Returns:
            Optional[int]: Session ID or None
        """
        match = re.search(r'session_(\d+)', file_path.name)
        if match:
            return int(match.group(1))
        return None
    
    def select_sessions(self, 
                       all_files: List[Path], 
                       num_sessions: Optional[int] = None,
                       session_range: Optional[Tuple[int, int]] = None,
                       skip_processed: bool = True) -> List[Path]:
        """
        Select sessions to process based on various criteria.
        
        Args:
            all_files: List of all session files
            num_sessions: Number of sessions to process (from start)
            session_range: Tuple of (start_session, end_session) for range-based selection
            skip_processed: Skip sessions that were already successfully processed
            
        Returns:
            List[Path]: Filtered list of session files to process
        """
        # Sort files by session ID
        sorted_files = sorted(all_files, 
                            key=lambda x: self.extract_session_id(x) or 0)
        
        # Filter out already processed sessions if requested
        if skip_processed:
            processed_sessions = set(self.progress_data.get('successful_sessions', []))
            sorted_files = [f for f in sorted_files 
                          if self.extract_session_id(f) not in processed_sessions]
            
            if processed_sessions:
                print(f"📋 Skipping {len(all_files) - len(sorted_files)} already processed sessions")
        
        # Apply range filter if specified
        if session_range:
            start, end = session_range
            sorted_files = [f for f in sorted_files 
                          if start <= (self.extract_session_id(f) or 0) <= end]
            print(f"🎯 Selected sessions in range [{start}, {end}]: {len(sorted_files)} files")
        
        # Apply number limit if specified
        if num_sessions is not None:
            sorted_files = sorted_files[:num_sessions]
            print(f"🎯 Limited to first {num_sessions} sessions")
        
        return sorted_files
    
    def analyze_directory(self, conv_dir: Path) -> Dict[str, Any]:
        """Analyze a conversation directory, including subdirectories like 'successful/'."""
        # Recursively search for session files in all subdirectories
        session_files = list(conv_dir.glob("session_*.json"))  # Direct files
        session_files.extend(list(conv_dir.glob("**/session_*.json")))  # Recursive search
        
        # Remove duplicates (in case files exist in multiple locations)
        session_files = list(set(session_files))
        
        # Sort by session ID for consistent ordering
        session_files.sort(key=lambda x: self.extract_session_id(x) or 0)
        
        # Count by session type
        type_counts = {}
        for file_path in session_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                session_type = data.get('session_type', 'unknown')
                type_counts[session_type] = type_counts.get(session_type, 0) + 1
            except:
                type_counts['error'] = type_counts.get('error', 0) + 1
        
        return {
            'directory': conv_dir.name,
            'total_files': len(session_files),
            'session_types': type_counts,
            'files': session_files
        }
    
    def process_directory(self, 
                         conv_dir: Path, 
                         num_sessions: Optional[int] = None,
                         session_range: Optional[Tuple[int, int]] = None,
                         skip_processed: bool = True,
                         save_interval: int = 5) -> Dict[str, Any]:
        """
        Process conversations in a directory with robust error handling.
        
        Args:
            conv_dir: Directory containing conversation files
            num_sessions: Number of sessions to process
            session_range: Tuple of (start, end) session IDs
            skip_processed: Skip already processed sessions
            save_interval: Save progress every N sessions
            
        Returns:
            Dict with processing results
        """
        print(f"\n📁 Processing: {conv_dir.name}")
        print("=" * 60)
        
        # Initialize progress tracking if not already started
        if not self.progress_data['processing_started']:
            self.progress_data['processing_started'] = datetime.now().isoformat()
        
        analysis = self.analyze_directory(conv_dir)
        print(f"📊 Directory Analysis:")
        print(f"   Total files: {analysis['total_files']}")
        print(f"   Session types: {analysis['session_types']}")
        
        # Select sessions to process
        files_to_process = self.select_sessions(
            analysis['files'],
            num_sessions=num_sessions,
            session_range=session_range,
            skip_processed=skip_processed
        )
        
        if not files_to_process:
            print(f"\n⚠️  No sessions to process (all may be already processed)")
            return {
                'directory': conv_dir.name,
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'skipped': len(analysis['files'])
            }
        
        print(f"   Processing: {len(files_to_process)} files")
        
        # Check if memory system supports batch processing (e.g., Cognee)
        if hasattr(self.memory_system, 'add_conversations_batch'):
            print(f"\n🚀 Using batch processing mode for {self.memory_system_name}...")
            return self._process_directory_batch(files_to_process, conv_dir, save_interval)
        
        # Process conversations one by one with error tracking
        print(f"\n🚀 Starting single-file processing...")
        successful_count = 0
        failed_count = 0
        
        # Create progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(files_to_process, desc="Processing sessions", unit="session")
        else:
            pbar = files_to_process
        
        for i, file_path in enumerate(pbar if not TQDM_AVAILABLE else pbar, 1):
            session_id = self.extract_session_id(file_path)
            
            try:
                if not TQDM_AVAILABLE:
                    print(f"📄 [{i}/{len(files_to_process)}] Processing session {session_id}: {file_path.name}")
                else:
                    # Update progress bar with current session
                    pbar.set_postfix({
                        'session': session_id,
                        'success': successful_count,
                        'failed': failed_count
                    })
                
                # Process the conversation file using the memory system
                result = self.memory_system.process_conversation_file(str(file_path))
                
                # Track success
                if session_id:
                    self.progress_data['successful_sessions'].append(session_id)
                successful_count += 1
                
                if not TQDM_AVAILABLE:
                    print(f"   ✅ Success")
                
            except Exception as e:
                if not TQDM_AVAILABLE:
                    print(f"   ❌ Failed: {e}")
                
                # Track failure with details
                if session_id:
                    self.progress_data['failed_sessions'].append({
                        'session_id': session_id,
                        'file': file_path.name,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                failed_count += 1
                
                # Update progress bar with error
                if TQDM_AVAILABLE:
                    pbar.set_postfix({
                        'session': session_id,
                        'success': successful_count,
                        'failed': failed_count,
                        'last_error': str(e)[:30]
                    })
                
                # Continue processing other files
                continue
            
            # Periodic progress save
            if i % save_interval == 0:
                self.progress_data['total_processed'] = successful_count
                self.progress_data['total_failed'] = failed_count
                self.save_progress(force=True)
        
        if TQDM_AVAILABLE:
            pbar.close()
        
        # Final progress save
        self.progress_data['total_processed'] = successful_count
        self.progress_data['total_failed'] = failed_count
        self.save_progress(force=True)
        
        print(f"\n✅ Processing Summary:")
        print(f"   Successfully processed: {successful_count}")
        print(f"   Failed: {failed_count}")
        
        return {
            'directory': conv_dir.name,
            'total_files': len(files_to_process),
            'successful': successful_count,
            'failed': failed_count,
            'skipped': len(analysis['files']) - len(files_to_process)
        }
    
    def _process_directory_batch(self, files_to_process: List[Path], conv_dir: Path, save_interval: int = 5) -> Dict[str, Any]:
        """
        Process conversations in batch mode (for systems like Cognee that support it).
        
        Args:
            files_to_process: List of conversation file paths
            conv_dir: Directory containing conversation files
            save_interval: Save progress every N sessions
            
        Returns:
            Dict with processing results
        """
        import json
        
        print(f"📦 Loading {len(files_to_process)} conversations for batch processing...")
        
        conversations = []
        successful_count = 0
        failed_count = 0
        
        # Load all conversations
        for file_path in files_to_process:
            session_id = self.extract_session_id(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    conv_data = json.load(f)
                    conversations.append(conv_data)
                    successful_count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to load {file_path.name}: {e}")
                if session_id:
                    self.progress_data['failed_sessions'].append({
                        'session_id': session_id,
                        'file': file_path.name,
                        'error': f"Failed to load: {e}",
                        'timestamp': datetime.now().isoformat()
                    })
                failed_count += 1
        
        print(f"✅ Loaded {successful_count} conversations ({failed_count} failed to load)")
        
        if not conversations:
            print("❌ No conversations loaded successfully")
            return {
                'directory': conv_dir.name,
                'total_files': len(files_to_process),
                'successful': 0,
                'failed': failed_count,
                'skipped': 0
            }
        
        # Batch add all conversations
        print(f"🚀 Batch processing {len(conversations)} conversations...")
        try:
            result = self.memory_system.add_conversations_batch(conversations)
            
            if result.get('status') == 'success' or result.get('status') == 'partial':
                added_count = result.get('added_count', 0)
                batch_failed = result.get('failed_count', 0)
                
                print(f"✅ Batch complete: {added_count} added, {batch_failed} failed")
                
                # Update progress tracking
                for conv in conversations[:added_count]:
                    session_id = conv.get('session_id', 'unknown')
                    if session_id != 'unknown':
                        self.progress_data['successful_sessions'].append(session_id)
                
                # Save progress
                self.save_progress()
                
                return {
                    'directory': conv_dir.name,
                    'total_files': len(files_to_process),
                    'successful': added_count,
                    'failed': failed_count + batch_failed,
                    'skipped': 0
                }
            else:
                print(f"❌ Batch processing failed: {result.get('message', 'Unknown error')}")
                return {
                    'directory': conv_dir.name,
                    'total_files': len(files_to_process),
                    'successful': 0,
                    'failed': len(files_to_process),
                    'skipped': 0
                }
                
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'directory': conv_dir.name,
                'total_files': len(files_to_process),
                'successful': 0,
                'failed': len(files_to_process),
                'skipped': 0
            }
    
    def run(self, 
           directory_path: str = None,
           conversation_directory: str = None, 
           num_sessions: int = None,
           session_range: Tuple[int, int] = None,
           resume: bool = False,
           progress_file: str = None,
           skip_processed: bool = True):
        """
        Run the bulk conversation processing with robust error handling.
        
        Args:
            directory_path: Path to session output directory (auto-detects conversations/)
            conversation_directory: Explicit path to conversation directory (takes priority)
            num_sessions: Number of sessions to process (from start)
            session_range: Tuple of (start, end) session IDs for range-based processing
            resume: Resume from previous progress file
            progress_file: Path to progress file (for resume or explicit tracking)
            skip_processed: Skip already processed sessions (default: True)
        """
        # Extract persona and timeline for display (user_id already set in __init__)
        persona, timeline = self.extract_persona_and_timeline(directory_path, conversation_directory)
        
        # Auto-generate user_id if it's still the default
        if self.user_id == "memora_user" or (not self.user_id):
            self.user_id = f"{persona}_{timeline}"
            print(f"📋 Auto-detected: Persona={persona}, Timeline={timeline}")
            print(f"📋 Generated User ID: {self.user_id}")
        
        print(f"🚀 STEP 1: Storing Conversations in {self.memory_system_name.upper()}")
        print("=" * 60)
        print(f"This script stores conversations in {self.memory_system_name} memory system.")
        print(f"")
        print(f"💡 NEXT STEP: After this completes, run Step 2:")
        print(f"   python memory_to_answer.py questions.json --system {self.memory_system_name} --user-id {self.user_id}")
        print(f"")
        print("=" * 60)
        print(f"Memory System: {self.memory_system_name}")
        print(f"User ID: {self.user_id} (use same ID in Step 2!)")
        
        # Find conversation directory first to determine base path
        # conversation_directory takes priority over directory_path
        conv_dir = self.find_conversation_directory(directory_path, conversation_directory)
        
        if not conv_dir:
            if conversation_directory:
                print(f"❌ Conversation directory not found or empty: {conversation_directory}")
            elif directory_path:
                print(f"❌ No conversation directory found in: {directory_path}")
                print(f"   Expected to find 'conversations/' subdirectory with session_*.json files")
            else:
                print(f"❌ No conversation directories found in project output/")
            return
        
        # Create memory_processing directory in the session output directory
        # If conversation directory is in conversations/successful, go up 2 levels
        # If it's in conversations/, go up 1 level
        # Otherwise, use the conversation directory itself
        if conv_dir.name == "successful" and conv_dir.parent.name == "conversations":
            # conversations/successful -> go up to session directory
            base_dir = conv_dir.parent.parent
        elif conv_dir.name == "conversations":
            # conversations/ -> go up to session directory
            base_dir = conv_dir.parent
        else:
            # Use conversation directory as base (user-specified path)
            base_dir = conv_dir
        
        memory_processing_dir = base_dir / "memory_processing"
        memory_processing_dir.mkdir(exist_ok=True)
        print(f"📁 Memory processing directory: {memory_processing_dir}")
        
        # Handle resume mode
        if resume or progress_file:
            if progress_file and not Path(progress_file).is_absolute():
                # If relative path provided, look in memory_processing dir
                progress_file = str(memory_processing_dir / progress_file)
            progress_file = progress_file or str(memory_processing_dir / f"{self.memory_system_name}_progress_latest.json")
            self.progress_file = progress_file
            self.load_progress(progress_file)
        else:
            # Auto-generate progress file in memory_processing directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.progress_file = str(memory_processing_dir / f"{self.memory_system_name}_progress_{timestamp}.json")
            print(f"📝 Progress will be saved to: {self.progress_file}")
        
        # Display processing parameters
        if num_sessions:
            print(f"Processing limit: {num_sessions} sessions")
        elif session_range:
            print(f"Processing range: sessions {session_range[0]} to {session_range[1]}")
        else:
            print(f"Processing limit: No limit (all sessions)")
        
        if skip_processed:
            print(f"Skip processed: Yes (resume-safe)")
        
        print(f"\n🔍 Found conversation directory:")
        print(f"   Path: {conv_dir}")
        
        # Analyze the directory
        analysis = self.analyze_directory(conv_dir)
        print(f"\n📊 Directory Analysis:")
        print(f"   Total files: {analysis['total_files']}")
        print(f"   Session types: {analysis['session_types']}")
        
        if analysis['total_files'] == 0:
            print(f"❌ No session files found in {conv_dir}")
            return
        
        # Process the directory with error tracking
        try:
            result = self.process_directory(
                conv_dir, 
                num_sessions=num_sessions,
                session_range=session_range,
                skip_processed=skip_processed
            )
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Processing interrupted by user!")
            print(f"💾 Progress has been saved to: {self.progress_file}")
            print(f"\n💡 To resume, run with: --resume --progress-file {self.progress_file}")
            return
        except Exception as e:
            print(f"\n❌ Fatal error during processing: {e}")
            print(f"💾 Progress has been saved to: {self.progress_file}")
            print(f"\n💡 To resume, run with: --resume --progress-file {self.progress_file}")
            raise
        
        # Final summary
        print(f"\n📊 Final Summary")
        print("=" * 60)
        print(f"Memory System: {self.memory_system_name}")
        print(f"Directory processed: {conv_dir.name}")
        print(f"Total conversations processed: {result.get('successful', 0)}")
        print(f"Failed: {result.get('failed', 0)}")
        print(f"Skipped (already processed): {result.get('skipped', 0)}")
        
        # Show failed sessions if any
        if result.get('failed', 0) > 0:
            print(f"\n⚠️  Failed Sessions:")
            for failure in self.progress_data.get('failed_sessions', [])[-5:]:  # Show last 5
                print(f"   - Session {failure.get('session_id')}: {failure.get('error', 'Unknown error')}")
            if len(self.progress_data.get('failed_sessions', [])) > 5:
                print(f"   ... and {len(self.progress_data.get('failed_sessions', [])) - 5} more")
            print(f"\n   Check {self.progress_file} for full details")
        
        if result.get('successful', 0) > 0:
            print(f"\n✅ Conversations successfully stored in {self.memory_system_name}!")
            print(f"🧠 {self.memory_system_name} can now create memories from these conversation patterns")
            print(f"💾 Progress saved to: {self.progress_file}")
        
        print(f"\n🎉 Bulk processing completed!")


def parse_range(range_str: str) -> Tuple[int, int]:
    """Parse a range string like '10-50' into a tuple (10, 50)."""
    try:
        start, end = range_str.split('-')
        return (int(start.strip()), int(end.strip()))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Range must be in format 'start-end', got: {range_str}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Bulk process conversations for any supported memory system with robust error handling and resume capability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Released layout is data/<period>/<persona>/conversations/, and the convention
  # is user_id="<persona>_<period>" so step-2 retrieval finds the same store.

  # Full run on one persona
  python conversation_to_memory.py \\
      --system mem_0 --user-id academic_researcher_weekly \\
      --conversation-directory data/weekly/academic_researcher/conversations

  # First 10 sessions only
  python conversation_to_memory.py \\
      --system a_mem --user-id academic_researcher_weekly \\
      --conversation-directory data/weekly/academic_researcher/conversations \\
      --num-sessions 10

  # Resume an interrupted run (auto-picks the latest progress file)
  python conversation_to_memory.py \\
      --system mem_0 --user-id academic_researcher_weekly \\
      --conversation-directory data/weekly/academic_researcher/conversations \\
      --resume

Released memory systems:
  a_mem, langmem, mem_0, memobase, memos, nemori

Features:
  - Automatic progress tracking and periodic saves
  - Resume capability if process is interrupted (Ctrl+C)
  - Detailed error tracking for failed sessions
  - Range-based or number-based session selection
  - Skip already processed sessions (resume-safe)
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "directory",
        nargs="?",
        help="Path to a persona directory (e.g., data/weekly/academic_researcher). "
             "Will look for a 'conversations/' subdirectory inside. If both this and "
             "--conversation-directory are omitted, looks for the most recent sessions directory."
    )

    # Required system argument
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        help="Memory system to use (a_mem, langmem, mem_0, memobase, memos, nemori)"
    )
    
    # Conversation directory argument
    parser.add_argument(
        "--conversation-directory",
        type=str,
        metavar="DIR",
        help="Explicit path to conversation directory (e.g., data/weekly/academic_researcher/conversations). "
             "Takes priority over directory argument. Recursively searches for session_*.json files."
    )
    
    # Session selection arguments
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "--num-sessions", 
        type=int, 
        metavar="N",
        help="Process first N sessions (default: process all)"
    )
    session_group.add_argument(
        "--range", 
        type=parse_range,
        metavar="START-END",
        help="Process sessions in range (e.g., --range 10-50)"
    )
    
    # Resume and progress tracking
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume from previous progress file"
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        help="Path to progress file (for resume or explicit tracking)"
    )
    parser.add_argument(
        "--no-skip-processed",
        action="store_true",
        help="Don't skip already processed sessions (force reprocess)"
    )
    
    # Configuration
    parser.add_argument(
        "--user-id", 
        type=str,
        help="User ID for memories (default: from MEMORA_USER_ID env var, or 'memora_user')"
    )
    
    # Retry configuration
    parser.add_argument(
        "--max-retries",
        type=int,
        default=-1,
        metavar="N",
        help="Maximum number of retry attempts (-1 for infinite retries, default: -1)"
    )
    parser.add_argument(
        "--initial-retry-delay",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Initial delay between retries in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--max-retry-delay",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help="Maximum delay between retries in seconds (default: 120.0)"
    )
    
    args = parser.parse_args()
    
    # List available systems if requested
    if args.system == "list":
        available_systems = list_available_systems()
        print("Available memory systems:")
        for system in available_systems:
            print(f"  - {system}")
        return
    
    try:
        processor = ConversationToMemoryProcessor(
            memory_system_name=args.system,
            user_id=args.user_id,
            progress_file=args.progress_file,
            max_retries=args.max_retries,
            initial_retry_delay=args.initial_retry_delay,
            max_retry_delay=args.max_retry_delay
        )
        
        processor.run(
            directory_path=args.directory,
            conversation_directory=args.conversation_directory,
            num_sessions=args.num_sessions,
            session_range=args.range,
            resume=args.resume,
            progress_file=args.progress_file,
            skip_processed=not args.no_skip_processed
        )
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Gracefully shut down. Progress has been saved.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()