"""
Conversation Slicing for Memory Evaluation

Extracts memory-related segments from conversations for quality evaluation.
"""

from typing import Dict, List, Any, Optional, Tuple
import os


def extract_memory_segment(conversation_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract memory-related segment from conversation data.
    
    Args:
        conversation_data: Full conversation data with session metadata
        
    Returns:
        Dict containing memory segment with context, or None if no memory segment found
    """
    try:
        conversation = conversation_data.get('conversation', [])
        if not conversation:
            return None
        
        # Find memory-related phases
        memory_phase_start = None
        memory_phase_end = None
        memory_turns = []
        
        for i, turn in enumerate(conversation):
            phase = turn.get('phase', '')
            share_memory = turn.get('share_memory', False)
            
            # Check for memory-related phases
            # For content_delete/content_update, include ALL turns in the phase (not just share_memory=True)
            # because the initial turn where user identifies the content is critical for evaluation
            if phase in ['memory_share', 'content_share', 'content_delete', 'content_update']:
                if memory_phase_start is None:
                    memory_phase_start = i
                memory_phase_end = i
                memory_turns.append(i)
        
        if memory_phase_start is None:
            return None
        
        # Extract context before memory sharing
        context_before = None
        if memory_phase_start > 0:
            before_turn = conversation[memory_phase_start - 1]
            context_before = {
                "turn": before_turn.get('turn', 0),
                "speaker": before_turn.get('speaker', ''),
                "message": before_turn.get('message', ''),
                "phase": before_turn.get('phase', ''),
                "intent": before_turn.get('intent', '')
            }
        
        # Extract context after memory sharing
        context_after = None
        if memory_phase_end < len(conversation) - 1:
            after_turn = conversation[memory_phase_end + 1]
            context_after = {
                "turn": after_turn.get('turn', 0),
                "speaker": after_turn.get('speaker', ''),
                "message": after_turn.get('message', ''),
                "phase": after_turn.get('phase', ''),
                "intent": after_turn.get('intent', '')
            }
        
        # Extract memory turns
        memory_turn_data = []
        for turn_idx in memory_turns:
            turn = conversation[turn_idx]
            memory_turn_data.append({
                "turn": turn.get('turn', 0),
                "speaker": turn.get('speaker', ''),
                "message": turn.get('message', ''),
                "phase": turn.get('phase', ''),
                "intent": turn.get('intent', ''),
                "share_memory": turn.get('share_memory', False)
            })
        
        return {
            "context_before": context_before,
            "memory_turns": memory_turn_data,
            "context_after": context_after,
            "memory_phase_start": memory_phase_start,
            "memory_phase_end": memory_phase_end,
            "total_memory_turns": len(memory_turn_data)
        }
        
    except Exception as e:
        print(f"Error extracting memory segment: {e}")
        return None


def test_memory_slicing():
    """Test the memory slicing functionality with sample data"""
    
    # Sample conversation data
    sample_conversation = {
        "session_id": 1,
        "session_type": "preference_memory",
        "conversation": [
            {
                "turn": 1,
                "speaker": "user_agent",
                "message": "That's really interesting about quantum computing!",
                "phase": "opening",
                "intent": "user_general_question"
            },
            {
                "turn": 2,
                "speaker": "ai_agent", 
                "message": "I'm glad you found it interesting!",
                "phase": "opening",
                "intent": "ai_general_response"
            },
            {
                "turn": 3,
                "speaker": "user_agent",
                "message": "Speaking of technology, I really like techno music.",
                "phase": "memory_share",
                "intent": "user_casual_memory_add",
                "share_memory": True
            },
            {
                "turn": 4,
                "speaker": "ai_agent",
                "message": "Thanks for sharing your music preference!",
                "phase": "memory_share", 
                "intent": "ai_acknowledge_preference_sharing",
                "share_memory": True
            },
            {
                "turn": 5,
                "speaker": "user_agent",
                "message": "What other genres do you think I might enjoy?",
                "phase": "closing",
                "intent": "user_continue_conversation"
            }
        ]
    }
    
    print("🧪 Testing conversation slicing...")
    print("=" * 50)
    
    # Extract memory segment
    memory_segment = extract_memory_segment(sample_conversation)
    
    if memory_segment:
        print("✅ Memory segment found!")
        print(f"📊 Memory turns: {memory_segment['total_memory_turns']}")
        print(f"📍 Phase range: {memory_segment['memory_phase_start']} - {memory_segment['memory_phase_end']}")
        
        if memory_segment['context_before']:
            print(f"\n🔍 Context Before:")
            print(f"   {memory_segment['context_before']['speaker']}: {memory_segment['context_before']['message'][:50]}...")
        
        print(f"\n💬 Memory Turns:")
        for turn in memory_segment['memory_turns']:
            print(f"   {turn['speaker']}: {turn['message'][:50]}...")
        
        if memory_segment['context_after']:
            print(f"\n🔍 Context After:")
            print(f"   {memory_segment['context_after']['speaker']}: {memory_segment['context_after']['message'][:50]}...")
    else:
        print("❌ No memory segment found")
    
    return memory_segment


def test_memory_slice_with_file(conversation_file: str):
    """
    Test memory slicing with an existing conversation file.
    
    Args:
        conversation_file: Path to the conversation JSON file
    """
    import json
    import os
    
    print(f"🔍 Testing memory slice with: {conversation_file}")
    print("=" * 60)
    
    # Load the conversation file
    try:
        with open(conversation_file, 'r') as f:
            conversation_data = json.load(f)
        print(f"✅ Loaded conversation file")
    except Exception as e:
        print(f"❌ Error loading conversation file: {e}")
        return None
    
    # Display basic conversation info
    print(f"📊 Conversation Info:")
    print(f"   - Session ID: {conversation_data.get('session_id', 'N/A')}")
    print(f"   - Session Type: {conversation_data.get('session_type', 'N/A')}")
    print(f"   - Operation: {conversation_data.get('operation', 'N/A')}")
    print(f"   - Item: {conversation_data.get('item', 'N/A')}")
    print(f"   - Total turns: {len(conversation_data.get('conversation', []))}")
    
    # Extract memory segment
    print(f"\n🔪 Extracting memory segment...")
    memory_segment = extract_memory_segment(conversation_data)
    
    if memory_segment is None:
        print("❌ No memory segment found in conversation")
        return None
    
    print(f"✅ Memory segment extracted successfully!")
    print(f"\n📊 Memory Segment Details:")
    print(f"   - Memory phase start: {memory_segment['memory_phase_start']}")
    print(f"   - Memory phase end: {memory_segment['memory_phase_end']}")
    print(f"   - Total memory turns: {memory_segment['total_memory_turns']}")
    
    # Show context before
    if memory_segment['context_before']:
        print(f"\n🔍 Context Before Memory:")
        before = memory_segment['context_before']
        print(f"   Turn {before['turn']}: {before['speaker']} - {before['message'][:80]}...")
        print(f"   Phase: {before['phase']} | Intent: {before['intent']}")
    else:
        print(f"\n🔍 Context Before: None (memory starts at beginning)")
    
    # Show memory turns
    print(f"\n💬 Memory Turns:")
    for i, turn in enumerate(memory_segment['memory_turns'], 1):
        print(f"   {i}. Turn {turn['turn']}: {turn['speaker']} - {turn['message'][:80]}...")
        print(f"      Phase: {turn['phase']} | Intent: {turn['intent']} | Share Memory: {turn['share_memory']}")
    
    # Show context after
    if memory_segment['context_after']:
        print(f"\n🔍 Context After Memory:")
        after = memory_segment['context_after']
        print(f"   Turn {after['turn']}: {after['speaker']} - {after['message'][:80]}...")
        print(f"   Phase: {after['phase']} | Intent: {after['intent']}")
    else:
        print(f"\n🔍 Context After: None (memory ends at end)")
    
    return memory_segment

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 2:
        # Test with provided JSON file
        conversation_file = sys.argv[1]
        if not os.path.exists(conversation_file):
            print(f"❌ Conversation file not found: {conversation_file}")
            sys.exit(1)
        
        memory_segment = test_memory_slice_with_file(conversation_file)
        
        if memory_segment:
            print(f"\n🎉 Memory slicing test completed successfully!")
            print(f"✅ Found {memory_segment['total_memory_turns']} memory turns")
        else:
            print(f"\n❌ Memory slicing test failed - no memory segment found")
    else:
        # Run the original test with sample data
        print("Usage: python memory_slice.py <conversation_file>")
        print("Example: python memory_slice.py ../output/sessions_<timestamp>_<persona>/conversations/session_0002_activity_memory.json")
        print("\nRunning sample test instead...")
        test_memory_slicing()
