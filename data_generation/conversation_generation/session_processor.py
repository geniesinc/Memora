"""
Session Processor

Handles traversing and processing session data from new_sessions.json
Extracts memory type, operation, and metadata for conversation generation.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class SessionInfo:
    """Structured session information for conversation generation"""
    session_id: int
    memory_type: str
    category: str
    subcategory: str
    preference_type: str
    item: str
    operation: str
    operation_details: Dict[str, Any]
    date: str
    
    def __str__(self):
        return f"Session {self.session_id}: {self.memory_type}.{self.operation} ({self.category}/{self.subcategory})"


class SessionProcessor:
    """Processes session data for memory-grounded conversation generation"""
    
    def __init__(self, session_file_path: str):
        self.session_file_path = session_file_path
        self.session_data = None
        self.persona = None
    
    def load_sessions(self) -> bool:
        """Load session data from JSON file"""
        try:
            with open(self.session_file_path, 'r', encoding='utf-8') as file:
                self.session_data = json.load(file)
                self.persona = self.session_data.get('persona', 'unknown')
                print(f"✅ Loaded {len(self.session_data.get('sessions', []))} sessions")
                print(f"👤 Persona: {self.persona}")
                return True
        except Exception as e:
            print(f"❌ Error loading sessions: {e}")
            return False
    
    def get_sessions(self, limit: Optional[int] = None) -> List[SessionInfo]:
        """Get structured session information"""
        if not self.session_data:
            return []
        
        sessions = self.session_data.get('sessions', [])
        if limit:
            sessions = sessions[:limit]
        
        session_info_list = []
        for session in sessions:
            operation_details = session.get('operation_details', {})
            memory_type = session.get('type')
            category = session.get('category')
            
            # Handle no_memory sessions differently
            if memory_type == 'no_memory':
                # No_memory sessions don't have memory-related fields
                subcategory = ''
                preference_type = ''
                item = ''
            elif memory_type == 'activity_memory':
                # For activity memory, derive subcategory, preference_type, and item from the activity data
                subcategory, preference_type, item = self._map_activity_memory_fields(category, operation_details)
            else:
                # For other memory types, use the existing mapping
                subcategory = operation_details.get('subcategory', '')
                preference_type = operation_details.get('preference', '')
                item = operation_details.get('item', '')
            
            session_info = SessionInfo(
                session_id=session.get('id'),
                memory_type=memory_type,
                category=category if category is not None else '',
                subcategory=subcategory,
                preference_type=preference_type,
                item=item,
                operation=session.get('operation'),
                operation_details=operation_details,
                date=session.get('date')
            )
            session_info_list.append(session_info)
        
        return session_info_list
    
    def _map_activity_memory_fields(self, category: str, operation_details: Dict[str, Any]) -> tuple[str, str, str]:
        """Map activity memory fields to subcategory, preference_type, and item"""
        item_data = operation_details.get('item', {})
        
        if category == 'todo_list':
            # Todo list activities
            task_type = item_data.get('task_type', 'general_tasks')
            description = item_data.get('description', 'task')
            return task_type, 'task_management', description
            
        elif category == 'step_tracker':
            # Step tracking activities
            step_count = item_data.get('step_count', 0)
            return 'fitness_tracking', 'activity_tracking', f"{step_count} steps"
            
        elif category == 'food_expenses':
            # Food expense activities
            amount = item_data.get('amount', 0)
            return 'food_expenses', 'expense_tracking', f"${amount:.2f} food expense"
            
        elif category == 'calendar_event':
            # Calendar event activities
            event_name = item_data.get('event_name', 'event')
            return 'scheduling', 'event_management', event_name
            
        else:
            # Default fallback for unknown activity categories
            return category, 'activity', str(item_data) if item_data else 'activity'
    
    def get_session_by_id(self, session_id: int) -> Optional[SessionInfo]:
        """Get specific session by ID"""
        sessions = self.get_sessions()
        for session in sessions:
            if session.session_id == session_id:
                return session
        return None
    
    def get_sessions_by_type(self, memory_type: str) -> List[SessionInfo]:
        """Get sessions of specific memory type"""
        sessions = self.get_sessions()
        return [s for s in sessions if s.memory_type == memory_type]
    
    def get_sessions_by_operation(self, operation: str) -> List[SessionInfo]:
        """Get sessions with specific operation"""
        sessions = self.get_sessions()
        return [s for s in sessions if s.operation == operation]
    
    def get_memory_types(self) -> List[str]:
        """Get all unique memory types in the session data"""
        sessions = self.get_sessions()
        return list(set(s.memory_type for s in sessions))
    
    def get_categories(self) -> List[str]:
        """Get all unique categories in the session data"""
        sessions = self.get_sessions()
        return list(set(s.category for s in sessions))
    
    def get_operations(self) -> List[str]:
        """Get all unique operations in the session data"""
        sessions = self.get_sessions()
        return list(set(s.operation for s in sessions))
    
    def analyze_sessions(self) -> Dict[str, Any]:
        """Analyze session data and provide summary"""
        sessions = self.get_sessions()
        
        if not sessions:
            return {"error": "No sessions loaded"}
        
        # Count by memory type
        memory_type_counts = {}
        for session in sessions:
            memory_type_counts[session.memory_type] = memory_type_counts.get(session.memory_type, 0) + 1
        
        # Count by operation
        operation_counts = {}
        for session in sessions:
            operation_counts[session.operation] = operation_counts.get(session.operation, 0) + 1
        
        # Count by category
        category_counts = {}
        for session in sessions:
            category_counts[session.category] = category_counts.get(session.category, 0) + 1
        
        return {
            "total_sessions": len(sessions),
            "persona": self.persona,
            "memory_types": memory_type_counts,
            "operations": operation_counts,
            "categories": category_counts,
            "date_range": {
                "start": min(s.date for s in sessions),
                "end": max(s.date for s in sessions)
            }
        }
    
    def extract_memory_content(self, session: SessionInfo) -> Dict[str, Any]:
        """Extract the actual memory content from session"""
        content = {
            "memory_type": session.memory_type,
            "category": session.category,
            "operation": session.operation
        }
        
        operation_details = session.operation_details
        
        if session.memory_type == "preference_memory":
            content.update({
                "item": operation_details.get("item", ""),
                "preference": operation_details.get("preference", ""),
                "subcategory": operation_details.get("subcategory", "")
            })
        
        elif session.memory_type == "activity_memory":
            item = operation_details.get("item", {})
            content.update({
                "activity_data": item,
                "category_specific": session.category
            })
        
        elif session.memory_type == "content_memory":
            content_data = operation_details.get("content_data", {})
            content.update({
                "item": operation_details.get("item", ""),
                "content_data": content_data
            })
        
        elif session.memory_type == "goal_memory":
            content.update({
                "subcategory": operation_details.get("subcategory", ""),
                "item": operation_details.get("item", ""),
                "actual_operation": operation_details.get("actual_operation", ""),
                "old_value": operation_details.get("old_value", None)
            })
        
        return content


def main():
    """Test the session processor"""
    print("📂 Session Processor Test")
    print("=" * 40)
    
    # Initialize processor
    processor = SessionProcessor("../output/new_sessions.json")
    
    # Load sessions
    if not processor.load_sessions():
        return
    
    # Analyze sessions
    analysis = processor.analyze_sessions()
    print(f"\n📊 Session Analysis:")
    print(f"Total sessions: {analysis['total_sessions']}")
    print(f"Persona: {analysis['persona']}")
    
    print(f"\nMemory types:")
    for mem_type, count in analysis['memory_types'].items():
        print(f"  • {mem_type}: {count}")
    
    print(f"\nOperations:")
    for operation, count in analysis['operations'].items():
        print(f"  • {operation}: {count}")
    
    # Show sample sessions
    sessions = processor.get_sessions(3)
    print(f"\n📝 Sample Sessions:")
    for session in sessions:
        print(f"  {session}")
        
        # Show extracted memory content
        content = processor.extract_memory_content(session)
        print(f"    Memory content: {content}")
        print()


if __name__ == "__main__":
    main()
