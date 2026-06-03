"""
Custom Instruction Generator

A dedicated class for generating contextual instructions for intents that require custom instructions.
This provides a clean, modular approach to instruction generation with dedicated methods for each memory type.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from flow_manager import SessionMetadata


class InstructionGenerator:
    """
    Dedicated class for generating custom instructions for memory-related intents
    """
    
    def __init__(self):
        pass
    
    def generate_custom_instruction(self, intent_id: str, intent_data: Dict[str, Any], 
                                  session: SessionMetadata, current_field: str = None, feedback: str = None) -> str:
        """
        Generate custom instruction for a specific intent
        
        Args:
            intent_id: The intent identifier
            intent_data: Intent configuration data
            session: Session metadata for context
            current_field: Current content field being discussed (for content memory)
            
        Returns:
            Contextual instruction string with memory metadata and specific guidance
        """
        # Get common context
        memory_metadata = session.get_memory_context_string()
        agent = intent_data.get('agent', 'unknown')
        
        # Add feedback to memory metadata if available
        if feedback:
            memory_metadata += f"\n\nEVALUATION FEEDBACK:\n{feedback}"
        
        # Route to appropriate instruction generator based on memory type
        if session.memory_type == "preference_memory":
            return self._generate_preference_memory_instruction(intent_id, agent, session, memory_metadata)
        elif session.memory_type == "activity_memory":
            return self._generate_activity_memory_instruction(intent_id, agent, session, memory_metadata)
        elif session.memory_type == "goal_memory":
            return self._generate_goal_memory_instruction(intent_id, agent, session, memory_metadata)
        elif session.memory_type == "content_memory":
            return self._generate_content_memory_instruction(intent_id, agent, session, memory_metadata, current_field)
        else:
            # Fallback for unknown memory types
            description = intent_data.get('description', f"Execute intent: {intent_id}")
            return f"{memory_metadata}\n\n{description}"
    
    def _generate_preference_memory_instruction(self, intent_id: str, agent: str, 
                                              session: SessionMetadata, memory_metadata: str) -> str:
        """
        Generate instructions for preference memory intents
        """
        item_desc = session.item if session.item else str(session.operation_details.get('item', 'item'))
        
        # AI Agent Instructions
        if intent_id == "ai_transition_to_add_memory":
            return f"{memory_metadata}\n\nAI should smoothly transition to asking about the user's preferences regarding {session.category}/{session.subcategory}. Ask an open-ended question to let the user naturally share what they like."
        
        elif intent_id == "ai_transition_to_update_memory":
            return f"{memory_metadata}\n\nAI should ask about changes in the user's preferences in {session.category}/{session.subcategory}. Ask openly about how their tastes might have evolved or changed recently."
        
        elif intent_id == "ai_transition_to_delete_memory":
            return f"{memory_metadata}\n\nAI should ask about items the user no longer likes or is interested in regarding {session.category}/{session.subcategory}. Let them naturally mention what they've lost interest in."
        
        elif intent_id == "ai_ask_about_new_discoveries":
            subcat = session.subcategory or session.category
            return f"{memory_metadata}\n\nAI should ask about new {subcat} items the user has discovered recently. Ask an open question without mentioning specific items - let the user share what they found."
        
        elif intent_id == "ai_ask_about_recent_consumption_memory":
            subcat = session.subcategory or session.category
            return f"{memory_metadata}\n\nAI should ask what {subcat} content the user has been enjoying recently. Keep the question open-ended so the user can share their recent favorites."
        
        elif intent_id == "ai_share_memory_then_ask_user_reflection":
            return f"{memory_metadata}\n\nAI should share its own opinion about {session.category}/{session.subcategory}, then ask the user to reflect on their preferences, creating space to discuss '{item_desc}'."
        
        elif intent_id == "ai_ask_about_memory_item":
            pref_type = session.preference_type or 'preference'
            return f"{memory_metadata}\n\nAI should ask the user about their {pref_type}s in {session.category}/{session.subcategory}, encouraging them to share specific items."
        
        elif intent_id == "ai_ask_about_user_preference_on_shared_memory":
            return f"{memory_metadata}\n\nAI should ask about the user's preference on the memory item they just shared, encouraging them to express their feelings about '{item_desc}'."
        
        elif intent_id == "ai_comparative_question_to_memory":
            subcat = session.subcategory or session.category
            return f"{memory_metadata}\n\nAI should ask the user to compare different {subcat} options, naturally leading to preference sharing about '{item_desc}'."
        
        elif intent_id == "ai_curiosity_driven_memory_question":
            subcat = session.subcategory or session.category
            return f"{memory_metadata}\n\nAI should express curiosity about the user's tastes in {subcat} in a casual, friendly way, asking them to share their favorites."
        
        elif intent_id == "ai_ask_why_changed":
            return f"{memory_metadata}\n\nAI should ask why the user's preference changed regarding '{item_desc}', showing genuine interest in their reasoning."
        
        elif intent_id == "ai_ask_why_preference":
            pref_type = session.preference_type or 'like'
            return f"{memory_metadata}\n\nAI should ask why the user {pref_type}s '{item_desc}', encouraging them to elaborate on their preference."
        
        elif intent_id == "ai_ask_about_previous_preference":
            subcat = session.subcategory or session.category
            return f"{memory_metadata}\n\nAI should ask about the user's previous preferences in {subcat}, creating context for discussing changes. Do NOT mention specific items - let the user bring them up naturally."
        
        elif intent_id == "ai_ask_about_changing_tastes":
            subcat = session.subcategory or session.category
            return f"{memory_metadata}\n\nAI should ask about how the user's tastes in {subcat} have evolved over time, encouraging them to share what has changed."
        
        elif intent_id == "ai_acknowledge_new_preference":
            pref_type = session.preference_type or 'preference'
            return f"AI should acknowledge and respond positively to the user's {pref_type} for '{item_desc}'. Show enthusiasm and interest in their taste."
        
        elif intent_id == "ai_acknowledge_change":
            return f"AI should acknowledge the user's preference change regarding '{item_desc}' and show understanding of their evolving tastes."
        
        elif intent_id == "ai_acknowledge_removal":
            return f"AI should acknowledge what the user accomplished and continue the conversation naturally. Do NOT mention memory management, adding, removing, or deleting anything."
        
        elif intent_id == "ai_continue_conversation_after_memory_share":
            return f"AI should continue the conversation naturally after the user shared their memory. Do NOT mention memory management, adding, removing, or deleting anything. Just acknowledge what they shared and continue the conversation flow."
        
        # User Agent Instructions
        elif intent_id == "user_add_memory_item_after_transition":
            pref_type = session.preference_type or 'like'
            subcat = session.subcategory or session.category
            return f"{memory_metadata}\n\nUser should share that they {pref_type} '{item_desc}' in the {subcat} context. Express this preference naturally without explicitly mentioning categories or memory."
        
        elif intent_id == "user_update_memory_item_after_transition":
            if session.update_type == 'preference_update':
                old_pref = session.old_preference or 'previous preference'
                new_pref = session.preference_type or 'new preference'
                return f"{memory_metadata}\n\nUser should update their preference about '{item_desc}' from '{old_pref}' to '{new_pref}'. Explain the change naturally."
            elif session.update_type == 'value_update':
                old_item = session.old_item or 'previous item'
                return f"{memory_metadata}\n\nUser should update from '{old_item}' to '{item_desc}' while keeping their {session.preference_type} preference."
            else:
                return f"{memory_metadata}\n\nUser should update their preference about '{item_desc}' naturally, explaining what changed."
        
        elif intent_id == "user_delete_memory_item_after_transition":
            pref_type = session.preference_type or 'like'
            return f"{memory_metadata}\n\nUser should express that they no longer {pref_type} '{item_desc}'. Share this change naturally, explaining why their preference shifted."
        
        elif intent_id == "user_share_memory_item_with_preference":
            pref_type = session.preference_type or 'like'
            return f"{memory_metadata}\n\nUser should share that they {pref_type} '{item_desc}' naturally as part of the conversation flow."
        
        elif intent_id == "user_share_memory_item_with_preference_in_response":
            pref_type = session.preference_type or 'like'
            return f"{memory_metadata}\n\nUser should respond to AI's question by sharing that they {pref_type} '{item_desc}', making the preference clear."
        
        elif intent_id == "user_share_memory_item_without_preference":
            return f"{memory_metadata}\n\nUser should mention '{item_desc}' without explicitly stating their preference yet, keeping it conversational."
        
        elif intent_id == "user_share_preference_on_shared_memory":
            pref_type = session.preference_type or 'preference'
            return f"{memory_metadata}\n\nUser should share their {pref_type} about '{item_desc}' that they just mentioned, making their feelings clear."
        
        elif intent_id == "user_share_comparative_preference_memory":
            pref_type = session.preference_type or 'preference'
            subcat = session.subcategory or session.category
            return f"{memory_metadata}\n\nUser should compare different {subcat} options and express their {pref_type} for '{item_desc}' in the comparison."
        
        elif intent_id == "user_reflect_on_ai_shared_memory":
            return f"{memory_metadata}\n\nUser should reflect on AI's shared opinion and express their own perspective about '{item_desc}'."
        
        elif intent_id == "user_transition_and_add_memory":
            pref_type = session.preference_type or 'like'
            return f"{memory_metadata}\n\nUser should naturally transition to sharing that they {pref_type} '{item_desc}', bringing it up organically in conversation."
        
        elif intent_id == "user_transition_and_update_memory":
            # Handle both preference updates and value updates
            if session.operation == "update" and session.update_type == 'preference_update' and session.old_preference and session.preference_type:
                old_pref = session.old_preference
                new_pref = session.preference_type
                return f"{memory_metadata}\n\nUser should naturally transition to updating their preference about '{item_desc}'. They used to {old_pref} it but now they {new_pref} it. Explain this change naturally."
            elif session.operation == "update" and session.update_type == 'value_update' and session.old_item:
                old_item = session.old_item
                return f"{memory_metadata}\n\nUser should naturally transition to updating their preference about '{item_desc}'. They used to prefer '{old_item}' but now they prefer '{item_desc}'. Explain this change naturally."
            else:
                return f"{memory_metadata}\n\nUser should naturally transition to updating their preference about '{item_desc}', explaining what changed."
        
        elif intent_id == "user_transition_and_delete_memory":
            return f"{memory_metadata}\n\nUser should naturally transition to expressing that they no longer like '{item_desc}', explaining the change."
        
        elif intent_id == "user_answer_ai_curiosity_with_memory":
            pref_type = session.preference_type or 'preference'
            return f"{memory_metadata}\n\nUser should satisfy AI's curiosity by sharing their {pref_type} for '{item_desc}', being enthusiastic about their taste."
        
        elif intent_id == "user_answer_ai_why_changed":
            # Handle both preference updates and value updates
            if session.operation == "update" and session.update_type == 'preference_update' and session.old_preference and session.preference_type:
                old_pref = session.old_preference
                new_pref = session.preference_type
                return f"{memory_metadata}\n\nUser should explain why their preference about '{item_desc}' changed from {old_pref} to {new_pref}, sharing their reasoning naturally."
            elif session.operation == "update" and session.update_type == 'value_update' and session.old_item:
                old_item = session.old_item
                return f"{memory_metadata}\n\nUser should explain why their preference changed from '{old_item}' to '{item_desc}', sharing their reasoning naturally."
            else:
                return f"{memory_metadata}\n\nUser should explain why their preference about '{item_desc}' changed, sharing their reasoning naturally."
        
        elif intent_id == "user_answer_ai_why_preference":
            pref_type = session.preference_type or 'like'
            return f"{memory_metadata}\n\nUser should explain why they {pref_type} '{item_desc}', sharing what appeals to them about it."
        
        elif intent_id == "user_express_taste_evolution":
            subcat = session.subcategory or session.category
            # Handle both preference updates and value updates
            if session.operation == "update" and session.update_type == 'preference_update' and session.old_preference and session.preference_type:
                old_pref = session.old_preference
                new_pref = session.preference_type
                return f"{memory_metadata}\n\nUser should express how their taste in {subcat} has evolved regarding '{item_desc}'. They used to {old_pref} it but now they {new_pref} it. Lead naturally to discussing this change."
            elif session.operation == "update" and session.update_type == 'value_update' and session.old_item:
                old_item = session.old_item
                return f"{memory_metadata}\n\nUser should express how their taste in {subcat} has evolved regarding '{item_desc}'. They used to prefer '{old_item}' but now they prefer '{item_desc}'. Lead naturally to discussing this change."
            else:
                return f"{memory_metadata}\n\nUser should express how their taste in {subcat} has evolved, naturally leading to discussing '{item_desc}'."
        
        elif intent_id == "user_mention_change_with_memory":
            # Handle both preference updates and value updates
            if session.operation == "update" and session.update_type == 'preference_update' and session.old_preference and session.preference_type:
                old_pref = session.old_preference
                new_pref = session.preference_type
                return f"{memory_metadata}\n\nUser should mention a change regarding '{item_desc}' naturally in conversation. The user used to {old_pref} it but now they {new_pref} it. Set up for further discussion."
            elif session.operation == "update" and session.update_type == 'value_update' and session.old_item:
                old_item = session.old_item
                return f"{memory_metadata}\n\nUser should mention a change regarding '{item_desc}' naturally in conversation. The user used to prefer '{old_item}' but now they prefer '{item_desc}'. Set up for further discussion."
            else:
                return f"{memory_metadata}\n\nUser should mention a change regarding '{item_desc}' naturally in conversation, setting up for further discussion."
        
        elif intent_id == "user_mention_the_disinterest_in_memory":
            return f"{memory_metadata}\n\nUser should express their disinterest in '{item_desc}', explaining why they no longer find it appealing."
        
        # Fallback for unknown preference intents
        return f"{memory_metadata}\n\nExecute preference memory intent: {intent_id}"
    
    def _generate_activity_memory_instruction(self, intent_id: str, agent: str, 
                                            session: SessionMetadata, memory_metadata: str) -> str:
        """
        Generate instructions for activity memory intents
        """
        item_desc = session.item if session.item else str(session.operation_details.get('item', 'activity'))
        
        if intent_id == 'ai_smooth_topic_bridge_to_memory':
            if session.memory_type == "activity_memory":
                return f"{memory_metadata}\n\nAI should smoothly transition to asking about the user's {session.category} activities in a natural, open-ended way. Do NOT mention specific amounts, items, or details. Let them share what they've been doing in this area."
            else:
                return f"{memory_metadata}\n\nAI should smoothly transition to asking about the user's activities or experiences in an open way, encouraging them to share what they've been up to."
        
        if intent_id == 'ai_ask_about_activity_memory':
            if session.memory_type == "activity_memory":
                # Make AI questions operation-aware
                if session.operation == "add":
                    if session.category == "todo_list":
                        return f"{memory_metadata}\n\nAI should ask about tasks the user needs to do or is planning to work on. Do NOT mention specific items or details. Let them share what they need to accomplish."
                    elif session.category == "food_expenses":
                        return f"{memory_metadata}\n\nAI should ask about food expenses the user had recently. Do NOT mention specific amounts or items. Let them share what they spent on."
                    elif session.category == "step_tracker":
                        return f"{memory_metadata}\n\nAI should ask about steps the user took today or recently. Do NOT mention specific amounts or details. Let them share their step count."
                    elif session.category == "calendar_event":
                        return f"{memory_metadata}\n\nAI should ask about events the user scheduled recently. Do NOT mention specific items or details. Let them share what they planned."
                    else:
                        return f"{memory_metadata}\n\nAI should ask about the user's {session.category} activities in a natural, open-ended way. Do NOT mention specific amounts, items, or details. Let them share what they've been doing in this area."
                elif session.operation == "delete":
                    if session.category == "todo_list":
                        return f"{memory_metadata}\n\nAI should ask about tasks the user completed or accomplished recently. Do NOT mention specific items or details. Let them share what they finished."
                    else:
                        return f"{memory_metadata}\n\nAI should ask about the user's {session.category} activities in a natural, open-ended way. Do NOT mention specific amounts, items, or details. Let them share what they've been doing in this area."
                else:
                    return f"{memory_metadata}\n\nAI should ask about the user's {session.category} activities in a natural, open-ended way. Do NOT mention specific amounts, items, or details. Let them share what they've been doing in this area."
            else:
                return f"{memory_metadata}\n\nAI should ask about the user's activities or experiences in an open way, encouraging them to share what they've been up to."
        
        elif intent_id == 'user_add_activity_memory':
            if session.memory_type == "activity_memory":
                # Handle different activity types appropriately for add operations
                if session.category == "todo_list":
                    # For to-do list, user should talk about something they need to do naturally
                    return f"{memory_metadata}\n\nUser should mention something they need to do: '{item_desc}'. Express this naturally as part of the conversation - just mention what they need to accomplish, don't mention to-do lists or memory management."
                elif session.category == "food_expenses" and session.operation_details and 'item' in session.operation_details:
                    item_details = session.operation_details['item']
                    if isinstance(item_details, dict) and 'expense_type' in item_details:
                        expense_type = item_details['expense_type']
                        amount = item_details.get('amount', '')
                        return f"{memory_metadata}\n\nUser should mention their {session.category} activity about spending ${amount} on {expense_type}. Express this naturally as part of the conversation - mention the activity without explicitly stating you want to add it to memory."
                elif session.category == "step_tracker":
                    # For step tracking, user should talk about steps they took today/recently
                    return f"{memory_metadata}\n\nUser should mention their {session.category} activity about '{item_desc}' that they did today. Express this naturally as part of the conversation - mention the activity without explicitly stating you want to add it to memory."
                elif session.category == "calendar_event":
                    # For calendar events, user should talk about something they scheduled
                    # Calculate and include the event date
                    event_date_str = self._calculate_calendar_event_date(session)
                    if event_date_str:
                        return f"{memory_metadata}\n\nUser should mention their {session.category} activity about '{item_desc}' scheduled for {event_date_str}. Express this naturally as part of the conversation - mention both the event name and when it's scheduled (e.g., 'I have a {item_desc} on {event_date_str}' or 'I have a {item_desc} in X days'). Do NOT mention memory management."
                    else:
                        return f"{memory_metadata}\n\nUser should mention their {session.category} activity about '{item_desc}'. Express this naturally as part of the conversation - mention the activity without explicitly stating you want to add it to memory."
                else:
                    return f"{memory_metadata}\n\nUser should mention their {session.category} activity about '{item_desc}'. Express this naturally as part of the conversation - mention the activity without explicitly stating you want to add it to memory."
            else:
                return f"{memory_metadata}\n\nUser should mention their activity or experience about '{item_desc}' naturally."
        
        elif intent_id == 'user_casual_memory_add':
            if session.memory_type == "activity_memory":
                # Handle different activity types appropriately
                if session.category == "food_expenses" and session.operation_details and 'item' in session.operation_details:
                    item_details = session.operation_details['item']
                    if isinstance(item_details, dict) and 'expense_type' in item_details:
                        expense_type = item_details['expense_type']
                        amount = item_details.get('amount', '')
                        return f"{memory_metadata}\n\nUser should casually mention their {session.category} activity about spending ${amount} on {expense_type}. Express this naturally as part of the conversation - mention the activity without explicitly stating you want to add it to memory."
                elif session.category == "todo_list":
                    # For to-do list, user should talk about something they need to do naturally
                    return f"{memory_metadata}\n\nUser should casually mention something they need to do: '{item_desc}'. Express this naturally as part of the conversation - just mention what they need to accomplish, don't mention to-do lists or memory management."
                elif session.category == "step_tracker":
                    # For step tracking, user should talk about steps they took today/recently
                    return f"{memory_metadata}\n\nUser should casually mention their {session.category} activity about '{item_desc}' that they did today. Express this naturally as part of the conversation - mention the activity without explicitly stating you want to add it to memory."
                elif session.category == "calendar_event":
                    # For calendar events, user should talk about something they scheduled
                    # Calculate and include the event date
                    event_date_str = self._calculate_calendar_event_date(session)
                    if event_date_str:
                        return f"{memory_metadata}\n\nUser should casually mention their {session.category} activity about '{item_desc}' scheduled for {event_date_str}. Express this naturally as part of the conversation - mention both the event name and when it's scheduled (e.g., 'I have a {item_desc} on {event_date_str}' or 'I have a {item_desc} in X days'). Do NOT mention memory management."
                    else:
                        return f"{memory_metadata}\n\nUser should casually mention their {session.category} activity about '{item_desc}'. Express this naturally as part of the conversation - mention the activity without explicitly stating you want to add it to memory."
                else:
                    return f"{memory_metadata}\n\nUser should casually mention their {session.category} activity about '{item_desc}'. Express this naturally as part of the conversation - mention the activity without explicitly stating you want to add it to memory."
            else:
                return f"{memory_metadata}\n\nUser should casually mention their activity or experience about '{item_desc}' naturally."
        
        elif intent_id == 'user_delete_activity_memory':
            if session.memory_type == "activity_memory":
                # Handle different activity types appropriately for delete operations
                if session.category == "todo_list":
                    # For to-do list, user should talk about completing something naturally
                    return f"{memory_metadata}\n\nUser should casually mention that they completed '{item_desc}'. Express this naturally as part of the conversation - just mention what they accomplished, don't mention to-do lists, memory management, or time references."
                elif session.category == "calendar_event":
                    # For calendar events, user should talk about missing or not being able to attend
                    return f"{memory_metadata}\n\nUser should casually mention that they missed or couldn't attend '{item_desc}'. Express this naturally as part of the conversation - just mention the situation, don't mention calendar management, memory deletion, or specific dates."
                else:
                    return f"{memory_metadata}\n\nUser should express that they want to remove or no longer track their {session.category} activity about '{item_desc}'. Share this change naturally."
            else:
                return f"{memory_metadata}\n\nUser should express removing their activity about '{item_desc}'."
        
        elif intent_id == 'user_update_activity_memory':
            if session.memory_type == "activity_memory":
                # Handle different activity types appropriately for update operations
                if session.category == "todo_list":
                    # For to-do list, user should talk about modifying a task naturally
                    return f"{memory_metadata}\n\nUser should casually mention that they need to update '{item_desc}'. Express this naturally as part of the conversation - just mention what they need to change, don't mention to-do lists or memory management."
                elif session.category == "calendar_event":
                    # For calendar events, user should talk about rescheduling something
                    return f"{memory_metadata}\n\nUser should casually mention their {session.category} activity about '{item_desc}'. Express this naturally as part of the conversation - mention the activity without explicitly stating you want to update it in memory."
                else:
                    return f"{memory_metadata}\n\nUser should share an update to their {session.category} activity about '{item_desc}'. Express the change or update naturally."
            else:
                return f"{memory_metadata}\n\nUser should share an update about their activity '{item_desc}'."
        
        elif intent_id == 'ai_continue_conversation_after_memory_share':
            if session.memory_type == "activity_memory":
                return f"{memory_metadata}\n\nAI should continue the conversation naturally after the user shared their memory. Do NOT mention specific amounts, items, or details. Do NOT mention memory management, adding, removing, or deleting anything. Just acknowledge what they shared and continue the conversation flow."
            else:
                return f"{memory_metadata}\n\nAI should continue the conversation naturally after the user shared their memory. Do NOT mention memory management, adding, removing, or deleting anything. Just acknowledge what they shared and continue the conversation flow."
        
        # Fallback for unknown activity intents
        return f"{memory_metadata}\n\nExecute activity memory intent: {intent_id}"
    
    def _generate_goal_memory_instruction(self, intent_id: str, agent: str, 
                                        session: SessionMetadata, memory_metadata: str) -> str:
        """
        Generate instructions for goal memory intents
        """
        goal_value = session.item if session.item else str(session.operation_details.get('item', 'goal'))
        subcategory = session.subcategory if session.subcategory else session.operation_details.get('subcategory', '')
        
        # Build natural goal description
        if session.category == "food_expenses":
            goal_desc = f"spending under ${goal_value} per week on {subcategory}"
        elif session.category == "step_tracker":
            goal_desc = f"walking {goal_value} steps per day"
        else:
            goal_desc = f"{goal_value} for {subcategory}"
        
        if intent_id == 'user_directly_share_goal':
            return f"{memory_metadata}\n\nUser should share their goal about {goal_desc} directly with the AI, expressing what they want to achieve. Make sure to specify it's PER WEEK (for food expenses) or PER DAY (for steps), NOT per month."
        
        elif intent_id == 'ai_respond_to_goal_positivly':
            return f"{memory_metadata}\n\nAI should respond positively to the user's goal about {goal_desc}, showing support and encouragement. Confirm the WEEKLY (food) or DAILY (steps) timeframe."
        
        elif intent_id == 'ai_ask_about_user_goal':
            return f"{memory_metadata}\n\nAI should ask about the user's goals in {session.category}/{subcategory} in an open-ended way, letting them share what they're working towards."
        
        elif intent_id == 'user_share_goal_in_response_to_ai_question':
            return f"{memory_metadata}\n\nUser should share their goal about {goal_desc} in response to AI's question, explaining their aspirations. Make sure to specify it's PER WEEK (for food expenses) or PER DAY (for steps), NOT per month."
        
        # Fallback for unknown goal intents
        return f"{memory_metadata}\n\nExecute goal memory intent: {intent_id}"
    
    def _generate_content_memory_instruction(self, intent_id: str, agent: str, 
                                           session: SessionMetadata, memory_metadata: str, current_field: str = None) -> str:
        """
        Generate instructions for content memory intents
        Content memory handles multiple pieces of information in a single session (field-by-field approach)
        """
        item_desc = session.item if session.item else str(session.operation_details.get('item', 'item'))
        
        # Content type mapping for natural language
        content_type_map = {
            'project_proposal': 'project proposal',
            'email_writeup': 'email',
            'social_media_post': 'social media post', 
            'meeting_notes': 'meeting notes'
        }
        content_type = content_type_map.get(session.category, session.category.replace('_', ' '))
        
        # === CONVERSATION INITIATION INTENTS ===
        if intent_id == 'user_initiate_content_conversation':
            return f"{memory_metadata}\n\nUser should ask for help with writing up a {content_type}. Be specific that they need assistance organizing and structuring their {content_type} information."
        
        elif intent_id == 'ai_agree_to_help_with_content_conversation':
            return f"{memory_metadata}\n\nAI should agree to help with the {content_type} writeup. This should be a simple acknowledgment - do NOT ask any questions yet. Questions will be asked in a later turn."
        
        # === FIELD-SPECIFIC INTENTS ===
        elif intent_id == 'ai_ask_about_content_fields':
            if current_field:
                # Create natural questions based on field name and content type
                field_questions = {
                    # Project proposal fields
                    'project_title': "What's the title of this project proposal?",
                    'project_description': "Could you describe what this project is about?",
                    'budget': "What's the proposed budget for this project?", 
                    'timeline': "What's the timeline or duration for this project?",
                    'project_lead': "Who will be leading this project?",
                    'stakeholders': "Who are the key stakeholders involved?",
                    'objectives': "What are the main objectives or goals?",
                    'deliverables': "What will be the key deliverables?",
                    'resources_needed': "What resources will be needed?",
                    'success_metrics': "How will success be measured?",
                    'risk_assessment': "What are the potential risks?",
                    
                    # Email fields
                    'subject': "What should the subject line be?",
                    'recipient': "Who is this email for?",
                    'purpose': "What's the main purpose of this email?",
                    'tone': "What tone should we use for this email?",
                    'key_points': "What are the key points to cover?",
                    'call_to_action': "What action do you want the recipient to take?",
                    
                    # Social media post fields
                    'platform': "Which social media platform is this for?",
                    'caption': "What should the caption say?",
                    'hashtags': "What hashtags should we include?",
                    'target_audience': "Who is the target audience?",
                    'post_type': "What type of post is this?",
                    
                    # Meeting notes fields
                    'meeting_title': "What was the title or purpose of the meeting?",
                    'attendees': "Who attended the meeting?",
                    'date': "When did this meeting take place?",
                    'agenda': "What was on the agenda?",
                    'key_decisions': "What key decisions were made?",
                    'action_items': "What are the action items from this meeting?",
                    'next_steps': "What are the next steps?"
                }
                
                natural_question = field_questions.get(current_field, f"What about the {current_field.replace('_', ' ')}?")
                return f"{memory_metadata}\n\nAI should ask about the '{current_field}' field. Ask naturally: '{natural_question}'"
            else:
                return f"{memory_metadata}\n\nAI should ask about a specific field of the {content_type}."
        
        elif intent_id == 'user_share_content_fields':
            if current_field:
                # Get the actual field value from content data
                content_data = session.operation_details.get('content_data', {})
                field_value = content_data.get(current_field, 'N/A')
                
                # Format the field value appropriately for natural conversation
                if isinstance(field_value, list):
                    # For instruction generation, show ALL items - user needs complete metadata
                    field_display = ', '.join(str(item) for item in field_value)
                elif isinstance(field_value, dict):
                    field_display = str(field_value)
                else:
                    field_display = str(field_value)
                
                content_context = content_type_map.get(session.category, 'content')
                return f"{memory_metadata}\n\nUser should share the '{current_field}' information naturally. The actual value is: {field_display}. Present this information conversationally as if discussing a real {content_context} - include all the specific details mentioned in the actual value, but do not mention any generic identifiers."
            else:
                return f"{memory_metadata}\n\nUser should share a specific field of their {content_type} naturally."
        
        elif intent_id == 'ai_confirm_content_writeup_complete':
            # This intent is used for both ADD/UPDATE and DELETE operations
            # For DELETE operations, clarify it's about updating by removing elements
            if session.operation == 'delete':
                return f"{memory_metadata}\n\nAI should confirm that all the requested removals/modifications have been completed. IMPORTANT: Frame this as updating the {content_type} by removing specific elements, NOT deleting the entire {content_type}. Be brief - use 'it' or 'the {content_type}' instead of repeating the full title. Offer summary or next steps if appropriate."
            else:
                return f"{memory_metadata}\n\nAI should confirm that all the necessary information has been gathered or updated. Be brief - use 'it' or 'the {content_type}' instead of repeating the full title. Offer summary or next steps if appropriate."
        
        # === CONTENT UPDATE INTENTS ===
        elif intent_id == 'user_initiate_content_update_conversation':
            # Use content-specific identifier instead of generic item_desc
            content_identifier = self._get_content_identifier(session)
            return f"{memory_metadata}\n\nUser should mention that they need to update their {content_type} titled '{content_identifier}'. Be explicit and clear about which {content_type} they are updating - mention the title/name early in the message. Then briefly express that they want to change some details."
        
        elif intent_id == 'ai_agree_to_help_with_content_update':
            return f"{memory_metadata}\n\nAI should agree to help with the update. If the user already mentioned specific changes, acknowledge them briefly. Otherwise, ask what information needs to be changed. Use 'it' or 'the {content_type}' instead of repeating the full title."
        
        elif intent_id == 'user_share_content_update':
            # Get the current update from memory_updates
            memory_updates = session.operation_details.get('memory_updates', [])
            if hasattr(session, '_current_update_index'):
                current_update = memory_updates[session._current_update_index] if session._current_update_index < len(memory_updates) else {}
            else:
                current_update = memory_updates[0] if memory_updates else {}
            
            field = current_update.get('field', 'field')
            added_item = current_update.get('added_item', 'new value')
            action = current_update.get('action', 'updated')
            
            if action == 'budget_revised':
                return f"{memory_metadata}\n\nUser should share the budget update naturally. Mention the revised budget amount without repeating the full project/content title (it was already established)."
            else:
                return f"{memory_metadata}\n\nUser should share that they want to update the '{field}' by adding: '{added_item}'. Present this update naturally - no need to repeat the full project/content title since it was already established. Just mention the field and the new value."
        
        elif intent_id == 'ai_acknowledge_content_update':
            return f"{memory_metadata}\n\nAI should acknowledge the specific update the user just mentioned - reference the field name and the value they're adding (e.g., 'Got it, I've added [stakeholder name] to stakeholders'). Do NOT repeat the full project title every time - it was already established. Do NOT ask for information already provided. Do NOT mention memory management. Just acknowledge briefly and ask if there's anything else to update."
        
        elif intent_id == 'user_confirm_content_updates_complete':
            return f"{memory_metadata}\n\nUser should confirm that they have shared all the updates they wanted to make. Keep it brief and natural - no need to repeat the project/content title."
        
        # === CONTENT DELETE INTENTS ===
        elif intent_id == 'user_initiate_content_delete_conversation':
            # Use content-specific identifier instead of generic item_desc
            content_identifier = self._get_content_identifier(session)
            return f"{memory_metadata}\n\nUser should mention that they need to remove or modify some information from their {content_type} titled '{content_identifier}'. IMPORTANT: This is NOT deleting the entire {content_type} - it's about removing specific fields/elements. Be explicit and clear about which {content_type} they are updating - mention the title/name early in the message. Do NOT mention specific fields, values, or what needs to be removed yet - just indicate that some information needs to be removed/modified. The specific details will be discussed in later turns."
        
        elif intent_id == 'ai_agree_to_help_with_content_delete':
            return f"{memory_metadata}\n\nAI should agree to help with removing or modifying specific elements from the {content_type}. IMPORTANT: This is NOT deleting the entire {content_type} - it's updating it by removing specific fields/elements. Use 'it' or 'the {content_type}' instead of repeating the full title. Ask what specific information should be removed or modified."
        
        elif intent_id == 'user_share_content_delete':
            # Get the current deletion from memory_deletes
            memory_deletes = session.operation_details.get('memory_deletes', [])
            if hasattr(session, '_current_delete_index'):
                current_delete = memory_deletes[session._current_delete_index] if session._current_delete_index < len(memory_deletes) else {}
            else:
                current_delete = memory_deletes[0] if memory_deletes else {}
            
            field = current_delete.get('field', 'field')
            removed_item = current_delete.get('removed_item', 'item')
            action = current_delete.get('action', 'removed')
            
            if action == 'budget_reverted':
                reverted_from = current_delete.get('reverted_from', '')
                reverted_to = current_delete.get('reverted_to', '')
                return f"{memory_metadata}\n\nUser should share that they want to revert the budget from {reverted_from} back to {reverted_to}. IMPORTANT: This is updating the {content_type} by changing a specific field value, NOT deleting the entire {content_type}. Mention the specific budget change naturally without repeating the full project/content title."
            else:
                return f"{memory_metadata}\n\nUser should share that they want to remove '{removed_item}' from the '{field}' field. IMPORTANT: This is updating the {content_type} by removing a specific element, NOT deleting the entire {content_type}. Present this removal naturally without repeating the full project/content title (already established). Just mention what specific element needs to be removed."
        
        elif intent_id == 'ai_acknowledge_content_delete':
            return f"{memory_metadata}\n\nAI should acknowledge the specific removal/modification the user just mentioned - reference the field name and what was removed or changed (e.g., 'Got it, I've removed [item] from [field]' or 'I've updated the budget back to [value]'). IMPORTANT: Frame this as updating the {content_type} by removing specific elements, NOT deleting the entire {content_type}. Do NOT repeat the full project title. Do NOT ask for information already provided. Do NOT mention memory management. Just acknowledge briefly and ask if there's anything else to remove or modify."
        
        elif intent_id == 'user_confirm_content_deletes_complete':
            return f"{memory_metadata}\n\nUser should confirm that they have shared all the removals/modifications they wanted to make to the {content_type}. Keep it brief - no need to repeat the project/content title. Remember: This is about removing specific elements, not deleting the entire {content_type}."
        
        # Fallback for unknown content intents
        return f"{memory_metadata}\n\nExecute content memory intent: {intent_id}"
    
    def _get_content_identifier(self, session: SessionMetadata) -> str:
        """Get appropriate content identifier based on content type"""
        if session.memory_type != "content_memory" or not session.operation_details:
            return session.item or "content"
        
        content_data = session.operation_details.get('content_data', {})
        
        # Use content-specific identifier based on category
        if session.category == "meeting_notes":
            return content_data.get('meeting_title', session.item or 'meeting')
        elif session.category == "email_writeup":
            return content_data.get('email_purpose', content_data.get('subject', session.item or 'email'))
        elif session.category == "project_proposal":
            return content_data.get('project_title', session.item or 'project')
        elif session.category == "social_media_post":
            # Social media posts use content_type as identifier (e.g., "Technical Insight Post")
            # Combine with platform for clarity (e.g., "LinkedIn Technical Insight Post")
            content_type = content_data.get('content_type', '')
            platform = content_data.get('platform', '')
            if content_type and platform:
                return f"{platform} {content_type}"
            elif content_type:
                return content_type
            elif platform:
                return f"{platform} post"
            else:
                return 'social media post'
        else:
            # Fallback to first available title field or item
            for field in ['title', 'name', 'subject', 'project_title', 'meeting_title', 'content_type']:
                if field in content_data and content_data[field]:
                    return content_data[field]
            return session.item or 'content'
    
    def _calculate_calendar_event_date(self, session: SessionMetadata) -> Optional[str]:
        """Calculate and format calendar event date for natural language instruction
        
        Args:
            session: Session metadata containing operation_details with date information
            
        Returns:
            Formatted date string (e.g., "June 18th", "in 17 days", "tomorrow") or None
        """
        if not session.operation_details or 'item' not in session.operation_details:
            return None
        
        item_data = session.operation_details['item']
        if not isinstance(item_data, dict):
            return None
        
        date_offset = item_data.get('date')
        created_at = item_data.get('created_at')
        
        if not date_offset or not created_at:
            return None
        
        try:
            # Parse the creation date
            created_datetime = datetime.strptime(created_at, "%Y-%m-%d")
            
            # Parse the offset (e.g., "+17 days", "+1 week")
            if date_offset.startswith("+"):
                offset_str = date_offset[1:].strip()
                
                if "day" in offset_str:
                    days = int(offset_str.split()[0])
                    event_datetime = created_datetime + timedelta(days=days)
                    
                    # Format naturally based on days
                    if days == 0:
                        return "today"
                    elif days == 1:
                        return "tomorrow"
                    elif days <= 7:
                        return f"in {days} days"
                    else:
                        # Format as actual date (e.g., "June 18th")
                        return event_datetime.strftime("%B %d")
                elif "week" in offset_str:
                    weeks = int(offset_str.split()[0])
                    event_datetime = created_datetime + timedelta(weeks=weeks)
                    if weeks == 1:
                        return "next week"
                    else:
                        return event_datetime.strftime("%B %d")
                else:
                    # Unknown format, return None
                    return None
            else:
                # If it's not a relative date, return as-is
                return date_offset
                
        except (ValueError, AttributeError, IndexError):
            return None
