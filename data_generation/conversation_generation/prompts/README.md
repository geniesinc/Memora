# System Prompts & Rules Organization

This directory contains all system prompts and rules used by the PromptManager, organized by type and purpose for easy maintenance and editing.

## Directory Structure

```
prompts/
├── ai_agent/                    # AI Assistant prompts
│   ├── general.md              # General conversation (no personal questions)
│   ├── preference_memory.md    # Preference memory (capture likes/dislikes)
│   ├── activity_memory.md      # Activity memory (track experiences)
│   ├── content_memory.md       # Content memory (organize writeups)
│   └── goal_memory.md          # Goal memory (capture aspirations)
├── user_agent/                 # User simulation prompts
│   ├── general.md              # General conversation (no personal info)
│   ├── preference_memory.md    # Share preferences when instructed
│   ├── activity_memory.md      # Report activities and experiences
│   ├── content_memory.md       # Provide content information
│   └── goal_memory.md          # Share goals and aspirations
└── README.md                   # This file
```

## Usage

The PromptManager automatically loads the appropriate content based on:

### System Prompts
- **Agent Type**: `ai_agent` or `user_agent`
- **Memory Type**: `general`, `preference_memory`, `activity_memory`, `content_memory`, `goal_memory`

### Memory Behavior
- **Instruction Generation**: Memory sharing behavior is handled by the instruction generator, which provides context-specific instructions for each conversation turn
- **Variable Substitution**: Some templates support dynamic content via `.format()`

## Benefits

1. **Easy Editing**: Edit prompts and rules directly in Markdown files
2. **Version Control**: Track changes to prompts and rules over time
3. **Modularity**: Each prompt/rule has a specific purpose and context
4. **Maintainability**: No more long embedded strings in code
5. **Caching**: Content is cached for performance
6. **Flexibility**: Support for template variables in rules

## Adding New Content

### New Memory Types
1. Create `ai_agent/new_memory_type.md`
2. Create `user_agent/new_memory_type.md`
3. The PromptManager will automatically load them when `memory_type="new_memory_type"`

### New Memory Rules
1. Create `rules/new_rule_type.md`
2. Update the logic in `get_memory_sharing_rules()` to use the new rule
3. Rules support variable substitution using Python's `.format()` method

## Fallback Behavior

If a prompt file is missing, the system falls back to simple default prompts to ensure graceful degradation.
