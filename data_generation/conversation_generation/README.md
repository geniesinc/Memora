# Conversation Generation System

A sophisticated memory-grounded conversation generation system that transforms structured session data into natural, multi-turn dialogues using LLM orchestration and intent-based flows.

## 🚀 Quick Start

### Prerequisites
```bash
# Activate the memora conda environment
conda activate memora

# Ensure you have your OpenRouter API key set
export OPEN_ROUTER_API_KEY="your_api_key_here"
```

### Basic Usage

Generate conversations with automatic individual file saving and error handling:

```bash
# Generate 3 conversations (default)
python memory_grounded_generator.py

# Generate 50 conversations with individual files
python memory_grounded_generator.py --num-sessions 50

# Generate 100 preference_memory conversations only
python memory_grounded_generator.py --memory-type preference_memory --num-sessions 100

# Test with no_memory conversations
python memory_grounded_generator.py --test-no-memory --num-sessions 10

# Generate specific session with individual file
python memory_grounded_generator.py --session-id 1234
```

## 📁 Output Structure

### Timestamped Folder Organization
Each batch run creates a timestamped folder to prevent data loss and enable easy organization:

```
conversations_YYYYMMDD_HHMMSS_[memory_type_filter]/
├── session_0001_activity_memory.json
├── session_0002_content_memory.json
├── session_0003_preference_memory.json
├── session_0004_preference_memory_FAILED.json  # Failed sessions preserved
└── generation_summary.json                     # Batch summary
```

### Individual Conversation Files
Each conversation is saved immediately upon generation as:
- **Successful**: `session_NNNN_[memory_type].json`
- **Failed**: `session_NNNN_[memory_type]_FAILED.json`

## 🎯 Command-Line Options

### Core Parameters
```bash
--num-sessions N          # Number of sessions to process (default: 3)
--session-id ID          # Generate conversation for specific session ID
--session-file PATH      # Path to sessions file (default: ../output/new_sessions.json)
```

### Memory Type Filtering
```bash
--memory-type TYPE       # Filter sessions by memory type
--test-no-memory        # Quick test with no_memory sessions only
```

**Available memory types:**
- `no_memory` - Pure general conversations (no memory operations)
- `preference_memory` - User preference conversations (add/update/delete preferences)
- `activity_memory` - User activity tracking conversations
- `goal_memory` - Goal setting and tracking conversations
- `content_memory` - Content creation assistance conversations

### Flow Control (Advanced)
```bash
--opening-flow ID        # Specific opening flow ID to use
--exploration-flow ID    # Specific exploration flow ID to use
--memory-flow ID         # Specific memory flow ID to use
--closing-flow ID        # Specific closing flow ID to use
```

## 📊 Memory Type Examples

### No-Memory Conversations
Pure general conversations without memory operations - perfect for baseline data:
```bash
python memory_grounded_generator.py --memory-type no_memory --num-sessions 20
```

### Preference Memory
Conversations about user likes/dislikes, taste evolution:
```bash
python memory_grounded_generator.py --memory-type preference_memory --num-sessions 30
```

### Content Memory
Field-by-field content creation assistance (project proposals, emails, etc.):
```bash
python memory_grounded_generator.py --memory-type content_memory --num-sessions 15
```

### Activity Memory
User activity tracking and task management:
```bash
python memory_grounded_generator.py --memory-type activity_memory --num-sessions 25
```

### Goal Memory
Goal setting, tracking, and achievement conversations:
```bash
python memory_grounded_generator.py --memory-type goal_memory --num-sessions 20
```

## 🛡️ Error Handling & Reliability

### Automatic Retry Logic
- **3 automatic retries** for failed LLM calls
- **Exponential backoff** (1s, 2s, 4s delays)
- **Individual session isolation** - one failure doesn't break the batch

### Data Loss Prevention
- **Individual file saving** - each conversation saved immediately
- **Failed session preservation** - errors captured with full context
- **Timestamped folders** - no conflicts between batch runs
- **Progress tracking** - clear logging of success/failure rates

### Resume Capability
Since each conversation is saved individually, you can:
1. Identify failed sessions from `*_FAILED.json` files
2. Retry specific sessions using `--session-id`
3. Continue from where previous batch left off

## 📈 Performance & Scale

### Recommended Batch Sizes
```bash
# Small batch testing
python memory_grounded_generator.py --num-sessions 10

# Medium batch processing
python memory_grounded_generator.py --num-sessions 50

# Large batch processing
python memory_grounded_generator.py --num-sessions 100

# Production batches
python memory_grounded_generator.py --num-sessions 500
```

### Model Configuration
The system uses **Google Gemini 2.5 Flash** by default for optimal multi-turn conversation generation. Model can be configured in the code if needed.

## 🔍 Output Analysis

### Individual Files
Each conversation file contains:
- Complete conversation turns with speaker labels
- Intent tracking and flow metadata
- Session context and memory operations
- Detailed prompt and instruction data

### Summary Files
Batch summary files include:
- Generation statistics and success rates
- Memory type distribution
- Error analysis and failed session details
- Complete conversation collection for analysis

## 🎭 Conversation Structure

### Flow Phases
1. **Opening Phase** (2-3 turns) - Greetings and initial connection
2. **Exploration Phase** (7-15 turns) - Topic exploration and general discussion
3. **Memory Phase** (2-6 turns) - Memory-grounded interactions (skipped for no_memory)
4. **Closing Phase** (1-3 turns) - Natural conversation endings

### Memory-Grounded Features
- **Natural AI questions** - AI asks open-ended questions about categories, not specific items
- **User-driven sharing** - Users naturally mention specific preferences/activities/goals
- **Contextual instructions** - Detailed guidance for each conversation turn
- **Persona consistency** - Aligned with software engineer persona from session data

## 🔧 Troubleshooting

### Common Issues

**LLM API Errors:**
- Check `OPEN_ROUTER_API_KEY` environment variable
- Verify network connectivity
- Review rate limits (system has automatic retry logic)

**Session Data Issues:**
- Ensure sessions file exists at specified path
- Check session data format and structure
- Verify memory type filter values

**Output Folder Permissions:**
- Ensure write permissions in conversation_generation directory
- Check disk space for large batch runs

### Debug Mode
For detailed debugging, check the console output which includes:
- Flow selection details
- Intent generation progress
- LLM call status and retries
- File saving confirmation

## 📚 System Architecture

The conversation generation system consists of:

- **SessionProcessor** - Parses and structures session data
- **FlowManager** - Manages conversation flows and intent selection
- **PromptManager** - LLM orchestration with sophisticated prompting
- **InstructionGenerator** - Contextual instruction generation
- **MemoryGroundedGenerator** - Main pipeline orchestrator

For detailed system architecture, see the main project README and system flowchart.

---

**Generated conversations are ready for training, evaluation, and analysis workflows!** 🚀
