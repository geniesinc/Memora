#!/bin/bash
# =============================================================================
# Generate Conversations for All Personas
# =============================================================================
# This script generates conversations for all personas in the output folder
# that don't already have conversations generated.
# =============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${SCRIPT_DIR}/../output"

# Configuration
ENABLE_EVALUATION=true
MAX_EVALUATION_ITERATIONS=5

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================================================="
echo "Conversation Generation for All Personas"
echo "============================================================================="
echo "Output directory: $OUTPUT_BASE"
echo "Evaluation: $ENABLE_EVALUATION"
echo "Max evaluation iterations: $MAX_EVALUATION_ITERATIONS"
echo "============================================================================="
echo ""

# Find all persona directories
PERSONA_DIRS=($(find "$OUTPUT_BASE" -maxdepth 1 -type d -name "sessions_*" | sort))

if [ ${#PERSONA_DIRS[@]} -eq 0 ]; then
    echo "❌ No persona directories found in $OUTPUT_BASE"
    exit 1
fi

echo "Found ${#PERSONA_DIRS[@]} persona directory(ies)"
echo ""

# Track progress
TOTAL=${#PERSONA_DIRS[@]}
CURRENT=0
SUCCESSFUL=0
SKIPPED=0
FAILED=0

# Change to script directory
cd "$SCRIPT_DIR"

# Process each persona
for PERSONA_DIR in "${PERSONA_DIRS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Extract persona name
    PERSONA_NAME=$(basename "$PERSONA_DIR" | sed 's/sessions_[0-9]*_[0-9]*_//')
    SESSIONS_FILE="${PERSONA_DIR}/new_sessions.json"
    CONVERSATIONS_DIR="${PERSONA_DIR}/conversations"
    
    echo "============================================================================="
    echo "[$CURRENT/$TOTAL] Processing: $PERSONA_NAME"
    echo "============================================================================="
    echo "Directory: $(basename "$PERSONA_DIR")"
    
    # Check if sessions file exists
    if [ ! -f "$SESSIONS_FILE" ]; then
        echo -e "${YELLOW}⚠️  Skipping: new_sessions.json not found${NC}"
        SKIPPED=$((SKIPPED + 1))
        echo ""
        continue
    fi
    
    # Check if conversations already exist
    if [ -d "$CONVERSATIONS_DIR" ] && [ -n "$(ls -A "$CONVERSATIONS_DIR/successful" 2>/dev/null)" ]; then
        CONVERSATION_COUNT=$(find "$CONVERSATIONS_DIR/successful" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
        echo -e "${BLUE}ℹ️  Conversations already exist ($CONVERSATION_COUNT files)${NC}"
        echo -e "${BLUE}   Skipping generation for this persona${NC}"
        SKIPPED=$((SKIPPED + 1))
        echo ""
        continue
    fi
    
    echo "Generating conversations..."
    echo "Sessions file: $SESSIONS_FILE"
    echo ""
    
    # Build command
    CMD="python memory_grounded_generator.py"
    CMD="$CMD --session-file \"$SESSIONS_FILE\""
    CMD="$CMD --generate-all"
    
    if [ "$ENABLE_EVALUATION" = "true" ]; then
        CMD="$CMD --enable-evaluation"
        CMD="$CMD --max-evaluation-iterations $MAX_EVALUATION_ITERATIONS"
    fi
    
    # Run generation
    START_TIME=$(date +%s)
    
    if eval $CMD; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo ""
        echo -e "${GREEN}✅ SUCCESS: Completed in ${DURATION}s${NC}"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo ""
        echo -e "${YELLOW}❌ FAILED: Error after ${DURATION}s${NC}"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
    echo ""
done

# Summary
echo "============================================================================="
echo "SUMMARY"
echo "============================================================================="
echo "Total personas: $TOTAL"
echo -e "${GREEN}✅ Successful: $SUCCESSFUL${NC}"
echo -e "${BLUE}⏭️  Skipped (already generated): $SKIPPED${NC}"
echo -e "${YELLOW}❌ Failed: $FAILED${NC}"
echo "============================================================================="