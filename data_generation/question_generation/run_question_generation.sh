#!/bin/bash

# Run unified question generation on every session folder under an output directory.
#
# Usage:
#   ./run_question_generation.sh [timeline_type] [output_dir]
#
# Arguments:
#   timeline_type  weekly (default) | monthly | quarterly
#   output_dir     Directory containing sessions_* folders (default: output)
#
# The script auto-discovers all sub-directories matching `sessions_*` that
# contain a `memory_states_by_session.json` file and runs the unified
# question generator on each.

set -e

# Locate paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Args
TIMELINE="${1:-weekly}"
OUTPUT_DIR="${2:-output}"

# Validate timeline
if [[ "$TIMELINE" != "weekly" && "$TIMELINE" != "monthly" && "$TIMELINE" != "quarterly" ]]; then
    echo "Invalid timeline type: $TIMELINE"
    echo "  Valid options: weekly, monthly, quarterly"
    exit 1
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Output directory not found: $OUTPUT_DIR"
    exit 1
fi

echo "============================================"
echo "Unified Question Generation"
echo "============================================"
echo "Timeline:   $TIMELINE"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Auto-discover session folders
mapfile -t SESSION_FOLDERS < <(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "sessions_*" | sort)

if [[ ${#SESSION_FOLDERS[@]} -eq 0 ]]; then
    echo "No sessions_* folders found in $OUTPUT_DIR"
    exit 1
fi

TOTAL=${#SESSION_FOLDERS[@]}
CURRENT=0
SUCCESS=0
FAILED=0

echo "Found $TOTAL session folder(s) to process"
echo ""

for SESSION_PATH in "${SESSION_FOLDERS[@]}"; do
    CURRENT=$((CURRENT + 1))
    FOLDER=$(basename "$SESSION_PATH")

    echo "--------------------------------------------"
    echo "[$CURRENT/$TOTAL] $FOLDER"
    echo "--------------------------------------------"

    MEMORY_FILE="$SESSION_PATH/memory_states_by_session.json"
    if [[ ! -f "$MEMORY_FILE" ]]; then
        echo "Skipping: memory_states_by_session.json not found"
        FAILED=$((FAILED + 1))
        continue
    fi

    if python question_generation/unified_question_generator.py \
        --session_directory "$SESSION_PATH" \
        --timeline "$TIMELINE"; then
        echo "Success: $FOLDER"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "Failed: $FOLDER"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "============================================"
echo "Summary"
echo "============================================"
echo "Total:      $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed:     $FAILED"

[[ $FAILED -eq 0 ]] && exit 0 || exit 1
