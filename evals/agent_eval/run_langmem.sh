#!/usr/bin/env bash
# =============================================================================
# Track 2 — langmem evaluation runner
# =============================================================================
# For each (period, persona) under data/, runs the two-step pipeline:
#   1. conversation_to_memory.py  — store all sessions in Mem0 under user_id={persona}_{period}
#   2. memory_to_answer.py        — retrieve memories and answer evaluation questions
#
# Mem0 is a cloud system; we wait a bit between steps so the backend can
# finish processing the uploaded conversations before we start querying them.
#
# Usage:
#   ./run_mem_0.sh                 # full pipeline (store + wait + answer)
#   ./run_mem_0.sh --skip-store    # answer only (memories already stored)
#   ./run_mem_0.sh --skip-answer   # store only
# =============================================================================

set -u

SYSTEM="langmem"
WAIT_TIME=0

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RELEASE_ROOT="$(cd "$EVALS_DIR/.." && pwd)"
DATA_DIR="${MEMORA_DATA_DIR:-$RELEASE_ROOT/data}"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# --- What to evaluate --------------------------------------------------------
PERIODS=(
    "weekly"
    # "monthly"
    # "quarterly"
)

PERSONAS=(
    "academic_researcher"
    # "business_executive"
    # "content_writer"
    # "creative_designer"
    # "financial_analyst"
    # "management_consultant"
    # "marketing_manager"
    # "sales_manager"
    # "software_engineer"
    # "startup_founder"
)

LLM_MODEL="gpt-4o-mini"   # answer-generation model
QUESTION_LIMIT=""         # set to a number for testing, "" for full run

# --- Args --------------------------------------------------------------------
SKIP_STORE=false
SKIP_ANSWER=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-store)  SKIP_STORE=true; shift ;;
        --skip-answer) SKIP_ANSWER=true; shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--skip-store] [--skip-answer]"; exit 1 ;;
    esac
done
if $SKIP_STORE && $SKIP_ANSWER; then
    echo "Cannot use both --skip-store and --skip-answer"; exit 1
fi

# --- Run loop ----------------------------------------------------------------
TOTAL=$(( ${#PERIODS[@]} * ${#PERSONAS[@]} ))
DONE=0
FAILED=0

echo "============================================================"
echo "Memora Track 2 — $SYSTEM"
echo "Data dir: $DATA_DIR"
echo "Total runs: $TOTAL"
echo "Mode: $($SKIP_STORE && echo 'answer-only' || ($SKIP_ANSWER && echo 'store-only' || echo 'full'))"
echo "============================================================"

for PERIOD in "${PERIODS[@]}"; do
    for PERSONA in "${PERSONAS[@]}"; do
        SESSIONS_DIR="$DATA_DIR/$PERIOD/$PERSONA"
        CONV_DIR="$SESSIONS_DIR/conversations"
        QUESTIONS_FILE="$SESSIONS_DIR/evaluation_questions_${PERSONA}.json"
        USER_ID="${PERSONA}_${PERIOD}"
        LOG_FILE="/tmp/${SYSTEM}_${PERIOD}_${PERSONA}.log"

        DONE=$((DONE + 1))
        echo ""
        echo "------------------------------------------------------------"
        echo "[$DONE/$TOTAL] $SYSTEM  --  $PERIOD/$PERSONA  (user_id=$USER_ID)"
        echo "------------------------------------------------------------"

        if [ ! -d "$CONV_DIR" ] || [ ! -f "$QUESTIONS_FILE" ]; then
            echo "SKIP: missing conversations dir or questions file"
            FAILED=$((FAILED + 1))
            continue
        fi

        # Step 1 — store
        if ! $SKIP_STORE; then
            echo "[1/2] Storing conversations..."
            python "$SCRIPT_DIR/conversation_to_memory.py" \
                --system "$SYSTEM" \
                --user-id "$USER_ID" \
                --conversation-directory "$CONV_DIR" \
                2>&1 | tee "$LOG_FILE" || true
            if ! $SKIP_ANSWER; then
                echo "Waiting ${WAIT_TIME}s for $SYSTEM backend to process..."
                sleep "$WAIT_TIME"
            fi
        fi

        # Step 2 — answer
        if ! $SKIP_ANSWER; then
            echo "[2/2] Answering questions..."
            CMD=(python "$SCRIPT_DIR/memory_to_answer.py" "$QUESTIONS_FILE"
                 --system "$SYSTEM"
                 --user-id "$USER_ID"
                 --model "$LLM_MODEL")
            [ -n "$QUESTION_LIMIT" ] && CMD+=(--limit "$QUESTION_LIMIT")
            "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE" || true
        fi
    done
done

echo ""
echo "============================================================"
echo "Done. $((DONE - FAILED))/$TOTAL succeeded, $FAILED skipped"
echo "============================================================"
