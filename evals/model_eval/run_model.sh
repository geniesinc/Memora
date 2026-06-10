#!/usr/bin/env bash
# =============================================================================
# Track 1 — LLM Evaluation Batch Driver
# =============================================================================
# Loops over (period, persona, model, reasoning_mode) and invokes
# model_based_evaluator.py against the released data layout:
#
#   data/<period>/<persona>/
#     ├── conversations/session_*.json
#     └── evaluation_questions_<persona>.json
#
# Results are written to:
#   data/<period>/<persona>/eval_results/<model>_<no_reasoning|reasoning>/
# =============================================================================

set -e

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RELEASE_ROOT="$(cd "$EVALS_DIR/.." && pwd)"
DATA_DIR="${MEMORA_DATA_DIR:-$RELEASE_ROOT/data}"

# --- What to evaluate --------------------------------------------------------
PERIODS=(
    # "weekly"
    "monthly"
    "quarterly"
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

# LLMs under evaluation. All routed through OpenRouter.
# The first block is the paper's Table 3 set; the second is the newer
# frontier models added in the 2026-05-28 spot check.
# Override for a one-off subset with: MODELS="a/b c/d" ./run_model.sh
if [ -n "${MODELS:-}" ]; then
    read -r -a MODELS <<< "$MODELS"
else
    MODELS=(
        "qwen/qwen3-32b"
        "anthropic/claude-sonnet-4.5"
        "google/gemini-3-pro-preview"
        "openai/gpt-5.2"
        "openai/gpt-5.5"
        "anthropic/claude-opus-4.7"
        "google/gemini-3.1-pro-preview"
        "anthropic/claude-fable-5"
    )
fi

# Run each model twice: once with reasoning OFF, once with reasoning ON.
EVALUATE_WITH_REASONING="true"
REASONING_EFFORT="high"          # OpenAI/GPT-5/Grok use reasoning.effort
REASONING_MAX_TOKENS="10000"     # Anthropic/Gemini/Qwen use reasoning.max_tokens

# Multi-judge config (must be OpenRouter model ids).
USE_MULTI_JUDGE="true"
JUDGE_OPENAI="openai/gpt-4.1"
JUDGE_ANTHROPIC="anthropic/claude-haiku-4.5"
JUDGE_GOOGLE="google/gemini-2.5-flash"

# Set to a number for testing, "" for full run. Overridable from the env.
QUESTION_LIMIT="${QUESTION_LIMIT-}"

# --- Helpers -----------------------------------------------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/evaluation_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

reasoning_kind_for_model() {
    # Pick the reasoning param style based on the model id.
    case "$1" in
        openai/o*|openai/gpt-5*|*grok*) echo "effort" ;;
        *anthropic*|*gemini*|*qwen*)    echo "max_tokens" ;;
        *)                              echo "max_tokens" ;;
    esac
}

run_one() {
    local PERIOD="$1" PERSONA="$2" MODEL="$3" REASONING="$4"

    local SESSIONS_DIR="$DATA_DIR/$PERIOD/$PERSONA"
    if [ ! -d "$SESSIONS_DIR" ]; then
        log "SKIP: $SESSIONS_DIR does not exist"
        return
    fi

    local MODEL_CLEAN
    MODEL_CLEAN=$(echo "$MODEL" | tr '/:' '__')
    local SUFFIX
    [ "$REASONING" = "true" ] && SUFFIX="reasoning" || SUFFIX="no_reasoning"
    local OUTPUT_DIR="$SESSIONS_DIR/eval_results/${MODEL_CLEAN}_${SUFFIX}"

    log ""
    log "[$((++CURRENT))/$TOTAL] $MODEL  reasoning=$SUFFIX  -- $PERIOD/$PERSONA"

    # Resume support: skip if a non-empty eval_report already exists for this slot.
    if [ "${SKIP_IF_DONE:-true}" = "true" ] && \
       ls "$OUTPUT_DIR"/eval_report_*.json >/dev/null 2>&1; then
        log "SKIP: report already present in $OUTPUT_DIR"
        SUCCESS=$((SUCCESS + 1))
        return
    fi

    local CMD=(python -u "$SCRIPT_DIR/model_based_evaluator.py"
               --sessions-dir "$SESSIONS_DIR"
               --model        "$MODEL"
               --output-dir   "$OUTPUT_DIR")

    if [ "$USE_MULTI_JUDGE" = "true" ]; then
        CMD+=(--judge-openai "$JUDGE_OPENAI"
              --judge-anthropic "$JUDGE_ANTHROPIC"
              --judge-google "$JUDGE_GOOGLE")
    else
        CMD+=(--no-multi-judge)
    fi

    [ -n "$QUESTION_LIMIT" ] && CMD+=(--limit "$QUESTION_LIMIT")

    if [ "$REASONING" = "true" ]; then
        local KIND; KIND=$(reasoning_kind_for_model "$MODEL")
        if [ "$KIND" = "effort" ]; then
            CMD+=(--reasoning-effort "$REASONING_EFFORT")
        else
            CMD+=(--reasoning-max-tokens "$REASONING_MAX_TOKENS")
        fi
    fi

    local START END
    START=$(date +%s)
    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        END=$(date +%s)
        log "OK in $((END - START))s"
        SUCCESS=$((SUCCESS + 1))
    else
        END=$(date +%s)
        log "FAIL after $((END - START))s"
        FAILED=$((FAILED + 1))
    fi

    sleep 2
}

# --- Main loop ---------------------------------------------------------------
RUNS_PER_MODEL=1
[ "$EVALUATE_WITH_REASONING" = "true" ] && RUNS_PER_MODEL=2
TOTAL=$(( ${#PERIODS[@]} * ${#PERSONAS[@]} * ${#MODELS[@]} * RUNS_PER_MODEL ))
CURRENT=0
SUCCESS=0
FAILED=0

log "============================================================"
log "Memora Track 1 — LLM Evaluation"
log "============================================================"
log "Data dir: $DATA_DIR"
log "Periods:  ${PERIODS[*]}"
log "Personas: ${#PERSONAS[@]}"
log "Models:   ${#MODELS[@]}"
log "Reasoning sweep: $EVALUATE_WITH_REASONING"
log "Total runs: $TOTAL"

for PERIOD in "${PERIODS[@]}"; do
    for PERSONA in "${PERSONAS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            run_one "$PERIOD" "$PERSONA" "$MODEL" "false"
            [ "$EVALUATE_WITH_REASONING" = "true" ] && run_one "$PERIOD" "$PERSONA" "$MODEL" "true"
        done
    done
done

log ""
log "============================================================"
log "Done. Success: $SUCCESS / Failed: $FAILED / Total: $TOTAL"
log "Log: $LOG_FILE"
log "============================================================"
