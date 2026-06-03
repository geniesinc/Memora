#!/bin/bash

# Run session_simulator.py across one or more personas and a chosen timeline.
#
# Usage:
#   ./run_session_simulator.sh [timeline] [start_date] [end_date] [persona ...]
#
# Arguments:
#   timeline    weekly (default) | monthly | quarterly
#   start_date  YYYY-MM-DD (defaults provided per timeline)
#   end_date    YYYY-MM-DD (defaults provided per timeline)
#   persona     One or more persona names. If omitted, runs all 10 personas.
#
# Examples:
#   # Weekly run for all personas with default dates
#   ./run_session_simulator.sh
#
#   # Monthly run for two specific personas
#   ./run_session_simulator.sh monthly 2025-06-01 2025-06-30 software_engineer marketing_manager
#
#   # Quarterly run for one persona
#   ./run_session_simulator.sh quarterly 2025-01-01 2025-03-31 startup_founder

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMELINE="${1:-weekly}"
shift || true

# Default date ranges per timeline
case "$TIMELINE" in
    weekly)
        DEFAULT_START="2025-06-01"
        DEFAULT_END="2025-06-07"
        ;;
    monthly)
        DEFAULT_START="2025-06-01"
        DEFAULT_END="2025-06-30"
        ;;
    quarterly)
        DEFAULT_START="2025-01-01"
        DEFAULT_END="2025-03-31"
        ;;
    *)
        echo "Invalid timeline: $TIMELINE (expected weekly|monthly|quarterly)"
        exit 1
        ;;
esac

START_DATE="${1:-$DEFAULT_START}"
END_DATE="${2:-$DEFAULT_END}"
[[ $# -gt 0 ]] && shift
[[ $# -gt 0 ]] && shift

# Personas: rest of args, or all 10 if none given
if [[ $# -eq 0 ]]; then
    PERSONAS=(
        academic_researcher
        business_executive
        content_writer
        creative_designer
        financial_analyst
        management_consultant
        marketing_manager
        sales_manager
        software_engineer
        startup_founder
    )
else
    PERSONAS=("$@")
fi

CONFIG="meta_data/memory_configs/memory_config_${TIMELINE}.json"
if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG"
    exit 1
fi

echo "============================================"
echo "Session Simulator"
echo "============================================"
echo "Timeline:   $TIMELINE"
echo "Config:     $CONFIG"
echo "Date range: $START_DATE -> $END_DATE"
echo "Personas:   ${PERSONAS[*]}"
echo "============================================"

for PERSONA in "${PERSONAS[@]}"; do
    echo ""
    echo "--- Simulating: $PERSONA ---"
    python session_simulator.py \
        --config "$CONFIG" \
        --persona "$PERSONA" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE"
    echo "--- Completed: $PERSONA ---"
done

echo ""
echo "Session generation completed. Output is under ./output/"
