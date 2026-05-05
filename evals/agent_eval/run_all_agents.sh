#!/usr/bin/env bash
# =============================================================================
# Track 2 — Run all 6 memory agents
# =============================================================================
# Usage:
#   ./run_all_agents.sh                       # all 6 sequentially
#   ./run_all_agents.sh --parallel            # all 6 in parallel
#   ./run_all_agents.sh a_mem mem_0           # specific subset, sequential
#   ./run_all_agents.sh --parallel a_mem mem_0
# =============================================================================

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

ALL_AGENTS=(a_mem langmem mem_0 memobase memos nemori)

PARALLEL=false
SUBSET=()
for arg in "$@"; do
    case $arg in
        --parallel) PARALLEL=true ;;
        *) SUBSET+=("$arg") ;;
    esac
done
[ ${#SUBSET[@]} -eq 0 ] && SUBSET=("${ALL_AGENTS[@]}")

PASSED=()
FAILED=()
PIDS=()

run_one() {
    local agent="$1"
    local script="./run_${agent}.sh"
    if [ ! -f "$script" ]; then
        echo "MISSING: $script"
        return 1
    fi
    echo "------------------------------------------------------------"
    echo "Running $agent"
    echo "------------------------------------------------------------"
    "$script"
}

if $PARALLEL; then
    echo "Running ${#SUBSET[@]} agents in parallel: ${SUBSET[*]}"
    for agent in "${SUBSET[@]}"; do
        ( run_one "$agent" ) &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do
        if wait "$pid"; then PASSED+=("pid=$pid"); else FAILED+=("pid=$pid"); fi
    done
else
    for agent in "${SUBSET[@]}"; do
        if run_one "$agent"; then PASSED+=("$agent"); else FAILED+=("$agent"); fi
    done
fi

echo ""
echo "============================================================"
echo "Summary: ${#PASSED[@]} passed / ${#FAILED[@]} failed (of ${#SUBSET[@]})"
[ ${#PASSED[@]} -gt 0 ] && echo "  passed: ${PASSED[*]}"
[ ${#FAILED[@]} -gt 0 ] && echo "  failed: ${FAILED[*]}"
echo "============================================================"
