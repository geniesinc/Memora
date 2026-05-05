#!/usr/bin/env python3
"""
aggregate_results.py — produce a Table-3-style summary across many
eval_report_*.json files.

The released benchmark reports FAMA (Forgetting-Aware Memory Accuracy, paper §4.2)
broken down by the three tasks (Remembering, Reasoning, Recommending) and the three
temporal durations (weekly, monthly, quarterly).

Both Track 1 (LLM eval) and Track 2 (memory-agent eval) write reports with the same
schema; this script ingests both. Each input report file is identified by the
metadata it carries (`model_name` for Track 1, `memory_system` for Track 2).

Usage:
    python aggregate_results.py data/                          # walk a release tree
    python aggregate_results.py data/weekly/academic_researcher/eval_results/*/eval_report_*.json
    python aggregate_results.py results.json --print           # one file, pretty-print
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


TASK_TYPES = ("Remembering", "Reasoning", "Recommending")


def find_reports(paths: List[Path]) -> List[Path]:
    """Expand the given paths into a list of eval_report_*.json files."""
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.rglob("eval_report_*.json")))
        elif p.is_file():
            out.append(p)
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def parse_release_path(report_path: Path) -> Dict[str, Optional[str]]:
    """Extract (period, persona, run_id) from a path like
    ``data/<period>/<persona>/eval_results/<run_id>/eval_report_*.json``."""
    parts = report_path.parts
    info: Dict[str, Optional[str]] = {"period": None, "persona": None, "run_id": None}
    if "eval_results" in parts:
        i = parts.index("eval_results")
        if i + 1 < len(parts):
            info["run_id"] = parts[i + 1]
        if i >= 1:
            info["persona"] = parts[i - 1]
        if i >= 2:
            info["period"] = parts[i - 2]
    return info


def report_subject(report: Dict[str, Any], run_id: Optional[str]) -> str:
    """Identify what the report is reporting on.

    For Track 1 (LLM eval) the subject is "<model> (reasoning)" or "<model> (no reasoning)";
    Table 3 reports them as separate rows. For Track 2 (memory-agent eval) the
    subject is just the memory-system name.
    """
    md = report.get("metadata", {})
    if md.get("model_name"):
        suffix = ""
        if run_id and run_id.endswith("_no_reasoning"):
            suffix = " (no reasoning)"
        elif run_id and run_id.endswith("_reasoning"):
            suffix = " (reasoning)"
        return md["model_name"] + suffix
    return md.get("memory_system") or "<unknown>"


def summarize_one(report_path: Path) -> Dict[str, Any]:
    """Pull the FAMA + accuracy numbers from a single report into a flat row."""
    with report_path.open() as f:
        report = json.load(f)

    overall = report.get("overall_metrics", {})
    by_task = report.get("by_task_type", {})
    location = parse_release_path(report_path)

    row: Dict[str, Any] = {
        "report": str(report_path),
        "subject": report_subject(report, location["run_id"]),
        "period": location["period"],
        "persona": location["persona"],
        "run_id": location["run_id"],
        "total_questions": overall.get("total_questions"),
        "fama": overall.get("fama"),
        "overall_accuracy": overall.get("overall_accuracy"),
        "memory_presence_accuracy": overall.get("memory_presence_accuracy"),
        "forgetting_absence_accuracy": overall.get("forgetting_absence_accuracy"),
    }
    for task in TASK_TYPES:
        stats = by_task.get(task, {})
        row[f"fama_{task.lower()}"] = stats.get("fama")
        row[f"questions_{task.lower()}"] = stats.get("total_questions", 0)
    return row


def aggregate_table3(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a Table-3-style nested dict: subject → task → period → mean(FAMA)."""
    nested: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in rows:
        subj = r["subject"]
        period = r["period"] or "unknown"
        for task in TASK_TYPES:
            v = r.get(f"fama_{task.lower()}")
            # Only count this cell if the run actually evaluated questions in
            # this task bucket — otherwise FAMA defaults to 0.0 from the empty
            # task and would skew the average.
            if v is not None and r.get(f"questions_{task.lower()}", 0) > 0:
                nested[subj][task][period].append(float(v))

    out: Dict[str, Any] = {}
    for subj, by_task in nested.items():
        out[subj] = {}
        for task, by_period in by_task.items():
            out[subj][task] = {
                p: (sum(vs) / len(vs)) if vs else None
                for p, vs in by_period.items()
            }
    return out


def print_table3(table: Dict[str, Any]) -> None:
    """Pretty-print the FAMA Table 3."""
    periods = ("weekly", "monthly", "quarterly")
    head = f"{'Subject':<45}"
    for task in TASK_TYPES:
        for p in periods:
            head += f" {task[:4]}/{p[:4]:>5}"
    print(head)
    print("-" * len(head))
    for subj in sorted(table):
        row = f"{subj:<45}"
        for task in TASK_TYPES:
            for p in periods:
                v = table[subj].get(task, {}).get(p)
                row += f" {v:>10.2f}" if v is not None else f" {'-':>10}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("paths", nargs="+", type=Path,
                    help="Report files or a directory tree to walk")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="Pretty-print Table 3 to stdout")
    ap.add_argument("--output", type=Path, default=None,
                    help="Write the aggregated JSON to this path")
    args = ap.parse_args()

    reports = find_reports(args.paths)
    if not reports:
        ap.error(f"No eval_report_*.json files found under: {[str(p) for p in args.paths]}")

    rows = [summarize_one(p) for p in reports]
    table = aggregate_table3(rows)

    if args.do_print:
        print_table3(table)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump({"rows": rows, "table3": table}, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
