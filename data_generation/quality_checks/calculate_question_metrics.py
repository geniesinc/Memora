#!/usr/bin/env python3
"""
Calculate Question Metrics

This script calculates:
1. Memory consolidation: How many sessions of conversations needed to be taken into account 
   for answering the questions correctly (counts unique sessions referenced in memory_evidence)
   - Calculated for ALL question types (remembering, reasoning, recommending)
2. Memory mutations: How many update and delete operations are in the memory evidence trail
   (counts update/delete operations, excluding add operations)
   - Calculated for recommendation and remembering questions only

For each output folder containing evaluation_questions_{persona}.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict
from datetime import datetime


def load_config(config_file: str) -> Dict[str, Any]:
    """Load memory config file to determine allowed operations"""
    config_path = Path(config_file)
    if not config_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / config_file
    
    with open(config_path, 'r') as f:
        return json.load(f)


def get_allowed_operations(config: Dict[str, Any], memory_type: str, category: str) -> List[str]:
    """Get allowed operations for a memory type and category"""
    if memory_type not in config:
        return []
    
    memory_config = config[memory_type]
    if "options" not in memory_config:
        return []
    
    options = memory_config["options"]
    if category not in options:
        return []
    
    return options[category].get("allowed_operations", [])


def extract_session_ids_from_memory_evidence(memory_evidence: Dict[str, Any]) -> Set[int]:
    """Extract all session IDs referenced in memory evidence"""
    session_ids = set()
    
    # Check for goal_session
    goal_session = memory_evidence.get("goal_session")
    if goal_session and isinstance(goal_session, dict):
        session_id = goal_session.get("session_id")
        if session_id is not None:
            session_ids.add(int(session_id))
    
    # Check for activity_sessions
    activity_sessions = memory_evidence.get("activity_sessions", [])
    if isinstance(activity_sessions, list):
        for session in activity_sessions:
            if isinstance(session, dict):
                session_id = session.get("session_id")
                if session_id is not None:
                    session_ids.add(int(session_id))
    
    # Check for session_id in nested structures
    # For content_data, check if there are session references
    # For activity items, check if they have session_id
    if "content_data" in memory_evidence:
        # Content memory doesn't typically have session_id in content_data
        pass
    
    # Check for calendar_events, food_expenses, step_tracker, etc. with session_id
    for key in ["calendar_events", "food_expenses", "step_tracker", "remaining_tasks"]:
        if key in memory_evidence:
            items = memory_evidence[key]
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        session_id = item.get("session_id")
                        if session_id is not None:
                            session_ids.add(int(session_id))
    
    # Check for budget_timeline (for comparative questions)
    budget_timeline = memory_evidence.get("budget_timeline")
    if isinstance(budget_timeline, dict):
        # Budget timeline keys are day offsets, not session IDs
        # But we can infer sessions from period_breakdown
        pass
    
    # Check for period_breakdown (for comparative questions)
    period_breakdown = memory_evidence.get("period_breakdown")
    if isinstance(period_breakdown, dict):
        for period_data in period_breakdown.values():
            if isinstance(period_data, dict):
                items = period_data.get("items", [])
                for item in items:
                    if isinstance(item, dict):
                        session_id = item.get("session_id")
                        if session_id is not None:
                            session_ids.add(int(session_id))
    
    return session_ids


def extract_session_ids_from_forgetting_evidence(forgetting_evidence: Dict[str, Any]) -> Set[int]:
    """Extract all session IDs referenced in forgetting evidence"""
    session_ids = set()
    
    forgotten_items = forgetting_evidence.get("forgotten_items", [])
    if isinstance(forgotten_items, list):
        for forgotten_item in forgotten_items:
            if isinstance(forgotten_item, dict):
                session_id = forgotten_item.get("session_id")
                if session_id is not None:
                    session_ids.add(int(session_id))
    
    return session_ids


def count_memory_references(question: Dict[str, Any]) -> int:
    """Count memory consolidation: how many unique sessions are referenced in memory_evidence
    
    This represents how many sessions of conversations needed to be taken into account
    for answering the question correctly.
    """
    session_ids = set()
    
    # Extract from memory_evidence
    memory_evidence = question.get("memory_evidence", {})
    session_ids.update(extract_session_ids_from_memory_evidence(memory_evidence))
    
    # Extract from forgetting_evidence
    forgetting_evidence = question.get("forgetting_evidence", {})
    session_ids.update(extract_session_ids_from_forgetting_evidence(forgetting_evidence))
    
    # Also check the question's own session_id
    question_session_id = question.get("session_id")
    if question_session_id is not None:
        session_ids.add(int(question_session_id))
    
    return len(session_ids)


def get_memory_mutations_from_question(question: Dict[str, Any]) -> int:
    """Get memory mutations from question's forgetting_evidence
    
    Memory mutations = total_forgotten_items, which represents items that were deleted or updated
    (where the old item was forgotten). This is already calculated and stored in forgetting_evidence.
    """
    forgetting_evidence = question.get("forgetting_evidence", {})
    return forgetting_evidence.get("total_forgotten_items", 0)


def analyze_questions_file(questions_file: Path, config: Dict[str, Any], 
                          session_file: Optional[Path] = None) -> Dict[str, Any]:
    """Analyze a single evaluation questions file"""
    with open(questions_file, 'r') as f:
        data = json.load(f)
    
    # Find session file if not provided
    if session_file is None:
        session_dir = questions_file.parent
        session_file = session_dir / "new_sessions.json"
        if not session_file.exists():
            print(f"  ⚠️  Warning: Could not find new_sessions.json in {session_dir}")
            session_file = None
    
    # Collect all questions with their task_type
    all_questions = []
    questions_by_category = data.get("questions", {})
    
    for task_type in ["remembering", "reasoning", "recommending"]:
        questions = questions_by_category.get(task_type, [])
        # Add task_type to each question for tracking
        for question in questions:
            question['_task_type'] = task_type
            all_questions.append(question)
    
    # Calculate metrics
    memory_references = []
    mutations_by_category = defaultdict(list)
    
    for question in all_questions:
        # Count memory references (for all question types)
        ref_count = count_memory_references(question)
        memory_references.append(ref_count)
        
        # Count mutations for recommendation and remembering questions
        # Mutations = total_forgotten_items from forgetting_evidence (items that were deleted or updated)
        task_type = question.get('_task_type', '')
        if task_type in ["recommending", "remembering"]:
            forgetting_evidence = question.get("forgetting_evidence", {})
            mutation_count = forgetting_evidence.get("total_forgotten_items", 0)
            
            # Include all questions (even with 0 mutations) for accurate averages
            memory_type = question.get("memory_type", "")
            category = question.get("category", "")
            key = f"{memory_type}/{category}"
            mutations_by_category[key].append(mutation_count)
    
    # Calculate averages
    avg_memory_reference = sum(memory_references) / len(memory_references) if memory_references else 0
    max_memory_reference = max(memory_references) if memory_references else 0
    
    # Calculate mutation statistics by category
    mutation_stats = {}
    for key, counts in mutations_by_category.items():
        if counts:
            mutation_stats[key] = {
                "average": sum(counts) / len(counts),
                "max": max(counts),
                "min": min(counts),
                "count": len(counts)
            }
    
    return {
        "persona": data.get("persona", "unknown"),
        "total_questions": len(all_questions),
        "memory_reference": {
            "average": avg_memory_reference,
            "max": max_memory_reference,
            "min": min(memory_references) if memory_references else 0,
            "distribution": {
                "1": sum(1 for r in memory_references if r == 1),
                "2-5": sum(1 for r in memory_references if 2 <= r <= 5),
                "6-10": sum(1 for r in memory_references if 6 <= r <= 10),
                "11+": sum(1 for r in memory_references if r > 10)
            }
        },
        "memory_mutations": mutation_stats
    }


def main():
    """Main function to analyze all evaluation question files"""
    import argparse
    
    # Get script directory and find output directory relative to project root
    script_dir = Path(__file__).parent.absolute()
    # Script is in quality_checks/, so output is in parent/output
    project_root = script_dir.parent
    
    parser = argparse.ArgumentParser(
        description='Calculate question metrics: memory consolidation and mutations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze weekly data
  python calculate_question_metrics.py --timeline weekly
  
  # Analyze monthly data
  python calculate_question_metrics.py --timeline monthly
  
  # Analyze quarterly data
  python calculate_question_metrics.py --timeline quarterly
  
  # Use custom output directory
  python calculate_question_metrics.py --output-dir output_custom
  
  # Analyze specific personas
  python calculate_question_metrics.py --timeline weekly --persona-filter academic_researcher software_engineer
        """
    )
    parser.add_argument(
        '--timeline',
        type=str,
        choices=['weekly', 'monthly', 'quarterly'],
        help='Timeline type: weekly, monthly, or quarterly (maps to output_weekly, output_monthly, output_quarterly)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory containing persona folders (overrides --timeline if specified)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Memory config file path (will try to detect from questions file if not provided)'
    )
    parser.add_argument(
        '--persona-filter',
        nargs='+',
        help='Only analyze specific personas'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        # Use explicitly provided output directory
        output_dir = Path(args.output_dir).resolve()
    elif args.timeline:
        # Map timeline to directory name
        timeline_dirs = {
            'weekly': 'output_weekly',
            'monthly': 'output_monthly',
            'quarterly': 'output_quarterly'
        }
        output_dir = project_root / timeline_dirs[args.timeline]
    else:
        # Default to output directory
        output_dir = project_root / 'output'
    
    # Resolve output_dir to absolute path
    output_dir = output_dir.resolve()
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    # Find all evaluation question files
    question_files = list(output_dir.glob("**/evaluation_questions_*.json"))
    
    if not question_files:
        print(f"❌ No evaluation question files found in {output_dir}")
        return
    
    print(f"📊 Found {len(question_files)} evaluation question file(s)")
    print("=" * 80)
    
    all_results = {}
    
    for questions_file in sorted(question_files):
        persona = questions_file.stem.replace("evaluation_questions_", "")
        
        if args.persona_filter and persona not in args.persona_filter:
            continue
        
        print(f"\n📁 Analyzing: {persona}")
        print(f"   File: {questions_file}")
        
        # Try to detect config file from questions file
        config_file = args.config
        if not config_file:
            try:
                with open(questions_file, 'r') as f:
                    questions_data = json.load(f)
                    config_file = questions_data.get("config_file")
            except:
                pass
        
        if not config_file:
            # Default config file based on timeline or default to weekly
            if args.timeline:
                config_file = f"meta_data/memory_configs/memory_config_{args.timeline}.json"
            else:
                config_file = "meta_data/memory_configs/memory_config_weekly.json"
            print(f"  ⚠️  Warning: No config file specified, using default: {config_file}")
        
        # Load config
        try:
            config = load_config(config_file)
        except Exception as e:
            print(f"  ❌ Error loading config: {e}")
            continue
        
        # Analyze questions
        try:
            result = analyze_questions_file(questions_file, config)
            all_results[persona] = result
            
            print(f"  ✅ Analyzed {result['total_questions']} questions")
            print(f"  📊 Average memory consolidation: {result['memory_reference']['average']:.2f}")
            print(f"  📊 Max memory consolidation: {result['memory_reference']['max']}")
            
            if result['memory_mutations']:
                print(f"  🔄 Memory mutations by category (recommendation and remembering questions):")
                for key, stats in result['memory_mutations'].items():
                    print(f"     {key}: avg={stats['average']:.2f}, max={stats['max']}, count={stats['count']}")
            
        except Exception as e:
            print(f"  ❌ Error analyzing: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate overall statistics
    print("\n" + "=" * 80)
    print("📊 STATISTICS BY CATEGORY")
    print("=" * 80)
    
    if not all_results:
        print("❌ No results to aggregate")
        return
    
    # Aggregate mutation stats by category (across all personas)
    all_mutation_stats = defaultdict(lambda: {"counts": [], "maxes": [], "question_count": 0})
    for result in all_results.values():
        for key, stats in result['memory_mutations'].items():
            all_mutation_stats[key]["counts"].extend([stats['average']] * stats['count'])
            all_mutation_stats[key]["maxes"].append(stats['max'])
            all_mutation_stats[key]["question_count"] += stats['count']
    
    # Display by category
    if all_mutation_stats:
        print(f"\n🔄 Memory Mutations by Category (Recommendation and Remembering Questions):")
        for key, stats in sorted(all_mutation_stats.items()):
            if stats["counts"]:
                avg_avg = sum(stats["counts"]) / len(stats["counts"])
                overall_max = max(stats["maxes"])
                print(f"   {key}:")
                print(f"      Average mutations: {avg_avg:.2f}")
                print(f"      Max mutations: {overall_max}")
                print(f"      Questions analyzed: {stats['question_count']}")
    
    # Aggregate memory reference stats by task type
    memory_ref_by_task_type = defaultdict(lambda: {"refs": [], "count": 0})
    for questions_file in sorted(question_files):
        persona = questions_file.stem.replace("evaluation_questions_", "")
        if args.persona_filter and persona not in args.persona_filter:
            continue
        
        try:
            with open(questions_file, 'r') as f:
                questions_data = json.load(f)
            
            questions_by_category = questions_data.get("questions", {})
            for task_type in ["remembering", "reasoning", "recommending"]:
                questions = questions_by_category.get(task_type, [])
                for q in questions:
                    ref_count = count_memory_references(q)
                    memory_ref_by_task_type[task_type]["refs"].append(ref_count)
                    memory_ref_by_task_type[task_type]["count"] += 1
        except:
            continue
    
    print(f"\n📈 Memory Consolidation by Task Type:")
    for task_type in ["remembering", "reasoning", "recommending"]:
        if task_type in memory_ref_by_task_type:
            refs = memory_ref_by_task_type[task_type]["refs"]
            if refs:
                print(f"   {task_type.capitalize()}:")
                print(f"      Average: {sum(refs) / len(refs):.2f}")
                print(f"      Max: {max(refs)}")
                print(f"      Min: {min(refs)}")
                print(f"      Questions: {len(refs)}")
    
    # Overall aggregates
    print("\n" + "=" * 80)
    print("📊 OVERALL AGGREGATE STATISTICS")
    print("=" * 80)
    
    # Aggregate memory consolidation stats
    all_avg_refs = [r['memory_reference']['average'] for r in all_results.values()]
    all_max_refs = [r['memory_reference']['max'] for r in all_results.values()]
    all_min_refs = [r['memory_reference']['min'] for r in all_results.values()]
    
    print(f"\n📈 Memory Consolidation (All Questions):")
    print(f"   Overall average: {sum(all_avg_refs) / len(all_avg_refs):.2f}")
    print(f"   Overall max: {max(all_max_refs)}")
    print(f"   Overall min: {min(all_min_refs)}")
    print(f"   Total personas: {len(all_results)}")
    
    if all_mutation_stats:
        print(f"\n🔄 Memory Mutations (All Categories - Recommendation and Remembering Questions):")
        all_mutation_counts = []
        all_mutation_maxes = []
        total_questions = 0
        for key, stats in all_mutation_stats.items():
            if stats["counts"]:
                all_mutation_counts.extend(stats["counts"])
                all_mutation_maxes.append(max(stats["maxes"]))
                total_questions += stats["question_count"]
        
        if all_mutation_counts:
            print(f"   Overall average mutations: {sum(all_mutation_counts) / len(all_mutation_counts):.2f}")
            print(f"   Overall max mutations: {max(all_mutation_maxes)}")
            print(f"   Total recommendation and remembering questions analyzed: {total_questions}")
            print(f"   Categories analyzed: {len(all_mutation_stats)}")


if __name__ == "__main__":
    main()

