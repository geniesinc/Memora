#!/usr/bin/env python3
"""
Batch Memory State Validator

Runs validation on multiple session directories and generates a summary report.
"""

import sys
import subprocess
from pathlib import Path
import json

def run_validation(session_dir: Path) -> dict:
    """Run validator on a session directory and return results"""
    try:
        result = subprocess.run(
            ["python", "quality_checks/memory_state_validator.py", str(session_dir)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Parse output to extract key info
        output = result.stdout
        
        # Extract statistics
        sessions_count = None
        errors_count = 0
        error_types = {}
        
        for line in output.split('\n'):
            if 'Loaded' in line and 'sessions from' in line:
                try:
                    sessions_count = int(line.split()[1])
                except:
                    pass
            
            if 'ERRORS FOUND:' in line:
                try:
                    errors_count = int(line.split(':')[1].strip().split()[0])
                except:
                    pass
            
            # Extract error type counts
            if 'occurrences' in line and '🔴' in line:
                try:
                    parts = line.split(':')
                    error_type = parts[0].replace('🔴', '').strip()
                    count = int(parts[1].split()[0])
                    error_types[error_type] = count
                except:
                    pass
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "sessions_count": sessions_count,
            "errors_count": errors_count,
            "error_types": error_types,
            "output": output
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "sessions_count": None,
            "errors_count": None,
            "error_types": {},
            "output": "Validation timed out"
        }
    except Exception as e:
        return {
            "status": "error",
            "sessions_count": None,
            "errors_count": None,
            "error_types": {},
            "output": str(e)
        }

def main():
    """Run validation on first 10 session directories"""
    output_dir = Path("output")
    
    # Get first 10 session directories
    session_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("sessions_")])[:10]
    
    print("="*80)
    print("🔍 BATCH MEMORY STATE VALIDATION")
    print("="*80)
    print(f"\n📁 Found {len(session_dirs)} session directories to validate\n")
    
    results = {}
    
    for i, session_dir in enumerate(session_dirs, 1):
        print(f"[{i}/{len(session_dirs)}] Validating {session_dir.name}...", end=" ", flush=True)
        result = run_validation(session_dir)
        results[session_dir.name] = result
        
        if result["status"] == "passed":
            print(f"✅ PASSED ({result['sessions_count']} sessions)")
        elif result["status"] == "failed":
            print(f"❌ FAILED ({result['errors_count']} errors)")
        else:
            print(f"⚠️  {result['status'].upper()}")
    
    # Generate summary report
    print("\n" + "="*80)
    print("📊 VALIDATION SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for r in results.values() if r["status"] == "passed")
    failed_count = sum(1 for r in results.values() if r["status"] == "failed")
    
    print(f"\n✅ Passed: {passed_count}/{len(session_dirs)}")
    print(f"❌ Failed: {failed_count}/{len(session_dirs)}")
    
    if failed_count > 0:
        print(f"\n🔴 Failed Directories:")
        print("-"*80)
        
        # Aggregate error types across all failures
        all_error_types = {}
        
        for dir_name, result in results.items():
            if result["status"] == "failed":
                print(f"\n📁 {dir_name}")
                print(f"   Sessions: {result['sessions_count']}, Errors: {result['errors_count']}")
                
                if result["error_types"]:
                    print(f"   Error types:")
                    for error_type, count in sorted(result["error_types"].items()):
                        print(f"     - {error_type}: {count}")
                        all_error_types[error_type] = all_error_types.get(error_type, 0) + count
        
        # Show aggregate error statistics
        if all_error_types:
            print(f"\n📈 Aggregate Error Statistics:")
            print("-"*80)
            for error_type, count in sorted(all_error_types.items(), key=lambda x: -x[1]):
                print(f"  {error_type}: {count} occurrences")
    
    # Save detailed results to JSON
    output_file = "validation_batch_report.json"
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {
                "total_directories": len(session_dirs),
                "passed": passed_count,
                "failed": failed_count
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {output_file}")
    print("="*80)
    
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

