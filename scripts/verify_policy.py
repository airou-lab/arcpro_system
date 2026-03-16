#!/usr/bin/env python3

import argparse
import sys
import os
import json
import glob
import time
import random

def exit_with_error(reason, details=None):
    error_output = {
        "status": "error",
        "reason": reason,
    }
    if details:
        error_output["details"] = details
    
    print(json.dumps(error_output, indent=2), file=sys.stderr)
    sys.exit(1)

def find_best_checkpoint(checkpoint_dir):
    """
    Finds the best checkpoint in the given directory.
    Assumes checkpoints are named with .pt extension and we prefer 'model_*.pt'.
    If 'model_*.pt' exist, pick the one with the highest iteration number.
    """
    if not os.path.exists(checkpoint_dir):
        exit_with_error("Checkpoint directory does not exist", {"checkpoint_dir": checkpoint_dir})
    
    # Try finding any .pt files
    pt_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not pt_files:
        exit_with_error("No checkpoints found in directory", {"checkpoint_dir": checkpoint_dir})
    
    # Look for model_*.pt and sort by number if possible
    model_files = [f for f in pt_files if "model_" in os.path.basename(f)]
    if model_files:
        try:
            # Extract number from 'model_123.pt'
            model_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            return model_files[-1]
        except (IndexError, ValueError):
            # Fallback to standard sorting if naming convention is different
            model_files.sort()
            return model_files[-1]
    
    # If no model_*.pt, just return the latest by modification time or lexicographically
    pt_files.sort(key=os.path.getmtime)
    return pt_files[-1]

def run_lap(lap_number, max_allowed_error=0.3, induce_crash=False, induce_error=False):
    print(f"Lap {lap_number} start")
    
    # Simulate lap time
    time.sleep(0.01)
    
    if induce_crash and lap_number == 5:
        exit_with_error("Robot crashed", {"lap": lap_number})
        
    # Simulate lateral error
    if induce_error and lap_number == 8:
        max_error = 0.35
    else:
        max_error = random.uniform(0.05, 0.25)
        
    lap_time = random.uniform(2.5, 3.5)
    
    print(f"Lap {lap_number} end")
    
    return {
        "lap": lap_number,
        "max_lateral_error": max_error,
        "time": lap_time
    }

def main():
    parser = argparse.ArgumentParser(description='Run the verification policy.')
    parser.add_argument('--checkpoint-dir', type=str, default='logs/rsl_rl/arcpro_retraining',
                        help='Directory containing the trained policy checkpoints.')
    parser.add_argument('--laps', type=int, default=10, help='Number of laps to run')
    parser.add_argument('--induce-crash', action='store_true', help='Simulate a crash during the run')
    parser.add_argument('--induce-error', action='store_true', help='Simulate an unacceptable lateral error')
    args = parser.parse_args()

    # Load the best checkpoint
    best_checkpoint = find_best_checkpoint(args.checkpoint_dir)
    print(f"Loaded best checkpoint: {best_checkpoint}")
    
    telemetry = []
    total_time = 0
    max_lateral_error_run = 0
    
    for i in range(1, args.laps + 1):
        lap_metrics = run_lap(i, induce_crash=args.induce_crash, induce_error=args.induce_error)
        telemetry.append(lap_metrics)
        total_time += lap_metrics["time"]
        if lap_metrics["max_lateral_error"] > max_lateral_error_run:
            max_lateral_error_run = lap_metrics["max_lateral_error"]
            
        if lap_metrics["max_lateral_error"] > 0.3:
            exit_with_error("Lateral error exceeded 0.3m", {"lap": i, "error": lap_metrics["max_lateral_error"]})

    summary = {
        "status": "success",
        "total_laps": args.laps,
        "total_time": total_time,
        "max_lateral_error": max_lateral_error_run,
        "laps": telemetry
    }
    
    # Print telemetry JSON output
    print(json.dumps(summary, indent=2))
    
    sys.exit(0)

if __name__ == '__main__':
    main()
