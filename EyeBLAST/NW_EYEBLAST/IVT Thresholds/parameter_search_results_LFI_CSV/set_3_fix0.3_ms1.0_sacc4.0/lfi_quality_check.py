
import os
import json
import glob
import pandas as pd
import numpy as np

def analyze_tokens(token_dir, raw_data_dir, thresh_fix, thresh_ms, thresh_sacc, log_file):
    """
    Analyze IVT tokenization results for LFI_CSV dataset.
    """
    
    # Get all token files
    token_files = glob.glob(os.path.join(token_dir, "*_tokens.json"))
    
    if not token_files:
        print("No token files found!")
        return 0, 0, 0, 0, 0, 0  # Return zeros instead of None
    
    total_fixations = 0
    total_microsaccades = 0
    total_saccades = 0
    total_tokens = 0
    
    # Estimate time spent in each category
    total_fix_time = 0
    total_ms_time = 0
    total_sacc_time = 0
    total_time = 0
    
    # Collect statistics by task
    task_stats = {}
    
    for token_file in token_files:
        with open(token_file, 'r') as f:
            tokens = json.load(f)
        
        # Extract task info from filename
        filename = os.path.basename(token_file)
        try:
            participant_task = filename.replace('_tokens.json', '')
            _, task_id = participant_task.split('_')
            task_name = {1: "TALK", 2: "READ", 3: "VIDEO", 4: "WALK", 5: "TYPE", 6: "SOLVE", 7: "CYCLE"}.get(int(task_id), f"TASK_{task_id}")
            
            if task_name not in task_stats:
                task_stats[task_name] = {'F': 0, 'M': 0, 'S': 0, 'total': 0}
        except:
            task_name = "UNKNOWN"
            if task_name not in task_stats:
                task_stats[task_name] = {'F': 0, 'M': 0, 'S': 0, 'total': 0}
        
        for token in tokens:
            duration = int(token[1:])  # Extract duration from token (e.g., "F15" -> 15)
            total_tokens += 1
            task_stats[task_name]['total'] += 1
            
            if token.startswith('F'):
                total_fixations += 1
                total_fix_time += duration
                task_stats[task_name]['F'] += 1
            elif token.startswith('M'):
                total_microsaccades += 1
                total_ms_time += duration
                task_stats[task_name]['M'] += 1
            elif token.startswith('S'):
                total_saccades += 1
                total_sacc_time += duration
                task_stats[task_name]['S'] += 1
            
            total_time += duration
    
    # Calculate percentages
    fix_pct = (total_fixations / total_tokens * 100) if total_tokens > 0 else 0
    ms_pct = (total_microsaccades / total_tokens * 100) if total_tokens > 0 else 0
    sacc_pct = (total_saccades / total_tokens * 100) if total_tokens > 0 else 0
    
    fix_time_pct = (total_fix_time / total_time * 100) if total_time > 0 else 0
    ms_time_pct = (total_ms_time / total_time * 100) if total_time > 0 else 0
    sacc_time_pct = (total_sacc_time / total_time * 100) if total_time > 0 else 0
    
    # Write to log file
    with open(log_file, 'w') as f:
        f.write(f"LFI_CSV IVT Quality Check Results\n")
        f.write(f"Parameters: thresh_fix={thresh_fix}, thresh_ms={thresh_ms}, thresh_sacc={thresh_sacc}\n")
        f.write(f"Files processed: {len(token_files)}\n")
        f.write(f"Total tokens: {total_tokens}\n\n")
        
        f.write(f"Overall Results:\n")
        f.write(f"By token count:\n")
        f.write(f"  Fixations: {fix_pct:.2f}%\n")
        f.write(f"  Microsaccades: {ms_pct:.2f}%\n")
        f.write(f"  Saccades: {sacc_pct:.2f}%\n\n")
        
        f.write(f"By time spent:\n")
        f.write(f"  Fixations: {fix_time_pct:.2f}%\n")
        f.write(f"  Microsaccades: {ms_time_pct:.2f}%\n")
        f.write(f"  Saccades: {sacc_time_pct:.2f}%\n\n")
        
        f.write(f"Results by Task:\n")
        for task_name, stats in task_stats.items():
            if stats['total'] > 0:
                task_fix_pct = (stats['F'] / stats['total'] * 100)
                task_ms_pct = (stats['M'] / stats['total'] * 100)
                task_sacc_pct = (stats['S'] / stats['total'] * 100)
                f.write(f"  {task_name}: F={task_fix_pct:.1f}%, M={task_ms_pct:.1f}%, S={task_sacc_pct:.1f}% ({stats['total']} tokens)\n")
    
    print(f"Quality check complete. Results saved to {log_file}")
    return fix_pct, ms_pct, sacc_pct, fix_time_pct, ms_time_pct, sacc_time_pct

# Run the analysis
fix_pct, ms_pct, sacc_pct, fix_time_pct, ms_time_pct, sacc_time_pct = analyze_tokens(
    r"parameter_search_results_LFI_CSV\set_3_fix0.3_ms1.0_sacc4.0\IVT_output",
    r"C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/LFI_CSV",
    0.3,
    1.0,
    4.0,
    r"parameter_search_results_LFI_CSV\set_3_fix0.3_ms1.0_sacc4.0\quality_check_log.txt"
)
