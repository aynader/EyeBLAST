
import os
import json
import glob
import pandas as pd
import numpy as np

def analyze_tokens(token_dir, raw_data_dir, thresh_fix, thresh_ms, thresh_sacc, log_file):
    """
    Analyze IVT tokenization results for JuDo1000 dataset.
    """
    
    # Get all token files
    token_files = glob.glob(os.path.join(token_dir, "*_tokens.json"))
    
    if not token_files:
        print("No token files found!")
        return
    
    total_fixations = 0
    total_microsaccades = 0
    total_saccades = 0
    total_tokens = 0
    
    # Estimate time spent in each category
    total_fix_time = 0
    total_ms_time = 0
    total_sacc_time = 0
    total_time = 0
    
    for token_file in token_files:
        with open(token_file, 'r') as f:
            tokens = json.load(f)
        
        for token in tokens:
            duration = int(token[1:])  # Extract duration from token (e.g., "F15" -> 15)
            total_tokens += 1
            
            if token.startswith('F'):
                total_fixations += 1
                total_fix_time += duration
            elif token.startswith('M'):
                total_microsaccades += 1
                total_ms_time += duration
            elif token.startswith('S'):
                total_saccades += 1
                total_sacc_time += duration
            
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
        f.write(f"JuDo1000 IVT Quality Check Results\n")
        f.write(f"Parameters: thresh_fix={thresh_fix}, thresh_ms={thresh_ms}, thresh_sacc={thresh_sacc}\n")
        f.write(f"Files processed: {len(token_files)}\n")
        f.write(f"Total tokens: {total_tokens}\n\n")
        
        f.write(f"By token count:\n")
        f.write(f"  Fixations: {fix_pct:.2f}%\n")
        f.write(f"  Microsaccades: {ms_pct:.2f}%\n")
        f.write(f"  Saccades: {sacc_pct:.2f}%\n\n")
        
        f.write(f"By time spent:\n")
        f.write(f"  Fixations: {fix_time_pct:.2f}%\n")
        f.write(f"  Microsaccades: {ms_time_pct:.2f}%\n")
        f.write(f"  Saccades: {sacc_time_pct:.2f}%\n")
    
    print(f"Quality check complete. Results saved to {log_file}")
    return fix_pct, ms_pct, sacc_pct, fix_time_pct, ms_time_pct, sacc_time_pct

# Run the analysis
fix_pct, ms_pct, sacc_pct, fix_time_pct, ms_time_pct, sacc_time_pct = analyze_tokens(
    "parameter_search_results_JuDo\set_7_fix200000_ms300000_sacc1000000\IVT_output",
    "C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/JuDo1000",
    200000,
    300000,
    1000000,
    "parameter_search_results_JuDo\set_7_fix200000_ms300000_sacc1000000\quality_check_log.txt"
)
