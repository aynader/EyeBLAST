import os
import glob
import json
import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze IVT tokens for MPIIDPEye dataset")
    parser.add_argument("--token-dir", required=True, help="Directory containing token JSON files")
    parser.add_argument("--raw-data-dir", required=True, help="Directory containing raw CSV files")
    parser.add_argument("--thresh-fix", type=float, default=4.0, help="Fixation threshold used")
    parser.add_argument("--thresh-ms", type=float, default=7.0, help="Microsaccade threshold used")
    parser.add_argument("--thresh-sacc", type=float, default=40.0, help="Saccade threshold used")
    parser.add_argument("--log-file", default=None, help="Output file for quality check log")
    args = parser.parse_args()

    # Initialize counters
    total_tokens = 0
    fix_tokens = 0
    ms_tokens = 0
    sacc_tokens = 0
    
    total_time = 0
    fix_time = 0
    ms_time = 0
    sacc_time = 0

    # Get all token files
    token_files = glob.glob(os.path.join(args.token_dir, "*_tokens.json"))
    print(f"Found {len(token_files)} token files in {args.token_dir}")

    for token_file in token_files:
        base_name = os.path.basename(token_file).replace("_tokens.json", "")
        raw_file = os.path.join(args.raw_data_dir, f"{base_name}.csv")
        
        # Load tokens
        with open(token_file, 'r') as f:
            tokens = json.load(f)
        
        # Load raw data to calculate time durations
        try:
            # MPIIDPEye uses semicolon delimiter and no header
            df = pd.read_csv(raw_file, sep=';', header=None)
            # Assuming columns: x, y, timestamp, confidence, pupil_diameter
            df.columns = ['x', 'y', 'timestamp', 'confidence', 'pupil_diameter']
            
            # Get time in seconds (if needed, convert from ms)
            if df['timestamp'].mean() > 1000:
                df['timestamp'] = df['timestamp'] / 1000.0
                
            # Calculate total recording time
            recording_time = df['timestamp'].max() - df['timestamp'].min()
            total_time += recording_time
            
            # Count tokens by type and calculate times
            file_fix_time = 0
            file_ms_time = 0
            file_sacc_time = 0
            
            for token in tokens:
                token_type = token[0]
                token_length = int(token[1:])
                total_tokens += 1
                
                if token_type == 'F':
                    fix_tokens += 1
                    # Calculate approximate time for this fixation
                    # Assuming average time step is total time / number of samples
                    avg_time_step = recording_time / len(df)
                    token_time = avg_time_step * token_length
                    file_fix_time += token_time
                elif token_type == 'M':
                    ms_tokens += 1
                    # Calculate approximate time for this microsaccade
                    avg_time_step = recording_time / len(df)
                    token_time = avg_time_step * token_length
                    file_ms_time += token_time
                elif token_type == 'S':
                    sacc_tokens += 1
                    # Calculate approximate time for this saccade
                    avg_time_step = recording_time / len(df)
                    token_time = avg_time_step * token_length
                    file_sacc_time += token_time
            
            fix_time += file_fix_time
            ms_time += file_ms_time
            sacc_time += file_sacc_time
            
        except Exception as e:
            print(f"Error processing {raw_file}: {str(e)}")
            continue

    # Calculate percentages
    fix_pct = (fix_tokens / total_tokens * 100) if total_tokens > 0 else 0
    ms_pct = (ms_tokens / total_tokens * 100) if total_tokens > 0 else 0
    sacc_pct = (sacc_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    fix_time_pct = (fix_time / total_time * 100) if total_time > 0 else 0
    ms_time_pct = (ms_time / total_time * 100) if total_time > 0 else 0
    sacc_time_pct = (sacc_time / total_time * 100) if total_time > 0 else 0

    # Create summary
    summary = f'''
======================================================================
IVT Quality Check for MPIIDPEye Dataset
======================================================================
Parameters:
  Fixation threshold: {args.thresh_fix} deg/s
  Microsaccade threshold: {args.thresh_ms} deg/s
  Saccade threshold: {args.thresh_sacc} deg/s

Token statistics:
  Total tokens: {total_tokens}
  Fixation tokens: {fix_tokens} ({fix_pct:.2f}%)
  Microsaccade tokens: {ms_tokens} ({ms_pct:.2f}%)
  Saccade tokens: {sacc_tokens} ({sacc_pct:.2f}%)

By token count:
  Fixations: {fix_pct:.2f}%
  Microsaccades: {ms_pct:.2f}%
  Saccades: {sacc_pct:.2f}%

By time spent:
  Fixations: {fix_time_pct:.2f}%
  Microsaccades: {ms_time_pct:.2f}%
  Saccades: {sacc_time_pct:.2f}%

Total recording time: {total_time:.2f} seconds
======================================================================
'''

    # Print or save to log file
    print(summary)
    if args.log_file:
        with open(args.log_file, 'w') as f:
            f.write(summary)
        print(f"Log saved to {args.log_file}")

if __name__ == "__main__":
    main()
