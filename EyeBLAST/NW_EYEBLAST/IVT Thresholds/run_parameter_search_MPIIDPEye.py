import os
import subprocess
import pandas as pd
import argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run parameter search for IVT tokenization on MPIIDPEye dataset")
parser.add_argument("--test-files", type=int, default=10,
                    help="Number of files to process in test mode (default: 10)")
parser.add_argument("--raw-data-dir", default="C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/MPIIDPEye/Eye_Tracking_Data",
                    help="Directory containing the raw CSV gaze data")
args = parser.parse_args()

# Define parameter sets to test
parameter_sets = [
    {"set": 1, "thresh_fix": 3.0, "thresh_ms": 6.0, "thresh_sacc": 30.0, 
     "notes": "very strict fix → lots of events; micro and saccade only the fastest"},
    {"set": 2, "thresh_fix": 3.0, "thresh_ms": 7.0, "thresh_sacc": 40.0, 
     "notes": "strict fix, baseline micro/sacc"},
    {"set": 3, "thresh_fix": 3.0, "thresh_ms": 10.0, "thresh_sacc": 50.0, 
     "notes": "strict fix, broad micro window"},
    {"set": 4, "thresh_fix": 4.0, "thresh_ms": 6.0, "thresh_sacc": 30.0, 
     "notes": "moderate fix, strict micro/sacc"},
    {"set": 5, "thresh_fix": 4.0, "thresh_ms": 7.0, "thresh_sacc": 40.0, 
     "notes": "current default"},
    {"set": 6, "thresh_fix": 4.0, "thresh_ms": 10.0, "thresh_sacc": 50.0, 
     "notes": "moderate fix, extended micro/sacc"},
    {"set": 7, "thresh_fix": 5.0, "thresh_ms": 6.0, "thresh_sacc": 30.0, 
     "notes": "liberal fix, strict micro/sacc"},
    {"set": 8, "thresh_fix": 5.0, "thresh_ms": 7.0, "thresh_sacc": 40.0, 
     "notes": "liberal fix, baseline micro/sacc"},
    {"set": 9, "thresh_fix": 5.0, "thresh_ms": 10.0, "thresh_sacc": 50.0, 
     "notes": "liberal fix, broad micro/sacc"}
]

# Raw data directory
RAW_DATA_DIR = args.raw_data_dir

# Create a base folder to store all results
BASE_RESULTS_DIR = "MPIIDPEye_parameter_search_results"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# Create a summary log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_file = os.path.join(BASE_RESULTS_DIR, f"summary_{timestamp}.csv")

# Path to the MPIIDPEye_IVT.py script in the parent directory
IVT_SCRIPT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "MPIIDPEye_IVT.py"))

# Create list to store summary metrics
summary_data = []

# Run each parameter set
for params in parameter_sets:
    set_number = params["set"]
    thresh_fix = params["thresh_fix"]
    thresh_ms = params["thresh_ms"]
    thresh_sacc = params["thresh_sacc"]
    notes = params["notes"]
    
    print(f"\n\n{'=' * 80}")
    print(f"Running parameter set {set_number}: thresh_fix={thresh_fix}, thresh_ms={thresh_ms}, thresh_sacc={thresh_sacc}")
    print(f"Notes: {notes}")
    print('=' * 80)
    
    # Create folder for this parameter set
    output_folder = os.path.join(BASE_RESULTS_DIR, f"set_{set_number}_fix{thresh_fix}_ms{thresh_ms}_sacc{thresh_sacc}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Create IVT output folder specific to this parameter set
    ivt_output_folder = os.path.join(output_folder, "IVT_output")
    os.makedirs(ivt_output_folder, exist_ok=True)
    
    # Create a custom test script to process multiple files
    if args.test_files > 0:
        test_script_path = os.path.join(output_folder, "process_test_files.py")
        with open(test_script_path, "w") as f:
            f.write(f"""import os
import glob
import subprocess

# Get list of CSV files from the raw data directory
files = glob.glob(os.path.join(r"{RAW_DATA_DIR}", "**", "*.csv"), recursive=True)
files = sorted(files)  # Sort to ensure consistent selection

# Process only the first {args.test_files} files
test_files = files[:{args.test_files}]
print(f"Processing {{len(test_files)}} test files")

# Path to the IVT script
ivt_script_path = r"{IVT_SCRIPT_PATH}"

# Process each test file with MPIIDPEye_IVT.py
for file_path in test_files:
    cmd = [
        "python", ivt_script_path,
        "--input_dir", os.path.dirname(file_path),
        "--output_dir", r"{ivt_output_folder}",
        "--thresh-fix", "{thresh_fix}",
        "--thresh-ms", "{thresh_ms}",
        "--thresh-sacc", "{thresh_sacc}"
    ]
    
    # Use only the specific file
    cmd.extend(["--single-file", os.path.basename(file_path)])
    
    print(f"Processing {{os.path.basename(file_path)}}...")
    subprocess.run(cmd)

print("All test files processed")
""")
        
        # Run the custom test script
        print(f"Running IVT tokenization on {args.test_files} files...")
        subprocess.run(["python", test_script_path])
    else:
        # Original approach - process with --test flag (just 1 file)
        ivt_command = [
            "python", IVT_SCRIPT_PATH,
            "--output_dir", ivt_output_folder,
            "--thresh-fix", str(thresh_fix),
            "--thresh-ms", str(thresh_ms),
            "--thresh-sacc", str(thresh_sacc),
            "--test"  # Process only one file for testing
        ]
        
        print("Running IVT tokenization...")
        subprocess.run(ivt_command)
    
    # Set up log file for quality check
    quality_check_log = os.path.join(output_folder, f"quality_check_log.txt")
    
    # Create the IVT quality check script for MPIIDPEye
    ivt_quality_check_path = os.path.join(output_folder, "IVT_quality_check_MPIIDPEye.py")
    with open(ivt_quality_check_path, "w") as f:
        f.write("""import os
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
""")
    
    # Run the quality check script we just created
    quality_check_command = [
        "python", ivt_quality_check_path,
        "--token-dir", ivt_output_folder,
        "--raw-data-dir", RAW_DATA_DIR,
        "--thresh-fix", str(thresh_fix),
        "--thresh-ms", str(thresh_ms),
        "--thresh-sacc", str(thresh_sacc),
        "--log-file", quality_check_log
    ]
    
    print("Running quality check...")
    subprocess.run(quality_check_command)
    
    # Extract key metrics from the quality check log
    fix_pct, ms_pct, sacc_pct = 0, 0, 0
    fix_time_pct, ms_time_pct, sacc_time_pct = 0, 0, 0
    
    try:
        with open(quality_check_log, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "By token count:" in line and i+3 < len(lines):
                    fix_line = lines[i+1]
                    ms_line = lines[i+2]
                    sacc_line = lines[i+3]
                    
                    try:
                        fix_pct = float(fix_line.split('%')[0].split(':')[-1].strip())
                        ms_pct = float(ms_line.split('%')[0].split(':')[-1].strip())
                        sacc_pct = float(sacc_line.split('%')[0].split(':')[-1].strip())
                    except (ValueError, IndexError):
                        print("Error parsing token count percentages")
                
                if "By time spent:" in line and i+3 < len(lines):
                    fix_time_line = lines[i+1]
                    ms_time_line = lines[i+2]
                    sacc_time_line = lines[i+3]
                    
                    try:
                        fix_time_pct = float(fix_time_line.split('%')[0].split(':')[-1].strip())
                        ms_time_pct = float(ms_time_line.split('%')[0].split(':')[-1].strip())
                        sacc_time_pct = float(sacc_time_line.split('%')[0].split(':')[-1].strip())
                    except (ValueError, IndexError):
                        print("Error parsing time spent percentages")
    except Exception as e:
        print(f"Error reading quality check log: {e}")
    
    # Record summary information
    summary_data.append({
        "set": set_number,
        "thresh_fix": thresh_fix,
        "thresh_ms": thresh_ms,
        "thresh_sacc": thresh_sacc,
        "notes": notes,
        "fix_token_pct": fix_pct,
        "ms_token_pct": ms_pct,
        "sacc_token_pct": sacc_pct,
        "fix_time_pct": fix_time_pct,
        "ms_time_pct": ms_time_pct,
        "sacc_time_pct": sacc_time_pct,
        "output_folder": output_folder,
        "files_processed": args.test_files if args.test_files > 0 else 1
    })
    
    print(f"Completed parameter set {set_number}")

# Create a summary dataframe and save it
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_file, index=False)

print(f"\n\nParameter search complete! Summary saved to {summary_file}")
print(f"Processed {args.test_files} files for each parameter set")
print("\nSummary of results:")
print(summary_df[["set", "thresh_fix", "thresh_ms", "thresh_sacc", 
                 "fix_token_pct", "ms_token_pct", "sacc_token_pct"]]) 