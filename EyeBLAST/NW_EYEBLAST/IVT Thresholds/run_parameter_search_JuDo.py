import os
import subprocess
import pandas as pd
import argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run parameter search for IVT tokenization on JuDo1000 dataset")
parser.add_argument("--test-files", type=int, default=10,
                    help="Number of files to process in test mode (default: 10)")
parser.add_argument("--raw-data-dir", default="C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/JuDo1000",
                    help="Directory containing the raw CSV gaze data")
args = parser.parse_args()

# Define parameter sets to test (pixel-based thresholds for JuDo1000)
parameter_sets = [
    {"set": 1, "thresh_fix": 50000, "thresh_ms": 200000, "thresh_sacc": 1000000, 
     "notes": "strict fix → lots of events; micro and saccade only the fastest"},
    {"set": 2, "thresh_fix": 50000, "thresh_ms": 300000, "thresh_sacc": 1500000, 
     "notes": "strict fix, baseline micro/sacc"},
    {"set": 3, "thresh_fix": 50000, "thresh_ms": 500000, "thresh_sacc": 2000000, 
     "notes": "strict fix, broad micro window"},
    {"set": 4, "thresh_fix": 100000, "thresh_ms": 200000, "thresh_sacc": 1000000, 
     "notes": "moderate fix, strict micro/sacc"},
    {"set": 5, "thresh_fix": 100000, "thresh_ms": 500000, "thresh_sacc": 2000000, 
     "notes": "current default"},
    {"set": 6, "thresh_fix": 100000, "thresh_ms": 700000, "thresh_sacc": 2500000, 
     "notes": "moderate fix, extended micro/sacc"},
    {"set": 7, "thresh_fix": 200000, "thresh_ms": 300000, "thresh_sacc": 1000000, 
     "notes": "liberal fix, strict micro/sacc"},
    {"set": 8, "thresh_fix": 200000, "thresh_ms": 500000, "thresh_sacc": 2000000, 
     "notes": "liberal fix, baseline micro/sacc"},
    {"set": 9, "thresh_fix": 200000, "thresh_ms": 700000, "thresh_sacc": 2500000, 
     "notes": "liberal fix, broad micro/sacc"}
]

# Raw data directory
RAW_DATA_DIR = args.raw_data_dir

# Create a base folder to store all results
BASE_RESULTS_DIR = "parameter_search_results_JuDo"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# Create a summary log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_file = os.path.join(BASE_RESULTS_DIR, f"summary_{timestamp}.csv")

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

# Get list of CSV files from the raw data directory (excluding TrialVars files)
files = glob.glob(os.path.join("{RAW_DATA_DIR}", "*.csv"))
files = [f for f in files if "TrialVars" not in f]
files = sorted(files)  # Sort to ensure consistent selection

# Process only the first {args.test_files} files
test_files = files[:{args.test_files}]
print(f"Processing {{len(test_files)}} test files")

# Process each test file with JuDo1000_IVT.py
for file_path in test_files:
    cmd = [
        "python", "../JuDo1000_IVT.py",
        "--input_dir", os.path.dirname(file_path),
        "--output_dir", "{ivt_output_folder}",
        "--thresh-fix", "{thresh_fix}",
        "--thresh-ms", "{thresh_ms}",
        "--thresh-sacc", "{thresh_sacc}",
        "--exclude-trialvars"
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
            "python", "../JuDo1000_IVT.py",
            "--output_dir", ivt_output_folder,
            "--thresh-fix", str(thresh_fix),
            "--thresh-ms", str(thresh_ms),
            "--thresh-sacc", str(thresh_sacc),
            "--test",  # Process only one file for testing
            "--exclude-trialvars"
        ]
        
        print("Running IVT tokenization...")
        subprocess.run(ivt_command)
    
    # Set up log file for quality check
    quality_check_log = os.path.join(output_folder, f"quality_check_log.txt")
    
    # Create a custom quality check script for JuDo1000
    quality_check_script = os.path.join(output_folder, "judo_quality_check.py")
    with open(quality_check_script, "w") as f:
        f.write(f"""
import os
import json
import glob
import pandas as pd
import numpy as np

def analyze_tokens(token_dir, raw_data_dir, thresh_fix, thresh_ms, thresh_sacc, log_file):
    \"\"\"
    Analyze IVT tokenization results for JuDo1000 dataset.
    \"\"\"
    
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
        f.write(f"JuDo1000 IVT Quality Check Results\\n")
        f.write(f"Parameters: thresh_fix={{thresh_fix}}, thresh_ms={{thresh_ms}}, thresh_sacc={{thresh_sacc}}\\n")
        f.write(f"Files processed: {{len(token_files)}}\\n")
        f.write(f"Total tokens: {{total_tokens}}\\n\\n")
        
        f.write(f"By token count:\\n")
        f.write(f"  Fixations: {{fix_pct:.2f}}%\\n")
        f.write(f"  Microsaccades: {{ms_pct:.2f}}%\\n")
        f.write(f"  Saccades: {{sacc_pct:.2f}}%\\n\\n")
        
        f.write(f"By time spent:\\n")
        f.write(f"  Fixations: {{fix_time_pct:.2f}}%\\n")
        f.write(f"  Microsaccades: {{ms_time_pct:.2f}}%\\n")
        f.write(f"  Saccades: {{sacc_time_pct:.2f}}%\\n")
    
    print(f"Quality check complete. Results saved to {{log_file}}")
    return fix_pct, ms_pct, sacc_pct, fix_time_pct, ms_time_pct, sacc_time_pct

# Run the analysis
fix_pct, ms_pct, sacc_pct, fix_time_pct, ms_time_pct, sacc_time_pct = analyze_tokens(
    "{ivt_output_folder}",
    "{RAW_DATA_DIR}",
    {thresh_fix},
    {thresh_ms},
    {thresh_sacc},
    "{quality_check_log}"
)
""")
    
    print("Running quality check...")
    subprocess.run(["python", quality_check_script])
    
    # Extract metrics from the quality check log
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