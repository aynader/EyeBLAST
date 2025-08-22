import os
import subprocess
import pandas as pd
import argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run parameter search for IVT tokenization")
parser.add_argument("--test-files", type=int, default=10,
                    help="Number of files to process in test mode (default: 10)")
parser.add_argument("--raw-data-dir", default="C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/GazeBaseVR/data",
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
BASE_RESULTS_DIR = "parameter_search_results"
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

# Get list of CSV files from the raw data directory
files = glob.glob(os.path.join("{RAW_DATA_DIR}", "**", "*.csv"), recursive=True)
files = sorted(files)  # Sort to ensure consistent selection

# Process only the first {args.test_files} files
test_files = files[:{args.test_files}]
print(f"Processing {{len(test_files)}} test files")

# Process each test file with IVT.py
for file_path in test_files:
    cmd = [
        "python", "IVT.py",
        "--input_dir", os.path.dirname(file_path),
        "--output_dir", "{ivt_output_folder}",
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
            "python", "IVT.py",
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
    
    # Run IVT_quality_check.py on the results
    quality_check_command = [
        "python", "IVT_quality_check.py",
        "--token-dir", ivt_output_folder,
        "--raw-data-dir", RAW_DATA_DIR,
        "--thresh-fix", str(thresh_fix),
        "--thresh-ms", str(thresh_ms),
        "--thresh-sacc", str(thresh_sacc),
        "--log-file", quality_check_log
    ]
    
    print("Running quality check...")
    subprocess.run(quality_check_command)
    
    # Extract some key metrics from the quality check log
    # This is a simplified example, you might need to adjust based on actual log format
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