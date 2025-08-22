import os
import subprocess
import pandas as pd
import argparse
import json
import glob
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run parameter search for IVT tokenization on GazeBase")
parser.add_argument("--test-files", type=int, default=10,
                    help="Number of files to process in test mode (default: 10)")
parser.add_argument("--raw-data-dir", default="C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/GazeBase_CSVs",
                    help="Directory containing the raw CSV gaze data")
args = parser.parse_args()

# Define parameter sets to test
parameter_sets = [
    {"set": 1, "thresh_fix": 3.0, "thresh_ms": 6.0, "thresh_sacc": 30.0, 
     "notes": "very strict fix - lots of events; micro and saccade only the fastest"},
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
BASE_RESULTS_DIR = "parameter_search_results_GazeBase"
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

# Process each test file with GazeBase_IVT.py
for file_path in test_files:
    cmd = [
        "python", "../GazeBase_IVT.py",
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
            "python", "../GazeBase_IVT.py",
            "--output_dir", ivt_output_folder,
            "--thresh-fix", str(thresh_fix),
            "--thresh-ms", str(thresh_ms),
            "--thresh-sacc", str(thresh_sacc),
            "--test"  # Process only one file for testing
        ]
        
        print("Running IVT tokenization...")
        subprocess.run(ivt_command)
    
    # Analyze token distribution for this parameter set
    token_counts = {'F': 0, 'M': 0, 'S': 0}
    total_tokens = 0
    
    # Process all token files in the output folder
    token_files = glob.glob(os.path.join(ivt_output_folder, "*_tokens.json"))
    for token_file in token_files:
        try:
            with open(token_file, 'r') as f:
                tokens = json.load(f)
                for token in tokens:
                    if token.startswith('F'):
                        token_counts['F'] += 1
                    elif token.startswith('M'):
                        token_counts['M'] += 1
                    elif token.startswith('S'):
                        token_counts['S'] += 1
                total_tokens += len(tokens)
        except Exception as e:
            print(f"Error analyzing token file {token_file}: {e}")
    
    # Calculate percentages
    if total_tokens > 0:
        fix_pct = (token_counts['F'] / total_tokens) * 100
        ms_pct = (token_counts['M'] / total_tokens) * 100
        sacc_pct = (token_counts['S'] / total_tokens) * 100
    else:
        fix_pct, ms_pct, sacc_pct = 0, 0, 0
    
    # Write analysis to a log file
    log_file = os.path.join(output_folder, "analysis.log")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Parameter Set {set_number} Analysis\n")
        f.write(f"=============================\n")
        f.write(f"thresh_fix: {thresh_fix}, thresh_ms: {thresh_ms}, thresh_sacc: {thresh_sacc}\n")
        f.write(f"Notes: {notes}\n\n")
        f.write(f"Token Distribution:\n")
        f.write(f"  Fixations (F): {token_counts['F']} ({fix_pct:.2f}%)\n")
        f.write(f"  Microsaccades (M): {token_counts['M']} ({ms_pct:.2f}%)\n")
        f.write(f"  Saccades (S): {token_counts['S']} ({sacc_pct:.2f}%)\n")
        f.write(f"  Total Tokens: {total_tokens}\n")
    
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
        "fix_tokens": token_counts['F'],
        "ms_tokens": token_counts['M'],
        "sacc_tokens": token_counts['S'],
        "total_tokens": total_tokens,
        "output_folder": output_folder,
        "files_processed": len(token_files)
    })
    
    print(f"Completed parameter set {set_number}")
    print(f"Token Distribution: F: {fix_pct:.2f}%, M: {ms_pct:.2f}%, S: {sacc_pct:.2f}%")

# Create a summary dataframe and save it
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_file, index=False)

print(f"\n\nParameter search complete! Summary saved to {summary_file}")
print(f"Processed {args.test_files} files for each parameter set")
print("\nSummary of results:")
print(summary_df[["set", "thresh_fix", "thresh_ms", "thresh_sacc", 
                 "fix_token_pct", "ms_token_pct", "sacc_token_pct"]]) 

# Find best parameter set based on criteria
# Example: highest fixation percentage
best_fix_set = summary_df.loc[summary_df['fix_token_pct'].idxmax()]
print(f"\nParameter set with highest fixation percentage: Set {best_fix_set['set']}")
print(f"  thresh_fix={best_fix_set['thresh_fix']}, thresh_ms={best_fix_set['thresh_ms']}, thresh_sacc={best_fix_set['thresh_sacc']}")
print(f"  fixations: {best_fix_set['fix_token_pct']:.2f}%, microsaccades: {best_fix_set['ms_token_pct']:.2f}%, saccades: {best_fix_set['sacc_token_pct']:.2f}%") 