import os
import glob
import subprocess

# Get list of CSV files from the raw data directory (excluding TrialVars files)
files = glob.glob(os.path.join("C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/JuDo1000", "*.csv"))
files = [f for f in files if "TrialVars" not in f]
files = sorted(files)  # Sort to ensure consistent selection

# Process only the first 10 files
test_files = files[:10]
print(f"Processing {len(test_files)} test files")

# Process each test file with JuDo1000_IVT.py
for file_path in test_files:
    cmd = [
        "python", "../JuDo1000_IVT.py",
        "--input_dir", os.path.dirname(file_path),
        "--output_dir", "parameter_search_results_JuDo\set_9_fix200000_ms700000_sacc2500000\IVT_output",
        "--thresh-fix", "200000",
        "--thresh-ms", "700000",
        "--thresh-sacc", "2500000",
        "--exclude-trialvars"
    ]
    
    # Use only the specific file
    cmd.extend(["--single-file", os.path.basename(file_path)])
    
    print(f"Processing {os.path.basename(file_path)}...")
    subprocess.run(cmd)

print("All test files processed")
