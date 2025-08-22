import os
import glob
import subprocess

# Get list of CSV files from the raw data directory
files = glob.glob(os.path.join("C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/GazeBase_CSVs", "**", "*.csv"), recursive=True)
files = sorted(files)  # Sort to ensure consistent selection

# Process only the first 200 files
test_files = files[:200]
print(f"Processing {len(test_files)} test files")

# Process each test file with GazeBase_IVT.py
for file_path in test_files:
    cmd = [
        "python", "../GazeBase_IVT.py",
        "--input_dir", os.path.dirname(file_path),
        "--output_dir", "parameter_search_results_GazeBase\set_4_fix4.0_ms6.0_sacc30.0\IVT_output",
        "--thresh-fix", "4.0",
        "--thresh-ms", "6.0",
        "--thresh-sacc", "30.0"
    ]
    
    # Use only the specific file
    cmd.extend(["--single-file", os.path.basename(file_path)])
    
    print(f"Processing {os.path.basename(file_path)}...")
    subprocess.run(cmd)

print("All test files processed")
