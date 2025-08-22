import os
import glob
import subprocess

# Get list of CSV files from the raw data directory
files = glob.glob(os.path.join("C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/LFI_CSV", "*.csv"))
files = sorted(files)  # Sort to ensure consistent selection

# Process only the first 10 files
test_files = files[:10]
print(f"Processing {len(test_files)} test files")

# Process each test file with LFI_CSV_IVT.py
for file_path in test_files:
    cmd = [
        "python", "C:/Users/aynad/Desktop/Home/TUM/Research/Ongoing_Projects/NW_EYEBLAST/EyeBLAST/LFI_CSV_IVT.py",
        "--input_dir", os.path.dirname(file_path),
        "--output_dir", "parameter_search_results_LFI_CSV\set_3_fix0.3_ms1.0_sacc4.0\IVT_output",
        "--thresh-fix", "0.3",
        "--thresh-ms", "1.0",
        "--thresh-sacc", "4.0"
    ]
    
    # Use only the specific file
    cmd.extend(["--single-file", os.path.basename(file_path)])
    
    print(f"Processing {os.path.basename(file_path)}...")
    subprocess.run(cmd)

print("All test files processed")
