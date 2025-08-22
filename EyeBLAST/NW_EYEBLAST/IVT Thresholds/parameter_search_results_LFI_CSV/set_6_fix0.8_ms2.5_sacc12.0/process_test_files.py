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
        "--output_dir", "parameter_search_results_LFI_CSV\set_6_fix0.8_ms2.5_sacc12.0\IVT_output",
        "--thresh-fix", "0.8",
        "--thresh-ms", "2.5",
        "--thresh-sacc", "12.0"
    ]
    
    # Use only the specific file
    cmd.extend(["--single-file", os.path.basename(file_path)])
    
    print(f"Processing {os.path.basename(file_path)}...")
    subprocess.run(cmd)

print("All test files processed")
