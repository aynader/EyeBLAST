import os
import glob
import subprocess

# Get list of CSV files from the raw data directory
files = glob.glob(os.path.join(r"C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/MPIIDPEye/Eye_Tracking_Data", "**", "*.csv"), recursive=True)
files = sorted(files)  # Sort to ensure consistent selection

# Process only the first 10 files
test_files = files[:10]
print(f"Processing {len(test_files)} test files")

# Path to the IVT script
ivt_script_path = r"C:\Users\aynad\Desktop\Home\TUM\Research\EyeBLAST\EyeBLASTv5 - VALIDATION\MPIIDPEye_IVT.py"

# Process each test file with MPIIDPEye_IVT.py
for file_path in test_files:
    cmd = [
        "python", ivt_script_path,
        "--input_dir", os.path.dirname(file_path),
        "--output_dir", r"MPIIDPEye_parameter_search_results\set_1_fix3.0_ms6.0_sacc30.0\IVT_output",
        "--thresh-fix", "3.0",
        "--thresh-ms", "6.0",
        "--thresh-sacc", "30.0"
    ]
    
    # Use only the specific file
    cmd.extend(["--single-file", os.path.basename(file_path)])
    
    print(f"Processing {os.path.basename(file_path)}...")
    subprocess.run(cmd)

print("All test files processed")
