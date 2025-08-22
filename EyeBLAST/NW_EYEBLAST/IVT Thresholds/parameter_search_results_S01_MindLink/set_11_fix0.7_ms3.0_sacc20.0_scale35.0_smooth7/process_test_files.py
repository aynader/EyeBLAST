import os
import glob
import subprocess
import json

# Get list of CSV files from the raw data directory matching the data source
pattern = "S01_MindLink_*.csv"
files = glob.glob(os.path.join("C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/SynchronEyes/data", pattern))
files = sorted(files)  # Sort to ensure consistent selection

# Process only the first 100 files
test_files = files[:100]
print(f"Processing {len(test_files)} test files")

# Lists to store percentage values
fixation_pcts = []
ms_pcts = []
saccade_pcts = []

# Process each test file with the modified SynchronEyes_IVT.py
for file_path in test_files:
    cmd = [
        "python", "parameter_search_results_S01_MindLink\set_11_fix0.7_ms3.0_sacc20.0_scale35.0_smooth7\temp_SynchronEyes_IVT.py",
        "--single-file", os.path.basename(file_path),
        "--input_dir", "C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/SynchronEyes/data",
        "--output_dir", "parameter_search_results_S01_MindLink\set_11_fix0.7_ms3.0_sacc20.0_scale35.0_smooth7\IVT_output",
        "--thresh-fix", "0.7",
        "--thresh-ms", "3.0",
        "--thresh-sacc", "20.0",
        "--smooth-window", "7",
        "--data-source", "S01_MindLink"
    ]
    
    print(f"Processing {os.path.basename(file_path)}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract percentages from output
    output = result.stdout
    lines = output.split('\n')
    for line in lines:
        if "Token distribution:" in line:
            parts = line.split("Token distribution:")[1].strip()
            fix_part = parts.split("Fixations:")[1].split("%")[0].strip()
            ms_part = parts.split("Microsaccades:")[1].split("%")[0].strip()
            sacc_part = parts.split("Saccades:")[1].split("%")[0].strip()
            
            fixation_pcts.append(float(fix_part))
            ms_pcts.append(float(ms_part))
            saccade_pcts.append(float(sacc_part))

# Calculate averages
avg_fix = sum(fixation_pcts) / len(fixation_pcts) if fixation_pcts else 0
avg_ms = sum(ms_pcts) / len(ms_pcts) if ms_pcts else 0
avg_sacc = sum(saccade_pcts) / len(saccade_pcts) if saccade_pcts else 0

# Write summary to file
with open("parameter_search_results_S01_MindLink\set_11_fix0.7_ms3.0_sacc20.0_scale35.0_smooth7\distribution_summary.json", "w") as f:
    json.dump({"fixation_pct": avg_fix, "microsaccade_pct": avg_ms, "saccade_pct": avg_sacc}, f, indent=2)

print("All test files processed")
print(f"Average distribution: Fixations: {avg_fix:.2f}%, Microsaccades: {avg_ms:.2f}%, Saccades: {avg_sacc:.2f}%")
