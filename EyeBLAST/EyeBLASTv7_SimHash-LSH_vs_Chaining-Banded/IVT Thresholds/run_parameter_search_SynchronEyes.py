import os
import subprocess
import pandas as pd
import argparse
import json
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run parameter search for IVT tokenization on SynchronEyes data")
parser.add_argument("--test-files", type=int, default=10,
                    help="Number of files to process in test mode (default: 10)")
parser.add_argument("--raw-data-dir", default="C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/SynchronEyes/data",
                    help="Directory containing the raw SynchronEyes CSV data")
parser.add_argument("--data-source", default="S02_EyeLink",
                    choices=["S01_MindLink", "S02_MindLink", "S02_EyeLink"],
                    help="Data source to process (S01_MindLink, S02_MindLink, S02_EyeLink)")
parser.add_argument("--target-fix", type=float, default=89.03,
                    help="Target percentage for fixations (default: 89.03)")
parser.add_argument("--target-ms", type=float, default=7.37,
                    help="Target percentage for microsaccades (default: 7.37)")
parser.add_argument("--target-sacc", type=float, default=3.60,
                    help="Target percentage for saccades (default: 3.60)")
args = parser.parse_args()

# Define parameter sets to test - adjusted for EyeLink data with higher thresholds and lower scale factors
parameter_sets = [
    # Variations around the successful parameters (50, 100, 200)
    {"set": 1, "thresh_fix": 50.0, "thresh_ms": 100.0, "thresh_sacc": 200.0, "scale_factor": 1.0, "smooth_window": 7,
     "notes": "Initial successful parameters"},
    {"set": 2, "thresh_fix": 60.0, "thresh_ms": 120.0, "thresh_sacc": 240.0, "scale_factor": 1.0, "smooth_window": 7,
     "notes": "Higher thresholds, same scale"},
    {"set": 3, "thresh_fix": 40.0, "thresh_ms": 80.0, "thresh_sacc": 160.0, "scale_factor": 1.0, "smooth_window": 7,
     "notes": "Lower thresholds, same scale"},
    {"set": 4, "thresh_fix": 50.0, "thresh_ms": 100.0, "thresh_sacc": 200.0, "scale_factor": 0.8, "smooth_window": 7,
     "notes": "Same thresholds, lower scale"},
    {"set": 5, "thresh_fix": 50.0, "thresh_ms": 100.0, "thresh_sacc": 200.0, "scale_factor": 1.2, "smooth_window": 7,
     "notes": "Same thresholds, higher scale"},
    # Variations to increase fixation percentage
    {"set": 6, "thresh_fix": 70.0, "thresh_ms": 100.0, "thresh_sacc": 200.0, "scale_factor": 1.0, "smooth_window": 7,
     "notes": "Higher fixation threshold only"},
    {"set": 7, "thresh_fix": 80.0, "thresh_ms": 120.0, "thresh_sacc": 200.0, "scale_factor": 1.0, "smooth_window": 7,
     "notes": "Higher fixation and microsaccade thresholds"},
    {"set": 8, "thresh_fix": 90.0, "thresh_ms": 130.0, "thresh_sacc": 220.0, "scale_factor": 1.0, "smooth_window": 7,
     "notes": "Much higher fixation threshold with adjusted others"},
    {"set": 9, "thresh_fix": 100.0, "thresh_ms": 140.0, "thresh_sacc": 240.0, "scale_factor": 0.9, "smooth_window": 9,
     "notes": "Very high thresholds with lower scale and more smoothing"},
    # Variations with different smoothing windows
    {"set": 10, "thresh_fix": 50.0, "thresh_ms": 100.0, "thresh_sacc": 200.0, "scale_factor": 1.0, "smooth_window": 5,
     "notes": "Less smoothing"},
    {"set": 11, "thresh_fix": 50.0, "thresh_ms": 100.0, "thresh_sacc": 200.0, "scale_factor": 1.0, "smooth_window": 9,
     "notes": "More smoothing"},
    {"set": 12, "thresh_fix": 60.0, "thresh_ms": 110.0, "thresh_sacc": 220.0, "scale_factor": 0.9, "smooth_window": 11,
     "notes": "Higher thresholds with maximum smoothing"},
    # More extreme variations
    {"set": 13, "thresh_fix": 120.0, "thresh_ms": 160.0, "thresh_sacc": 250.0, "scale_factor": 0.8, "smooth_window": 9,
     "notes": "Very high thresholds with lower scale"},
    {"set": 14, "thresh_fix": 150.0, "thresh_ms": 200.0, "thresh_sacc": 300.0, "scale_factor": 0.7, "smooth_window": 11,
     "notes": "Maximum thresholds with lowest scale"},
    {"set": 15, "thresh_fix": 80.0, "thresh_ms": 150.0, "thresh_sacc": 250.0, "scale_factor": 0.5, "smooth_window": 9,
     "notes": "High thresholds with very low scale"}
]

# Raw data directory
RAW_DATA_DIR = args.raw_data_dir

# Create a base folder to store all results
BASE_RESULTS_DIR = f"parameter_search_results_{args.data_source}"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# Create a summary log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_file = os.path.join(BASE_RESULTS_DIR, f"summary_{timestamp}.csv")

# Create list to store summary metrics
summary_data = []

# Target distribution
target_fix = args.target_fix
target_ms = args.target_ms
target_sacc = args.target_sacc

print(f"Searching for parameters to match target distribution:")
print(f"  Fixations: {target_fix:.2f}%")
print(f"  Microsaccades: {target_ms:.2f}%")
print(f"  Saccades: {target_sacc:.2f}%")

# Run each parameter set
for params in parameter_sets:
    set_number = params["set"]
    thresh_fix = params["thresh_fix"]
    thresh_ms = params["thresh_ms"]
    thresh_sacc = params["thresh_sacc"]
    scale_factor = params["scale_factor"]
    smooth_window = params["smooth_window"]
    notes = params["notes"]
    
    print(f"\n\n{'=' * 80}")
    print(f"Running parameter set {set_number}")
    print(f"thresh_fix={thresh_fix}, thresh_ms={thresh_ms}, thresh_sacc={thresh_sacc}")
    print(f"scale_factor={scale_factor}, smooth_window={smooth_window}")
    print(f"Notes: {notes}")
    print('=' * 80)
    
    # Create folder for this parameter set
    output_folder = os.path.join(BASE_RESULTS_DIR, 
                               f"set_{set_number}_fix{thresh_fix}_ms{thresh_ms}_sacc{thresh_sacc}_scale{scale_factor}_smooth{smooth_window}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Create IVT output folder specific to this parameter set
    ivt_output_folder = os.path.join(output_folder, "IVT_output")
    os.makedirs(ivt_output_folder, exist_ok=True)
    
    # Create a temporary modified version of SynchronEyes_IVT.py with updated scale_factor
    temp_script_path = os.path.join(output_folder, "temp_SynchronEyes_IVT.py")
    
    # Read the original script
    with open("SynchronEyes_IVT.py", "r") as f:
        script_content = f.read()
    
    # Replace the scale_factor value
    script_content = script_content.replace("scale_factor = 1.0", f"scale_factor = {scale_factor}")
    
    # Write the modified script
    with open(temp_script_path, "w") as f:
        f.write(script_content)
    
    # Process a few test files directly instead of creating a separate script
    if args.test_files > 0:
        import glob
        
        # Get list of CSV files from the raw data directory matching the data source
        pattern = f"{args.data_source}_*.csv"
        files = glob.glob(os.path.join(RAW_DATA_DIR, pattern))
        files = sorted(files)  # Sort to ensure consistent selection
        
        # Process only the first n files
        test_files = files[:args.test_files]
        print(f"Processing {len(test_files)} test files")
        
        # Lists to store percentage values
        fixation_pcts = []
        ms_pcts = []
        saccade_pcts = []
        
        # Process each test file with the modified SynchronEyes_IVT.py
        for file_path in test_files:
            cmd = [
                "python", temp_script_path,
                "--single-file", os.path.basename(file_path),
                "--input_dir", RAW_DATA_DIR,
                "--output_dir", ivt_output_folder,
                "--thresh-fix", str(thresh_fix),
                "--thresh-ms", str(thresh_ms),
                "--thresh-sacc", str(thresh_sacc),
                "--smooth-window", str(smooth_window),
                "--data-source", args.data_source
            ]
            
            print(f"Processing {os.path.basename(file_path)}...")
            # Use shell=True for Windows to better capture output
            process = subprocess.run(" ".join(cmd), capture_output=True, text=True, shell=True)
            
            # Debug: Print the full output
            print("Subprocess stdout:")
            print(process.stdout)
            
            if "Token distribution:" in process.stdout:
                # Parse the distribution line and extract percentages
                for line in process.stdout.split('\n'):
                    if "Token distribution:" in line:
                        try:
                            # More robust parsing
                            parts = line.split("Token distribution:")[1].strip()
                            fix_part = parts.split("Fixations:")[1].split("%,")[0].strip()
                            ms_part = parts.split("Microsaccades:")[1].split("%,")[0].strip()
                            sacc_part = parts.split("Saccades:")[1].split("%")[0].strip()
                            
                            fixation_pcts.append(float(fix_part))
                            ms_pcts.append(float(ms_part))
                            saccade_pcts.append(float(sacc_part))
                            
                            print(f"Extracted: Fix={fix_part}%, MS={ms_part}%, Sacc={sacc_part}%")
                        except Exception as e:
                            print(f"Error parsing distribution: {e}")
                            print(f"Line: {line}")
            else:
                print("Warning: Could not find 'Token distribution:' in the output")
                # Check stderr for errors
                if process.stderr:
                    print("Subprocess stderr:")
                    print(process.stderr)
        
        # Calculate averages
        avg_fix = sum(fixation_pcts) / len(fixation_pcts) if fixation_pcts else 0
        avg_ms = sum(ms_pcts) / len(ms_pcts) if ms_pcts else 0
        avg_sacc = sum(saccade_pcts) / len(saccade_pcts) if saccade_pcts else 0
        
        # Write summary to file
        with open(os.path.join(output_folder, 'distribution_summary.json'), "w") as f:
            json.dump({
                "fixation_pct": avg_fix, 
                "microsaccade_pct": avg_ms, 
                "saccade_pct": avg_sacc,
                "num_files_processed": len(test_files),
                "files_with_data": len(fixation_pcts)
            }, f, indent=2)
        
        print(f"Average distribution: Fixations: {avg_fix:.2f}%, Microsaccades: {avg_ms:.2f}%, Saccades: {avg_sacc:.2f}%")
    
    # Read the distribution summary
    summary_path = os.path.join(output_folder, 'distribution_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            distribution = json.load(f)
        
        fix_pct = distribution.get("fixation_pct", 0)
        ms_pct = distribution.get("microsaccade_pct", 0)
        sacc_pct = distribution.get("saccade_pct", 0)
    else:
        fix_pct, ms_pct, sacc_pct = 0, 0, 0
    
    # Calculate distance from target distribution
    target_dist = [target_fix, target_ms, target_sacc]
    actual_dist = [fix_pct, ms_pct, sacc_pct]
    
    # Mean squared error
    mse = sum([(t - a)**2 for t, a in zip(target_dist, actual_dist)]) / 3
    
    # Record summary information
    summary_data.append({
        "set": set_number,
        "thresh_fix": thresh_fix,
        "thresh_ms": thresh_ms,
        "thresh_sacc": thresh_sacc,
        "scale_factor": scale_factor,
        "smooth_window": smooth_window,
        "notes": notes,
        "fix_pct": fix_pct,
        "ms_pct": ms_pct,
        "sacc_pct": sacc_pct,
        "mse": mse,
        "output_folder": output_folder,
        "files_processed": args.test_files if args.test_files > 0 else 1
    })
    
    print(f"Completed parameter set {set_number}")
    print(f"Distribution: Fixations: {fix_pct:.2f}%, Microsaccades: {ms_pct:.2f}%, Saccades: {sacc_pct:.2f}%")
    print(f"Distance from target (MSE): {mse:.4f}")

# Create a summary dataframe and save it
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_file, index=False)

# Find the best set (lowest MSE)
if summary_df.empty:
    print("No results to analyze!")
else:
    best_set = summary_df.loc[summary_df['mse'].idxmin()]
    
    print(f"\n\nParameter search complete! Summary saved to {summary_file}")
    print(f"Processed {args.test_files} files for each parameter set")
    
    print("\nBest parameter set:")
    print(f"Set {best_set['set']} - MSE: {best_set['mse']:.4f}")
    print(f"thresh_fix={best_set['thresh_fix']}, thresh_ms={best_set['thresh_ms']}, thresh_sacc={best_set['thresh_sacc']}")
    print(f"scale_factor={best_set['scale_factor']}, smooth_window={best_set['smooth_window']}")
    print(f"Distribution: Fixations: {best_set['fix_pct']:.2f}%, Microsaccades: {best_set['ms_pct']:.2f}%, Saccades: {best_set['sacc_pct']:.2f}%")
    print(f"Target: Fixations: {target_fix:.2f}%, Microsaccades: {target_ms:.2f}%, Saccades: {target_sacc:.2f}%")
    
    print("\nSummary of top 5 results (by MSE):")
    top5 = summary_df.sort_values(by='mse').head(5)
    print(top5[["set", "thresh_fix", "thresh_ms", "thresh_sacc", "scale_factor", "smooth_window", 
               "fix_pct", "ms_pct", "sacc_pct", "mse"]])

# Clean up temporary files
for params in parameter_sets:
    output_folder = os.path.join(BASE_RESULTS_DIR, 
                               f"set_{params['set']}_fix{params['thresh_fix']}_ms{params['thresh_ms']}_sacc{params['thresh_sacc']}_scale{params['scale_factor']}_smooth{params['smooth_window']}")
    temp_script = os.path.join(output_folder, "temp_SynchronEyes_IVT.py")
    if os.path.exists(temp_script):
        os.remove(temp_script) 