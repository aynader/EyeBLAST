import os
import glob
import json
import argparse

import numpy as np
import pandas as pd

def ivt_tokenize(df,
                 vel_col='velocity',
                 time_col='t',                 # Changed from 'Timestamp' to 't'
                 x_col='x',                    # Changed from 'Gaze_X' to 'x'
                 y_col='y',                    # Changed from 'Gaze_Y' to 'y'
                 smooth_window=9,              # Smaller smoothing window to maintain sensitivity
                 thresh_fix=100.0,               # Lower threshold to detect more fixations
                 thresh_ms=140.0,                # Threshold for microsaccades
                 thresh_sacc=240.0,             # Higher threshold for saccades
                 dur_fix_min=0.080,
                 dur_thresh=0.150,
                 fix1_max=0.3,
                 fix2_max=0.5,
                 ms1_max=4.0,
                 ms2_max=7.0,
                 s1_max=25.0,
                 s2_max=60.0):
    """
    I-VT (Velocity-Threshold Identification) tokenization.
    Classifies gaze samples into fixations (F), micro-saccades (M), and saccades (S)
    based on angular velocity thresholds.
    """

    # compute time steps - t is in milliseconds, convert to seconds for calculations
    dt = df[time_col].diff().fillna(1).values / 1000.0
    
    # Handle potential NaN values in gaze coordinates
    x_values = df[x_col].fillna(method='ffill').fillna(method='bfill').values
    y_values = df[y_col].fillna(method='ffill').fillna(method='bfill').values
    
    # compute velocities
    dx = np.diff(x_values, prepend=x_values[0])
    dy = np.diff(y_values, prepend=y_values[0])
    
    # Use a much smaller scale factor for EyeLink data
    # EyeLink values already represent visual degrees, so we need a smaller scale
    scale_factor = 0.9  # Significantly reduced scale factor
    vel = np.sqrt(dx**2 + dy**2) * scale_factor / dt
    
    # smooth velocity with appropriate window
    vel_smooth = pd.Series(vel).rolling(window=smooth_window,
                                        center=True,
                                        min_periods=1).mean().values
    df['velocity'] = vel_smooth

    # Initial classification of samples
    classifications = []
    for v in vel_smooth:
        if v <= thresh_fix:
            classifications.append('F')  # Fixation
        elif v <= thresh_ms:
            classifications.append('M')  # Microsaccade
        else:
            classifications.append('S')  # Saccade

    # Post-processing: ensure physiological constraints
    # 1. Ensure fixations have a minimum duration
    # 2. Merge adjacent fixations separated by very short saccades/microsaccades
    # 3. Fill in short gaps between fixations
    
    # Convert classifications to tokens
    tokens = []
    current_type = classifications[0]
    count = 1
    
    for i in range(1, len(classifications)):
        if classifications[i] == current_type:
            count += 1
        else:
            tokens.append(current_type + str(count))
            current_type = classifications[i]
            count = 1
    
    # Add the last token
    tokens.append(current_type + str(count))
    
    # Merge very short tokens (< 4 samples)
    merged_tokens = []
    i = 0
    
    while i < len(tokens):
        current = tokens[i]
        current_type = current[0]
        current_length = int(current[1:])
        
        # If this is a very short token and not the last one
        if current_length < 4 and i < len(tokens) - 1:
            next_token = tokens[i+1]
            next_type = next_token[0]
            
            # If previous exists and matches next type, merge with previous
            if i > 0 and merged_tokens and merged_tokens[-1][0] == next_type:
                prev_token = merged_tokens.pop()
                prev_type = prev_token[0]
                prev_length = int(prev_token[1:])
                merged_tokens.append(f"{prev_type}{prev_length + current_length}")
                i += 1  # Skip the current token
            else:
                # Convert very short fixations to microsaccades
                if current_type == 'F':
                    merged_tokens.append(f"M{current_length}")
                else:
                    merged_tokens.append(current)
        else:
            merged_tokens.append(current)
        i += 1

    # Final pass: merge adjacent tokens of the same type
    final_tokens = []
    i = 0
    
    while i < len(merged_tokens):
        current = merged_tokens[i]
        current_type = current[0]
        current_length = int(current[1:])
        
        # Look ahead to merge adjacent tokens of the same type
        j = i + 1
        while j < len(merged_tokens) and merged_tokens[j][0] == current_type:
            current_length += int(merged_tokens[j][1:])
            j += 1
        
        final_tokens.append(f"{current_type}{current_length}")
        i = j
    
    # Process again: join microsaccades between fixations if they're short
    processed_tokens = []
    i = 0
    
    while i < len(final_tokens):
        current = final_tokens[i]
        
        # If this is a fixation and there's more tokens
        if current[0] == 'F' and i < len(final_tokens) - 2:
            next_token = final_tokens[i+1]
            next_next_token = final_tokens[i+2]
            
            # If pattern is F-M-F and M is short, join all as one fixation
            if (next_token[0] == 'M' and next_next_token[0] == 'F' and 
                int(next_token[1:]) < 5):
                
                f1_length = int(current[1:])
                m_length = int(next_token[1:])
                f2_length = int(next_next_token[1:])
                
                processed_tokens.append(f"F{f1_length + m_length + f2_length}")
                i += 3  # Skip the 3 tokens we just merged
            else:
                processed_tokens.append(current)
                i += 1
        else:
            processed_tokens.append(current)
            i += 1
    
    # Calculate distribution percentages
    total_samples = sum(int(token[1:]) for token in processed_tokens)
    fixation_samples = sum(int(token[1:]) for token in processed_tokens if token[0] == 'F')
    ms_samples = sum(int(token[1:]) for token in processed_tokens if token[0] == 'M')
    saccade_samples = sum(int(token[1:]) for token in processed_tokens if token[0] == 'S')
    
    if total_samples > 0:
        fixation_pct = (fixation_samples / total_samples) * 100
        ms_pct = (ms_samples / total_samples) * 100
        saccade_pct = (saccade_samples / total_samples) * 100
        
        print(f"Token distribution: Fixations: {fixation_pct:.2f}%, Microsaccades: {ms_pct:.2f}%, Saccades: {saccade_pct:.2f}%")
    
    return processed_tokens

def process_file(infile, outdir, **ivt_kwargs):
    df = pd.read_csv(infile)
    
    # Skip conversion from ms to seconds as we'll handle it in the function
    tokens = ivt_tokenize(df, **ivt_kwargs)

    # prepare output path
    base = os.path.basename(infile)
    name, _ = os.path.splitext(base)
    outpath = os.path.join(outdir, f"{name}_tokens.json")

    # save tokens list as JSON
    with open(outpath, 'w') as f:
        json.dump(tokens, f)

    print(f"Processed {infile} → {outpath} ({len(tokens)} tokens)")

def main():
    parser = argparse.ArgumentParser(
        description="Apply I-VT tokenization to SynchronEyes CSV files"
    )
    parser.add_argument("--input_dir", default="C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/SynchronEyes/data",
                        help="Base directory containing SynchronEyes CSV files")
    parser.add_argument("--output_dir", default="SynchronEyes_IVT",
                        help="Directory to save the token JSONs")
    parser.add_argument("--time-col", default="t",              # Changed default to 't'
                        help="Name of the time column in the CSV")
    parser.add_argument("--x-col", default="x",                 # Changed default to 'x'
                        help="Name of the horizontal gaze coordinate column")
    parser.add_argument("--y-col", default="y",                 # Changed default to 'y'
                        help="Name of the vertical gaze coordinate column")
    parser.add_argument("--smooth-window", type=int, default=7,
                        help="Window size for velocity smoothing")
    parser.add_argument("--thresh-fix", type=float, default=2.5,
                        help="Velocity threshold for fixations (deg/s)")
    parser.add_argument("--thresh-ms", type=float, default=6.0,
                        help="Velocity threshold for microsaccades (deg/s)")
    parser.add_argument("--thresh-sacc", type=float, default=35.0,
                        help="Velocity threshold for saccades (deg/s)")
    parser.add_argument("--dur-fix-min", type=float, default=0.080,
                        help="Minimum duration for a fixation (s)")
    parser.add_argument("--dur-thresh", type=float, default=0.150,
                        help="Duration threshold (s) to split D1 vs D2")
    parser.add_argument("--fix1-max", type=float, default=0.3,
                        help="Peak velocity max for F1")
    parser.add_argument("--fix2-max", type=float, default=0.5,
                        help="Peak velocity max for F2")
    parser.add_argument("--ms1-max", type=float, default=4.0,
                        help="Peak velocity max for MS1")
    parser.add_argument("--ms2-max", type=float, default=7.0,
                        help="Peak velocity max for MS2")
    parser.add_argument("--s1-max", type=float, default=25.0,
                        help="Peak velocity max for S1")
    parser.add_argument("--s2-max", type=float, default=60.0,
                        help="Peak velocity max for S2")
    parser.add_argument("--test", action="store_true",
                        help="Process only the first file for testing")
    parser.add_argument("--vel-col", default="velocity",
                        help="Name of the computed velocity column")
    parser.add_argument("--single-file", default=None,
                        help="Process a specific file from the input directory")
    parser.add_argument("--data-source", default="S02_EyeLink",  # Changed default to S02_EyeLink
                        choices=["S01_MindLink", "S02_MindLink", "S02_EyeLink"],
                        help="Data source to process (S01_MindLink, S02_MindLink, S02_EyeLink)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # find all CSVs under input_dir matching the specified data source
    pattern = f"{args.data_source}_*.csv"
    files = glob.glob(os.path.join(args.input_dir, pattern))
    print(f"Found {len(files)} {args.data_source} CSV files in {args.input_dir}")

    # Filter for a specific file if provided
    if args.single_file:
        files = [f for f in files if os.path.basename(f) == args.single_file]
        if not files:
            print(f"Error: Could not find file {args.single_file} in {args.input_dir}")
            return
        print(f"Selected 1 file: {os.path.basename(files[0])}")

    if args.test:
        if files:
            test_file = files[0]
            process_file(test_file,
                         args.output_dir,
                         time_col=args.time_col,
                         x_col=args.x_col,
                         y_col=args.y_col,
                         smooth_window=args.smooth_window,
                         thresh_fix=args.thresh_fix,
                         thresh_ms=args.thresh_ms,
                         thresh_sacc=args.thresh_sacc,
                         dur_fix_min=args.dur_fix_min,
                         dur_thresh=args.dur_thresh,
                         fix1_max=args.fix1_max,
                         fix2_max=args.fix2_max,
                         ms1_max=args.ms1_max,
                         ms2_max=args.ms2_max,
                         s1_max=args.s1_max,
                         s2_max=args.s2_max,
                         vel_col=args.vel_col)
            print("Test complete. Processed one file.")
    else:
        for infile in files:
            process_file(infile,
                         args.output_dir,
                         time_col=args.time_col,
                         x_col=args.x_col,
                         y_col=args.y_col,
                         smooth_window=args.smooth_window,
                         thresh_fix=args.thresh_fix,
                         thresh_ms=args.thresh_ms,
                         thresh_sacc=args.thresh_sacc,
                         dur_fix_min=args.dur_fix_min,
                         dur_thresh=args.dur_thresh,
                         fix1_max=args.fix1_max,
                         fix2_max=args.fix2_max,
                         ms1_max=args.ms1_max,
                         ms2_max=args.ms2_max,
                         s1_max=args.s1_max,
                         s2_max=args.s2_max,
                         vel_col=args.vel_col)

    print(f"All files processed. Tokens saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 