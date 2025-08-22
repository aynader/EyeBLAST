import os
import glob
import json
import argparse

import numpy as np
import pandas as pd

# Task mapping for LFI dataset
NUM_TO_TASK = {
    1: "TALK",
    2: "READ", 
    3: "VIDEO",
    4: "WALK",
    5: "TYPE",
    6: "SOLVE",
    7: "CYCLE",
}

def ivt_tokenize(df,
                 vel_col='velocity',
                 time_col='high_tstp_internal',
                 x_col='high_cube1_vel',
                 y_col='high_cube2_vel',
                 smooth_window=7,
                 thresh_fix=2.0,                # fixations below 2.0 units/s
                 thresh_ms=5.0,                 # microsaccades between 2.0–5.0 units/s
                 thresh_sacc=25.0,              # saccades above 25.0 units/s
                 dur_fix_min=0.080,
                 dur_thresh=0.150,
                 fix1_max=0.3,
                 fix2_max=0.5,
                 ms1_max=4.0,
                 ms2_max=7.0,
                 s1_max=25.0,
                 s2_max=60.0):
    """
    I-VT (Velocity-Threshold Identification) tokenization for LFI_CSV dataset.
    Classifies gaze samples into fixations (F), micro-saccades (M), and saccades (S)
    based on velocity thresholds using cube1 and cube2 velocities as spatial coordinates.
    
    Note: LFI uses laser interferometry measurements, thresholds are in measurement units/second.
    """

    # Handle time column - use internal timestamp if available
    if time_col not in df.columns:
        if 'high_tstp_internal' in df.columns:
            time_col = 'high_tstp_internal'
        elif 'high_tstp_unix' in df.columns:
            time_col = 'high_tstp_unix'
            # Convert unix timestamp to seconds if needed
            if df[time_col].mean() > 1e9:  # Unix timestamps are large
                df[time_col] = df[time_col] - df[time_col].min()  # Relative time
        else:
            # Create artificial time column
            df[time_col] = np.arange(len(df)) * 0.001  # Assume 1ms sampling
    
    # Compute time steps
    dt = df[time_col].diff().fillna(1e-4).values
    # Handle zero or negative time steps
    dt = np.where(dt <= 0, 1e-4, dt)
    
    # Use cube velocities directly as position proxies
    # Take differences to get velocity changes (analogous to position changes)
    dx = np.diff(df[x_col].values, prepend=df[x_col].values[0])
    dy = np.diff(df[y_col].values, prepend=df[y_col].values[0])
    
    # Instantaneous speed (units/s) - using velocity changes over time
    vel = np.sqrt(dx**2 + dy**2) / dt
    
    # Smooth velocity
    vel_smooth = pd.Series(vel).rolling(window=smooth_window,
                                        center=True,
                                        min_periods=1).mean().values
    df['velocity'] = vel_smooth
    
    print(f"Velocity stats: min={vel_smooth.min():.4f}, max={vel_smooth.max():.4f}, mean={vel_smooth.mean():.4f}")
    print(f"Time delta stats: min={dt.min():.6f}, max={dt.max():.6f}, mean={dt.mean():.6f}")
    print(f"Thresholds: fix={thresh_fix}, ms={thresh_ms}, sacc={thresh_sacc} (units/s)")

    tokens = []
    current_token = None
    current_durations = []

    for v in df['velocity']:
        if v <= thresh_fix:
            label = 'F'
        elif v <= thresh_ms:
            label = 'M'
        elif v > thresh_sacc:
            label = 'S'
        else:
            # ambiguous region between micro-saccade and saccade thresholds
            label = 'M'

        if current_token is None:
            current_token = label
            current_durations = [v]
        elif label == current_token:
            current_durations.append(v)
        else:
            # flush previous token
            tokens.append(current_token + str(len(current_durations)))
            current_token = label
            current_durations = [v]

    # append last token
    if current_token is not None:
        tokens.append(current_token + str(len(current_durations)))

    return tokens

def process_file(infile, outdir, **ivt_kwargs):
    """Process a single LFI_CSV file and extract IVT tokens."""
    try:
        df = pd.read_csv(infile)
        print(f"Successfully read {infile} with comma separator")
    except Exception as e:
        # Try with semicolon separator
        try:
            df = pd.read_csv(infile, sep=';')
            print(f"Successfully read {infile} with semicolon separator")
        except Exception as e2:
            # Try with tab separator
            try:
                df = pd.read_csv(infile, sep='\t')
                print(f"Successfully read {infile} with tab separator")
            except Exception as e3:
                raise ValueError(f"Failed to read file {infile}: {str(e3)}")
    
    # Remove rows with missing cube velocity data
    required_cols = ['high_cube1_vel', 'high_cube2_vel']
    df = df.dropna(subset=required_cols)
    
    if len(df) == 0:
        print(f"Warning: No valid data in {infile}")
        return
    
    print(f"Processing {len(df)} samples from {infile}")
    
    # Extract participant and task info from filename
    base = os.path.basename(infile)
    name, _ = os.path.splitext(base)
    try:
        participant_id, task_id = name.split('_')
        task_name = NUM_TO_TASK.get(int(task_id), f"TASK_{task_id}")
        print(f"Participant: {participant_id}, Task: {task_name} ({task_id})")
    except (ValueError, KeyError):
        print(f"Could not parse participant/task from filename: {name}")
    
    # Print data ranges
    print(f"Cube1 velocity range: {df['high_cube1_vel'].min():.4f} to {df['high_cube1_vel'].max():.4f}")
    print(f"Cube2 velocity range: {df['high_cube2_vel'].min():.4f} to {df['high_cube2_vel'].max():.4f}")
    
    # Handle time column
    time_col = ivt_kwargs.get('time_col', 'high_tstp_internal')
    if time_col in df.columns:
        print(f"Time range: {df[time_col].min()} to {df[time_col].max()}")
    
    tokens = ivt_tokenize(df, **ivt_kwargs)

    # prepare output path
    outpath = os.path.join(outdir, f"{name}_tokens.json")

    # save tokens list as JSON
    with open(outpath, 'w') as f:
        json.dump(tokens, f)

    print(f"Processed {infile} → {outpath} ({len(tokens)} tokens)")
    if len(tokens) <= 10:  # If few tokens, print them for debugging
        print(f"Tokens: {tokens}")

def main():
    parser = argparse.ArgumentParser(
        description="Apply I-VT tokenization to LFI_CSV files"
    )
    parser.add_argument("--input_dir", default="C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/LFI_CSV",
                        help="Base directory containing LFI_CSV files")
    parser.add_argument("--output_dir", default="LFI_CSV_IVT",
                        help="Directory to save the token JSONs")
    parser.add_argument("--time-col", default="high_tstp_internal",
                        help="Name of the time column in the CSV")
    parser.add_argument("--x-col", default="high_cube1_vel",
                        help="Name of the cube1 velocity column (x-analogue)")
    parser.add_argument("--y-col", default="high_cube2_vel",
                        help="Name of the cube2 velocity column (y-analogue)")
    parser.add_argument("--smooth-window", type=int, default=7,
                        help="Window size for velocity smoothing")
    parser.add_argument("--thresh-fix", type=float, default=2.0,
                        help="Velocity threshold for fixations (units/s)")
    parser.add_argument("--thresh-ms", type=float, default=5.0,
                        help="Velocity threshold for microsaccades (units/s)")
    parser.add_argument("--thresh-sacc", type=float, default=25.0,
                        help="Velocity threshold for saccades (units/s)")
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
    parser.add_argument("--participant", default=None,
                        help="Process only files from a specific participant (e.g., '003')")
    parser.add_argument("--task", default=None,
                        help="Process only files from a specific task (e.g., '001' for TALK)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # find all CSVs under input_dir
    files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    print(f"Found {len(files)} CSV files in {args.input_dir}")

    # Filter by participant if specified
    if args.participant:
        files = [f for f in files if os.path.basename(f).startswith(args.participant + '_')]
        print(f"Filtered to {len(files)} files for participant {args.participant}")

    # Filter by task if specified
    if args.task:
        files = [f for f in files if os.path.basename(f).endswith('_' + args.task + '.csv')]
        print(f"Filtered to {len(files)} files for task {args.task}")

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
            try:
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
            except Exception as e:
                print(f"Error processing test file {test_file}: {str(e)}")
    else:
        for infile in files:
            try:
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
            except Exception as e:
                print(f"Error processing {infile}: {str(e)}")

    print(f"All files processed. Tokens saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 