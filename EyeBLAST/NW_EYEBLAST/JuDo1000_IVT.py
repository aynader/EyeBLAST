import os
import glob
import json
import argparse

import numpy as np
import pandas as pd

def ivt_tokenize(df,
                 vel_col='velocity',
                 time_col='time',
                 x_col='x_avg',
                 y_col='y_avg',
                 smooth_window=7,
                 thresh_fix=100000.0,           # fixations below 100k pixels/s
                 thresh_ms=500000.0,            # microsaccades between 100k–500k pixels/s
                 thresh_sacc=2000000.0,         # saccades above 2M pixels/s
                 dur_fix_min=0.080,
                 dur_thresh=0.150,
                 fix1_max=0.3,
                 fix2_max=0.5,
                 ms1_max=4.0,
                 ms2_max=7.0,
                 s1_max=25.0,
                 s2_max=60.0):
    """
    I-VT (Velocity-Threshold Identification) tokenization for JuDo1000 dataset.
    Classifies gaze samples into fixations (F), micro-saccades (M), and saccades (S)
    based on angular velocity thresholds.
    
    Note: JuDo1000 uses pixel coordinates, so thresholds are in pixels/second.
    """

    # compute time steps (time is already in microseconds, convert to seconds)
    df['time_seconds'] = df[time_col] / 1e6
    dt = df['time_seconds'].diff().fillna(1e-4).values
    
    # compute velocities using average of left and right eye
    dx = np.diff(df[x_col].values, prepend=df[x_col].values[0])
    dy = np.diff(df[y_col].values, prepend=df[y_col].values[0])
    
    # instantaneous speed (pixels/s)
    vel = np.sqrt(dx**2 + dy**2) / dt
    
    # smooth velocity
    vel_smooth = pd.Series(vel).rolling(window=smooth_window,
                                        center=True,
                                        min_periods=1).mean().values
    df['velocity'] = vel_smooth
    
    print(f"Velocity stats: min={vel_smooth.min():.2f}, max={vel_smooth.max():.2f}, mean={vel_smooth.mean():.2f}")
    print(f"Time delta stats: min={dt.min():.6f}, max={dt.max():.6f}, mean={dt.mean():.6f}")
    print(f"Thresholds: fix={thresh_fix}, ms={thresh_ms}, sacc={thresh_sacc} (pixels/s)")

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
    """Process a single JuDo1000 CSV file and extract IVT tokens."""
    # Try reading with different separators
    try:
        df = pd.read_csv(infile, sep=r'\s+')  # Use whitespace as separator
    except:
        try:
            df = pd.read_csv(infile, sep='\t')  # Try tab separator
        except:
            df = pd.read_csv(infile)  # Default comma separator
    
    # Average left and right eye coordinates
    df['x_avg'] = (df['x_left'] + df['x_right']) / 2.0
    df['y_avg'] = (df['y_left'] + df['y_right']) / 2.0
    
    # Remove rows with missing data
    df = df.dropna(subset=['x_left', 'y_left', 'x_right', 'y_right'])
    
    if len(df) == 0:
        print(f"Warning: No valid data in {infile}")
        return
    
    print(f"Processing {len(df)} samples from {infile}")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"X range: {df['x_avg'].min():.2f} to {df['x_avg'].max():.2f}")
    print(f"Y range: {df['y_avg'].min():.2f} to {df['y_avg'].max():.2f}")
    
    tokens = ivt_tokenize(df, **ivt_kwargs)

    # prepare output path
    base = os.path.basename(infile)
    name, _ = os.path.splitext(base)
    outpath = os.path.join(outdir, f"{name}_tokens.json")

    # save tokens list as JSON
    with open(outpath, 'w') as f:
        json.dump(tokens, f)

    print(f"Processed {infile} → {outpath} ({len(tokens)} tokens)")
    if len(tokens) <= 10:  # If few tokens, print them for debugging
        print(f"Tokens: {tokens}")

def main():
    parser = argparse.ArgumentParser(
        description="Apply I-VT tokenization to JuDo1000 CSV files"
    )
    parser.add_argument("--input_dir", default="C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/JuDo1000",
                        help="Base directory containing JuDo1000 CSV files")
    parser.add_argument("--output_dir", default="JuDo1000_IVT",
                        help="Directory to save the token JSONs")
    parser.add_argument("--time-col", default="time",
                        help="Name of the time column (in microseconds) in the CSV")
    parser.add_argument("--x-col", default="x_avg",
                        help="Name of the horizontal gaze coordinate column")
    parser.add_argument("--y-col", default="y_avg",
                        help="Name of the vertical gaze coordinate column")
    parser.add_argument("--smooth-window", type=int, default=7,
                        help="Window size for velocity smoothing")
    parser.add_argument("--thresh-fix", type=float, default=100000.0,
                        help="Velocity threshold for fixations (pixels/s)")
    parser.add_argument("--thresh-ms", type=float, default=500000.0,
                        help="Velocity threshold for microsaccades (pixels/s)")
    parser.add_argument("--thresh-sacc", type=float, default=2000000.0,
                        help="Velocity threshold for saccades (pixels/s)")
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
    parser.add_argument("--exclude-trialvars", action="store_true",
                        help="Exclude TrialVars files from processing")
    parser.add_argument("--include-trialvars", action="store_true", default=False,
                        help="Include TrialVars files in processing (they usually don't contain eye tracking data)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # find all CSVs under input_dir, excluding TrialVars files
    files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    if args.exclude_trialvars:
        files = [f for f in files if "TrialVars" not in f]
    
    # find all CSVs under input_dir, excluding TrialVars files by default
    files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    if not args.include_trialvars:  # Exclude TrialVars by default
        files = [f for f in files if "TrialVars" not in f]
    
    print(f"Found {len(files)} CSV files in {args.input_dir}")

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
                print(f"Error processing {infile}: {e}")

    print(f"All files processed. Tokens saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 