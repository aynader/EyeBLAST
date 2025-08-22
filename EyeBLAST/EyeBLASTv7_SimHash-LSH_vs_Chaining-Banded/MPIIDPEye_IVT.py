import os
import glob
import json
import argparse

import numpy as np
import pandas as pd

def ivt_tokenize(df,
                 vel_col='velocity',
                 time_col='timestamp',
                 x_col='x',
                 y_col='y',
                 smooth_window=7,
                 thresh_fix=5.0,                # fixations below 5.0 deg/s
                 thresh_ms=6.0,                 # microsaccades between 5.0–6.0 deg/s
                 thresh_sacc=30.0,              # saccades above 30.0 deg/s
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

    # compute time steps
    dt = df[time_col].diff().fillna(1e-4).values
    # compute velocities
    dx = np.diff(df[x_col].values, prepend=df[x_col].values[0])
    dy = np.diff(df[y_col].values, prepend=df[y_col].values[0])
    # instantaneous angular speed (deg/s)
    vel = np.sqrt(dx**2 + dy**2) / dt
    # smooth velocity
    vel_smooth = pd.Series(vel).rolling(window=smooth_window,
                                        center=True,
                                        min_periods=1).mean().values
    df['velocity'] = vel_smooth

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
    # For MPIIDPEye files, use semicolon as separator and no header
    try:
        # Try reading with semicolon delimiter and no header
        df = pd.read_csv(infile, sep=';', header=None)
        print(f"Reading file with semicolon delimiter and no header: {infile}")
        
        # Based on the sample data, assign column names
        # MPIIDPEye eye tracking format: x, y, timestamp, confidence, pupil_diameter
        df.columns = ['x', 'y', 'timestamp', 'confidence', 'pupil_diameter']
    except Exception as e:
        # Fallback to try with header
        try:
            df = pd.read_csv(infile, sep=';')
            print(f"Reading file with semicolon delimiter and header: {infile}")
        except Exception as e1:
            # Try with comma delimiter
            try:
                df = pd.read_csv(infile, sep=',')
                print(f"Reading file with comma delimiter: {infile}")
            except Exception as e2:
                raise ValueError(f"Failed to read file: {str(e2)}")
    
    # Set default column names if needed
    time_col = ivt_kwargs.get('time_col', 'timestamp')
    x_col = ivt_kwargs.get('x_col', 'x')
    y_col = ivt_kwargs.get('y_col', 'y')
    
    # If timestamp not found, try to find the right column
    if time_col not in df.columns:
        # For MPIIDPEye data, timestamp is the 3rd column (index 2)
        if len(df.columns) >= 3:
            time_col = df.columns[2]
            ivt_kwargs['time_col'] = time_col
            print(f"Using column at position 3 as timestamp: '{time_col}'")
        else:
            # Try standard names
            possible_time_cols = ['timestamp', 'time', 't']
            for col in possible_time_cols:
                if col in df.columns:
                    time_col = col
                    ivt_kwargs['time_col'] = col
                    print(f"Using '{col}' as time column")
                    break
    
    # If x/y coordinates not found, try to find the right columns
    if x_col not in df.columns:
        # For MPIIDPEye data, x is the 1st column (index 0)
        if len(df.columns) >= 1:
            x_col = df.columns[0]
            ivt_kwargs['x_col'] = x_col
            print(f"Using column at position 1 as x-coordinate: '{x_col}'")
        else:
            possible_x_cols = ['x', 'gaze_x', 'gaze_point_x']
            for x in possible_x_cols:
                if x in df.columns:
                    x_col = x
                    ivt_kwargs['x_col'] = x
                    print(f"Using '{x}' as x-coordinate column")
                    break
                    
    if y_col not in df.columns:
        # For MPIIDPEye data, y is the 2nd column (index 1)
        if len(df.columns) >= 2:
            y_col = df.columns[1]
            ivt_kwargs['y_col'] = y_col
            print(f"Using column at position 2 as y-coordinate: '{y_col}'")
        else:
            possible_y_cols = ['y', 'gaze_y', 'gaze_point_y']
            for y in possible_y_cols:
                if y in df.columns:
                    y_col = y
                    ivt_kwargs['y_col'] = y
                    print(f"Using '{y}' as y-coordinate column")
                    break
    
    # Ensure we have necessary columns
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in {infile}")
    if x_col not in df.columns:
        raise ValueError(f"X-coordinate column '{x_col}' not found in {infile}")
    if y_col not in df.columns:
        raise ValueError(f"Y-coordinate column '{y_col}' not found in {infile}")
    
    print(f"Processing with columns - x: {x_col}, y: {y_col}, time: {time_col}")
    
    # Convert time to seconds if needed (assuming it might be in ms)
    # Check if the time values are large, indicating milliseconds
    if df[time_col].mean() > 1000:  # Heuristic to detect milliseconds
        df[time_col] = df[time_col] / 1000.0  # convert time from ms to seconds
        print(f"Converting {time_col} from milliseconds to seconds")
    
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
        description="Apply I-VT tokenization to MPIIDPEye CSV files"
    )
    parser.add_argument("--input_dir", default="C:/Users/aynad/Desktop/Home/TUM/Research/Datasets/MPIIDPEye/Eye_Tracking_Data",
                        help="Base directory containing MPIIDPEye CSV files")
    parser.add_argument("--output_dir", default="MPIIDPEye_IVT",
                        help="Directory to save the token JSONs")
    parser.add_argument("--time-col", default="timestamp",
                        help="Name of the time column in the CSV")
    parser.add_argument("--x-col", default="x",
                        help="Name of the horizontal gaze coordinate column")
    parser.add_argument("--y-col", default="y",
                        help="Name of the vertical gaze coordinate column")
    parser.add_argument("--smooth-window", type=int, default=7,
                        help="Window size for velocity smoothing")
    parser.add_argument("--thresh-fix", type=float, default=5.0,
                        help="Velocity threshold for fixations (deg/s)")
    parser.add_argument("--thresh-ms", type=float, default=6.0,
                        help="Velocity threshold for microsaccades (deg/s)")
    parser.add_argument("--thresh-sacc", type=float, default=30.0,
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # find all CSVs under input_dir
    files = glob.glob(os.path.join(args.input_dir, "**", "*.csv"),
                      recursive=True)
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