#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import json
from pathlib import Path

# Constants
NX_BINS = 4
NY_BINS = 4

# Hardcoded dataset path
DEFAULT_INPUT_DIR = r"C:\Users\aynad\Desktop\Home\TUM\Research\Datasets\GazeBaseVR\data"
DEFAULT_OUTPUT_DIR = r"GazeBaseVR_AOIs"

def load_all_xy(csv_paths, x_col='x', y_col='y'):
    """Load all X,Y coordinates from CSV files"""
    xs, ys = [], []
    print("Loading coordinates from CSV files:")
    for p in tqdm(csv_paths, desc="Loading CSVs"):
        try:
            # GazeBaseVR files are standard CSV with headers
            df = pd.read_csv(p)
            
            # Check if columns exist
            if x_col in df.columns and y_col in df.columns:
                x_data = df[x_col].values
                y_data = df[y_col].values
                # Filter out NaN values
                valid_indices = ~(np.isnan(x_data) | np.isnan(y_data))
                if valid_indices.any():
                    xs.append(x_data[valid_indices])
                    ys.append(y_data[valid_indices])
                    print(f"Successfully read {len(x_data[valid_indices])} points from {p}")
                else:
                    print(f"No valid data points in {p}")
            else:
                print(f"Warning: {x_col} or {y_col} column not found in {p}")
                
        except Exception as e:
            print(f"Error reading {p}: {e}")
            
    if xs and ys:
        return np.concatenate(xs), np.concatenate(ys)
    else:
        raise ValueError("No valid data found in CSV files")

def compute_quantile_edges(vals, n_bins):
    """Compute quantile edges for binning"""
    return np.quantile(vals, q=np.linspace(0, 1, n_bins + 1))

def build_bin_lattice(all_x, all_y, nx, ny):
    """Build quantile-based bin lattice"""
    # 1) x-edges for nx quantiles
    print("Computing x-quantiles...")
    x_edges = compute_quantile_edges(all_x, nx)
    
    # assign every sample to an x-bin 0..nx-1
    print("Assigning samples to x-bins...")
    x_idx = np.minimum(np.digitize(all_x,
                                  bins=x_edges[1:-1],
                                  right=True),
                      nx - 1)
    
    # 2) for each x-bin, compute y-edges for ny quantiles on that slice
    print("Computing y-quantiles for each x-bin...")
    y_edges_dict = {}
    for xi in tqdm(range(nx), desc="Processing x-bins"):
        mask = (x_idx == xi)
        if not mask.any():
            # fallback if no samples
            y_edges_dict[xi] = compute_quantile_edges(all_y, ny)
        else:
            y_edges_dict[xi] = compute_quantile_edges(all_y[mask], ny)
    
    return x_edges, y_edges_dict

def make_symbol_alphabet(n_cells):
    """Generate printable ASCII symbols for the alphabet"""
    # Use printable ASCII starting from '!' (33)
    symbols = [chr(i) for i in range(33, 33 + n_cells)]
    if n_cells > len(symbols):
        raise ValueError(f"Need {n_cells} symbols, but only {len(symbols)} available")
    return symbols[:n_cells]

def discover_csv_files(indir, pattern="**/*.csv"):
    """Find all CSV files in the input directory"""
    return sorted(glob.glob(os.path.join(indir, pattern), recursive=True))

def convert_csv_to_symbol_sequence(csv_path, x_edges, y_edges_dict, nx, ny, symbols, 
                                  x_col='x', y_col='y'):
    """Convert CSV coordinates to symbol sequence using quantile bins"""
    try:
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Warning: Required columns not found in {csv_path}")
            return ""
            
        print(f"Processing {csv_path} with x={x_col}, y={y_col}")
            
        # Use tqdm to show progress for long sequences
        seq = []
        for x, y in tqdm(zip(df[x_col], df[y_col]), desc="Mapping coordinates", 
                         total=len(df), leave=False):
            # Skip NaN values
            if np.isnan(x) or np.isnan(y):
                continue
                
            xi = np.minimum(np.digitize([x], bins=x_edges[1:-1], right=True)[0], nx-1)
            ye = y_edges_dict[xi]
            yi = np.minimum(np.digitize([y], bins=ye[1:-1], right=True)[0], ny-1)
            cell_id = xi * ny + yi
            seq.append(symbols[cell_id])
        
        return ''.join(seq)
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return ""

def process_dataset(input_dir, output_dir, x_col, y_col, file_pattern="**/*.csv", nx=NX_BINS, ny=NY_BINS):
    """Process the entire dataset and save encoded sequences"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = discover_csv_files(input_dir, pattern=file_pattern)
    print(f"Found {len(csv_files)} files to process")
    
    if not csv_files:
        print(f"No CSV files found in {input_dir} with pattern {file_pattern}")
        return
    
    # Load all x,y coordinates to calculate percentiles
    all_x, all_y = load_all_xy(csv_files, x_col=x_col, y_col=y_col)
    print(f"Loaded {len(all_x)} data points for binning")
    
    # Build bin lattice
    x_edges, y_edges = build_bin_lattice(all_x, all_y, nx, ny)
    
    # Generate symbol alphabet
    symbols = make_symbol_alphabet(nx * ny)
    
    # Save bin lattice for potential future use
    np.savez(os.path.join(output_dir, 'bin_lattice.npz'),
             x_edges=x_edges,
             y_edges=[y_edges[i] for i in range(nx)],
             symbols=symbols)
    print(f"Saved bin lattice to {output_dir}/bin_lattice.npz")
    
    # Process each file and save encoded sequence
    results = {}
    for csv_path in tqdm(csv_files, desc="Encoding files"):
        try:
            # Get filename without extension as key
            file_id = os.path.splitext(os.path.basename(csv_path))[0]
            
            # Convert to symbol sequence
            seq = convert_csv_to_symbol_sequence(
                csv_path, x_edges, y_edges, nx, ny, symbols, 
                x_col=x_col, y_col=y_col
            )
            
            # Store in results
            results[file_id] = {
                'file_path': csv_path,
                'sequence': seq,
                'length': len(seq)
            }
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
    
    # Save all sequences to a single JSON file
    with open(os.path.join(output_dir, 'encoded_sequences.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save each sequence to individual files
    sequences_dir = os.path.join(output_dir, 'sequences')
    os.makedirs(sequences_dir, exist_ok=True)
    
    for file_id, data in tqdm(results.items(), desc="Saving individual files"):
        output_file = os.path.join(sequences_dir, f"{file_id}.json")
        with open(output_file, 'w') as f:
            json.dump({
                'file_path': data['file_path'],
                'sequence': data['sequence'],
                'length': data['length']
            }, f)
    
    print(f"Encoded {len(results)} files")
    print(f"Results saved to {output_dir}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Encode GazeBaseVR dataset into percentile-based spatial bins")
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                        help='Directory containing GazeBaseVR CSV files')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save encoded sequences')
    parser.add_argument('--x_col', type=str, default='x',
                        help='Column name for X coordinates in CSV')
    parser.add_argument('--y_col', type=str, default='y',
                        help='Column name for Y coordinates in CSV')
    parser.add_argument('--file_pattern', type=str, default='**/*.csv',
                        help='File pattern to match CSV files')
    parser.add_argument('--nx_bins', type=int, default=NX_BINS,
                        help='Number of bins along X axis')
    parser.add_argument('--ny_bins', type=int, default=NY_BINS,
                        help='Number of bins along Y axis')
    parser.add_argument('--test', action='store_true',
                        help='Process only the first file for testing')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Processing GazeBaseVR dataset from {args.input_dir}")
    print(f"Output will be saved to {args.output_dir}")
    print(f"Using {args.nx_bins} x {args.ny_bins} bins")
    
    # For test mode, only process first file
    if args.test and os.path.exists(args.input_dir):
        csv_files = discover_csv_files(args.input_dir, pattern=args.file_pattern)
        if csv_files:
            test_file = csv_files[0]
            print(f"Test mode: Processing only {test_file}")
            test_files = [test_file]
            # Create temporary directory with just this file
            os.makedirs("test_tmp", exist_ok=True)
            process_dataset(
                args.input_dir, 
                args.output_dir,
                args.x_col,
                args.y_col,
                file_pattern=os.path.basename(test_file),
                nx=args.nx_bins,
                ny=args.ny_bins
            )
    else:
        # Process dataset with all files
        process_dataset(
            args.input_dir,
            args.output_dir,
            args.x_col,
            args.y_col,
            args.file_pattern,
            nx=args.nx_bins,
            ny=args.ny_bins
        )

# Alternative direct execution without command line arguments
def run_with_defaults():
    print(f"Processing GazeBaseVR dataset from {DEFAULT_INPUT_DIR}")
    print(f"Output will be saved to {DEFAULT_OUTPUT_DIR}")
    print(f"Using {NX_BINS} x {NY_BINS} bins")
    
    # Process dataset with default parameters
    process_dataset(
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        'x',
        'y',
        '**/*.csv',
        nx=NX_BINS,
        ny=NY_BINS
    )

# Comment out the default parser-based main function
# if __name__ == "__main__":
#     main() 

# Use the hardcoded defaults version
if __name__ == "__main__":
    run_with_defaults() 