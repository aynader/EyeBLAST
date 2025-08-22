#!/usr/bin/env python3
"""
Comparison script for FastEyeBLAST Python vs C++ implementations.
This script runs both implementations on the same data and compares results.
"""

import os
import sys
import json
import subprocess
import argparse
import numpy as np
from pathlib import Path

def run_python_implementation(input_dir, output_dir, args):
    """Run the Python implementation."""
    print("Running Python implementation...")
    
    cmd = [
        sys.executable, "Fast_EyeBLAST_IVT.py",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--seed", str(args.seed),
        "--sample_percentage", str(args.sample_percentage),
        "--test_size", str(args.test_size),
        "--num_neighbors", str(args.num_neighbors)
    ]
    
    if args.parameter_sweep:
        cmd.append("--parameter_sweep")
    
    try:
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"Python implementation failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("Python implementation timed out after 1 hour")
        return False
    except Exception as e:
        print(f"Error running Python implementation: {e}")
        return False

def run_cpp_implementation(input_dir, output_dir, args):
    """Run the C++ implementation."""
    print("Running C++ implementation...")
    
    # Automatically build the C++ executable
    print("Building C++ executable...")
    import platform
    
    if platform.system() == "Windows":
        # Run build commands directly without interactive script
        try:
            # Check if we're in the right directory
            if not os.path.exists("Fast_EyeBLAST_IVT.cpp"):
                print("Error: Fast_EyeBLAST_IVT.cpp not found in current directory")
                return False
            
            # Create build directory
            os.makedirs("build", exist_ok=True)
            
            # Check for CMake
            cmake_check = subprocess.run(["where", "cmake"], capture_output=True, text=True)
            if cmake_check.returncode != 0:
                print("CMake not found. Please install CMake or Visual Studio Build Tools.")
                print("You can download CMake from: https://cmake.org/download/")
                return False
            
            # Configure with CMake
            print("Configuring build with CMake...")
            config_result = subprocess.run(
                ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                cwd="build", capture_output=True, text=True, timeout=120
            )
            if config_result.returncode != 0:
                print(f"CMake configuration failed:")
                print("STDOUT:", config_result.stdout)
                print("STDERR:", config_result.stderr)
                return False
            
            # Build the project
            print("Building FastEyeBLAST...")
            build_result = subprocess.run(
                ["cmake", "--build", ".", "--config", "Release", "--parallel"],
                cwd="build", capture_output=True, text=True, timeout=300
            )
            if build_result.returncode != 0:
                print(f"Build failed:")
                print("STDOUT:", build_result.stdout)
                print("STDERR:", build_result.stderr)
                return False
            
            print("Build completed successfully!")
            
        except subprocess.TimeoutExpired:
            print("Build timed out")
            return False
        except Exception as e:
            print(f"Error during build process: {e}")
            return False
            
    else:
        # Unix/Linux - use build.sh
        build_script = "./build.sh"
        try:
            build_result = subprocess.run([build_script], cwd=".", capture_output=True, text=True, timeout=300)
            if build_result.returncode != 0:
                print(f"Build failed with return code {build_result.returncode}")
                print("Build STDOUT:", build_result.stdout)
                print("Build STDERR:", build_result.stderr)
                return False
            print("Build completed successfully!")
        except subprocess.TimeoutExpired:
            print("Build timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"Error running build script: {e}")
            return False
    
    # Try to find the C++ executable
    cpp_exe_paths = [
        "build/bin/fast_eyeblast",
        "build/bin/fast_eyeblast.exe",
        "build/bin/Release/fast_eyeblast.exe",
        "build/fast_eyeblast",
        "build/fast_eyeblast.exe"
    ]
    
    cpp_exe = None
    for path in cpp_exe_paths:
        if os.path.exists(path):
            cpp_exe = path
            break
    
    if cpp_exe is None:
        print("C++ executable not found even after building.")
        print("Please check the build script and CMake configuration.")
        return False
    
    print(f"Using C++ executable: {cpp_exe}")
    
    cmd = [
        cpp_exe,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--sample_percentage", str(args.sample_percentage),
        "--test_size", str(args.test_size),
        "--num_neighbors", str(args.num_neighbors)
    ]
    
    if args.parameter_sweep:
        cmd.append("--parameter_sweep")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"C++ implementation failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("C++ implementation timed out after 1 hour")
        return False
    except Exception as e:
        print(f"Error running C++ implementation: {e}")
        return False

def load_results(results_file):
    """Load results from JSON file."""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None

def load_predictions(pred_file):
    """Load predictions from JSON file."""
    try:
        with open(pred_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {pred_file}: {e}")
        return None

def compare_results(py_results, cpp_results, tolerance=1e-6):
    """Compare results between implementations."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Compare overall accuracy
    py_acc = py_results.get('accuracy', 0)
    cpp_acc = cpp_results.get('accuracy', 0)
    acc_diff = abs(py_acc - cpp_acc)
    
    print(f"Accuracy:")
    print(f"  Python: {py_acc:.6f}")
    print(f"  C++:    {cpp_acc:.6f}")
    print(f"  Diff:   {acc_diff:.6f} {'✓' if acc_diff < tolerance else '✗'}")
    
    # Compare macro averages
    py_macro = py_results.get('macro_avg', {})
    cpp_macro = cpp_results.get('macro_avg', {})
    
    print(f"\nMacro-averaged metrics:")
    for metric in ['precision', 'recall', 'f1']:
        py_val = py_macro.get(metric, 0)
        cpp_val = cpp_macro.get(metric, 0)
        diff = abs(py_val - cpp_val)
        print(f"  {metric.capitalize()}:")
        print(f"    Python: {py_val:.6f}")
        print(f"    C++:    {cpp_val:.6f}")
        print(f"    Diff:   {diff:.6f} {'✓' if diff < tolerance else '✗'}")
    
    # Compare confusion matrices
    py_cm = py_results.get('confusion_matrix', [])
    cpp_cm = cpp_results.get('confusion_matrix', [])
    
    if py_cm and cpp_cm:
        py_cm_arr = np.array(py_cm)
        cpp_cm_arr = np.array(cpp_cm)
        
        if py_cm_arr.shape == cpp_cm_arr.shape:
            cm_diff = np.abs(py_cm_arr - cpp_cm_arr)
            max_cm_diff = np.max(cm_diff)
            print(f"\nConfusion Matrix:")
            print(f"  Max difference: {max_cm_diff} {'✓' if max_cm_diff == 0 else '✗'}")
            
            if max_cm_diff > 0:
                print(f"  Total differing entries: {np.sum(cm_diff > 0)}")
        else:
            print(f"\nConfusion Matrix:")
            print(f"  Shape mismatch: Python {py_cm_arr.shape} vs C++ {cpp_cm_arr.shape} ✗")
    
    # Compare timing (informational only)
    py_timing = py_results.get('timing_data', {})
    cpp_timing = cpp_results.get('timing_data', {})
    
    print(f"\nTiming comparison (informational):")
    py_total = py_timing.get('total_time', 0)
    cpp_total = cpp_timing.get('total_time', 0)
    speedup = py_total / cpp_total if cpp_total > 0 else 0
    
    print(f"  Python total time: {py_total:.2f}s")
    print(f"  C++ total time:    {cpp_total:.2f}s")
    print(f"  Speedup:           {speedup:.2f}x")
    
    return acc_diff < tolerance

def compare_predictions(py_pred, cpp_pred):
    """Compare predictions between implementations."""
    py_true = py_pred.get('true', [])
    py_pred_vals = py_pred.get('pred', [])
    cpp_true = cpp_pred.get('true', [])
    cpp_pred_vals = cpp_pred.get('pred', [])
    
    print(f"\nPredictions comparison:")
    print(f"  Python predictions: {len(py_pred_vals)}")
    print(f"  C++ predictions:    {len(cpp_pred_vals)}")
    
    if len(py_pred_vals) == len(cpp_pred_vals):
        matches = sum(1 for p, c in zip(py_pred_vals, cpp_pred_vals) if p == c)
        match_rate = matches / len(py_pred_vals) if py_pred_vals else 0
        print(f"  Matching predictions: {matches}/{len(py_pred_vals)} ({match_rate:.1%})")
        
        # Check if true labels match
        if py_true == cpp_true:
            print(f"  True labels: ✓ Match")
        else:
            print(f"  True labels: ✗ Mismatch")
        
        return match_rate > 0.95  # Allow for small differences due to randomness
    else:
        print(f"  ✗ Different number of predictions")
        return False

def run_plot_creation(cpp_output_dir):
    """Run the plot creation script for C++ results."""
    print("Creating plots from C++ results...")
    
    cmd = [
        sys.executable, "create_plots_from_cpp_data.py",
        "--data_dir", cpp_output_dir,
        "--output_dir", cpp_output_dir
    ]
    
    try:
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Plot creation failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        print("Plot creation completed successfully!")
        return True
    except subprocess.TimeoutExpired:
        print("Plot creation timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running plot creation script: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Compare Python and C++ FastEyeBLAST implementations")
    parser.add_argument("--input_dir", default="GazeBaseVR_IVT",
                       help="Input directory with scanpath JSON files")
    parser.add_argument("--sample_percentage", type=int, default=10,
                       help="Percentage of data to use for testing (default: 10 for faster comparison)")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Test set proportion")
    parser.add_argument("--num_neighbors", type=int, default=5,
                       help="Number of neighbors for k-NN")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--parameter_sweep", action="store_true",
                       help="Run parameter sweep")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                       help="Tolerance for numerical comparisons")
    parser.add_argument("--skip_python", action="store_true",
                       help="Skip running Python implementation (use existing results)")
    parser.add_argument("--skip_cpp", action="store_true",
                       help="Skip running C++ implementation (use existing results)")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found")
        return 1
    
    # Create organized output directory structure
    from datetime import datetime
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generate parameter string for folder name
    param_parts = [
        f"sample{args.sample_percentage}",
        f"test{args.test_size}",
        f"k{args.num_neighbors}"
    ]
    if args.parameter_sweep:
        param_parts.append("sweep")
    param_string = "_".join(param_parts)
    
    # Create directory structure
    base_output = "CPP_vs_Python_Comparison_Results"
    run_folder = f"{timestamp}_{param_string}"
    run_output_dir = os.path.join(base_output, run_folder)
    
    py_output = os.path.join(run_output_dir, "Python")
    cpp_output = os.path.join(run_output_dir, "CPP")
    
    os.makedirs(py_output, exist_ok=True)
    os.makedirs(cpp_output, exist_ok=True)
    
    print("="*60)
    print("FastEyeBLAST Implementation Comparison")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Sample percentage: {args.sample_percentage}%")
    print(f"Test size: {args.test_size}")
    print(f"Number of neighbors: {args.num_neighbors}")
    print(f"Random seed: {args.seed}")
    print(f"Parameter sweep: {args.parameter_sweep}")
    print(f"Results will be saved to: {run_output_dir}")
    
    success = True
    
    # Run implementations
    if not args.skip_python:
        if not run_python_implementation(args.input_dir, py_output, args):
            print("Failed to run Python implementation")
            success = False
    
    if not args.skip_cpp:
        if not run_cpp_implementation(args.input_dir, cpp_output, args):
            print("Failed to run C++ implementation")
            success = False
    
    if not success:
        return 1
    
    # Load and compare results
    py_results_file = os.path.join(py_output, "results.json")
    cpp_results_file = os.path.join(cpp_output, "results.json")
    py_pred_file = os.path.join(py_output, "predictions.json")
    cpp_pred_file = os.path.join(cpp_output, "predictions.json")
    
    # Check if result files exist
    missing_files = []
    for file_path in [py_results_file, cpp_results_file, py_pred_file, cpp_pred_file]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing result files: {missing_files}")
        return 1
    
    # Load results
    py_results = load_results(py_results_file)
    cpp_results = load_results(cpp_results_file)
    py_pred = load_predictions(py_pred_file)
    cpp_pred = load_predictions(cpp_pred_file)
    
    if not all([py_results, cpp_results, py_pred, cpp_pred]):
        print("Failed to load all result files")
        return 1
    
    # Compare results
    results_match = compare_results(py_results, cpp_results, args.tolerance)
    predictions_match = compare_predictions(py_pred, cpp_pred)
    
    # Always create plots from C++ results for analysis
    print("\n" + "="*60)
    print("CREATING PLOTS FROM C++ RESULTS")
    print("="*60)
    plot_success = run_plot_creation(cpp_output)
    if not plot_success:
        print("Warning: Plot creation failed")
    else:
        print("Plots created successfully!")
    
    # Create a summary file with comparison info
    summary_file = os.path.join(run_output_dir, "comparison_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("FastEyeBLAST Implementation Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Sample percentage: {args.sample_percentage}%\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Number of neighbors: {args.num_neighbors}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Parameter sweep: {args.parameter_sweep}\n\n")
        
        f.write("Results:\n")
        f.write(f"- Results match: {'✓' if results_match else '✗'}\n")
        f.write(f"- Predictions match: {'✓' if predictions_match else '✗'}\n\n")
        
        if py_results and cpp_results:
            f.write("Performance Metrics:\n")
            f.write(f"- Python Accuracy: {py_results.get('accuracy', 0):.6f}\n")
            f.write(f"- C++ Accuracy: {cpp_results.get('accuracy', 0):.6f}\n")
            f.write(f"- Accuracy Difference: {abs(py_results.get('accuracy', 0) - cpp_results.get('accuracy', 0)):.6f}\n\n")
            
            py_timing = py_results.get('timing_data', {})
            cpp_timing = cpp_results.get('timing_data', {})
            py_total = py_timing.get('total_time', 0)
            cpp_total = cpp_timing.get('total_time', 0)
            speedup = py_total / cpp_total if cpp_total > 0 else 0
            
            f.write("Timing Comparison:\n")
            f.write(f"- Python total time: {py_total:.2f}s\n")
            f.write(f"- C++ total time: {cpp_total:.2f}s\n")
            f.write(f"- C++ speedup: {speedup:.2f}x\n\n")
        
        f.write("Output Structure:\n")
        f.write(f"- Python results: {py_output}\n")
        f.write(f"- C++ results: {cpp_output}\n")
        f.write(f"- This summary: {summary_file}\n")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if results_match and predictions_match:
        print("✓ Both implementations produce consistent results!")
        print("  The C++ implementation is working correctly.")
        print(f"  Results saved to: {run_output_dir}")
        print(f"  - Python results: {py_output}")
        print(f"  - C++ results and plots: {cpp_output}")
        print(f"  - Summary file: {summary_file}")
        return 0
    else:
        print("✗ Implementations produce different results")
        if not results_match:
            print("  - Performance metrics differ significantly")
        if not predictions_match:
            print("  - Predictions differ significantly")
        print(f"  Results saved to: {run_output_dir}")
        print(f"  - Python results: {py_output}")
        print(f"  - C++ results and plots: {cpp_output}")
        print(f"  - Summary file: {summary_file}")
        print("  Please investigate the differences.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 