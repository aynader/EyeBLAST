#!/usr/bin/env python3
"""
Quick 1:1 test to generate data for plotting
"""

import subprocess
import sys

def main():
    print("🚀 Running quick 1:1 test...")
    
    # Run 1:1 only test
    result = subprocess.run([
        "python3", "benchmark_plain_vs_pp.py", 
        "--runs", "3",
        "--skip_1_to_n"
    ], capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    print(f"\nReturn code: {result.returncode}")
    
    if result.returncode == 0:
        print("\n✅ Success! Now run:")
        print("python3 plot_timing_comparison.py")
    else:
        print("\n❌ Failed!")

if __name__ == "__main__":
    main()