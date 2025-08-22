#!/usr/bin/env python3
"""
Benchmark Plain EyeBLAST vs PP_EyeBLAST Performance Comparison

This script automates timing comparisons between:
1. Plain EyeBLAST (using test_alicebob.py approach)
2. PP_EyeBLAST (using pp_main Alice/Bob processes)

Supports both 1:1 and 1:N comparison scenarios.
"""

import json
import time
import subprocess
import os
import sys
import threading
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional

# Add PythonVsCpp to path to import FastEyeBLAST
pythonvscpp_path = Path(__file__).parent / "PythonVsCpp"
sys.path.insert(0, str(pythonvscpp_path))

try:
    from Fast_EyeBLAST_IVT import FastEyeBLAST
except ImportError as e:
    print("Error: Could not import FastEyeBLAST from PythonVsCpp directory")
    print(f"Looking in: {pythonvscpp_path}")
    print(f"Import error: {e}")
    print("Make sure Fast_EyeBLAST_IVT.py exists in PythonVsCpp/")
    sys.exit(1)


class PlainEyeBLASTBenchmark:
    """Benchmark plain EyeBLAST implementation"""
    
    def __init__(self):
        self.eyeblast = FastEyeBLAST()
    
    def benchmark_1_to_1(self, alice_file: str, bob_file: str) -> float:
        """
        Benchmark 1:1 comparison using plain EyeBLAST
        
        Args:
            alice_file: Path to Alice's scanpath JSON
            bob_file: Path to Bob's scanpath JSON
            
        Returns:
            Execution time in seconds
        """
        # Load scanpaths
        with open(alice_file) as f:
            scanpath1 = json.load(f)
        with open(bob_file) as f:
            scanpath2 = json.load(f)
        
        # Start timing
        start_time = time.perf_counter()
        
        # Encode scanpaths
        encoded1 = self.eyeblast._encode_scanpath(scanpath1)
        encoded2 = self.eyeblast._encode_scanpath(scanpath2)
        
        # Initialize bit-packing
        self.eyeblast._initialize_bit_packing()
        
        # Compute similarity
        score = self.eyeblast._similarity_score(encoded1, encoded2)
        
        # Stop timing
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        return elapsed
    
    def benchmark_1_to_n(self, alice_file: str, bob_files: List[str]) -> float:
        """
        Benchmark 1:N comparison using plain EyeBLAST
        
        Args:
            alice_file: Path to Alice's scanpath JSON
            bob_files: List of paths to Bob's scanpath JSONs
            
        Returns:
            Total execution time in seconds
        """
        # Load Alice's scanpath once
        with open(alice_file) as f:
            alice_scanpath = json.load(f)
        
        encoded_alice = self.eyeblast._encode_scanpath(alice_scanpath)
        self.eyeblast._initialize_bit_packing()
        
        # Start timing for the search phase
        start_time = time.perf_counter()
        
        scores = []
        for bob_file in bob_files:
            with open(bob_file) as f:
                bob_scanpath = json.load(f)
            
            encoded_bob = self.eyeblast._encode_scanpath(bob_scanpath)
            score = self.eyeblast._similarity_score(encoded_alice, encoded_bob)
            scores.append(score)
        
        # Stop timing
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        return elapsed


class PPEyeBLASTBenchmark:
    """Benchmark PP_EyeBLAST implementation"""
    
    def __init__(self, pp_main_path: str = "./build/pp_main"):
        # Auto-detect if we're on Windows and suggest WSL
        import platform
        if platform.system() == "Windows" and not pp_main_path.endswith(".exe"):
            print("Warning: Detected Windows system. PP_EyeBLAST requires Linux environment.")
            print("Please run this script in WSL (Windows Subsystem for Linux)")
            print("Or build a Windows version of pp_main")
            # Don't exit here, let user override
        self.pp_main_path = pp_main_path
        self.base_coarse_port = 12345
        self.base_refine_port = 12346
    
    def _get_available_ports(self) -> Tuple[int, int]:
        """Get available port pair"""
        # Simple port allocation - in practice, could check if ports are free
        import random
        import time
        # Use timestamp + random to ensure unique ports for concurrent runs
        offset = int(time.time() * 1000) % 1000 + random.randint(0, 100)
        return self.base_coarse_port + offset, self.base_refine_port + offset
    
    def benchmark_1_to_1(self, alice_file: str, bob_file: str, 
                        tau: int = 20, band_w: int = 5) -> float:
        """
        Benchmark 1:1 comparison using PP_EyeBLAST
        
        Args:
            alice_file: Path to Alice's scanpath JSON
            bob_file: Path to Bob's scanpath JSON
            tau: Tau parameter for PP_EyeBLAST
            band_w: Band width parameter for PP_EyeBLAST
            
        Returns:
            Execution time in seconds
        """
        coarse_port, refine_port = self._get_available_ports()
        
        # Prepare directories for Alice and Bob
        alice_dir = Path(alice_file).parent
        bob_dir = Path(bob_file).parent
        
        # Command templates
        # Adjust paths for running from build directory
        relative_alice_dir = f"../{alice_dir}"
        relative_bob_dir = f"../{bob_dir}"
        pp_main_name = "./pp_main"  # Relative path in build directory
        
        bob_cmd = [
            pp_main_name,
            "--role", "bob",
            "--infile", relative_bob_dir,
            "--ip", "127.0.0.1",
            "--coarse_port", str(coarse_port),
            "--refine_port", str(refine_port),
            "--tau", str(tau),
            "--band_w", str(band_w)
        ]
        
        alice_cmd = [
            pp_main_name,
            "--role", "alice", 
            "--infile", relative_alice_dir,
            "--ip", "127.0.0.1",
            "--coarse_port", str(coarse_port),
            "--refine_port", str(refine_port),
            "--tau", str(tau),
            "--band_w", str(band_w),
            "--output", "alice_out.json"
        ]
        
        # Start timing
        start_time = time.perf_counter()
        
        # Start Bob process (from build directory)
        bob_process = subprocess.Popen(
            bob_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="build"  # Run from build directory
        )
        
        # Give Bob a moment to start
        time.sleep(0.5)
        
        # Start Alice process (from build directory)
        alice_process = subprocess.Popen(
            alice_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="build"  # Run from build directory
        )
        
        # Wait for both processes to complete (no timeout)
        alice_stdout, alice_stderr = alice_process.communicate()
        bob_stdout, bob_stderr = bob_process.communicate()
        
        # Stop timing
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        # Check for errors and print debug info
        if alice_process.returncode != 0:
            print(f"Alice process failed: {alice_stderr}")
            print(f"Alice stdout: {alice_stdout}")
            print(f"Alice command: {' '.join(alice_cmd)}")
            return float('inf')
        if bob_process.returncode != 0:
            print(f"Bob process failed: {bob_stderr}")
            print(f"Bob stdout: {bob_stdout}")
            print(f"Bob command: {' '.join(bob_cmd)}")
            return float('inf')
        
        # Clean up output file (in build directory)
        output_file = Path("build/alice_out.json")
        if output_file.exists():
            output_file.unlink()
        
        return elapsed
    
    def benchmark_1_to_n(self, alice_file: str, bob_files: List[str],
                        tau: int = 20, band_w: int = 5) -> float:
        """
        Benchmark 1:N comparison using PP_EyeBLAST
        
        Args:
            alice_file: Path to Alice's scanpath JSON  
            bob_files: List of paths to Bob's scanpath JSONs
            tau: Tau parameter for PP_EyeBLAST
            band_w: Band width parameter for PP_EyeBLAST
            
        Returns:
            Total execution time in seconds
        """
        total_time = 0.0
        
        for i, bob_file in enumerate(bob_files):
            print(f"Running PP comparison {i+1}/{len(bob_files)}")
            run_time = self.benchmark_1_to_1(alice_file, bob_file, tau, band_w)
            if run_time == float('inf'):
                print(f"Failed comparison with {bob_file}")
                continue
            total_time += run_time
            
            # Add small delay between runs to avoid port conflicts
            if i < len(bob_files) - 1:  # Don't wait after last run
                time.sleep(1)
        
        return total_time


class BenchmarkRunner:
    """Main benchmark coordination class"""
    
    def __init__(self, pp_main_path: str = "./build/pp_main"):
        self.plain_benchmark = PlainEyeBLASTBenchmark()
        self.pp_benchmark = PPEyeBLASTBenchmark(pp_main_path)
        self.results = {
            "plain_eyeblast": {},
            "pp_eyeblast": {},
            "metadata": {}
        }
    
    def run_1_to_1_comparison(self, alice_file: str, bob_file: str, 
                            runs: int = 3) -> Dict:
        """
        Run 1:1 comparison benchmark
        
        Args:
            alice_file: Path to Alice's scanpath JSON
            bob_file: Path to Bob's scanpath JSON
            runs: Number of runs to average
            
        Returns:
            Dictionary with timing results
        """
        print(f"Running 1:1 comparison: {Path(alice_file).name} vs {Path(bob_file).name}")
        
        plain_times = []
        pp_times = []
        
        for run in range(runs):
            print(f"  Run {run + 1}/{runs}")
            
            # Plain EyeBLAST
            plain_time = self.plain_benchmark.benchmark_1_to_1(alice_file, bob_file)
            plain_times.append(plain_time)
            print(f"    Plain: {plain_time:.4f}s")
            
            # PP_EyeBLAST
            pp_time = self.pp_benchmark.benchmark_1_to_1(alice_file, bob_file)
            pp_times.append(pp_time)
            print(f"    PP: {pp_time:.4f}s")
        
        # Calculate averages
        avg_plain = sum(plain_times) / len(plain_times)
        avg_pp = sum(pp_times) / len(pp_times)
        
        result = {
            "alice_file": alice_file,
            "bob_file": bob_file,
            "runs": runs,
            "plain_times": plain_times,
            "pp_times": pp_times,
            "avg_plain": avg_plain,
            "avg_pp": avg_pp,
            "speedup_factor": avg_pp / avg_plain if avg_plain > 0 else float('inf')
        }
        
        return result
    
    def run_1_to_n_comparison(self, alice_file: str, bob_files: List[str],
                            runs: int = 1) -> Dict:
        """
        Run 1:N comparison benchmark
        
        Args:
            alice_file: Path to Alice's scanpath JSON
            bob_files: List of paths to Bob's scanpath JSONs
            runs: Number of runs to average
            
        Returns:
            Dictionary with timing results
        """
        print(f"Running 1:N comparison: {Path(alice_file).name} vs {len(bob_files)} files")
        
        plain_times = []
        pp_times = []
        
        for run in range(runs):
            print(f"  Run {run + 1}/{runs}")
            
            # Plain EyeBLAST
            plain_time = self.plain_benchmark.benchmark_1_to_n(alice_file, bob_files)
            plain_times.append(plain_time)
            print(f"    Plain: {plain_time:.4f}s")
            
            # PP_EyeBLAST  
            pp_time = self.pp_benchmark.benchmark_1_to_n(alice_file, bob_files)
            pp_times.append(pp_time)
            print(f"    PP: {pp_time:.4f}s")
        
        # Calculate averages
        avg_plain = sum(plain_times) / len(plain_times)
        avg_pp = sum(pp_times) / len(pp_times)
        
        result = {
            "alice_file": alice_file,
            "bob_files": bob_files,
            "database_size": len(bob_files),
            "runs": runs,
            "plain_times": plain_times,
            "pp_times": pp_times,
            "avg_plain": avg_plain,
            "avg_pp": avg_pp,
            "speedup_factor": avg_pp / avg_plain if avg_plain > 0 else float('inf')
        }
        
        return result
    
    def save_results(self, filename: str = "timing_results.json"):
        """Save benchmark results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")


def find_bob_files(bob_dir: str, limit: Optional[int] = None) -> List[str]:
    """Find Bob scanpath files in directory"""
    bob_path = Path(bob_dir)
    json_files = list(bob_path.glob("*.json"))
    
    if limit:
        json_files = json_files[:limit]
    
    return [str(f) for f in json_files]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Plain EyeBLAST vs PP_EyeBLAST Performance"
    )
    parser.add_argument("--alice", default="Alice/query.json",
                       help="Alice's scanpath file")
    parser.add_argument("--bob_single", default="Bob_Single/S_1002_S1_1_VRG_tokens.json",
                       help="Single Bob file for 1:1 comparison")
    parser.add_argument("--bob_dir", default="Bob",
                       help="Directory containing Bob files for 1:N comparison")
    parser.add_argument("--pp_main", default="pp_main",
                       help="pp_main executable name (runs from build/ directory)")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs to average")
    parser.add_argument("--limit_n", type=int, default=10,
                       help="Limit number of Bob files for 1:N test")
    parser.add_argument("--skip_1_to_1", action="store_true",
                       help="Skip 1:1 comparison")
    parser.add_argument("--skip_1_to_n", action="store_true", 
                       help="Skip 1:N comparison")
    parser.add_argument("--output", default="timing_results.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    # Check if pp_main exists and is executable
    pp_main_path = Path("build") / args.pp_main
    if not pp_main_path.exists():
        print(f"Error: pp_main not found at {pp_main_path}")
        print("Make sure to build the project first: cd build && make")
        return
    
    # Test if the executable works
    print(f"Testing executable: {pp_main_path}")
    try:
        test_result = subprocess.run(["./pp_main", "--help"], 
                                   capture_output=True, text=True, timeout=5, cwd="build")
        print(f"Executable test: {'✅ OK' if test_result.returncode == 0 else '❌ Failed'}")
        if test_result.returncode != 0:
            print(f"Error output: {test_result.stderr}")
    except Exception as e:
        print(f"❌ Executable test failed: {e}")
        print("Try running manually first to verify it works")
    
    runner = BenchmarkRunner(args.pp_main)
    
    print("=" * 60)
    print("Plain EyeBLAST vs PP_EyeBLAST Performance Benchmark")
    print("=" * 60)
    
    # 1:1 Comparison
    if not args.skip_1_to_1:
        print("\n1:1 Comparison Benchmark")
        print("-" * 30)
        
        result_1_to_1 = runner.run_1_to_1_comparison(
            args.alice, args.bob_single, args.runs
        )
        runner.results["plain_eyeblast"]["1_to_1"] = result_1_to_1
        runner.results["pp_eyeblast"]["1_to_1"] = result_1_to_1
        
        print(f"\n1:1 Results:")
        print(f"  Plain avg: {result_1_to_1['avg_plain']:.4f}s")
        print(f"  PP avg: {result_1_to_1['avg_pp']:.4f}s") 
        print(f"  PP/Plain ratio: {result_1_to_1['speedup_factor']:.2f}x")
    
    # 1:N Comparison
    if not args.skip_1_to_n:
        print(f"\n1:N Comparison Benchmark (max {args.limit_n} files)")
        print("-" * 30)
        
        bob_files = find_bob_files(args.bob_dir, args.limit_n)
        if not bob_files:
            print(f"No JSON files found in {args.bob_dir}")
        else:
            result_1_to_n = runner.run_1_to_n_comparison(
                args.alice, bob_files, args.runs
            )
            runner.results["plain_eyeblast"]["1_to_n"] = result_1_to_n
            runner.results["pp_eyeblast"]["1_to_n"] = result_1_to_n
            
            print(f"\n1:N Results ({len(bob_files)} files):")
            print(f"  Plain avg: {result_1_to_n['avg_plain']:.4f}s")
            print(f"  PP avg: {result_1_to_n['avg_pp']:.4f}s")
            print(f"  PP/Plain ratio: {result_1_to_n['speedup_factor']:.2f}x")
    
    # Save results
    runner.results["metadata"] = {
        "alice_file": args.alice,
        "bob_single": args.bob_single,
        "bob_dir": args.bob_dir,
        "runs": args.runs,
        "timestamp": time.time()
    }
    
    runner.save_results(args.output)
    
    print(f"\n" + "=" * 60)
    print("Benchmark completed!")
    print(f"Results saved to: {args.output}")
    print("Run plot_timing_comparison.py to generate graphs")
    print("=" * 60)


if __name__ == "__main__":
    main()