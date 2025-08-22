#!/usr/bin/env python3
"""
Test script to run ONLY Plain EyeBLAST timing
This avoids the PP_EyeBLAST subprocess issues
"""

import json
import time
import sys
from pathlib import Path

# Add PythonVsCpp to path
pythonvscpp_path = Path(__file__).parent / "PythonVsCpp"
sys.path.insert(0, str(pythonvscpp_path))

try:
    from Fast_EyeBLAST_IVT import FastEyeBLAST
    print("✅ Successfully imported FastEyeBLAST")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Looking in: {pythonvscpp_path}")
    sys.exit(1)

def test_plain_eyeblast():
    """Test plain EyeBLAST timing"""
    print("Testing Plain EyeBLAST timing...")
    
    # Initialize
    eyeblast = FastEyeBLAST()
    
    # Load test files
    alice_file = "Alice/query.json"
    bob_file = "Bob_Single/S_1002_S1_1_VRG_tokens.json"
    
    if not Path(alice_file).exists():
        print(f"❌ Alice file not found: {alice_file}")
        return
    if not Path(bob_file).exists():
        print(f"❌ Bob file not found: {bob_file}")
        return
    
    print(f"📁 Alice: {alice_file}")
    print(f"📁 Bob: {bob_file}")
    
    # Load scanpaths
    with open(alice_file) as f:
        scanpath1 = json.load(f)
    with open(bob_file) as f:
        scanpath2 = json.load(f)
    
    # Test multiple runs
    times = []
    for run in range(3):
        print(f"🏃 Run {run + 1}/3")
        
        start_time = time.perf_counter()
        
        # Encode scanpaths
        encoded1 = eyeblast._encode_scanpath(scanpath1)
        encoded2 = eyeblast._encode_scanpath(scanpath2)
        
        # Initialize bit-packing
        eyeblast._initialize_bit_packing()
        
        # Compute similarity
        score = eyeblast._similarity_score(encoded1, encoded2)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        times.append(elapsed)
        print(f"  ⏱️ Time: {elapsed:.4f}s, Score: {score:.4f}")
    
    avg_time = sum(times) / len(times)
    print(f"\n📊 Results:")
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  All times: {[f'{t:.4f}s' for t in times]}")
    
    return avg_time, times

if __name__ == "__main__":
    print("=" * 50)
    print("Plain EyeBLAST Test")
    print("=" * 50)
    
    test_plain_eyeblast()
    
    print("\n✅ Plain EyeBLAST test completed!")
    print("\nTo run the full benchmark with PP_EyeBLAST:")
    print("1. Open WSL terminal")
    print("2. Navigate to this directory")
    print("3. Run: python3 benchmark_plain_vs_pp.py --skip_1_to_n --runs 1")