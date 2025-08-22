#!/usr/bin/env python3
"""
Quick test to diagnose PP_EyeBLAST issues
"""

import subprocess
import sys
from pathlib import Path

def test_pp_main():
    pp_main = "./build/pp_main"
    
    print("🔍 Diagnosing PP_EyeBLAST issues...")
    print("=" * 50)
    
    # Check if file exists
    if not Path(pp_main).exists():
        print(f"❌ {pp_main} not found")
        return
    else:
        print(f"✅ {pp_main} exists")
    
    # Check if executable
    if not Path(pp_main).is_file():
        print(f"❌ {pp_main} is not a file")
        return
    else:
        print(f"✅ {pp_main} is a file")
    
    # Test help
    print(f"\n🧪 Testing --help:")
    try:
        result = subprocess.run([pp_main, "--help"], 
                              capture_output=True, text=True, timeout=5)
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Stdout: {result.stdout[:200]}...")
        if result.stderr:
            print(f"Stderr: {result.stderr[:200]}...")
    except Exception as e:
        print(f"❌ Help test failed: {e}")
    
    # Test version or basic run
    print(f"\n🧪 Testing basic run (Alice):")
    try:
        result = subprocess.run([pp_main, "--role", "alice", "--infile", "Alice"], 
                              capture_output=True, text=True, timeout=2)
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Stdout: {result.stdout[:300]}...")
        if result.stderr:
            print(f"Stderr: {result.stderr[:300]}...")
    except subprocess.TimeoutExpired:
        print("⏰ Timeout (expected for Alice waiting for Bob)")
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Diagnosis complete!")

if __name__ == "__main__":
    test_pp_main()