#!/usr/bin/env python3
"""
Script to compute plain EyeBLAST similarity between Alice and Bob token files.
This will help verify the expected score of 0.8375 and understand if the issue is with the PP implementation.
"""

import json
import os
import sys
from Fast_EyeBLAST_IVT import FastEyeBLAST

def load_token_file(filepath):
    """Load tokens from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    # File paths
    alice_file = "ALICEBOB/alice_tokens.json"
    bob_file = "ALICEBOB/bob_tokens.json"
    
    print("Loading token files...")
    
    # Load Alice and Bob's tokens
    alice_tokens = load_token_file(alice_file)
    bob_tokens = load_token_file(bob_file)
    
    print(f"Alice has {len(alice_tokens)} tokens")
    print(f"Bob has {len(bob_tokens)} tokens")
    
    # Preview first few tokens
    print(f"Alice first 10 tokens: {alice_tokens[:10]}")
    print(f"Bob first 10 tokens: {bob_tokens[:10]}")
    
    # Initialize FastEyeBLAST
    print("\nInitializing FastEyeBLAST...")
    fe = FastEyeBLAST(use_parallel=False)
    
    # We need to "train" the model first to build the token vocabulary
    # Use both Alice and Bob's data for training
    scanpaths = [alice_tokens, bob_tokens]
    labels = [0, 1]  # Dummy labels
    
    print("Training FastEyeBLAST (building vocabulary)...")
    fe.fit(scanpaths, labels)
    
    # Encode the scanpaths
    print("Encoding scanpaths...")
    alice_encoded = fe._encode_scanpath(alice_tokens)
    bob_encoded = fe._encode_scanpath(bob_tokens)
    
    print(f"Alice encoded length: {len(alice_encoded)}")
    print(f"Bob encoded length: {len(bob_encoded)}")
    
    # Compute similarity
    print("\nComputing similarity score...")
    similarity = fe._similarity_score(alice_encoded, bob_encoded)
    
    print(f"\nPlain EyeBLAST similarity between Alice and Bob: {similarity:.6f}")
    print(f"Expected score: 0.8375")
    print(f"Difference from expected: {abs(similarity - 0.8375):.6f}")
    
    if abs(similarity - 0.8375) < 0.001:
        print("Score matches expected value!")
    else:
        print("Score differs from expected value")
        print("This could indicate:")
        print("  - Different algorithm parameters")
        print("  - Different input data")
        print("  - Implementation differences")
        
    # Debug: Let's try to understand what's happening step by step
    print("\n=== DEBUGGING ===")
    
    # Pack sequences into words
    packed1 = fe._pack_ids_into_words(alice_encoded)
    packed2 = fe._pack_ids_into_words(bob_encoded)
    print(f"Packed Alice length: {len(packed1)}")
    print(f"Packed Bob length: {len(packed2)}")
    
    # Extract seeds
    seeds = fe._extract_seeds(packed1, packed2)
    print(f"Number of seeds found: {len(seeds)}")
    
    if len(seeds) > 0:
        print(f"First few seeds: {seeds[:5]}")
        
        # Chain seeds
        chain = fe._greedy_chain_seeds(seeds, len(alice_encoded), len(bob_encoded))
        print(f"Chain length: {len(chain)}")
        
        # Calculate banded Levenshtein distance
        score, alignment = fe._banded_levenshtein(alice_encoded, bob_encoded, fe.band_width, chain)
        print(f"Raw score: {score}")
        
        # Check normalization
        max_possible_score = max(len(alice_encoded), len(bob_encoded)) * 2
        print(f"Max possible score: {max_possible_score}")
        normalized_score = max(0, score) / max_possible_score
        print(f"Normalized score: {normalized_score}")
    else:
        print("No seeds found - this explains the 0 similarity!")
        print("Let's check the token differences...")
        
        # Compare some tokens
        alice_set = set(alice_tokens)
        bob_set = set(bob_tokens)
        common_tokens = alice_set & bob_set
        print(f"Common tokens between Alice and Bob: {len(common_tokens)}")
        print(f"Alice unique tokens: {len(alice_set - bob_set)}")
        print(f"Bob unique tokens: {len(bob_set - alice_set)}")
        
        if len(common_tokens) > 0:
            print(f"Sample common tokens: {list(common_tokens)[:10]}")
        
        alice_only = alice_set - bob_set
        if len(alice_only) > 0:
            print(f"Sample Alice-only tokens: {list(alice_only)[:10]}")
            
        bob_only = bob_set - alice_set
        if len(bob_only) > 0:
            print(f"Sample Bob-only tokens: {list(bob_only)[:10]}")

if __name__ == "__main__":
    main()