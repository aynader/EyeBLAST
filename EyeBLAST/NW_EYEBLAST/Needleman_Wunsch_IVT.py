import os
import time
import gc
import psutil
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from itertools import combinations
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        # Dummy decorator when numba is not available
        return func
from joblib import Parallel, delayed
from tqdm import tqdm
import json
import glob
from collections import defaultdict

# -----------------------------------------------------------------------------
# String-based Needleman-Wunsch with optional Numba JIT
# -----------------------------------------------------------------------------
def preprocess_tokens(seq1, seq2):
    """Preprocess tokens to extract types and lengths (non-JIT function)"""
    seq1_types = np.zeros(len(seq1), dtype=np.int32)
    seq1_lengths = np.zeros(len(seq1), dtype=np.int32)
    seq2_types = np.zeros(len(seq2), dtype=np.int32)
    seq2_lengths = np.zeros(len(seq2), dtype=np.int32)
    
    for i, token in enumerate(seq1):
        # Make sure we're looking at a string
        if isinstance(token, str):
            token_type = ord(token[0])  # Convert char to int
            token_length = int(token[1:]) if token[1:].isdigit() else 0
            seq1_types[i] = token_type
            seq1_lengths[i] = token_length
    
    for i, token in enumerate(seq2):
        # Make sure we're looking at a string
        if isinstance(token, str):
            token_type = ord(token[0])  # Convert char to int
            token_length = int(token[1:]) if token[1:].isdigit() else 0
            seq2_types[i] = token_type
            seq2_lengths[i] = token_length
    
    return seq1_types, seq1_lengths, seq2_types, seq2_lengths

# Non-JIT version of Needleman-Wunsch
def nw_score_no_jit(seq1_types, seq1_lengths, seq2_types, seq2_lengths, match=1, mismatch=-1, gap=-2):
    """Compute Needleman-Wunsch score without using Numba JIT"""
    n, m = len(seq1_types), len(seq2_types)
    
    # Memory optimization: only keep current and previous rows
    # If n > m, swap sequences to minimize memory usage
    if n > m:
        seq1_types, seq2_types = seq2_types, seq1_types
        seq1_lengths, seq2_lengths = seq2_lengths, seq1_lengths
        n, m = m, n
    
    # Only need two rows: previous and current
    prev_row = np.zeros(m+1, dtype=np.float32)
    curr_row = np.zeros(m+1, dtype=np.float32)
    
    # Initialize first row with gap penalties
    for j in range(m+1):
        prev_row[j] = j * gap
    
    # Fill the score matrix row by row
    for i in range(1, n+1):
        # Initialize first column of current row
        curr_row[0] = i * gap
        
        for j in range(1, m+1):
            # If token types match
            if seq1_types[i-1] == seq2_types[j-1]:
                # Calculate length similarity ratio
                min_len = min(seq1_lengths[i-1], seq2_lengths[j-1])
                max_len = max(seq1_lengths[i-1], seq2_lengths[j-1])
                if max_len == 0:
                    similarity = 0.0
                else:
                    similarity = min_len / max_len
                
                match_score = match * similarity  # Scale match score by similarity
            else:
                match_score = mismatch
            
            # Choose the maximum score
            curr_row[j] = max(
                prev_row[j-1] + match_score,  # Match or mismatch
                prev_row[j] + gap,            # Gap in seq2
                curr_row[j-1] + gap           # Gap in seq1
            )
        
        # Swap rows for next iteration
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[m]

@njit
def nw_score_numba(seq1_types, seq1_lengths, seq2_types, seq2_lengths, match=1, mismatch=-1, gap=-2):
    """Compute Needleman-Wunsch score using preprocessed token types and lengths with Numba JIT (space-optimized)"""
    n, m = len(seq1_types), len(seq2_types)
    
    # Memory optimization: only keep current and previous rows
    # If n > m, swap sequences to minimize memory usage
    if n > m:
        seq1_types, seq2_types = seq2_types, seq1_types
        seq1_lengths, seq2_lengths = seq2_lengths, seq1_lengths
        n, m = m, n
    
    # Only need two rows: previous and current
    prev_row = np.zeros(m+1, dtype=np.float32)
    curr_row = np.zeros(m+1, dtype=np.float32)
    
    # Initialize first row with gap penalties
    for j in range(m+1):
        prev_row[j] = j * gap
    
    # Fill the score matrix row by row
    for i in range(1, n+1):
        # Initialize first column of current row
        curr_row[0] = i * gap
        
        for j in range(1, m+1):
            # If token types match
            if seq1_types[i-1] == seq2_types[j-1]:
                # Calculate length similarity ratio
                min_len = min(seq1_lengths[i-1], seq2_lengths[j-1])
                max_len = max(seq1_lengths[i-1], seq2_lengths[j-1])
                if max_len == 0:
                    similarity = 0.0
                else:
                    similarity = min_len / max_len
                
                match_score = match * similarity  # Scale match score by similarity
            else:
                match_score = mismatch
            
            # Choose the maximum score
            curr_row[j] = max(
                prev_row[j-1] + match_score,  # Match or mismatch
                prev_row[j] + gap,            # Gap in seq2
                curr_row[j-1] + gap           # Gap in seq1
            )
        
        # Swap rows for next iteration
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[m]

def nw_score(seq1_types, seq1_lengths, seq2_types, seq2_lengths, match=1, mismatch=-1, gap=-2, use_numba=True):
    """
    Compute Needleman-Wunsch score with option to use Numba JIT or not
    
    Parameters:
    -----------
    seq1_types, seq2_types : np.ndarray
        Token type arrays
    seq1_lengths, seq2_lengths : np.ndarray
        Token length arrays
    match : float
        Match score
    mismatch : float
        Mismatch penalty
    gap : float
        Gap penalty
    use_numba : bool
        Whether to use Numba JIT acceleration
        
    Returns:
    --------
    float
        Alignment score
    """
    if use_numba and NUMBA_AVAILABLE:
        return nw_score_numba(seq1_types, seq1_lengths, seq2_types, seq2_lengths, match, mismatch, gap)
    else:
        return nw_score_no_jit(seq1_types, seq1_lengths, seq2_types, seq2_lengths, match, mismatch, gap)

def nw_score_string(seq1, seq2, match=1, mismatch=-1, gap=-2, use_numba=True):
    """Wrapper function that preprocesses tokens and calls the appropriate scoring function"""
    # Preprocess tokens (extract types and lengths)
    seq1_types, seq1_lengths, seq2_types, seq2_lengths = preprocess_tokens(seq1, seq2)
    
    # Call the appropriate scoring function
    score = nw_score(seq1_types, seq1_lengths, seq2_types, seq2_lengths, match, mismatch, gap, use_numba)
    
    return score

def compute_pair(args):
    """Compute similarity for a pair of scanpaths"""
    k1, k2, seq1, seq2, use_numba = args
    sim = nw_score_string(seq1, seq2, use_numba=use_numba)
    
    # Normalize by the length of the shorter sequence
    shorter_len = min(len(seq1), len(seq2))
    if shorter_len > 0:
        sim = sim / (shorter_len * 1.0)  # Assuming match=1
        
    return k1, k2, sim

def parse_args():
    parser = argparse.ArgumentParser(description='Compute Needleman-Wunsch similarity matrix')
    parser.add_argument('--input_dir', type=str, default='GazeBaseVR_IVT',
                        help='Directory containing tokenized scanpath JSON files')
    parser.add_argument('--output_dir', type=str, default='NW_Results',
                        help='Directory to save results')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--sample_percentage', type=int, default=100,
                        help='Percentage of data to use (randomly sampled)')
    parser.add_argument('--use_numba', action='store_true', default=True,
                        help='Use Numba JIT acceleration (default: True)')
    return parser.parse_args()

def load_and_prepare_data(input_dir, sample_percentage=100):
    """
    Load tokenized scanpaths and extract task labels
    
    Parameters:
    -----------
    input_dir : str
        Directory containing token JSON files
    sample_percentage : int
        Percentage of data to use (randomly sampled)
        
    Returns:
    --------
    list
        List of scanpaths (token lists)
    list
        Task labels
    dict
        Mapping of subject IDs to indices
    """
    files = glob.glob(os.path.join(input_dir, "*.json"))
    
    # Sample X% of files if requested
    if sample_percentage < 100:
        import random
        num_files = int(len(files) * sample_percentage / 100)
        print(f"Sampling {num_files} files ({sample_percentage}% of {len(files)})")
        files = random.sample(files, num_files)
    
    scanpaths = []
    labels = []
    subjects = {}
    
    # Tasks in GazeBaseVR
    # 1_VRG: VR Game
    # 2_PUR: Pursuit
    # 3_VID: Video
    # 4_TEX: Text
    # 5_RAN: Random
    
    print(f"Loading {len(files)} scanpath files...")
    for file_path in tqdm(files, desc="Loading scanpaths", leave=True):
        filename = os.path.basename(file_path)
        # Parse filename: S_XXXX_SY_Z_TTT_tokens.json
        # XXXX: subject ID, Y: session, Z: task number, TTT: task code
        parts = filename.split('_')
        subject_id = parts[1]
        task_num = int(parts[3])
        
        with open(file_path, 'r') as f:
            scanpath = json.load(f)
            
        scanpaths.append(scanpath)
        labels.append(task_num)
        
        # Track subject IDs for potential subject-wise split
        if subject_id not in subjects:
            subjects[subject_id] = []
        subjects[subject_id].append(len(scanpaths) - 1)
    
    return scanpaths, labels, subjects

def compute_similarity_matrix(scanpaths, n_jobs=-1, use_numba=True):
    """Compute pairwise Needleman-Wunsch similarity matrix for all scanpaths"""
    n = len(scanpaths)
    print(f"Computing pairwise similarities for {n} scanpaths ({n*(n-1)//2} pairs)...")
    
    # Check sequence lengths and warn about memory usage
    max_len = max(len(seq) for seq in scanpaths)
    avg_len = np.mean([len(seq) for seq in scanpaths])
    print(f"Sequence length stats: max={max_len}, avg={avg_len:.1f}")
    
    # Estimate memory usage per comparison (space-optimized version uses O(min(n,m)) space)
    estimated_memory_mb = (max_len * 4) / (1024 * 1024)  # 4 bytes per float32, only need min(n,m) space
    print(f"Estimated memory per NW comparison: {estimated_memory_mb:.1f} MB")
    
    # Adjust batch size based on sequence lengths and available memory
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Available system memory: {available_memory_gb:.1f} GB")
    
    # Calculate safe batch size based on memory
    if estimated_memory_mb > 50:  # If each comparison uses >50MB
        max_concurrent = max(1, int(available_memory_gb * 1024 / estimated_memory_mb))
        if n_jobs > 0:
            safe_batch_size = max(1, min(100, max_concurrent // n_jobs))
        else:
            safe_batch_size = max(1, min(100, max_concurrent // 4))  # Assume 4 cores if n_jobs=-1
        print(f"Using smaller batch size due to large sequences: {safe_batch_size}")
    else:
        safe_batch_size = min(500, len(scanpaths) * (len(scanpaths) - 1) // 2)
    
    # Prepare all pairs for parallel processing
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, scanpaths[i], scanpaths[j], use_numba))
    
    # Process in smaller batches to manage memory
    batch_size = min(safe_batch_size, len(pairs))
    results = []
    
    print(f"Processing {len(pairs)} pairs in batches of {batch_size}")
    
    for batch_idx, i in enumerate(range(0, len(pairs), batch_size)):
        batch = pairs[i:i+batch_size]
        print(f"Processing batch {batch_idx + 1}/{(len(pairs) + batch_size - 1)//batch_size} ({len(batch)} pairs)")
        
        # Monitor memory before batch
        mem_before = psutil.virtual_memory().percent
        print(f"Memory usage before batch: {mem_before:.1f}%")
        
        try:
            batch_results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(compute_pair)(pair) for pair in batch
            )
            results.extend(batch_results)
            
            # Monitor memory after batch
            mem_after = psutil.virtual_memory().percent
            print(f"Memory usage after batch: {mem_after:.1f}%")
            
        except MemoryError as e:
            print(f"MemoryError in batch {batch_idx + 1}: {e}")
            print("Falling back to sequential processing for this batch...")
            
            # Process batch sequentially as fallback
            batch_results = []
            for pair in tqdm(batch, desc=f"Sequential batch {batch_idx + 1}"):
                try:
                    result = compute_pair(pair)
                    batch_results.append(result)
                    gc.collect()  # Force garbage collection after each pair
                except Exception as pair_error:
                    print(f"Error processing pair: {pair_error}")
                    # Skip this pair and continue
                    continue
            
            results.extend(batch_results)
        
        # Force garbage collection after each batch
        gc.collect()
        
        # If memory usage is high, wait a bit
        if psutil.virtual_memory().percent > 85:
            print("High memory usage detected, waiting for cleanup...")
            time.sleep(2)
            gc.collect()
    
    print(f"Successfully computed {len(results)} similarity pairs")
    
    # Create similarity matrix
    sim_matrix = np.zeros((n, n))
    for i, j, sim in results:
        sim_matrix[i, j] = sim
        sim_matrix[j, i] = sim  # Symmetric
    
    # Set diagonal to 1 (self-similarity)
    np.fill_diagonal(sim_matrix, 1.0)
    
    return sim_matrix

def knn_classification(sim_matrix, labels, test_indices, k=5):
    """Perform k-NN classification based on similarity matrix"""
    train_indices = [i for i in range(len(labels)) if i not in test_indices]
    
    predictions = []
    for test_idx in test_indices:
        # Get similarities to all training samples
        similarities = [(train_idx, sim_matrix[test_idx, train_idx]) 
                        for train_idx in train_indices]
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k neighbors
        top_k = similarities[:k]
        
        # Get labels of neighbors
        neighbor_labels = [labels[idx] for idx, _ in top_k]
        
        # Majority vote
        from collections import Counter
        prediction = Counter(neighbor_labels).most_common(1)[0][0]
        predictions.append(prediction)
    
    return predictions

def evaluate_classification(true_labels, pred_labels):
    """Evaluate classification performance"""
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    accuracy = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, output_dict=True)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report
    }

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    scanpaths, labels, subjects = load_and_prepare_data(args.input_dir, args.sample_percentage)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Split data into train/test (80/20)
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(scanpaths))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Compute similarity matrix
    start_time = time.time()
    sim_matrix = compute_similarity_matrix(scanpaths, n_jobs=args.n_jobs, use_numba=args.use_numba)
    nw_time = time.time() - start_time
    print(f"Needleman-Wunsch similarity computation completed in {nw_time:.2f} seconds")
    
    # Save similarity matrix
    np.save(os.path.join(args.output_dir, 'nw_similarity_matrix.npy'), sim_matrix)
    
    # Perform classification
    print("Performing k-NN classification...")
    k_values = [1, 3, 5, 7, 9]
    results = {}
    
    for k in k_values:
        print(f"Using k={k}...")
        y_pred = knn_classification(sim_matrix, labels, test_indices, k=k)
        y_true = [labels[i] for i in test_indices]
        
        # Evaluate
        eval_results = evaluate_classification(y_true, y_pred)
        results[k] = eval_results
        print(f"k={k}, Accuracy: {eval_results['accuracy']:.4f}")
    
    # Save results
    with open(os.path.join(args.output_dir, 'nw_classification_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print best result
    best_k = max(results.keys(), key=lambda k: results[k]['accuracy'])
    print(f"Best k={best_k}, Accuracy: {results[best_k]['accuracy']:.4f}")
    
    print(f"All results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
