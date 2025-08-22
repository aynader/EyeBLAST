import os
import time
import gc
import psutil
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
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
# Character-based Needleman-Wunsch with optional Numba JIT
# -----------------------------------------------------------------------------
@njit
def nw_score_numba(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """Compute Needleman-Wunsch score for character sequences with Numba JIT"""
    n, m = len(seq1), len(seq2)
    
    # Initialize the score matrix
    score = np.zeros((n+1, m+1), dtype=np.float32)
    
    # Initialize first row and column with gap penalties
    for i in range(n+1):
        score[i, 0] = i * gap
    for j in range(m+1):
        score[0, j] = j * gap
    
    # Fill the score matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            # If symbols match
            if seq1[i-1] == seq2[j-1]:
                match_score = match
            else:
                match_score = mismatch
            
            # Choose the maximum score
            score[i, j] = max(
                score[i-1, j-1] + match_score,  # Match or mismatch
                score[i-1, j] + gap,            # Gap in seq2
                score[i, j-1] + gap             # Gap in seq1
            )
    
    return score[n, m]

@njit
def banded_nw_score_numba(seq1, seq2, band_width=100, match=1, mismatch=-1, gap=-2):
    """
    Compute banded Needleman-Wunsch score for character sequences with Numba JIT.
    Uses significantly less memory by only computing a band around the diagonal.
    """
    n, m = len(seq1), len(seq2)
    
    # For highly different length sequences, adapt the band width
    if abs(n - m) > band_width:
        band_width = min(abs(n - m) + 50, min(n, m))
    
    # Initialize the score matrix (only store the band)
    # We store a matrix of size n+1 x (2*band_width+1)
    score = np.zeros((n+1, 2*band_width+1), dtype=np.float32)
    
    # Fill with a low value to mark cells outside the band
    inf_val = -1e9
    score.fill(inf_val)
    
    # Initialize first row and column with gap penalties
    # Only initialize cells within the band
    for i in range(min(n+1, band_width+1)):
        score[i, band_width-i] = i * gap
    
    for j in range(min(m+1, band_width+1)):
        if j < 2*band_width+1:
            score[0, j+band_width] = j * gap
    
    # Fill the score matrix
    for i in range(1, n+1):
        # Calculate range of j values that fall within the band
        j_start = max(1, i - band_width)
        j_end = min(m+1, i + band_width + 1)
        
        for j in range(j_start, j_end):
            # Map j to the band index
            band_j = band_width + j - i
            
            # If within valid band index range
            if 0 <= band_j < 2*band_width+1:
                # Diagonal score (match/mismatch)
                if 0 <= band_width + (j-1) - (i-1) < 2*band_width+1:
                    diag = score[i-1, band_width + (j-1) - (i-1)]
                else:
                    diag = inf_val
                
                # Gap in seq1
                if 0 <= band_width + (j-1) - i < 2*band_width+1:
                    left = score[i, band_width + (j-1) - i]
                else:
                    left = inf_val
                
                # Gap in seq2
                if 0 <= band_width + j - (i-1) < 2*band_width+1:
                    up = score[i-1, band_width + j - (i-1)]
                else:
                    up = inf_val
                
                # If symbols match
                if seq1[i-1] == seq2[j-1]:
                    match_score = match
                else:
                    match_score = mismatch
                
                # Choose the maximum score if all valid
                if diag != inf_val or left != inf_val or up != inf_val:
                    score[i, band_j] = max(
                        diag + match_score if diag != inf_val else inf_val,  # Match or mismatch
                        up + gap if up != inf_val else inf_val,              # Gap in seq2
                        left + gap if left != inf_val else inf_val           # Gap in seq1
                    )
    
    # Return the final score if it was computed, otherwise return a large negative value
    if 0 <= band_width + m - n < 2*band_width+1:
        return score[n, band_width + m - n]
    else:
        return inf_val

def nw_score_no_jit(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """Compute Needleman-Wunsch score for character sequences without using Numba JIT"""
    n, m = len(seq1), len(seq2)
    
    # Check if sequences are too large for full matrix - use more conservative threshold
    if n*m > 10_000_000:  # 40MB for float32 matrix (reduced from 100_000_000)
        # Use a simpler approach - count matching characters
        matches = sum(1 for i, j in zip(seq1, seq2) if i == j)
        if min(n, m) > 0:
            return matches / min(n, m) * match
        else:
            return 0
    
    # Initialize the score matrix
    score = np.zeros((n+1, m+1), dtype=np.float32)
    
    # Initialize first row and column with gap penalties
    for i in range(n+1):
        score[i, 0] = i * gap
    for j in range(m+1):
        score[0, j] = j * gap
    
    # Fill the score matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            # If symbols match
            if seq1[i-1] == seq2[j-1]:
                match_score = match
            else:
                match_score = mismatch
            
            # Choose the maximum score
            score[i, j] = max(
                score[i-1, j-1] + match_score,  # Match or mismatch
                score[i-1, j] + gap,            # Gap in seq2
                score[i, j-1] + gap             # Gap in seq1
            )
    
    return score[n, m]

def nw_score(seq1, seq2, match=1, mismatch=-1, gap=-2, use_numba=True, max_seq_length=5000, band_width=100):
    """
    Compute Needleman-Wunsch score with option to use Numba JIT or not
    
    Parameters:
    -----------
    seq1, seq2 : str
        Symbol sequences to compare
    match : float
        Match score
    mismatch : float
        Mismatch penalty
    gap : float
        Gap penalty
    use_numba : bool
        Whether to use Numba JIT acceleration
    max_seq_length : int
        Maximum sequence length to process (longer sequences will be truncated)
    band_width : int
        Width of the band for banded alignment (used for memory efficiency)
        
    Returns:
    --------
    float
        Alignment score
    """
    # Truncate very long sequences to avoid memory issues (reduced from 10000)
    if len(seq1) > max_seq_length:
        seq1 = seq1[:max_seq_length]
    if len(seq2) > max_seq_length:
        seq2 = seq2[:max_seq_length]
    
    # Convert strings to numpy arrays for Numba compatibility
    seq1_arr = np.array([ord(c) for c in seq1], dtype=np.int32)
    seq2_arr = np.array([ord(c) for c in seq2], dtype=np.int32)
    
    # Check if sequences are too large for standard alignment - reduced threshold
    total_cells = len(seq1) * len(seq2)
    if use_numba and NUMBA_AVAILABLE:
        if total_cells > 5_000_000:  # 20MB for float32 matrix (reduced from 25_000_000)
            # Use banded alignment for large sequences
            return banded_nw_score_numba(seq1_arr, seq2_arr, band_width, match, mismatch, gap)
        else:
            # Use standard alignment for smaller sequences
            return nw_score_numba(seq1_arr, seq2_arr, match, mismatch, gap)
    else:
        return nw_score_no_jit(seq1_arr, seq2_arr, match, mismatch, gap)

def compute_pair(args):
    """Compute similarity for a pair of scanpaths"""
    k1, k2, seq1, seq2, use_numba, max_seq_length, band_width = args
    try:
        sim = nw_score(seq1, seq2, use_numba=use_numba, max_seq_length=max_seq_length, band_width=band_width)
        
        # Normalize by the length of the shorter sequence
        shorter_len = min(len(seq1), len(seq2))
        if shorter_len > 0:
            sim = sim / (shorter_len * 1.0)  # Assuming match=1
            
        return k1, k2, sim
    except MemoryError:
        print(f"Memory error processing pair ({k1}, {k2}) - sequences might be too large")
        # Return a default similarity of 0 in case of memory error
        return k1, k2, 0.0

def estimate_memory_usage(scanpaths, batch_size):
    """Estimate memory usage for processing a batch of scanpath pairs"""
    if not scanpaths:
        return 0
    
    # Calculate average sequence length
    avg_len = sum(len(seq) for seq in scanpaths) / len(scanpaths)
    
    # Estimate memory per comparison (rough estimate)
    # Dynamic programming matrix + overhead
    mem_per_comparison = (avg_len ** 2) * 4  # 4 bytes per float32
    
    # Total memory for batch
    total_mem = mem_per_comparison * batch_size
    
    return total_mem

def load_and_prepare_data(input_dir, sample_percentage=100, max_seq_length=None):
    """
    Load AOI-encoded scanpaths from JSON files
    
    Parameters:
    -----------
    input_dir : str
        Directory containing encoded_sequences.json
    sample_percentage : int
        Percentage of data to use (randomly sampled)
    max_seq_length : int or None
        Maximum sequence length to load (None for no limit)
        
    Returns:
    --------
    dict
        Dictionary of scanpath sequences
    list
        Task labels (if available)
    dict
        Subject information
    """
    json_path = os.path.join(input_dir, "encoded_sequences.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No encoded sequences found at {json_path}")
    
    print(f"Loading encoded sequences from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Sample X% of data if requested
    if sample_percentage < 100:
        import random
        num_items = int(len(data) * sample_percentage / 100)
        print(f"Sampling {num_items} sequences ({sample_percentage}% of {len(data)})")
        keys = random.sample(list(data.keys()), num_items)
        data = {k: data[k] for k in keys}
    
    print(f"Loaded {len(data)} sequences")
    
    # Extract sequences and file paths
    sequences = {}
    labels = []
    subjects = {}
    
    # Track sequence lengths
    seq_lengths = []
    
    for file_id, item in data.items():
        sequence = item['sequence']
        file_path = item['file_path']
        
        # Skip empty sequences
        if not sequence:
            continue
        
        # Truncate very long sequences if max_seq_length is specified
        if max_seq_length and len(sequence) > max_seq_length:
            print(f"Truncating sequence {file_id} from {len(sequence)} to {max_seq_length} characters")
            sequence = sequence[:max_seq_length]
        
        sequences[file_id] = sequence
        seq_lengths.append(len(sequence))
        
        # Extract task information from filename if possible
        # Format varies by dataset, here we try to extract task number
        parts = os.path.basename(file_path).split('_')
        
        # For GazeBaseVR format: S_XXXX_SY_Z_TTT.csv
        # XXXX: subject ID, Y: session, Z: task number, TTT: task code
        try:
            if len(parts) >= 4:
                subject_id = parts[1] if parts[0] == 'S' else parts[0]
                task_num = int(parts[3]) if parts[3].isdigit() else 0
                
                # Store task label
                labels.append(task_num)
                
                # Track subject IDs
                if subject_id not in subjects:
                    subjects[subject_id] = []
                subjects[subject_id].append(len(sequences) - 1)
        except Exception as e:
            # If task extraction fails, use a default label
            labels.append(0)
    
    # Print sequence length statistics
    if seq_lengths:
        min_len = min(seq_lengths)
        max_len = max(seq_lengths)
        avg_len = sum(seq_lengths) / len(seq_lengths)
        print(f"Sequence length statistics: min={min_len}, max={max_len}, avg={avg_len:.2f}")
    
    # Convert sequences to list form for easier handling
    seq_list = list(sequences.values())
    file_ids = list(sequences.keys())
    
    return seq_list, labels, subjects, file_ids

def compute_similarity_matrix(scanpaths, n_jobs=-1, use_numba=True, max_seq_length=5000, band_width=100, batch_size=None):
    """Compute pairwise Needleman-Wunsch similarity matrix for all scanpaths"""
    n = len(scanpaths)
    print(f"Computing pairwise similarities for {n} scanpaths ({n*(n-1)//2} pairs)...")
    
    # Calculate average sequence length for memory estimates
    avg_len = sum(len(seq) for seq in scanpaths) / max(1, len(scanpaths))
    print(f"Average sequence length: {avg_len:.2f} characters")
    
    # Dynamically adjust max_seq_length based on average length
    if avg_len > 5000:
        orig_max_seq_length = max_seq_length
        max_seq_length = min(max_seq_length, 3000)  # More aggressive truncation for very long sequences
        print(f"Adjusted max_seq_length from {orig_max_seq_length} to {max_seq_length} due to long sequences")
    
    # If batch_size is not specified, determine based on sequence lengths
    if not batch_size:
        if avg_len > 5000:
            batch_size = 50
        elif avg_len > 2000:
            batch_size = 100
        elif avg_len > 1000:
            batch_size = 250
        else:
            batch_size = 500
    
    print(f"Using batch size of {batch_size} pairs")
    
    # Prepare all pairs for parallel processing
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, scanpaths[i], scanpaths[j], use_numba, max_seq_length, band_width))
    
    # Process in batches to manage memory
    sim_matrix = np.zeros((n, n))
    num_batches = (len(pairs) + batch_size - 1) // batch_size
    
    # Reduce the number of parallel jobs if dealing with very long sequences
    if avg_len > 3000 and n_jobs > 2:
        orig_n_jobs = n_jobs
        n_jobs = max(1, n_jobs // 2)
        print(f"Reducing number of parallel jobs from {orig_n_jobs} to {n_jobs} due to long sequences")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(pairs))
        batch_pairs = pairs[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_pairs)} pairs)...")
        
        # Clear memory before processing each batch
        gc.collect()
        
        # Monitor memory usage
        start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        # Compute batch in parallel
        try:
            results = Parallel(n_jobs=n_jobs, verbose=10, prefer="threads")(
                delayed(compute_pair)(pair) for pair in batch_pairs
            )
            
            # Update similarity matrix
            for i, j, sim in results:
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim  # Symmetric
            
            # Force garbage collection between batches
            gc.collect()
            
            # Monitor memory usage after batch
            end_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
            print(f"Memory usage: {end_mem:.1f} MB (change: {end_mem - start_mem:.1f} MB)")
            
            # Dynamically reduce batch size if memory usage is high
            if end_mem > 2000 and batch_size > 20:  # If using more than 2GB and batch size can be reduced
                new_batch_size = max(10, batch_size // 2)
                print(f"High memory usage detected. Reducing batch size from {batch_size} to {new_batch_size}")
                batch_size = new_batch_size
                # Recalculate number of batches
                num_batches = (len(pairs) - end_idx + batch_size - 1) // batch_size + batch_idx + 1
            
        except MemoryError:
            print(f"Memory error processing batch {batch_idx+1}. Reducing batch size and retrying...")
            # Try with a smaller batch size for this batch
            half_batch = len(batch_pairs) // 2
            if half_batch < 1:
                print(f"Cannot further reduce batch size. Skipping problematic pairs.")
                continue
                
            for sub_batch_idx in range(0, len(batch_pairs), half_batch):
                sub_batch = batch_pairs[sub_batch_idx:sub_batch_idx + half_batch]
                try:
                    # Reduce number of jobs for sub-batch
                    sub_jobs = max(1, n_jobs//2)
                    print(f"Using {sub_jobs} jobs for sub-batch")
                    
                    sub_results = Parallel(n_jobs=sub_jobs, verbose=10, prefer="threads")(
                        delayed(compute_pair)(pair) for pair in sub_batch
                    )
                    
                    for i, j, sim in sub_results:
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim  # Symmetric
                        
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing sub-batch: {e}")
                    # Process one by one as a last resort
                    for pair in sub_batch:
                        try:
                            i, j, sim = compute_pair(pair)
                            sim_matrix[i, j] = sim
                            sim_matrix[j, i] = sim
                        except Exception as e2:
                            print(f"Error processing pair: {e2}. Skipping.")
    
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

def parse_args():
    parser = argparse.ArgumentParser(description='Compute Needleman-Wunsch similarity matrix for AOI-encoded scanpaths')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing encoded_sequences.json file')
    parser.add_argument('--output_dir', type=str, default='NW_Results',
                        help='Directory to save results')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--sample_percentage', type=int, default=100,
                        help='Percentage of data to use (randomly sampled)')
    parser.add_argument('--use_numba', action='store_true', default=True,
                        help='Use Numba JIT acceleration (default: True)')
    parser.add_argument('--task_classification', action='store_true', default=True,
                        help='Perform task classification (if labels available)')
    parser.add_argument('--max_seq_length', type=int, default=5000,
                        help='Maximum sequence length to process (default: 5000)')
    parser.add_argument('--band_width', type=int, default=100,
                        help='Width of band for banded alignment (default: 100)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Number of pairs to process in each batch (default: auto)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract dataset name from input directory
    dataset_name = os.path.basename(os.path.normpath(args.input_dir))
    
    print(f"Processing AOI-encoded scanpaths from {args.input_dir}")
    print(f"Dataset: {dataset_name}")
    print(f"Using Numba: {args.use_numba}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Band width: {args.band_width}")
    
    # Load data
    start_time = time.time()
    scanpaths, labels, subjects, file_ids = load_and_prepare_data(
        args.input_dir, 
        args.sample_percentage, 
        max_seq_length=args.max_seq_length
    )
    load_time = time.time() - start_time
    print(f"Data loading completed in {load_time:.2f} seconds")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Split data into train/test (80/20)
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(scanpaths))
    
    # Only perform classification if we have task labels
    task_classification = args.task_classification and len(labels) == len(scanpaths) and len(set(labels)) > 1
    
    if task_classification:
        print(f"Found {len(set(labels))} different task labels")
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    else:
        print("No task labels found or classification disabled, skipping classification")
        train_indices, test_indices = [], []  # Empty lists as we won't use them
    
    # Compute similarity matrix
    start_time = time.time()
    sim_matrix = compute_similarity_matrix(
        scanpaths, 
        n_jobs=args.n_jobs, 
        use_numba=args.use_numba,
        max_seq_length=args.max_seq_length,
        band_width=args.band_width,
        batch_size=args.batch_size
    )
    nw_time = time.time() - start_time
    print(f"Needleman-Wunsch similarity computation completed in {nw_time:.2f} seconds")
    
    # Save similarity matrix with file IDs as metadata
    np.savez(os.path.join(args.output_dir, f'{dataset_name}_nw_similarity.npz'), 
             similarity=sim_matrix, 
             file_ids=file_ids)
    print(f"Similarity matrix saved to {args.output_dir}/{dataset_name}_nw_similarity.npz")
    
    # Perform classification if task labels are available
    if task_classification:
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
        with open(os.path.join(args.output_dir, f'{dataset_name}_nw_classification_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print best result
        best_k = max(results.keys(), key=lambda k: results[k]['accuracy'])
        print(f"Best k={best_k}, Accuracy: {results[best_k]['accuracy']:.4f}")
    
    print(f"All results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 