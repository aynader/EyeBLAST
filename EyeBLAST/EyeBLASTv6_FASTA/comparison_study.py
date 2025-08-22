#!/usr/bin/env python3
import os
import argparse
import random
import json
import tempfile
import subprocess
import logging
import time
import sys
import gc
import psutil
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
from tqdm import tqdm
from Bio import SeqIO
from joblib import Parallel, delayed

from Fast_EyeBLAST_DNA import (
    FastEyeBLASTDNA,
    load_fasta_dir,
    DEFAULT_SEED_LENGTH,
    DEFAULT_BAND_WIDTH
)

class NCBIDnaMegaBLASTClassifier:
    def __init__(self, k=5, word_size=28, perc_identity=90, evalue=0.001):
        self.k = k
        self.word_size = word_size
        self.perc_identity = perc_identity
        self.evalue = evalue
        self.temp_dir = None
        self.db_path = None
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.temp_dir = tempfile.mkdtemp(prefix="megablast_")
        fasta_all = os.path.join(self.temp_dir, "train_all.fasta")
        
        print("Creating BLAST database...")
        with open(fasta_all, "w") as outf:
            for idx, seq in enumerate(tqdm(self.X_train, desc="Writing sequences")):
                outf.write(f">train_{idx}\n{seq}\n")
                
        self.db_path = os.path.join(self.temp_dir, "train_db")
        
        print("Running makeblastdb...")
        subprocess.run([
            "makeblastdb", "-in", fasta_all,
            "-dbtype", "nucl", "-out", self.db_path,
            "-parse_seqids"
        ], check=True)
        
        return self

    def predict(self, X_test):
        preds = []
        print("Running MegaBLAST predictions...")
        for idx, seq in enumerate(tqdm(X_test, desc="BLAST queries")):
            qfasta = os.path.join(self.temp_dir, f"query_{idx}.fasta")
            out_tsv = os.path.join(self.temp_dir, f"out_{idx}.tsv")
            with open(qfasta, "w") as f:
                f.write(f">q{idx}\n{seq}\n")
            subprocess.run([
                "blastn", "-task", "megablast",
                "-query", qfasta, "-db", self.db_path,
                "-outfmt", "6 qseqid sseqid bitscore",
                "-word_size", str(self.word_size),
                "-perc_identity", str(self.perc_identity),
                "-evalue", str(self.evalue),
                "-out", out_tsv
            ], check=True)

            hits = []
            with open(out_tsv) as h:
                for line in h:
                    qid, sid, bits = line.strip().split()
                    # sseqid is like 'train_5'
                    train_idx = int(sid.split("_")[1])
                    bitscore = float(bits)
                    hits.append((train_idx, bitscore))

            # take top-k by bitscore
            topk = sorted(hits, key=lambda x: x[1], reverse=True)[: self.k]
            if topk:
                labels = [self.y_train[i] for i, _ in topk]
                pred = Counter(labels).most_common(1)[0][0]
            else:
                pred = Counter(self.y_train).most_common(1)[0][0]
            preds.append(pred)
        return preds


# Needleman-Wunsch implementation from the other project
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        # Dummy decorator when numba is not available
        return func

# Needleman-Wunsch scoring function with numba acceleration
@njit
def nw_score_numba(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """
    Compute Needleman-Wunsch alignment score for two DNA sequences with numba acceleration
    
    Parameters:
    -----------
    seq1 : str or list
        First DNA sequence
    seq2 : str or list
        Second DNA sequence
    match : int
        Score for matching characters
    mismatch : int
        Score for mismatching characters
    gap : int
        Score for gaps
        
    Returns:
    --------
    float
        Alignment score
    """
    m, n = len(seq1), len(seq2)
    
    # Initialize score matrix
    dp = np.zeros((m+1, n+1), dtype=np.float32)
    
    # Initialize first row and column with gap penalties
    for i in range(1, m+1):
        dp[i, 0] = dp[i-1, 0] + gap
    for j in range(1, n+1):
        dp[0, j] = dp[0, j-1] + gap
    
    # Fill the matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Match or mismatch
            if seq1[i-1] == seq2[j-1]:
                diag_score = dp[i-1, j-1] + match
            else:
                diag_score = dp[i-1, j-1] + mismatch
            
            # Gap in seq1 or seq2
            up_score = dp[i-1, j] + gap
            left_score = dp[i, j-1] + gap
            
            # Take the maximum
            dp[i, j] = max(diag_score, up_score, left_score)
    
    # Return final score
    return dp[m, n]

# Non-JIT version for when numba is not available
def nw_score_no_jit(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """
    Compute Needleman-Wunsch alignment score for two DNA sequences without numba
    
    Parameters:
    -----------
    seq1 : str or list
        First DNA sequence
    seq2 : str or list
        Second DNA sequence
    match : int
        Score for matching characters
    mismatch : int
        Score for mismatching characters
    gap : int
        Score for gaps
        
    Returns:
    --------
    float
        Alignment score
    """
    m, n = len(seq1), len(seq2)
    
    # Initialize score matrix
    dp = np.zeros((m+1, n+1), dtype=np.float32)
    
    # Initialize first row and column with gap penalties
    for i in range(1, m+1):
        dp[i, 0] = dp[i-1, 0] + gap
    for j in range(1, n+1):
        dp[0, j] = dp[0, j-1] + gap
    
    # Fill the matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Match or mismatch
            if seq1[i-1] == seq2[j-1]:
                diag_score = dp[i-1, j-1] + match
            else:
                diag_score = dp[i-1, j-1] + mismatch
            
            # Gap in seq1 or seq2
            up_score = dp[i-1, j] + gap
            left_score = dp[i, j-1] + gap
            
            # Take the maximum
            dp[i, j] = max(diag_score, up_score, left_score)
    
    # Return final score
    return dp[m, n]

def nw_score(seq1, seq2, match=1, mismatch=-1, gap=-2, use_numba=True):
    """
    Compute Needleman-Wunsch alignment score for two DNA sequences
    
    Parameters:
    -----------
    seq1 : str or list
        First DNA sequence
    seq2 : str or list
        Second DNA sequence
    match : int
        Score for matching characters
    mismatch : int
        Score for mismatching characters
    gap : int
        Score for gaps
    use_numba : bool
        Whether to use numba acceleration
        
    Returns:
    --------
    float
        Alignment score
    """
    if use_numba and NUMBA_AVAILABLE:
        return nw_score_numba(seq1, seq2, match, mismatch, gap)
    else:
        return nw_score_no_jit(seq1, seq2, match, mismatch, gap)

class NeedlemanWunschDNAClassifier:
    """Classifier using Needleman-Wunsch alignment for DNA sequence comparison"""
    
    def __init__(self, k=5, match=1, mismatch=-1, gap=-2, use_numba=True, n_jobs=-1, batch_size=10, max_seq_length=500):
        """
        Initialize the classifier
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors for classification
        match : int
            Score for matching nucleotides
        mismatch : int
            Score for mismatching nucleotides
        gap : int
            Score for gaps in alignment
        use_numba : bool
            Whether to use numba acceleration
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        batch_size : int
            Batch size for parallel processing
        max_seq_length : int
            Maximum sequence length to consider (longer sequences will be sampled)
        """
        self.k = k
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        self.use_numba = use_numba
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        
        # Check if numba is available
        if use_numba and not NUMBA_AVAILABLE:
            print("Warning: Numba not available, falling back to non-JIT version")
            self.use_numba = False
            
    def _sample_sequence(self, seq):
        """
        Sample a sequence if it's too long
        
        Parameters:
        -----------
        seq : list
            Sequence to sample
            
        Returns:
        --------
        list
            Sampled sequence
        """
        if len(seq) <= self.max_seq_length:
            return seq
            
        # Sample from beginning, middle, and end
        sample_size = self.max_seq_length // 3
        return (seq[:sample_size] + 
                seq[len(seq)//2-sample_size//2:len(seq)//2+sample_size//2] + 
                seq[-sample_size:])
            
    def fit(self, X_train, y_train):
        """
        Train the classifier
        
        Parameters:
        -----------
        X_train : list
            List of training sequences
        y_train : list
            List of training labels
            
        Returns:
        --------
        self : NeedlemanWunschDNAClassifier
            Trained classifier
        """
        self.X_train = X_train
        self.y_train = y_train
        
        # Convert sequences to lists of characters for faster processing
        print("Processing training sequences...")
        self.X_train_processed = []
        
        for seq in tqdm(X_train, desc="Preprocessing"):
            # Convert to list and sample if too long
            processed_seq = self._sample_sequence(list(seq))
            self.X_train_processed.append(processed_seq)
        
        # Precompute sequence characteristics for faster filtering
        self.train_lengths = [len(seq) for seq in self.X_train_processed]
        
        # Precompute 5-mer frequencies for quick filtering
        self.kmer_size = 5
        self.train_kmers = []
        
        for seq in tqdm(self.X_train_processed, desc="Computing k-mers"):
            kmers = {}
            for i in range(len(seq) - self.kmer_size + 1):
                kmer = tuple(seq[i:i+self.kmer_size])
                kmers[kmer] = kmers.get(kmer, 0) + 1
            self.train_kmers.append(kmers)
        
        return self
    
    def _compute_similarity(self, seq1, seq2):
        """
        Compute similarity between two sequences using Needleman-Wunsch
        
        Parameters:
        -----------
        seq1 : list or str
            First sequence
        seq2 : list or str
            Second sequence
            
        Returns:
        --------
        float
            Similarity score between 0 and 1
        """
        # Quick filter: compare lengths
        len_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
        if len_ratio < 0.5:  # If length difference is too large, return low similarity
            return 0.2 * len_ratio
        
        # Quick filter: compare nucleotide composition
        seq1_counts = Counter(seq1)
        seq2_counts = Counter(seq2)
        
        gc1 = (seq1_counts.get('G', 0) + seq1_counts.get('C', 0)) / len(seq1) if len(seq1) > 0 else 0
        gc2 = (seq2_counts.get('G', 0) + seq2_counts.get('C', 0)) / len(seq2) if len(seq2) > 0 else 0
        
        if abs(gc1 - gc2) > 0.15:  # If GC content differs significantly
            return 0.3 * (1 - abs(gc1 - gc2))
        
        # Compute alignment score using Needleman-Wunsch
        score = nw_score(seq1, seq2, self.match, self.mismatch, self.gap, self.use_numba)
        
        # Normalize by the length of the shorter sequence
        max_possible_score = min(len(seq1), len(seq2)) * self.match
        
        if max_possible_score == 0:
            return 0
        
        similarity = score / max_possible_score
        
        # Ensure similarity is between 0 and 1
        return max(0, min(1, similarity))
    
    def _kmer_similarity(self, kmers1, kmers2):
        """
        Compute Jaccard similarity between k-mer sets
        
        Parameters:
        -----------
        kmers1 : dict
            K-mer counts for first sequence
        kmers2 : dict
            K-mer counts for second sequence
            
        Returns:
        --------
        float
            Similarity score between 0 and 1
        """
        # Set of k-mers in both sequences
        kmers1_set = set(kmers1.keys())
        kmers2_set = set(kmers2.keys())
        
        # Compute Jaccard similarity
        intersection = len(kmers1_set.intersection(kmers2_set))
        union = len(kmers1_set.union(kmers2_set))
        
        return intersection / union if union > 0 else 0
    
    def _process_batch(self, batch_indices, test_seq, test_kmers):
        """
        Process a batch of training sequences
        
        Parameters:
        -----------
        batch_indices : list
            Indices of training sequences to compare
        test_seq : list
            Test sequence to compare against
        test_kmers : dict
            K-mer frequencies of test sequence
            
        Returns:
        --------
        list
            List of (index, similarity) tuples
        """
        similarities = []
        
        for idx in batch_indices:
            train_seq = self.X_train_processed[idx]
            train_kmers = self.train_kmers[idx]
            
            # Quick k-mer filter
            kmer_sim = self._kmer_similarity(test_kmers, train_kmers)
            
            # Skip full alignment if k-mer similarity is too low
            if kmer_sim < 0.2:
                similarities.append((idx, kmer_sim * 0.5))  # Scale down the similarity
                continue
                
            # Compute full alignment similarity for promising candidates
            similarity = self._compute_similarity(test_seq, train_seq)
            similarities.append((idx, similarity))
            
        return similarities
    
    def predict(self, X_test):
        """
        Predict labels for test sequences
        
        Parameters:
        -----------
        X_test : list
            List of test sequences
            
        Returns:
        --------
        list
            Predicted labels
        """
        predictions = []
        
        # Convert test sequences to lists and sample if too long
        X_test_processed = [self._sample_sequence(list(seq)) for seq in X_test]
        
        # Precompute k-mer frequencies for test sequences
        test_kmers = []
        for seq in X_test_processed:
            kmers = {}
            for i in range(len(seq) - self.kmer_size + 1):
                kmer = tuple(seq[i:i+self.kmer_size])
                kmers[kmer] = kmers.get(kmer, 0) + 1
            test_kmers.append(kmers)
        
        # Use joblib for parallel processing
        n_jobs = self.n_jobs if self.n_jobs > 0 else -1
        
        print(f"Predicting {len(X_test)} sequences using Needleman-Wunsch...")
        
        for test_idx, (test_seq, test_kmer) in enumerate(tqdm(zip(X_test_processed, test_kmers), 
                                                           desc="NW Predictions", 
                                                           total=len(X_test_processed))):
            # Prepare batches for parallel processing
            all_indices = list(range(len(self.X_train)))
            
            # Pre-filter based on sequence length and k-mer similarity
            filtered_indices = []
            
            # Length filter
            test_len = len(test_seq)
            for i in all_indices:
                train_len = self.train_lengths[i]
                len_ratio = min(test_len, train_len) / max(test_len, train_len)
                
                if len_ratio >= 0.5:
                    # Quick k-mer similarity check
                    kmer_sim = self._kmer_similarity(test_kmer, self.train_kmers[i])
                    if kmer_sim >= 0.1:  # Only consider if there's some k-mer overlap
                        filtered_indices.append(i)
            
            # If too few candidates after filtering, use all
            if len(filtered_indices) < self.k * 2:
                filtered_indices = all_indices
                
            # Create batches
            batch_size = min(self.batch_size, len(filtered_indices))
            batches = [
                filtered_indices[i:i+batch_size] 
                for i in range(0, len(filtered_indices), batch_size)
            ]
            
            # Process batches in parallel
            if n_jobs != 1 and len(batches) > 1:
                # Parallel processing
                batch_results = Parallel(n_jobs=n_jobs)(
                    delayed(self._process_batch)(batch, test_seq, test_kmer) for batch in batches
                )
                # Flatten results
                all_similarities = []
                for batch_result in batch_results:
                    all_similarities.extend(batch_result)
            else:
                # Sequential processing
                all_similarities = []
                for batch in batches:
                    batch_result = self._process_batch(batch, test_seq, test_kmer)
                    all_similarities.extend(batch_result)
            
            # Sort by similarity (highest first)
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top k neighbors
            top_k = all_similarities[:self.k]
            
            # Majority vote
            if top_k:
                neighbor_labels = [self.y_train[idx] for idx, _ in top_k]
                prediction = Counter(neighbor_labels).most_common(1)[0][0]
            else:
                # If no neighbors found, use most common class
                prediction = Counter(self.y_train).most_common(1)[0][0]
                
            predictions.append(prediction)
            
        return predictions

# Helper function to set up logging
def setup_logging(output_dir):
    """Setup logging to both console and file"""
    log_file = os.path.join(output_dir, 'run.log')
    
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Redirect print statements to logger
    class PrintToLogger:
        def write(self, buf):
            for line in buf.rstrip().splitlines():
                logger.info(line.rstrip())
        def flush(self):
            pass
    
    # Keep a reference to the original stdout
    original_stdout = sys.stdout
    
    # Replace stdout with logging
    sys.stdout = PrintToLogger()
    
    return original_stdout


# Function to generate run name based on parameters
def generate_run_name(args):
    """Generate a descriptive run name based on parameters"""
    # Create timestamp for ordering
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract dataset name from input directory
    dataset_name = os.path.basename(args.input_dir)
    
    # Test size as percentage
    test_pct = int(args.test_size * 100)
    
    # Create name with timestamp first
    run_name = f"{timestamp}_{dataset_name}_test{test_pct}"
    
    # Add sample percentage if not 100
    if args.sample_percentage != 100:
        run_name += f"_sample{args.sample_percentage}"
    
    # Add balanced flag if used
    if args.balanced:
        run_name += "_balanced"
    
    # Add algorithms used
    run_name += f"_{args.algorithms}"
    
    # Add optional parameters if non-default
    if args.k != 5:
        run_name += f"_k{args.k}"
    
    if args.word_size != 28:
        run_name += f"_ws{args.word_size}"
        
    if args.seed_length != DEFAULT_SEED_LENGTH:
        run_name += f"_sl{args.seed_length}"
        
    if args.band_width != DEFAULT_BAND_WIDTH:
        run_name += f"_bw{args.band_width}"
        
    # Add Needleman-Wunsch parameters if not default
    if args.nw_match != 1 or args.nw_mismatch != -1 or args.nw_gap != -2:
        run_name += f"_nw{args.nw_match}_{args.nw_mismatch}_{args.nw_gap}"
        
    if args.no_numba:
        run_name += "_no_numba"
        
    if args.batch_size != 10:
        run_name += f"_bs{args.batch_size}"
        
    if args.max_seq_length != 500:
        run_name += f"_msl{args.max_seq_length}"
    
    return run_name


# Helper function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, output_path, title=None):
    """Plot confusion matrix and save to file"""
    plt.figure(figsize=(max(10, len(class_names) * 0.8), max(8, len(class_names) * 0.7)))
    
    # Check if confusion matrix contains float values (from averaging)
    # Use 'd' format for integers, '.1f' for floats
    fmt = '.1f' if np.issubdtype(cm.dtype, np.floating) else 'd'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title or 'Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# Helper function to plot performance comparison
def plot_performance_comparison(results, output_path):
    """Plot performance comparison between algorithms"""
    # Define the order and colors for the algorithms
    algorithm_order = ["Needleman-Wunsch", "NCBI MegaBLAST", "FastEyeBLAST_DNA"]
    algorithm_colors = {
        "Needleman-Wunsch": "#3498db",  # Blue
        "NCBI MegaBLAST": "#2ecc71",     # Green
        "FastEyeBLAST_DNA": "#e74c3c"   # Red
    }
    
    # Filter and order the algorithms based on what's in the results
    algorithms = []
    for algo in algorithm_order:
        if algo in results:
            algorithms.append(algo)
    
    # Add any remaining algorithms that weren't in our predefined order
    for algo in results.keys():
        if algo not in algorithms:
            algorithms.append(algo)
    
    # Accuracy comparison
    accuracies = [results[algo]["accuracy"] for algo in algorithms]
    
    # Timing comparison
    times = [results[algo]["train_time"] + results[algo]["prediction_time"] for algo in algorithms]
    
    # Memory usage
    memories = [results[algo]["memory_mb"] for algo in algorithms]
    
    # Create figure with multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Color list based on our order
    colors = [algorithm_colors.get(algo, "#9b59b6") for algo in algorithms]
    
    # Accuracy plot
    ax1.bar(algorithms, accuracies, color=colors)
    ax1.set_title('Classification Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Time plot
    ax2.bar(algorithms, times, color=colors)
    ax2.set_title('Total Processing Time')
    ax2.set_ylabel('Time (seconds)')
    for i, v in enumerate(times):
        ax2.text(i, v + 0.02 * max(times), f"{v:.2f}s", ha='center')
    
    # Memory plot
    ax3.bar(algorithms, memories, color=colors)
    ax3.set_title('Memory Usage')
    ax3.set_ylabel('Memory (MB)')
    for i, v in enumerate(memories):
        ax3.text(i, v + 0.02 * max(memories), f"{v:.1f}MB", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_classifier(classifier, X_test, y_test, label_names=None):
    """Evaluate classifier performance"""
    logging.info(f"Evaluating classifier on {len(X_test)} test samples...")
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate accuracy and confusion matrix
    logging.info("Computing evaluation metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate classification report
    if label_names:
        # Make sure all classes in test set are in label_names
        test_classes = sorted(set(y_test))
        for cls in test_classes:
            if cls not in label_names:
                label_names[cls] = f"Class {cls}"
                
        target_names = [label_names[i] for i in test_classes]
    else:
        target_names = None
    
    try:
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    except ValueError as e:
        logging.warning(f"Error generating classification report: {e}")
        # Fallback to a simpler report without target_names
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "prediction_time": prediction_time,
        "predictions": y_pred
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare FastEyeBLAST_DNA, Needleman-Wunsch, and MegaBLAST on a DNA dataset"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Root folder containing one subfolder per label, each with .fasta files"
    )
    parser.add_argument(
        "--output_dir", default="Results",
        help="Where to write dna_comparison_results.json"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of sequences to hold out for testing"
    )
    parser.add_argument("--k", type=int, default=5,
                        help="Number of neighbors for all methods")
    parser.add_argument(
        "--seed_length", type=int, default=DEFAULT_SEED_LENGTH,
        help="Seed length for FastEyeBLAST_DNA"
    )
    parser.add_argument(
        "--band_width", type=int, default=DEFAULT_BAND_WIDTH,
        help="Band width for FastEyeBLAST_DNA"
    )
    parser.add_argument(
        "--word_size", type=int, default=28,
        help="Word size for MegaBLAST"
    )
    # Needleman-Wunsch specific parameters
    parser.add_argument(
        "--nw_match", type=int, default=1,
        help="Match score for Needleman-Wunsch (default: 1)"
    )
    parser.add_argument(
        "--nw_mismatch", type=int, default=-1,
        help="Mismatch penalty for Needleman-Wunsch (default: -1)"
    )
    parser.add_argument(
        "--nw_gap", type=int, default=-2,
        help="Gap penalty for Needleman-Wunsch (default: -2)"
    )
    parser.add_argument(
        "--no_numba", action="store_true",
        help="Disable numba acceleration for Needleman-Wunsch (enabled by default if available)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Batch size for parallel processing in Needleman-Wunsch"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=500,
        help="Maximum sequence length to consider for Needleman-Wunsch (longer sequences will be sampled)"
    )
    parser.add_argument(
        "--algorithms", type=str, default="all",
        choices=["all", "fe", "mb", "nw", "fe-mb", "fe-nw", "mb-nw"],
        help="Which algorithms to run: fe=FastEyeBLAST, mb=MegaBLAST, nw=Needleman-Wunsch"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 for all CPUs)")
    parser.add_argument("--sample_percentage", type=int, default=100,
                       help="Percentage of data to use (1-100)")
    parser.add_argument("--balanced", action="store_true",
                       help="Ensure balanced class representation in the sample")
    # Add parameter for number of FastEyeBLAST runs
    parser.add_argument("--fe_runs", type=int, default=1,
                       help="Number of times to run FastEyeBLAST and average results (default: 1)")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create unique output directory with parameters
    run_id = generate_run_name(args)
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    original_stdout = setup_logging(output_dir)
    
    # Log all arguments
    logging.info(f"Starting comparison study with the following parameters:")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"  {arg}: {value}")
    
    # Save run parameters
    params = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input_dir": args.input_dir,
        "test_size": args.test_size,
        "n_jobs": args.n_jobs,
        "k": args.k,
        "word_size": args.word_size,
        "seed": args.seed,
        "seed_length": args.seed_length,
        "band_width": args.band_width,
        "sample_percentage": args.sample_percentage,
        "balanced": args.balanced,
        "algorithms": args.algorithms,
        # Needleman-Wunsch parameters
        "nw_match": args.nw_match,
        "nw_mismatch": args.nw_mismatch,
        "nw_gap": args.nw_gap,
        "no_numba": args.no_numba,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "fe_runs": args.fe_runs,
        "command": " ".join(["python"] + [os.path.basename(__file__)] + [f"--{k.replace('_', '-')}={v}" for k, v in vars(args).items() if v is not None and v is not False])
    }
    
    # Save parameters to JSON file
    params_file = os.path.join(output_dir, "run_parameters.json")
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    # Also save as readable text file
    params_txt = os.path.join(output_dir, "run_parameters.txt")
    with open(params_txt, 'w') as f:
        f.write(f"Comparison Study Run: {params['timestamp']}\n")
        f.write(f"==================================\n\n")
        f.write(f"Command: {params['command']}\n\n")
        f.write(f"Parameters:\n")
        for k, v in params.items():
            if k != "command" and k != "timestamp":
                f.write(f"  {k}: {v}\n")

    seqs, labels = load_fasta_dir(args.input_dir, sample_percentage=args.sample_percentage, balanced=args.balanced)
    if not seqs:
        raise SystemExit("No sequences found under " + args.input_dir)

    # Ensure balanced classes if requested
    if args.balanced and args.sample_percentage < 100:
        logging.info("Ensuring balanced class distribution...")
        # Group sequences by label
        sequences_by_label = {}
        for seq, label in zip(seqs, labels):
            if label not in sequences_by_label:
                sequences_by_label[label] = []
            sequences_by_label[label].append(seq)
        
        # Determine how many samples per class
        min_class_size = min(len(seqs_in_class) for seqs_in_class in sequences_by_label.values())
        samples_per_class = max(1, int(min_class_size * args.sample_percentage / 100))
        
        # Sample evenly from each class
        balanced_seqs = []
        balanced_labels = []
        for label, class_seqs in sequences_by_label.items():
            sampled_seqs = random.sample(class_seqs, samples_per_class)
            balanced_seqs.extend(sampled_seqs)
            balanced_labels.extend([label] * len(sampled_seqs))
        
        # Shuffle the balanced dataset
        combined = list(zip(balanced_seqs, balanced_labels))
        random.shuffle(combined)
        seqs, labels = zip(*combined)
        
        logging.info(f"Created balanced dataset with {len(seqs)} sequences")
        class_counts = Counter(labels)
        for label, count in sorted(class_counts.items()):
            logging.info(f"  {label}: {count} sequences")

    logging.info(f"Splitting data: {len(seqs)} sequences with {len(set(labels))} classes")
    X_train, X_test, y_train, y_test = train_test_split(
        seqs, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    logging.info(f"Train set: {len(X_train)} sequences")
    logging.info(f"Test set: {len(X_test)} sequences")
    
    # Save dataset split information
    split_info = {
        "total_samples": len(seqs),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "class_distribution": {str(k): v for k, v in Counter(labels).items()},
        "train_class_distribution": {str(k): v for k, v in Counter(y_train).items()},
        "test_class_distribution": {str(k): v for k, v in Counter(y_test).items()}
    }
    split_file = os.path.join(output_dir, "dataset_info.json")
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)

    # Get label names for nicer display
    label_names = {label: label for label in set(labels)}
    
    # Initialize results dictionary
    results = {}
    
    # Determine which algorithms to run
    run_fe = args.algorithms in ["all", "fe", "fe-mb", "fe-nw"]
    run_mb = args.algorithms in ["all", "mb", "fe-mb", "mb-nw"]
    run_nw = args.algorithms in ["all", "nw", "fe-nw", "mb-nw"]

    # --- FastEyeBLAST_DNA ---
    if run_fe:
        logging.info("\n" + "="*50)
        if args.fe_runs > 1:
            logging.info(f"Running FastEyeBLAST_DNA ({args.fe_runs} runs)")
        else:
            logging.info("Running FastEyeBLAST_DNA")
        logging.info("="*50)
        
        # Lists to store results from multiple runs
        fe_accuracies = []
        fe_confusion_matrices = []
        fe_reports = []
        fe_train_times = []
        fe_pred_times = []
        fe_memories = []
        fe_predictions = []
        
        # Run FastEyeBLAST multiple times
        for run in range(args.fe_runs):
            if args.fe_runs > 1:
                logging.info(f"\nFastEyeBLAST_DNA Run {run+1}/{args.fe_runs}")
                
                # Create a new random seed for each run
                run_seed = args.seed + run
                np.random.seed(run_seed)
                random.seed(run_seed)
                
                # Re-split data for this run to ensure varied test sets
                X_train_run, X_test_run, y_train_run, y_test_run = train_test_split(
                    seqs, labels, test_size=args.test_size, random_state=run_seed, stratify=labels
                )
            else:
                # Just use the original split for a single run
                X_train_run, X_test_run, y_train_run, y_test_run = X_train, X_test, y_train, y_test
            
            # Measure memory usage during training
            gc.collect()  # Clear memory before training
            start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
            
            fe = FastEyeBLASTDNA(
                seed_length=args.seed_length,
                band_width=args.band_width,
                num_neighbors=args.k,
                use_parallel=True,
                num_processes=args.n_jobs
            )
            
            logging.info("Training FastEyeBLAST_DNA...")
            start_time = time.time()
            fe.fit(X_train_run, y_train_run)
            train_time = time.time() - start_time
            
            current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
            memory_used_train = max(0, current_mem - start_mem)  # Ensure non-negative
            
            logging.info(f"FastEyeBLAST_DNA training completed in {train_time:.2f} seconds")
            logging.info(f"Training memory usage: {memory_used_train:.1f} MB")
            
            # Measure memory usage during prediction
            gc.collect()  # Clear memory before prediction
            pred_start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
            
            y_pred_fe_results = evaluate_classifier(fe, X_test_run, y_test_run, label_names)
            
            pred_current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
            memory_used_pred = max(0, pred_current_mem - pred_start_mem)  # Ensure non-negative
            
            # Total memory usage is the sum of training and prediction memory
            memory_used = memory_used_train + memory_used_pred
            
            fe_accuracy = y_pred_fe_results["accuracy"]
            logging.info(f"FastEyeBLAST_DNA Accuracy: {fe_accuracy:.4f}")
            logging.info(f"Prediction time: {y_pred_fe_results['prediction_time']:.2f} seconds")
            logging.info(f"Prediction memory usage: {memory_used_pred:.1f} MB")
            logging.info(f"Total memory usage: {memory_used:.1f} MB")
            
            # Store results for this run
            fe_accuracies.append(fe_accuracy)
            fe_confusion_matrices.append(y_pred_fe_results["confusion_matrix"])
            fe_reports.append(y_pred_fe_results["report"])
            fe_train_times.append(train_time)
            fe_pred_times.append(y_pred_fe_results["prediction_time"])
            fe_memories.append(memory_used)
            fe_predictions.append(y_pred_fe_results["predictions"])
        
        # Calculate averages if multiple runs were performed
        if args.fe_runs > 1:
            # Compute average metrics
            avg_accuracy = sum(fe_accuracies) / args.fe_runs
            avg_train_time = sum(fe_train_times) / args.fe_runs
            avg_pred_time = sum(fe_pred_times) / args.fe_runs
            avg_memory = sum(fe_memories) / args.fe_runs
            
            # Compute standard deviations
            std_accuracy = np.std(fe_accuracies)
            std_train_time = np.std(fe_train_times)
            std_pred_time = np.std(fe_pred_times)
            std_memory = np.std(fe_memories)
            
            # Log average results with standard deviations
            logging.info("\nFastEyeBLAST_DNA Average Results over {} runs:".format(args.fe_runs))
            logging.info(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            logging.info(f"Average Training time: {avg_train_time:.2f} ± {std_train_time:.2f} seconds")
            logging.info(f"Average Prediction time: {avg_pred_time:.2f} ± {std_pred_time:.2f} seconds")
            logging.info(f"Average Memory usage: {avg_memory:.1f} ± {std_memory:.1f} MB")
            
            # Create average confusion matrix
            # Get all unique labels from the original dataset
            all_classes = sorted(set(labels))
            n_classes = len(all_classes)
            
            # Initialize empty average confusion matrix
            avg_cm = np.zeros((n_classes, n_classes))
            
            # Sum all confusion matrices (they might have different shapes)
            for cm in fe_confusion_matrices:
                # Add values from this confusion matrix to the average
                for i in range(min(cm.shape[0], n_classes)):
                    for j in range(min(cm.shape[1], n_classes)):
                        avg_cm[i, j] += cm[i, j]
            
            # Divide by number of runs
            avg_cm = avg_cm / args.fe_runs
            
            # Use the last run's report as representative (or could average key metrics)
            representative_report = fe_reports[-1]
            
            # Get class names from the report for better labeling
            report_class_names = []
            for class_name in representative_report:
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report_class_names.append(class_name)
            
            # Sort class names to match the order in the confusion matrix
            report_class_names = sorted(report_class_names)
            
            # Save average confusion matrix
            fe_cm_path = os.path.join(output_dir, 'fe_confusion_matrix.png')
            plot_confusion_matrix(
                avg_cm,
                report_class_names,
                fe_cm_path,
                f"FastEyeBLAST_DNA Confusion Matrix\nAccuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f} (Average of {args.fe_runs} runs)"
            )
            
            # Store averaged results
            results["FastEyeBLAST_DNA"] = {
                "accuracy": avg_accuracy,
                "accuracy_std": std_accuracy,
                "confusion_matrix": avg_cm.tolist(),
                "report": representative_report,
                "train_time": avg_train_time,
                "train_time_std": std_train_time,
                "prediction_time": avg_pred_time,
                "prediction_time_std": std_pred_time,
                "memory_mb": avg_memory,
                "memory_mb_std": std_memory,
                "num_runs": args.fe_runs,
                "individual_runs": {
                    "accuracies": fe_accuracies,
                    "train_times": fe_train_times,
                    "prediction_times": fe_pred_times
                }
            }
        else:
            # Get class names from the report for better labeling
            report_class_names = []
            for class_name in fe_reports[0]:
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report_class_names.append(class_name)
            
            # Sort class names to match the order in the confusion matrix
            report_class_names = sorted(report_class_names)
            
            # Save confusion matrix for single run
            fe_cm_path = os.path.join(output_dir, 'fe_confusion_matrix.png')
            plot_confusion_matrix(
                fe_confusion_matrices[0],
                report_class_names,
                fe_cm_path,
                f"FastEyeBLAST_DNA Confusion Matrix\nAccuracy: {fe_accuracies[0]:.4f}"
            )
            
            # Store single run results
            results["FastEyeBLAST_DNA"] = {
                "accuracy": fe_accuracies[0],
                "confusion_matrix": fe_confusion_matrices[0].tolist(),
                "report": fe_reports[0],
                "train_time": fe_train_times[0],
                "prediction_time": fe_pred_times[0],
                "memory_mb": fe_memories[0]
            }

    # --- Needleman-Wunsch ---
    if run_nw:
        logging.info("\n" + "="*50)
        logging.info("Running Needleman-Wunsch")
        logging.info("="*50)
        
        # Measure memory usage during training
        gc.collect()  # Clear memory before training
        start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        nw = NeedlemanWunschDNAClassifier(
            k=args.k,
            match=args.nw_match,
            mismatch=args.nw_mismatch,
            gap=args.nw_gap,
            use_numba=not args.no_numba,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length
        )
        
        logging.info("Training Needleman-Wunsch classifier...")
        start_time = time.time()
        nw.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        memory_used_train = max(0, current_mem - start_mem)  # Ensure non-negative
        
        logging.info(f"Needleman-Wunsch training completed in {train_time:.2f} seconds")
        logging.info(f"Training memory usage: {memory_used_train:.1f} MB")
        
        # Measure memory usage during prediction
        gc.collect()  # Clear memory before prediction
        pred_start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        nw_results = evaluate_classifier(nw, X_test, y_test, label_names)
        
        pred_current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        memory_used_pred = max(0, pred_current_mem - pred_start_mem)  # Ensure non-negative
        
        # Total memory usage is the sum of training and prediction memory
        memory_used = memory_used_train + memory_used_pred
        
        nw_accuracy = nw_results["accuracy"]
        logging.info(f"Needleman-Wunsch Accuracy: {nw_accuracy:.4f}")
        logging.info(f"Prediction time: {nw_results['prediction_time']:.2f} seconds")
        logging.info(f"Prediction memory usage: {memory_used_pred:.1f} MB")
        logging.info(f"Total memory usage: {memory_used:.1f} MB")
        
        # Get class names from the report for better labeling
        report_class_names = []
        for class_name in nw_results['report']:
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                report_class_names.append(class_name)
        
        # Sort class names to match the order in the confusion matrix
        report_class_names = sorted(report_class_names)
        
        # Save confusion matrix
        nw_cm_path = os.path.join(output_dir, 'nw_confusion_matrix.png')
        plot_confusion_matrix(
            nw_results['confusion_matrix'],
            report_class_names,
            nw_cm_path,
            f"Needleman-Wunsch Confusion Matrix\nAccuracy: {nw_accuracy:.4f}"
        )
        
        results["Needleman-Wunsch"] = {
            "accuracy": nw_accuracy,
            "confusion_matrix": nw_results["confusion_matrix"].tolist(),
            "report": nw_results["report"],
            "train_time": train_time,
            "prediction_time": nw_results["prediction_time"],
            "memory_mb": memory_used
        }

    # --- MegaBLAST ---
    if run_mb:
        logging.info("\n" + "="*50)
        logging.info("Running MegaBLAST")
        logging.info("="*50)
        
        # Measure memory usage during training
        gc.collect()  # Clear memory before training
        start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        mb = NCBIDnaMegaBLASTClassifier(
            k=args.k,
            word_size=args.word_size
        )
        
        logging.info("Training MegaBLAST classifier...")
        start_time = time.time()
        mb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        memory_used_train = max(0, current_mem - start_mem)  # Ensure non-negative
        
        logging.info(f"MegaBLAST training completed in {train_time:.2f} seconds")
        logging.info(f"Training memory usage: {memory_used_train:.1f} MB")
        
        # Measure memory usage during prediction
        gc.collect()  # Clear memory before prediction
        pred_start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        mb_results = evaluate_classifier(mb, X_test, y_test, label_names)
        
        pred_current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        memory_used_pred = max(0, pred_current_mem - pred_start_mem)  # Ensure non-negative
        
        # Total memory usage is the sum of training and prediction memory
        memory_used = memory_used_train + memory_used_pred
        
        mb_accuracy = mb_results["accuracy"]
        logging.info(f"MegaBLAST Accuracy: {mb_accuracy:.4f}")
        logging.info(f"Prediction time: {mb_results['prediction_time']:.2f} seconds")
        logging.info(f"Prediction memory usage: {memory_used_pred:.1f} MB")
        logging.info(f"Total memory usage: {memory_used:.1f} MB")
        
        # Get class names from the report for better labeling
        report_class_names = []
        for class_name in mb_results['report']:
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                report_class_names.append(class_name)
        
        # Sort class names to match the order in the confusion matrix
        report_class_names = sorted(report_class_names)
        
        # Save confusion matrix
        mb_cm_path = os.path.join(output_dir, 'mb_confusion_matrix.png')
        plot_confusion_matrix(
            mb_results['confusion_matrix'],
            report_class_names,
            mb_cm_path,
            f"NCBI MegaBLAST Confusion Matrix\nAccuracy: {mb_accuracy:.4f}"
        )
        
        results["NCBI MegaBLAST"] = {
            "accuracy": mb_accuracy,
            "confusion_matrix": mb_results["confusion_matrix"].tolist(),
            "report": mb_results["report"],
            "train_time": train_time,
            "prediction_time": mb_results["prediction_time"],
            "memory_mb": memory_used
        }

    # Create performance comparison plot
    comparison_path = os.path.join(output_dir, 'performance_comparison.png')
    logging.info(f"Creating performance comparison chart at {comparison_path}...")
    plot_performance_comparison(results, comparison_path)

    # write metrics
    out_json = os.path.join(output_dir, "dna_comparison_results.json")
    logging.info(f"\nWriting results to {out_json}")
    with open(out_json, "w") as oj:
        json.dump(results, oj, indent=2)

    # print summary
    logging.info("\n" + "="*50)
    logging.info("RESULTS SUMMARY")
    logging.info("="*50)
    
    # Create a summary dictionary with all algorithms
    summary_results = {}
    for algo in sorted(results.keys()):
        summary_results[algo] = results[algo]["accuracy"]
    
    logging.info(json.dumps(summary_results, indent=2))
    
    # Restore original stdout
    sys.stdout = original_stdout
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
