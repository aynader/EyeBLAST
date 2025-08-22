import os
import time
import argparse
import numpy as np
import pandas as pd
import json
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
from collections import defaultdict, Counter
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import Parallel, delayed
import psutil
import gc
import sys
import tempfile
import subprocess
from io import StringIO
import logging

# Import implementations (ensure these files are in the same directory)
# IVT implementations
from Needleman_Wunsch_IVT import nw_score, nw_score_string, preprocess_tokens, compute_similarity_matrix, NUMBA_AVAILABLE as NUMBA_AVAILABLE_IVT
from Fast_EyeBLAST_IVT import FastEyeBLAST as FastEyeBLAST_IVT
from Fast_EyeBLAST_IVT_Variable import FastEyeBLAST as FastEyeBLAST_IVT_Variable
from MegaBLAST_IVT import tokens_to_fasta, create_blast_database, run_megablast
# Import the new comparative implementation
from Fast_EyeBLAST_IVT_Compare import FastEyeBLAST_Compare

# Add the current directory to path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# AOI implementations
try:
    from Needleman_Wunsch_AOI import nw_score as nw_score_aoi, NUMBA_AVAILABLE as NUMBA_AVAILABLE_AOI
    from Fast_EyeBLAST_AOI import FastEyeBLAST_AOI
    from MegaBLAST_AOI import sequence_to_fasta
    AOI_MODULES_AVAILABLE = True
    logging.info("Successfully imported AOI modules")
except ImportError as e:
    AOI_MODULES_AVAILABLE = False
    logging.warning(f"AOI modules not found. Only IVT data processing will be available. Error: {e}")

# Add these functions after the imports section, around line 35
def char_to_nucleotide(char):
    """
    Convert a character to a nucleotide sequence
    Uses a simple encoding scheme that preserves the ordering of ASCII values
    
    Parameters:
    -----------
    char : str
        Single character to convert
    
    Returns:
    --------
    str
        Nucleotide sequence (ACGT)
    """
    # Use ASCII value to determine nucleotide pattern
    ascii_val = ord(char)
    
    # Convert to binary (4 bits)
    binary = format(ascii_val % 16, '04b')
    
    # Map 00 -> A, 01 -> C, 10 -> G, 11 -> T
    nucleotides = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}
    
    # Convert binary to nucleotides
    nuc_seq = ''
    for i in range(0, len(binary), 2):
        if i+1 < len(binary):
            nuc_seq += nucleotides[binary[i:i+2]]
    
    return nuc_seq

def aoi_sequence_to_fasta(sequence, description):
    """
    Convert AOI character sequence to FASTA format string with proper nucleotide encoding
    Used as fallback when AOI modules aren't available
    
    Parameters:
    -----------
    sequence : str
        AOI character sequence (e.g., "!@#$%^&*()")
    description : str
        Description for the FASTA header
        
    Returns:
    --------
    str
        FASTA formatted string with encoded nucleotides
    """
    # Convert character sequence to nucleotides
    nucleotide_sequence = ""
    for char in sequence:
        nucleotide_sequence += char_to_nucleotide(char)
    
    # Create FASTA formatted string
    fasta = f">{description}\n{nucleotide_sequence}\n"
    return fasta

# -----------------------------------------------------------------------------
# Needleman-Wunsch classifier for scanpaths
# -----------------------------------------------------------------------------
class NeedlemanWunschClassifier:
    """
    Classifier using Needleman-Wunsch for scanpath comparison
    """
    def __init__(self, k=5, match=1, mismatch=-1, gap=-2, n_jobs=-1, use_numba=True):
        """Initialize the classifier"""
        self.k = k
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        self.n_jobs = n_jobs
        self.token_type_cache = {}
        self.token_length_cache = {}
        self.use_numba = use_numba
        self.is_aoi = False  # Will be determined during fit
        self.max_seq_length = 10000  # Default, may be adjusted based on data type
        self.batch_size = None  # Will be determined based on data
    
    def fit(self, X_train, y_train):
        """Train the classifier"""
        self.X_train = X_train
        self.y_train = y_train
        
        # Detect if we're dealing with AOI or IVT data
        # If the first item is a string, it's AOI data
        if X_train and isinstance(X_train[0], str):
            self.is_aoi = True
            logging.info("Detected AOI-encoded data for Needleman-Wunsch")
            
            # For AOI data
            avg_len = sum(len(seq) for seq in X_train) / max(1, len(X_train))
            logging.info(f"Average AOI sequence length: {avg_len:.2f} characters")
            
            # Adjust parameters for AOI
            if avg_len > 5000:
                self.max_seq_length = 3000
            elif avg_len > 2000:
                self.max_seq_length = 5000
            else:
                self.max_seq_length = 10000
                
            logging.info(f"Using max_seq_length of {self.max_seq_length} for AOI data")
            
            if avg_len > 5000:
                self.batch_size = 5
            elif avg_len > 2000:
                self.batch_size = 10
            elif avg_len > 1000:
                self.batch_size = 20
            else:
                self.batch_size = 50
                
            logging.info(f"Using batch size of {self.batch_size} for AOI data")
            
            self.X_train_preprocessed = X_train
            return self
        else:
            self.is_aoi = False
            logging.info("Detected IVT-encoded data for Needleman-Wunsch")
            
            # Preprocess training data for faster alignment (original IVT implementation)
            self.X_train_preprocessed = []
            for scanpath in custom_tqdm(X_train, desc="Preprocessing training data for NW", leave=True):
                # Preprocess scanpath for numba
                seq_types = np.zeros(len(scanpath), dtype=np.int32)
                seq_lengths = np.zeros(len(scanpath), dtype=np.int32)
                
                for i, token in enumerate(scanpath):
                    token_type = ord(token[0])  # Convert char to int
                    token_length = int(token[1:]) if token[1:].isdigit() else 0
                    seq_types[i] = token_type
                    seq_lengths[i] = token_length
                
                self.X_train_preprocessed.append((seq_types, seq_lengths))
            
            return self
    
    def _compute_pair_aoi(self, args):
        """Compute similarity for a pair of AOI scanpaths"""
        i, j, seq1, seq2 = args
        
        # Truncate very long sequences to avoid memory issues
        if len(seq1) > self.max_seq_length:
            seq1 = seq1[:self.max_seq_length]
        if len(seq2) > self.max_seq_length:
            seq2 = seq2[:self.max_seq_length]
        
        try:
            # Use AOI-specific implementation
            from Needleman_Wunsch_AOI import nw_score as nw_score_aoi
            score = nw_score_aoi(seq1, seq2, 
                               match=self.match, 
                               mismatch=self.mismatch, 
                               gap=self.gap, 
                               use_numba=self.use_numba,
                               max_seq_length=self.max_seq_length)
            
            # Normalize by the length of the shorter sequence
            shorter_len = min(len(seq1), len(seq2))
            if shorter_len > 0:
                sim = score / (shorter_len * self.match)
                # Clip to [0, 1] range
                sim = max(0, min(1, sim))
            else:
                sim = 0
                
            return i, j, sim
        except Exception as e:
            logging.warning(f"Error computing AOI similarity: {e}. Returning default similarity of 0.")
            return i, j, 0.0
    
    def _compute_pair(self, args):
        """Compute similarity for a pair of scanpaths using preprocessed data"""
        if self.is_aoi:
            return self._compute_pair_aoi(args)
            
        # Original IVT implementation
        i, j, seq1_types, seq1_lengths, seq2_types, seq2_lengths = args
        score = nw_score(seq1_types, seq1_lengths, seq2_types, seq2_lengths, 
                         self.match, self.mismatch, self.gap, self.use_numba)
        
        # Normalize by the length of the shorter sequence
        shorter_len = min(len(seq1_types), len(seq2_types))
        if shorter_len > 0:
            sim = score / (shorter_len * self.match)
            # Clip to [0, 1] range
            sim = max(0, min(1, sim))
        else:
            sim = 0
            
        return i, j, sim
    
    def _compute_test_similarities_aoi(self, test_idx, test_seq):
        """Compute similarities between a test AOI sample and all training samples"""
        # Truncate test sequence if needed
        if len(test_seq) > self.max_seq_length:
            test_seq = test_seq[:self.max_seq_length]
        
        similarities = []
        
        # Determine batch size based on sequence length
        avg_len = len(test_seq)
        for train_seq in self.X_train_preprocessed:
            avg_len += len(train_seq)
        avg_len /= (1 + len(self.X_train_preprocessed))
        
        batch_size = self.batch_size
        if not batch_size:
            if avg_len > 5000:
                batch_size = 5
            elif avg_len > 2000:
                batch_size = 10
            else:
                batch_size = 20
        
        # Create batches of training samples
        train_batches = []
        current_batch = []
        for train_idx, train_seq in enumerate(self.X_train_preprocessed):
            current_batch.append((train_idx, train_seq))
            if len(current_batch) >= batch_size:
                train_batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            train_batches.append(current_batch)
        
        # Process each batch
        for batch_idx, batch in enumerate(tqdm(train_batches, desc=f"Processing AOI batches for test idx {test_idx}", leave=False, ncols=100)):
            # Create pairs for parallel processing
            pairs = [(test_idx, train_idx, test_seq, train_seq) 
                    for train_idx, train_seq in batch]
            
            # Use at most half the CPUs for very long sequences
            n_jobs = self.n_jobs
            if avg_len > 3000 and n_jobs > 2:
                n_jobs = max(1, n_jobs // 2)
            
            # Process batch in parallel
            try:
                results = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(self._compute_pair_aoi)(pair) for pair in tqdm(pairs, desc=f"Computing AOI pairs (batch {batch_idx+1}/{len(train_batches)})", leave=False, ncols=100)
                )
                
                # Extract similarities
                for _, train_idx, sim in results:
                    similarities.append((train_idx, sim))
                
                # Force garbage collection between batches
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                # Process one by one as fallback
                for pair in tqdm(pairs, desc="Fallback processing (sequential)", leave=False, ncols=100):
                    try:
                        _, train_idx, sim = self._compute_pair_aoi(pair)
                        similarities.append((train_idx, sim))
                    except Exception as e2:
                        logging.error(f"Error processing pair: {e2}")
                        similarities.append((pair[1], 0.0))  # Default similarity
        
        return similarities
    
    def _compute_test_similarities(self, test_seq):
        """Compute similarities between a test sample and all training samples"""
        if self.is_aoi:
            return self._compute_test_similarities_aoi(0, test_seq)
            
        # Original IVT implementation
        # Preprocess test sequence
        test_types = np.zeros(len(test_seq), dtype=np.int32)
        test_lengths = np.zeros(len(test_seq), dtype=np.int32)
        
        for i, token in enumerate(test_seq):
            token_type = ord(token[0])  # Convert char to int
            token_length = int(token[1:]) if token[1:].isdigit() else 0
            test_types[i] = token_type
            test_lengths[i] = token_length
        
        similarities = []
        
        # Add progress bar for larger datasets
        if len(self.X_train_preprocessed) > 50:
            train_iter = custom_tqdm(enumerate(self.X_train_preprocessed), 
                                   desc="Computing NW similarities", 
                                   total=len(self.X_train_preprocessed), 
                                   leave=False,
                                   ncols=100)
        else:
            train_iter = enumerate(self.X_train_preprocessed)
            
        for i, (train_types, train_lengths) in train_iter:
            score = nw_score(
                test_types, test_lengths, 
                train_types, train_lengths,
                self.match, self.mismatch, self.gap, self.use_numba
            )
            
            # Normalize
            shorter_len = min(len(test_types), len(train_types))
            if shorter_len > 0:
                sim = score / (shorter_len * self.match)
                # Clip to [0, 1] range
                sim = max(0, min(1, sim))
            else:
                sim = 0
                
            similarities.append((i, sim))
        
        return similarities
    
    def _process_test_batch(self, batch):
        """Process a batch of test samples"""
        results = []
        for test_idx, test_seq in custom_tqdm(batch, desc="Processing NW test batch", leave=False, ncols=100):
            # Compute similarities with all training samples
            similarities = self._compute_test_similarities(test_seq)
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top k neighbors
            top_k = similarities[:self.k]
            
            # Get labels of neighbors
            neighbor_labels = [self.y_train[idx] for idx, _ in top_k]
            
            # Majority vote
            prediction = Counter(neighbor_labels).most_common(1)[0][0]
            results.append((test_idx, prediction))
        
        return results
    
    def predict(self, X_test):
        """Predict labels for test samples"""
        logging.info(f"Predicting labels for {len(X_test)} test samples using Needleman-Wunsch...")
        
        # For AOI data with very long sequences, use a more memory-efficient approach
        if self.is_aoi:
            logging.info("Using memory-optimized approach for AOI data")
            
            # Calculate average sequence length for better parameter tuning
            avg_test_len = sum(len(seq) for seq in X_test) / max(1, len(X_test))
            logging.info(f"Average test sequence length: {avg_test_len:.2f} characters")
            
            # Adjust batch size based on sequence length
            batch_size = self.batch_size
            if not batch_size:
                if avg_test_len > 5000:
                    batch_size = 2
                elif avg_test_len > 2000:
                    batch_size = 5
                else:
                    batch_size = 10
            
            logging.info(f"Using test batch size of {batch_size}")
                
            # Process in smaller batches
            all_results = []
            for test_idx in range(0, len(X_test), batch_size):
                batch_end = min(test_idx + batch_size, len(X_test))
                logging.info(f"Processing test batch {test_idx//batch_size + 1}/{(len(X_test) + batch_size - 1)//batch_size}: samples {test_idx}-{batch_end-1}")
                
                batch_results = []
                for i in range(test_idx, batch_end):
                    # For each test sequence
                    test_seq = X_test[i]
                    
                    # Compute similarities with all training samples (with batching)
                    similarities = self._compute_test_similarities_aoi(i, test_seq)
                    
                    # Sort by similarity (descending)
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # Get top k neighbors
                    top_k = similarities[:self.k]
                    
                    # Get labels of neighbors
                    neighbor_labels = [self.y_train[idx] for idx, _ in top_k]
                    
                    # Majority vote
                    prediction = Counter(neighbor_labels).most_common(1)[0][0]
                    batch_results.append((i, prediction))
                
                all_results.extend(batch_results)
                gc.collect()  # Force garbage collection between batches
            
            # Sort by test index and extract predictions
            all_results.sort(key=lambda x: x[0])
            predictions = [pred for _, pred in all_results]
            
            return predictions
        
        # Original IVT implementation
        # Prepare batches for parallel processing
        n_test = len(X_test)
        n_jobs = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
        batch_size = max(1, min(10, n_test // max(1, n_jobs)))
        batches = []
        
        for i in range(0, n_test, batch_size):
            batch = [(j, X_test[j]) for j in range(i, min(i + batch_size, n_test))]
            batches.append(batch)
        
        logging.info(f"Split test data into {len(batches)} batches")
        
        # Process in parallel
        if self.n_jobs != 1 and len(batches) > 1:
            logging.info(f"Processing {len(batches)} batches in parallel using {n_jobs} jobs")
            # Show overall progress bar for batch processing
            with custom_tqdm(total=len(batches), desc="Processing NW batches in parallel", leave=True, ncols=100) as pbar:
                results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(self._process_test_batch)(batch) for batch in batches
                )
                # Update progress bar manually
                for _ in range(len(batches)):
                    pbar.update(1)
            
            # Flatten results
            all_results = []
            for batch_result in results:
                all_results.extend(batch_result)
        else:
            # Sequential processing
            logging.info("Processing batches sequentially")
            all_results = []
            for batch in custom_tqdm(batches, desc="Processing NW test batches sequentially", leave=True, ncols=100):
                batch_result = self._process_test_batch(batch)
                all_results.extend(batch_result)
        
        # Sort by test index and extract predictions
        all_results.sort(key=lambda x: x[0])
        predictions = [pred for _, pred in all_results]
        
        return predictions

# Add MegaBLAST classifier implementation
class NCBIMegaBLASTScanpathClassifier:
    """
    Classifier using NCBI MegaBLAST for scanpath comparison
    """
    def __init__(self, k=5, word_size=28, perc_identity=90, evalue=0.001, n_jobs=-1):
        """Initialize the classifier"""
        self.k = k
        self.word_size = word_size
        self.perc_identity = perc_identity
        self.evalue = evalue
        self.n_jobs = n_jobs
        self.temp_dir = None
        self.db_path = None
        self.fasta_files = {}
        # Detect data type
        self.is_aoi = False
        
    def fit(self, X_train, y_train):
        """Train the classifier"""
        self.X_train = X_train
        self.y_train = y_train
        
        # Detect if we're dealing with AOI or IVT data
        if X_train and isinstance(X_train[0], str):
            self.is_aoi = True
            logging.info("Detected AOI-encoded data for MegaBLAST")
        else:
            self.is_aoi = False
            logging.info("Detected IVT-encoded data for MegaBLAST")
        
        # Create temporary directory for BLAST files
        self.temp_dir = tempfile.mkdtemp()
        logging.info(f"Created temporary directory for BLAST: {self.temp_dir}")
        
        # Create FASTA files for training data
        logging.info("Converting training scanpaths to FASTA format...")
        self.fasta_files = {}
        for i, scanpath in enumerate(X_train):
            fasta_file = os.path.join(self.temp_dir, f"train_{i}.fasta")
            with open(fasta_file, 'w') as f:
                if self.is_aoi:
                    # Use sequence_to_fasta for AOI data if available
                    if AOI_MODULES_AVAILABLE:
                        f.write(sequence_to_fasta(scanpath, f"train_{i}"))
                    else:
                        # Use our fallback implementation when AOI module not available
                        f.write(aoi_sequence_to_fasta(scanpath, f"train_{i}"))
                else:
                    # Use tokens_to_fasta for IVT data
                    f.write(tokens_to_fasta(scanpath, f"train_{i}"))
            self.fasta_files[i] = fasta_file
        
        # Create a concatenated FASTA file for the database
        concat_file = os.path.join(self.temp_dir, "train_all.fasta")
        with open(concat_file, 'w') as outf:
            for fasta_file in self.fasta_files.values():
                with open(fasta_file, 'r') as inf:
                    outf.write(inf.read())
        
        # Create BLAST database
        logging.info("Creating BLAST database from training data...")
        self.db_path = os.path.join(self.temp_dir, "train_db")
        db_cmd = [
            "makeblastdb",
            "-in", concat_file,
            "-dbtype", "nucl",
            "-out", self.db_path
        ]
        subprocess.run(db_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return self
    
    def predict(self, X_test):
        """Predict labels for test samples"""
        logging.info(f"Predicting labels for {len(X_test)} test samples using MegaBLAST...")
        predictions = []
        
        for test_idx, test_scanpath in enumerate(tqdm(X_test, desc="MegaBLAST predictions", ncols=100)):
            # Create FASTA for test scanpath
            test_fasta = os.path.join(self.temp_dir, f"test_{test_idx}.fasta")
            with open(test_fasta, 'w') as f:
                if self.is_aoi:
                    # Use sequence_to_fasta for AOI data if available
                    if AOI_MODULES_AVAILABLE:
                        f.write(sequence_to_fasta(test_scanpath, f"test_{test_idx}"))
                    else:
                        # Use our fallback implementation when AOI module not available
                        f.write(aoi_sequence_to_fasta(test_scanpath, f"test_{test_idx}"))
                else:
                    # Use tokens_to_fasta for IVT data
                    f.write(tokens_to_fasta(test_scanpath, f"test_{test_idx}"))
            
            # Run BLAST
            blast_results = os.path.join(self.temp_dir, f"test_{test_idx}_results.tsv")
            run_megablast(
                test_fasta, 
                self.db_path, 
                blast_results, 
                evalue=self.evalue, 
                word_size=self.word_size, 
                perc_identity=self.perc_identity
            )
            
            # Parse BLAST results
            similarities = []
            try:
                with open(blast_results, 'r') as f:
                    for line in f:
                        fields = line.strip().split('\t')
                        if len(fields) >= 12:
                            subject_id = fields[1]
                            train_idx = int(subject_id.split('_')[1])
                            bit_score = float(fields[-1])
                            similarities.append((train_idx, bit_score))
            except FileNotFoundError:
                # No matches found
                pass
            
            # Sort by bit score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top k neighbors
            top_k = similarities[:self.k]
            
            if top_k:
                # Get labels of neighbors
                neighbor_labels = [self.y_train[idx] for idx, _ in top_k]
                
                # Majority vote
                prediction = Counter(neighbor_labels).most_common(1)[0][0]
            else:
                # No matches, use most common class
                prediction = Counter(self.y_train).most_common(1)[0][0]
            
            predictions.append(prediction)
        
        return predictions
    
    def __del__(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

# Custom tqdm for better progress display
def custom_tqdm(*args, **kwargs):
    """Enhanced tqdm with better formatting"""
    if 'ncols' not in kwargs:
        kwargs['ncols'] = 100
    if 'bar_format' not in kwargs:
        kwargs['bar_format'] = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    if 'dynamic_ncols' not in kwargs:
        kwargs['dynamic_ncols'] = True
    return tqdm(*args, **kwargs)

# Setup logging to capture terminal output
def setup_logging(output_dir):
    """Setup logging to both console and file"""
    log_file = os.path.join(output_dir, 'run.log')
    
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
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

# -----------------------------------------------------------------------------
# Data loading and preparation
# -----------------------------------------------------------------------------
def load_and_prepare_data(input_dir, data_type="task", sample_percentage=100, gender_file=None, task_filter=None):
    """
    Load tokenized scanpaths and extract task labels or gender information
    
    Parameters:
    -----------
    input_dir : str
        Directory containing tokenized scanpath JSON files
    data_type : str
        Type of classification task: "task" or "gender"
    sample_percentage : int
        Percentage of data to use (randomly sampled)
    gender_file : str
        Path to Excel file containing gender information
    task_filter : int
        Optional task ID to filter by when doing gender classification
        
    Returns:
    --------
    list
        List of scanpaths
    list
        Labels
    dict
        Subjects
    """
    # Determine if we're dealing with AOI or IVT data
    is_aoi = is_aoi_dataset(input_dir)
    logging.info(f"Detected data format: {'AOI' if is_aoi else 'IVT'}")
    
    if is_aoi:
        # Load AOI-encoded data from encoded_sequences.json
        json_path = os.path.join(input_dir, "encoded_sequences.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"No encoded sequences found at {json_path}")
        
        logging.info(f"Loading AOI-encoded sequences from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Sample X% of data if requested
        if sample_percentage < 100:
            num_items = int(len(data) * sample_percentage / 100)
            logging.info(f"Sampling {num_items} sequences ({sample_percentage}% of {len(data)})")
            keys = random.sample(list(data.keys()), num_items)
            data = {k: data[k] for k in keys}
        
        scanpaths = []
        labels = []
        subjects = {}
        
        dataset_name = os.path.basename(input_dir)
        is_mpii = "MPII" in dataset_name
        is_gazebase = "GazeBase" in dataset_name and not "GazeBaseVR" in dataset_name
        
        # Load gender data if needed
        gender_data = {}
        if data_type == "gender" and gender_file:
            logging.info(f"Loading gender information from {gender_file}")
            try:
                # Load gender data based on file extension
                file_ext = os.path.splitext(gender_file)[1].lower()
                if file_ext == '.xlsx' or file_ext == '.xls':
                    gender_df = pd.read_excel(gender_file)
                elif file_ext == '.csv':
                    gender_df = pd.read_csv(gender_file)
                else:
                    raise ValueError(f"Unsupported gender file format: {file_ext}")
                
                # Check for required columns
                if 'Participant ID' not in gender_df.columns or 'Gender' not in gender_df.columns:
                    # Try alternative column names for MPIIDPEye Ground_Truth.csv
                    if 'participant' in gender_df.columns and 'gender' in gender_df.columns:
                        gender_df = gender_df.rename(columns={'participant': 'Participant ID', 'gender': 'Gender'})
                    else:
                        logging.warning(f"Gender file columns not recognized. Available columns: {gender_df.columns.tolist()}")
                        # Try to guess columns - first column as ID, look for gender-like column
                        id_col = gender_df.columns[0]
                        gender_cols = [col for col in gender_df.columns if 'gender' in col.lower() or 'sex' in col.lower()]
                        if gender_cols:
                            gender_col = gender_cols[0]
                            gender_df = gender_df.rename(columns={id_col: 'Participant ID', gender_col: 'Gender'})
                        else:
                            raise ValueError(f"Could not find required columns in gender file")
                
                # Sort by Recording round if it exists
                if 'Recording round' in gender_df.columns:
                    gender_df = gender_df.sort_values(by=['Participant ID', 'Recording round'])
                
                # Group by Participant ID and take the first (earliest) record for each
                unique_participants = gender_df.drop_duplicates(subset=['Participant ID'], keep='first')
                
                # Filter out any 'Other' gender
                binary_gender = unique_participants[unique_participants['Gender'].isin(['Male', 'Female', 'male', 'female', 'M', 'F', 'm', 'f'])]
                
                # Standardize gender values
                gender_map = {
                    'Male': 'Male', 'male': 'Male', 'M': 'Male', 'm': 'Male',
                    'Female': 'Female', 'female': 'Female', 'F': 'Female', 'f': 'Female'
                }
                binary_gender['Gender'] = binary_gender['Gender'].map(gender_map)
                
                logging.info(f"After filtering duplicates and non-binary genders, found {len(binary_gender)} participants")
                
                # Excel file has 'Participant ID' and 'Gender' columns
                for _, row in binary_gender.iterrows():
                    subject_id = str(row['Participant ID'])
                    gender_data[subject_id] = row['Gender']
                
                logging.info(f"Loaded gender information for {len(gender_data)} subjects (Male/Female only)")
            except Exception as e:
                logging.error(f"Error loading gender file: {e}")
                raise
        
        # Extract sequences and task labels
        for file_id, item in data.items():
            sequence = item['sequence']
            file_path = item['file_path']
            
            # Skip empty sequences
            if not sequence:
                continue
            
            # Parse filename based on dataset
            parts = os.path.basename(file_path).split('_')
            
            if is_mpii:
                # Check file format
                if parts[0] == "Data" and parts[1].startswith("P"):
                    # Format: Data_PX_RY.csv
                    # Where X is the participant number and Y is the recording/task number
                    try:
                        subject_id = parts[1].replace("P", "")  # Extract participant number
                        task_id = parts[2].replace("R", "")     # Extract recording/task number
                        # Remove file extension if present
                        task_id = task_id.split('.')[0]
                        task_num = int(task_id)                # Convert to integer
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Could not parse filename {file_path}: {e}")
                        continue
                else:
                    # Format: MPII_Subject_XX_TaskY.csv
                    try:
                        subject_part = [p for p in parts if p.startswith("Subject")]
                        if subject_part:
                            subject_id = subject_part[0].replace("Subject_", "")
                        else:
                            subject_id = "unknown"
                            
                        task_part = [p for p in parts if p.startswith("Task")]
                        if task_part:
                            # Remove file extension if present
                            task_str = task_part[0].replace("Task", "").split('.')[0]
                            task_num = int(task_str)
                        else:
                            task_num = 0
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Could not parse filename {file_path}: {e}")
                        continue
            elif "SynchronEyes" in dataset_name:
                # Format: S02_EyeLink_S_5030_S02_VID_tokens.json
                try:
                    subject_id = parts[3]  # subject ID is the part after S_XXXX
                    # Extract task type based on the suffix (VID, TEX, RAN, HSS, FIX)
                    task_type = parts[5]
                    # Map task types to numbers
                    task_map = {"VID": 1, "TEX": 2, "RAN": 3, "HSS": 4, "FIX": 5}
                    task_num = task_map.get(task_type, 0)
                except (IndexError, ValueError) as e:
                    logging.warning(f"Could not parse filename {file_path}: {e}")
                    continue
            elif is_gazebase:
                # Format: S_XXXX_SY_ZZZ_tokens.json
                try:
                    subject_id = parts[1]  # subject ID is the 4-digit number
                    # Extract task type (BLG, FXS, VD1, VD2, TEX, RAN, HSS)
                    task_type = parts[3].split('.')[0]  # Remove file extension if present
                    # Map GazeBase task types to numbers
                    task_map = {"BLG": 1, "FXS": 2, "VD1": 3, "VD2": 4, "TEX": 5, "RAN": 6, "HSS": 7}
                    task_num = task_map.get(task_type, 0)
                except (IndexError, ValueError) as e:
                    logging.warning(f"Could not parse filename {file_path}: {e}")
                    continue
            else:
                # GazeBaseVR format: S_XXXX_SY_Z_TTT.csv
                try:
                    raw_subject_id = parts[1] if len(parts) > 1 else "unknown"
                    
                    # GazeBaseVR has two ID patterns:
                    # 1. S_3XXX_SY_Z_TTT.csv -> Remove leading '3' to get XXX
                    # 2. S_1XXX_SY_Z_TTT.csv -> Use last 3 digits (XXX)
                    
                    if raw_subject_id.startswith('3') and len(raw_subject_id) > 1:
                        # Pattern 1: Remove leading '3'
                        subject_id = raw_subject_id[1:]
                    elif len(raw_subject_id) > 3:
                        # Pattern 2: Use last 3 digits
                        subject_id = raw_subject_id[-3:]
                    else:
                        # Default: use as is
                        subject_id = raw_subject_id
                    
                    # Extract task number, remove file extension if present
                    if len(parts) > 3:
                        task_part = parts[3].split('.')[0]
                        task_num = int(task_part)
                    else:
                        task_num = 0
                except (IndexError, ValueError) as e:
                    logging.warning(f"Could not parse filename {file_path}: {e}")
                    continue
            
            # Filter by task if task_filter is specified and we're doing gender classification
            if data_type == "gender" and task_filter is not None and task_num != task_filter:
                continue
                
            # Set label based on data_type
            if data_type == "task":
                label = task_num
                # For task classification, always add the scanpath
                scanpaths.append(sequence)
            elif data_type == "gender":
                # Skip if gender information is not available
                if subject_id not in gender_data:
                    continue
                label = gender_data[subject_id]
                # Only add the scanpath if we have gender information
                scanpaths.append(sequence)
            else:
                raise ValueError(f"Unknown data_type: {data_type}")
            
            labels.append(label)
            
            # Track subject IDs for potential subject-wise split
            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append(len(scanpaths) - 1)
    else:
        # Original code for IVT data
        files = glob.glob(os.path.join(input_dir, "*.json"))
        
        # Sample X% of files if requested
        if sample_percentage < 100:
            num_files = int(len(files) * sample_percentage / 100)
            logging.info(f"Sampling {num_files} files ({sample_percentage}% of {len(files)})")
            files = random.sample(files, num_files)
        
        scanpaths = []
        labels = []
        subjects = {}
        
        # Load gender data if needed
        gender_data = {}
        if data_type == "gender" and gender_file:
            logging.info(f"Loading gender information from {gender_file}")
            try:
                # Load gender data based on file extension
                file_ext = os.path.splitext(gender_file)[1].lower()
                if file_ext == '.xlsx' or file_ext == '.xls':
                    gender_df = pd.read_excel(gender_file)
                elif file_ext == '.csv':
                    gender_df = pd.read_csv(gender_file)
                else:
                    raise ValueError(f"Unsupported gender file format: {file_ext}")
                
                # Check for required columns
                if 'Participant ID' not in gender_df.columns or 'Gender' not in gender_df.columns:
                    # Try alternative column names for MPIIDPEye Ground_Truth.csv
                    if 'participant' in gender_df.columns and 'gender' in gender_df.columns:
                        gender_df = gender_df.rename(columns={'participant': 'Participant ID', 'gender': 'Gender'})
                    else:
                        logging.warning(f"Gender file columns not recognized. Available columns: {gender_df.columns.tolist()}")
                        # Try to guess columns - first column as ID, look for gender-like column
                        id_col = gender_df.columns[0]
                        gender_cols = [col for col in gender_df.columns if 'gender' in col.lower() or 'sex' in col.lower()]
                        if gender_cols:
                            gender_col = gender_cols[0]
                            gender_df = gender_df.rename(columns={id_col: 'Participant ID', gender_col: 'Gender'})
                        else:
                            raise ValueError(f"Could not find required columns in gender file")
                
                # Sort by Recording round if it exists
                if 'Recording round' in gender_df.columns:
                    gender_df = gender_df.sort_values(by=['Participant ID', 'Recording round'])
                
                # Group by Participant ID and take the first (earliest) record for each
                unique_participants = gender_df.drop_duplicates(subset=['Participant ID'], keep='first')
                
                # Filter out any 'Other' gender
                binary_gender = unique_participants[unique_participants['Gender'].isin(['Male', 'Female', 'male', 'female', 'M', 'F', 'm', 'f'])]
                
                # Standardize gender values
                gender_map = {
                    'Male': 'Male', 'male': 'Male', 'M': 'Male', 'm': 'Male',
                    'Female': 'Female', 'female': 'Female', 'F': 'Female', 'f': 'Female'
                }
                binary_gender['Gender'] = binary_gender['Gender'].map(gender_map)
                
                logging.info(f"After filtering duplicates and non-binary genders, found {len(binary_gender)} participants")
                
                # Excel file has 'Participant ID' and 'Gender' columns
                for _, row in binary_gender.iterrows():
                    subject_id = str(row['Participant ID'])
                    gender_data[subject_id] = row['Gender']
                
                logging.info(f"Loaded gender information for {len(gender_data)} subjects (Male/Female only)")
            except Exception as e:
                logging.error(f"Error loading gender file: {e}")
                raise
        
        dataset_name = os.path.basename(input_dir)
        is_gazebase = "GazeBase" in dataset_name and not "GazeBaseVR" in dataset_name
        
        if "MPII" in dataset_name:
            # MPIIDPEye dataset
            # Tasks:
            # 1: Comic
            # 2: Online newspaper 
            # 3: Textbook
            task_names = {1: "Comic", 2: "Online newspaper", 3: "Textbook"}
        elif is_gazebase:
            # GazeBase dataset
            # Tasks:
            # 1: BLG (Blog reading)
            # 2: FXS (Fixation)
            # 3: VD1 (Video 1)
            # 4: VD2 (Video 2)
            # 5: TEX (Text reading)
            # 6: RAN (Random)
            # 7: HSS (Hidden search/scene)
            task_names = {1: "Balura Game", 2: "Fixation", 3: "Video 1", 4: "Video 2", 5: "Text", 6: "Random", 7: "Horizontal Saccade"}
            # Map GazeBase task types to numbers
            task_map = {"BLG": 1, "FXS": 2, "VD1": 3, "VD2": 4, "TEX": 5, "RAN": 6, "HSS": 7}
        else:
            # GazeBaseVR dataset
            # Tasks:
            # 1: VRG (VR Game)
            # 2: PUR (Pursuit)
            # 3: VID (Video)
            # 4: TEX (Text)
            # 5: RAN (Random)
            task_names = {1: "Vergence", 2: "Pursuit", 3: "Video", 4: "Text", 5: "Random"}
        
        logging.info(f"Loading {len(files)} scanpath files from {input_dir}...")
        for file_path in custom_tqdm(files, desc="Loading scanpaths", leave=True, ncols=100):
            filename = os.path.basename(file_path)
            
            # Parse filename based on dataset
            parts = filename.split('_')
            
            if "MPII" in dataset_name:
                # Check file format
                if filename.startswith("Data_P"):
                    # Format: Data_PX_RY_tokens.json
                    # Where X is the participant number and Y is the recording/task number
                    try:
                        subject_id = parts[1].replace("P", "")  # Extract participant number
                        task_id = parts[2].replace("R", "")     # Extract recording/task number
                        # Remove file extension if present
                        task_id = task_id.split('.')[0]
                        task_num = int(task_id)                # Convert to integer
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Could not parse filename {filename}: {e}")
                        continue
                else:
                    # Format: MPII_Subject_XX_TaskY_tokens.json
                    try:
                        subject_id = parts[2]
                        # Remove file extension if present
                        task_str = parts[3].replace("Task", "").split('.')[0]
                        task_num = int(task_str)
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Could not parse filename {filename}: {e}")
                        continue
            elif "SynchronEyes" in dataset_name:
                # Format: S02_EyeLink_S_5030_S02_VID_tokens.json
                try:
                    subject_id = parts[3]  # subject ID is the part after S_XXXX
                    # Extract task type based on the suffix (VID, TEX, RAN, HSS, FIX)
                    task_type = parts[5]
                    # Map task types to numbers
                    task_map = {"VID": 1, "TEX": 2, "RAN": 3, "HSS": 4, "FIX": 5}
                    task_num = task_map.get(task_type, 0)
                except (IndexError, ValueError) as e:
                    logging.warning(f"Could not parse filename {filename}: {e}")
                    continue
            elif is_gazebase:
                # Format: S_XXXX_SY_ZZZ_tokens.json
                try:
                    subject_id = parts[1]  # subject ID is the 4-digit number
                    # Extract task type (FXS, BLG, VD1, VD2, TEX, RAN, HSS)
                    task_type = parts[3].split('.')[0]  # Remove file extension if present
                    task_num = task_map.get(task_type, 0)
                except (IndexError, ValueError) as e:
                    logging.warning(f"Could not parse filename {filename}: {e}")
                    continue
            else:
                # Format: S_XXXX_SY_Z_TTT_tokens.json
                try:
                    raw_subject_id = parts[1]
                    
                    # GazeBaseVR has two ID patterns:
                    # 1. S_3XXX_SY_Z_TTT_tokens.json -> Remove leading '3' to get XXX
                    # 2. S_1XXX_SY_Z_TTT_tokens.json -> Use last 3 digits (XXX)
                    
                    if raw_subject_id.startswith('3') and len(raw_subject_id) > 1:
                        subject_id = raw_subject_id[1:]
                    elif len(raw_subject_id) > 3:
                        subject_id = raw_subject_id[-3:]
                    else:
                        subject_id = raw_subject_id
                    
                    # Extract task number, remove file extension if present
                    task_part = parts[3].split('.')[0]
                    task_num = int(task_part)
                except (IndexError, ValueError) as e:
                    logging.warning(f"Could not parse filename {filename}: {e}")
                    continue
            
            with open(file_path, 'r') as f:
                scanpath = json.load(f)
            
            # Filter by task if task_filter is specified and we're doing gender classification
            if data_type == "gender" and task_filter is not None and task_num != task_filter:
                continue
            
            # Set label based on data_type
            if data_type == "task":
                label = task_num
                # For task classification, always add the scanpath
                scanpaths.append(scanpath)
            elif data_type == "gender":
                # Skip if gender information is not available
                if subject_id not in gender_data:
                    continue
                label = gender_data[subject_id]
                # Only add the scanpath if we have gender information
                scanpaths.append(scanpath)
            else:
                raise ValueError(f"Unknown data_type: {data_type}")
            
            labels.append(label)
            
            # Track subject IDs for potential subject-wise split
            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append(len(scanpaths) - 1)
    
    # Get task names based on dataset
    if "MPII" in dataset_name:
        task_names = {1: "Comic", 2: "Online newspaper", 3: "Textbook"}
    elif "SynchronEyes" in dataset_name:
        task_names = {1: "VID", 2: "TEX", 3: "RAN", 4: "HSS", 5: "FIX"}
    elif "GazeBase" in dataset_name and not "GazeBaseVR" in dataset_name:
        task_names = {1: "Balura Game", 2: "Fixation", 3: "Video 1", 4: "Video 2", 5: "Text", 6: "Random", 7: "Horizontal Saccade"}
    else:
        task_names = {1: "Vergence", 2: "Pursuit", 3: "Video", 4: "Text", 5: "Random"}
    
    logging.info(f"Loaded {len(scanpaths)} scanpaths with {len(set(labels))} different labels")
    
    # Display label distribution
    label_counts = Counter(labels)
    logging.info("Label distribution:")
    for label, count in sorted(label_counts.items()):
        if data_type == "task":
            label_name = task_names.get(label, f"Task {label}")
            logging.info(f"  Task {label} ({label_name}): {count} samples")
        else:
            logging.info(f"  {label}: {count} samples")
    
    return scanpaths, labels, subjects

def evaluate_classifier(classifier, X_test, y_test, label_names=None):
    """
    Evaluate classifier performance
    
    Parameters:
    -----------
    classifier : object
        Classifier with predict method
    X_test : list
        Test scanpaths
    y_test : list
        True labels
    label_names : dict or None
        Mapping of label IDs to names
        
    Returns:
    --------
    dict
        Evaluation results
    """
    logging.info(f"Evaluating classifier on {len(X_test)} test samples...")
    start_time = time.time()
    
    # Make predictions - don't wrap with progress bar as the predict methods 
    # now have their own progress tracking
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
                label_names[cls] = f"Task {cls}"
                
        target_names = [label_names[i] for i in test_classes]
    else:
        target_names = [f"Label {i}" for i in sorted(set(y_test))]
    
    # Make sure all predicted classes are also represented
    pred_classes = sorted(set(y_pred))
    for cls in pred_classes:
        if cls not in label_names:
            label_names[cls] = f"Task {cls}"
    
    # Get unique classes from both test and predictions
    all_classes = sorted(set(y_test) | set(y_pred))
    
    try:
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    except ValueError as e:
        logging.warning(f"Error generating classification report: {e}")
        # Fallback to a simpler report without target_names
        report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "prediction_time": prediction_time,
        "predictions": y_pred
    }

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot confusion matrix and save to file
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    class_names : list
        List of class names
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_performance_comparison(results, output_path):
    """
    Plot performance comparison between algorithms
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each algorithm
    output_path : str
        Path to save the plot
    """
    algorithms = list(results.keys())
    
    # Create shortened display names for algorithms to avoid overlapping text
    short_names = []
    for algo in algorithms:
        # For FastEyeBLAST component modes, extract just the mode
        if "FastEyeBLAST" in algo and "mode" in algo:
            # Extract the mode name from between parentheses
            mode = algo.split('(')[1].split(' mode')[0]
            short_names.append(mode)
        # For other algorithms, use shorter names
        elif "FastEyeBLAST" in algo:
            short_names.append("FastEye")
        elif "Needleman-Wunsch" in algo:
            short_names.append("NW")
        elif "NCBI MegaBLAST" in algo:
            short_names.append("MegaBLAST")
        else:
            # If no simplification matches, use full name
            short_names.append(algo)
    
    # Accuracy comparison
    accuracies = [results[algo]["accuracy"] for algo in algorithms]
    
    # Timing comparison
    times = [results[algo]["train_time"] + results[algo]["prediction_time"] for algo in algorithms]
    
    # Format time values for better readability
    time_labels = []
    for t in times:
        if t < 60:  # Less than a minute
            time_labels.append(f"{t:.1f}s")
        else:  # Minutes and seconds
            minutes = int(t // 60)
            seconds = int(t % 60)
            time_labels.append(f"{minutes}m {seconds}s")
    
    # Memory usage
    memories = [results[algo]["memory_mb"] for algo in algorithms]
    
    # Create figure with multiple subplots (increase height for better spacing)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    
    # Set bar width and positions
    bar_width = 0.7
    bar_positions = np.arange(len(algorithms))
    
    # Color map
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c'][:len(algorithms)]
    
    # Accuracy plot
    ax1.bar(bar_positions, accuracies, width=bar_width, color=colors)
    ax1.set_title('Classification Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1.05)  # Add more space at top for labels
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(short_names, rotation=0 if len(short_names) <= 3 else 15)
    
    # Add value labels above bars
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.03, f"{v:.4f}", ha='center', fontsize=11, fontweight='bold')
    
    # Time plot
    ax2.bar(bar_positions, times, width=bar_width, color=colors)
    ax2.set_title('Total Processing Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    
    # Set appropriate y-axis limit with extra space for labels
    max_time = max(times)
    ax2.set_ylim(0, max_time * 1.15)
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(short_names, rotation=0 if len(short_names) <= 3 else 15)
    
    # Add value labels above bars
    for i, (v, label) in enumerate(zip(times, time_labels)):
        ax2.text(i, v + (max_time * 0.03), f"{label}", ha='center', fontsize=11, fontweight='bold')
    
    # Memory plot
    ax3.bar(bar_positions, memories, width=bar_width, color=colors)
    ax3.set_title('Memory Usage', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Memory (MB)', fontsize=12)
    
    # Set appropriate y-axis limit
    max_memory = max(memories)
    ax3.set_ylim(0, max_memory * 1.15)
    ax3.set_xticks(bar_positions)
    ax3.set_xticklabels(short_names, rotation=0 if len(short_names) <= 3 else 15)
    
    # Add value labels above bars
    for i, v in enumerate(memories):
        ax3.text(i, v + (max_memory * 0.03), f"{v:.1f}MB", ha='center', fontsize=11, fontweight='bold')
    
    # Add grid for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and spacing
    plt.tight_layout(pad=3.0)
    
    # Add a title for the overall figure
    fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    # Save the plot with high DPI for better quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_run_name(args):
    """Generate a descriptive run name based on parameters"""
    # Create timestamp for ordering
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract dataset name from input directory
    dataset_full = os.path.basename(args.input_dir)
    dataset = dataset_full.replace('_IVT', '').replace('_AOI', '').replace('_AOIs', '')
    
    # Determine data type (IVT or AOI)
    if '_IVT' in dataset_full:
        data_format = 'IVT'
    elif '_AOI' in dataset_full or '_AOIs' in dataset_full:
        data_format = 'AOI'
    else:
        data_format = 'unknown'
    
    # Data type
    data_type = args.data_type
    
    # Add task filter info if gender classification with task filter
    if args.data_type == "gender" and args.task_filter is not None:
        # Get appropriate task name based on dataset
        task_name = None
        if "MPII" in dataset:
            task_names = {1: "Comic", 2: "OnlineNewspaper", 3: "Textbook"}
            task_name = task_names.get(args.task_filter)
        elif "SynchronEyes" in dataset:
            task_names = {1: "Video", 2: "Text", 3: "Random", 4: "HiddenSearch", 5: "Fixation"}
            task_name = task_names.get(args.task_filter)
        elif "GazeBase" in dataset and not "GazeBaseVR" in dataset:
            task_names = {1: "BaluraGame", 2: "Fixation", 3: "Video1", 4: "Video2", 5: "Text", 6: "Random", 7: "HorizontalSaccade"}
            task_name = task_names.get(args.task_filter)
        else:  # GazeBaseVR
            task_names = {1: "Vergence", 2: "Pursuit", 3: "Video", 4: "Text", 5: "Random"}
            task_name = task_names.get(args.task_filter)
        
        # Add task name if available, otherwise use task number
        if task_name:
            data_type += f"_{task_name}"
        else:
            data_type += f"_task{args.task_filter}"
    
    # Sample percentage
    sample_pct = args.sample_percentage
    
    # Test size as percentage
    test_pct = int(args.test_size * 100)
    
    # Algorithms
    if args.comparison == 'all':
        algo = 'all'
    else:
        algo = args.comparison
    
    # Create name with timestamp first
    run_name = f"{timestamp}_{dataset}_{data_format}_{data_type}_sample{sample_pct}_test{test_pct}_{algo}"
    
    # Add optional parameters if non-default
    if args.k != 5:
        run_name += f"_k{args.k}"
    
    if args.word_size != 28:
        run_name += f"_ws{args.word_size}"
        
    if args.seed_length != 3:
        run_name += f"_sl{args.seed_length}"
    
    if args.variable_word_size != 64:
        run_name += f"_vws{args.variable_word_size}"
    
    return run_name

def is_aoi_dataset(input_dir):
    """
    Determine if the input directory contains AOI-encoded data
    
    Parameters:
    -----------
    input_dir : str
        Directory to check
        
    Returns:
    --------
    bool
        True if AOI data, False if IVT data
    """
    # Check for encoded_sequences.json which is typically used in AOI datasets
    if os.path.exists(os.path.join(input_dir, 'encoded_sequences.json')):
        return True
    
    # Check if the directory name contains 'AOI'
    if 'AOI' in os.path.basename(input_dir):
        return True
    
    # Default to IVT
    return False

def main():
    """Main function to handle command line arguments and run comparisons"""
    parser = argparse.ArgumentParser(description='Compare Needleman-Wunsch, MegaBLAST, and FastEyeBLAST on scanpath classification')
    
    # Input data options
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing tokenized scanpath JSON files')
    parser.add_argument('--output_dir', type=str, default='Comparison_Results',
                        help='Base directory to save results')
    parser.add_argument('--data_type', type=str, choices=['task', 'gender'], default='task',
                        help='Type of classification task: task or gender')
    parser.add_argument('--gender_file', type=str, default=None,
                       help='Path to Excel/CSV file containing gender information (if not provided, will be auto-selected based on dataset)')
    parser.add_argument('--task_filter', type=int, default=None,
                       help='Filter samples by task ID when doing gender classification (e.g., only use samples from task 1)')
    
    # Algorithm selection
    parser.add_argument('--comparison', type=str, default='all',
                        choices=['all', 'nw-fb-mb', 'nw-fb', 'nw-mb', 'fb-mb', 'nw-only', 'fb-only', 'mb-only', 
                                'fb-var', 'nw-fb-var', 'mb-fb-var', 'fb-var-only', 'all-with-var',
                                'fb-component-comparison'],
                        help='Which algorithms to compare')
    
    # Sampling options
    parser.add_argument('--sample_percentage', type=int, default=100,
                        help='Percentage of data to use (randomly sampled)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    
    # Algorithm parameters
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors for k-NN classification')
    parser.add_argument('--nw_nj', type=str, choices=['true', 'false'], default='true',
                        help='Use Numba JIT for Needleman-Wunsch (default: true)')
    parser.add_argument('--word_size', type=int, default=28, 
                        help='BLAST word size parameter')
    parser.add_argument('--seed_length', type=int, default=3,
                        help='Seed length for FastEyeBLAST')
    parser.add_argument('--band_width', type=int, default=5,
                        help='Band width for FastEyeBLAST banded alignment')
    parser.add_argument('--max_candidates', type=int, default=20,
                        help='Maximum number of candidates for FastEyeBLAST similarity calculation')
    parser.add_argument('--variable_word_size', type=int, default=64, choices=[8, 16, 32, 64, 128, 256],
                        help='Word size for FastEyeBLAST Variable (default: 64)')
    
    # Component comparison options
    parser.add_argument('--component_modes', type=str, nargs='+', 
                        choices=['full', 'simhash_only', 'alignment_only'],
                        default=['full', 'simhash_only', 'alignment_only'],
                        help='Which component modes to compare in FastEyeBLAST')
    
    # Other options
    parser.add_argument('--random', action='store_true',
                        help='Use random seed instead of fixed seed (default: False)')
    
    args = parser.parse_args()
    
    # Create unique output directory with parameters
    run_id = generate_run_name(args)
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging to capture terminal output
    original_stdout = setup_logging(output_dir)
    
    # Log all arguments
    logging.info(f"Starting comparison study with the following parameters:")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"  {arg}: {value}")
    
    # Save run parameters
    params = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input_dir": args.input_dir,
        "data_type": args.data_type,
        "gender_file": args.gender_file,
        "task_filter": args.task_filter,
        "sample_percentage": args.sample_percentage,
        "test_size": args.test_size,
        "n_jobs": args.n_jobs,
        "k": args.k,
        "word_size": args.word_size,
        "random": args.random,
        "nw_nj": args.nw_nj,
        "comparison": args.comparison,
        "seed_length": args.seed_length,
        "band_width": args.band_width,
        "max_candidates": args.max_candidates,
        "variable_word_size": args.variable_word_size,
        "component_modes": args.component_modes,
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
    
    # Set random seed for reproducibility if not using random mode
    if not args.random:
        np.random.seed(42)
        random.seed(42)
        logging.info("Using fixed seed for reproducibility (42)")
    else:
        logging.info("Using random seed (results will vary between runs)")
    
    # Auto-select gender file if doing gender classification and no file specified
    if args.data_type == 'gender' and args.gender_file is None:
        # Determine which dataset we're using
        dataset_name = os.path.basename(args.input_dir)
        if "MPII" in dataset_name:
            args.gender_file = r"C:\Users\aynad\Desktop\Home\TUM\Research\Datasets\MPIIDPEye\Ground_Truth.csv"
            logging.info(f"Auto-selected MPIIDPEye gender file: {args.gender_file}")
        elif "GazeBase" in dataset_name and not "GazeBaseVR" in dataset_name:
            args.gender_file = r"C:\Users\aynad\Desktop\Home\TUM\Research\Datasets\GazeBase\GazeBaseDemoInfo.xlsx"
            logging.info(f"Auto-selected GazeBase gender file: {args.gender_file}")
        else:
            args.gender_file = r"C:\Users\aynad\Desktop\Home\TUM\Research\Datasets\GazeBaseVR\participant_details.xlsx"
            logging.info(f"Auto-selected GazeBaseVR gender file: {args.gender_file}")
    
    # Load data
    logging.info(f"Loading data from {args.input_dir}.")
    
    # Log if we're filtering by task for gender classification
    if args.data_type == "gender" and args.task_filter is not None:
        logging.info(f"Filtering gender classification to only use data from task {args.task_filter}")
        
    scanpaths, labels, subjects = load_and_prepare_data(
        args.input_dir, 
        data_type=args.data_type,
        sample_percentage=args.sample_percentage,
        gender_file=args.gender_file,
        task_filter=args.task_filter
    )
    
    # Check if we have enough data
    if len(scanpaths) < 10:
        logging.error(f"Not enough data: only {len(scanpaths)} samples loaded. Please check input directory and gender file if applicable.")
        sys.exit(1)
    
    # --------------------------------------------------------------------------
    # FULL PAIRWISE SIMILARITY MATRIX COMPUTATION & COMPARISON
    # --------------------------------------------------------------------------
    # Ensure reproducible ordering: store indices 0..n-1
    n = len(scanpaths)  # number of samples
    indices = list(range(n))
    
    # (A) Compute & save Needleman-Wunsch similarity matrix
    print(f"Computing full NW similarity matrix for {n} samples...")
    # (A) First compute the raw DP scores (integers)
    raw_nw = compute_similarity_matrix(
        scanpaths,
        n_jobs=args.n_jobs,
        use_numba=(args.nw_nj.lower() == 'true')
    )

    # (B) Now normalize every entry so that it falls in [0,1].
    #     Since our NW uses match=+1, the maximum achievable score for (i,j)
    #     is simply "shorter_length_of_seq_i_and_seq_j".
    nw_sim_matrix = raw_nw.astype(np.float64)  # copy into float array

    for i in range(n):
        len_i = len(scanpaths[i])
        for j in range(i, n):
            len_j = len(scanpaths[j])
            shorter = min(len_i, len_j)
            if shorter > 0:
                # divide raw DP score by (shorter * match=1)
                normalized_val = raw_nw[i, j] / float(shorter)
            else:
                normalized_val = 0.0

            nw_sim_matrix[i, j] = normalized_val
            nw_sim_matrix[j, i] = normalized_val

    # Force the diagonal to be exactly 1.0 (self-similarity)
    np.fill_diagonal(nw_sim_matrix, 1.0)

    # Save the normalized NW matrix
    nw_matrix_path = os.path.join(output_dir, "full_nw_similarity.npy")
    np.save(nw_matrix_path, nw_sim_matrix)
    logging.info(f"Saved NW similarity matrix to {nw_matrix_path}")
    
    # (B) Compute & save FastEyeBLAST similarity matrix
    print(f"Building FB index and computing full similarity matrix for {n} samples...")
    fb_index = FastEyeBLAST_IVT(
        seed_length=args.seed_length,
        band_width=args.band_width,
        num_neighbors=args.k,
        max_candidates=args.max_candidates,
        use_duration_bins=True,
        use_parallel=True,
        num_processes=args.n_jobs if args.n_jobs > 0 else None
    )
    # Build index on all scanpaths
    fb_index.fit(scanpaths, labels)
    
    fb_sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n), desc="FB similarity rows", ncols=100):
        for j in range(i+1, n):
            # Choose method based on available API
            if hasattr(fb_index, "_compute_pair"):
                _, _, sim_ij = fb_index._compute_pair((i, j, scanpaths[i], scanpaths[j]))
            elif hasattr(fb_index, "compute_similarity"):
                sim_ij = fb_index.compute_similarity(scanpaths[i], scanpaths[j])
            else:
                raise AttributeError("FastEyeBLAST object has no method to compute pairwise similarity")
            fb_sim_matrix[i, j] = sim_ij
            fb_sim_matrix[j, i] = sim_ij

    # Normalize FB similarity matrix to [0,1] based on max off-diagonal value
    iu = np.triu_indices(n, k=1)
    if iu[0].size > 0:
        max_fb = np.max(fb_sim_matrix[iu])
        if max_fb > 0:
            fb_sim_matrix = fb_sim_matrix / max_fb

    # Force the diagonal to be exactly 1.0 (self-similarity)
    np.fill_diagonal(fb_sim_matrix, 1.0)

    fb_matrix_path = os.path.join(output_dir, "full_fb_similarity.npy")
    np.save(fb_matrix_path, fb_sim_matrix)
    logging.info(f"Saved FB similarity matrix to {fb_matrix_path}")
    
    # (C) Extract upper triangular entries (i < j) to compare
    iu = np.triu_indices(n, k=1)
    nw_flat = nw_sim_matrix[iu]
    fb_flat = fb_sim_matrix[iu]
    
    # Compute Pearson & Spearman correlations
    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(nw_flat, fb_flat)
    spearman_rho, spearman_p = spearmanr(nw_flat, fb_flat)
    
    stats = {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }
    stats_path = os.path.join(output_dir, "matrix_correlation_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Saved NW vs. FB correlation stats to {stats_path}")
    
    # (D) Produce enhanced scatter plot with density coloring and annotated stats
    # Prepare error metric (absolute difference)
    errors = np.abs(fb_flat - nw_flat)
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    pct_close = np.sum(errors < 0.05) / len(errors) * 100  # percentage within ±0.05
    
    # Create scatter plot colored by error
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(nw_flat, fb_flat, c=errors, cmap='coolwarm', s=10, alpha=0.6)
    plt.colorbar(scatter, label='|FB - NW| (Absolute Difference)')
    plt.xlabel("NW similarity (normalized)")
    plt.ylabel("FB similarity (normalized)")
    plt.title("FB vs. NW similarity scores (upper triangle)\nColored by absolute difference")
    lims = [0, 1]
    plt.plot(lims, lims, "--", color="gray", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Annotate Pearson and Spearman on plot
    text_str = (f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.2e})\n"
                f"Spearman ρ = {spearman_rho:.3f} (p = {spearman_p:.2e})\n"
                f"Mean |error| = {mean_error:.3f}\n"
                f"% pairs |error|<0.05 = {pct_close:.1f}%")
    plt.annotate(text_str, xy=(0.50, -0.20), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    scatter_path = os.path.join(output_dir, "fb_vs_nw_scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    logging.info(f"Saved enhanced FB vs. NW similarity scatter to {scatter_path}")
    
    # Create scatter_report.log with extensive interpretation
    report_lines = []
    report_lines.append("Scatter Plot Interpretation Report")
    report_lines.append("===============================")
    report_lines.append(f"Total pairs compared: {len(nw_flat)}")
    report_lines.append(f"Pearson correlation (r): {pearson_r:.4f} (p = {pearson_p:.2e})")
    if pearson_r > 0.8:
        report_lines.append("  - Interpretation: Strong linear correlation between NW and FB scores.")
    elif pearson_r > 0.5:
        report_lines.append("  - Interpretation: Moderate linear correlation.")
    else:
        report_lines.append("  - Interpretation: Weak linear correlation.")
    report_lines.append(f"Spearman correlation (ρ): {spearman_rho:.4f} (p = {spearman_p:.2e})")
    if spearman_rho > 0.8:
        report_lines.append("  - Interpretation: Strong rank-order agreement between NW and FB.")
    elif spearman_rho > 0.5:
        report_lines.append("  - Interpretation: Moderate rank-order agreement.")
    else:
        report_lines.append("  - Interpretation: Weak rank-order agreement.")
    report_lines.append(f"Mean absolute error between scores: {mean_error:.4f}")
    report_lines.append(f"Median absolute error: {median_error:.4f}")
    report_lines.append(f"Percentage of pairs with |error| < 0.05: {pct_close:.2f}%")
    report_lines.append("")
    report_lines.append("Points close to diagonal:")
    report_lines.append("  - These represent pairs where FB and NW agree closely (similarity difference < 0.05).")
    report_lines.append("Points above diagonal (FB > NW):")
    report_lines.append("  - FB tends to overestimate similarity compared to NW for these pairs.")
    report_lines.append("Points below diagonal (FB < NW):")
    report_lines.append("  - NW tends to give higher similarity than FB for these pairs.")
    report_lines.append("")
    report_lines.append("Density observations:")
    report_lines.append("  - Regions with warmer colors (red) indicate larger disagreements between FB and NW.")
    report_lines.append("  - Regions with cooler colors (blue) indicate closer agreement.")
    report_lines.append("")
    report_lines.append("Overall assessment:")
    if pearson_r > 0.8 and spearman_rho > 0.8 and pct_close > 70:
        report_lines.append("  - FB's two-stage pipeline aligns very well with full NW, both in value and rank-order.")
    elif pearson_r > 0.5 and spearman_rho > 0.5:
        report_lines.append("  - FB shows reasonable agreement with NW but with notable outliers.")
    else:
        report_lines.append("  - FB diverges significantly from NW; consider revisiting LSH or banded alignment parameters.")
    report_lines.append("")
    report_lines.append("Detailed suggestions:")
    report_lines.append("  - Investigate points with high |error| to identify systematic biases.")
    report_lines.append("  - If many points lie below the diagonal, FB may be missing alignments that NW captures.")
    report_lines.append("  - If many points lie above the diagonal, FB may be producing false positives due to LSH.")
    
    report_path = os.path.join(output_dir, "scatter_report.log")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    logging.info(f"Saved scatter interpretation report to {report_path}")
    
    # --------------------------------------------------------------------------
    
    # Get label names based on dataset and data type
    label_names = {}
    dataset_name = os.path.basename(args.input_dir)
    
    if args.data_type == "task":
        if "MPII" in dataset_name:
            # MPIIDPEye task names
            label_names = {1: "Comic", 2: "Online newspaper", 3: "Textbook"}
        elif "SynchronEyes" in dataset_name:
            # SynchronEyes task names
            label_names = {
                1: "VID",
                2: "TEX",
                3: "RAN",
                4: "HSS",
                5: "FIX"
            }
        elif "GazeBase" in dataset_name and not "GazeBaseVR" in dataset_name:
            # GazeBase task names
            label_names = {
                1: "Balura Game",
                2: "Fixation",
                3: "Video 1",
                4: "Video 2",
                5: "Text",
                6: "Random",
                7: "Horizontal Saccade"
            }
        else:
            # GazeBaseVR task names
            label_names = {1: "Vergence", 2: "Pursuit", 3: "Video", 4: "Text", 5: "Random"}
    else:
        # For gender classification, just use the labels as names
        label_names = {label: str(label) for label in set(labels)}
    
    # Split data into train/test with configurable ratio
    train_percent = int((1 - args.test_size) * 100)
    test_percent = int(args.test_size * 100)
    logging.info(f"Splitting data into train/test sets ({train_percent}/{test_percent}).")
    X_train, X_test, y_train, y_test = train_test_split(
        scanpaths, labels, test_size=args.test_size, random_state=42 if not args.random else None
    )
    
    logging.info(f"Training set: {len(X_train)} samples")
    logging.info(f"Test set: {len(X_test)} samples")
    
    # Save dataset split information
    split_info = {
        "total_samples": len(scanpaths),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "task_distribution": {str(k): v for k, v in Counter(labels).items()}
    }
    split_file = os.path.join(output_dir, "dataset_info.json")
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Initialize results dictionary
    results = {}
    
    # Determine which algorithms to run based on the comparison parameter
    run_nw = args.comparison in ['all', 'all-with-var', 'nw-fb-mb', 'nw-fb', 'nw-mb', 'nw-only', 'nw-fb-var']
    run_mb = args.comparison in ['all', 'all-with-var', 'nw-fb-mb', 'fb-mb', 'nw-mb', 'mb-only', 'mb-fb-var']
    run_fb = args.comparison in ['all', 'all-with-var', 'nw-fb-mb', 'nw-fb', 'fb-mb', 'fb-only']
    run_fb_var = args.comparison in ['all-with-var', 'fb-var', 'nw-fb-var', 'mb-fb-var', 'fb-var-only']
    run_fb_component = args.comparison in ['fb-component-comparison']
    
    # -------------------------------------------------------------------------
    # 1. Needleman-Wunsch Classification
    # -------------------------------------------------------------------------
    if run_nw:
        logging.info("\n" + "="*50)
        logging.info("1. Needleman-Wunsch Classification")
        logging.info("="*50)
        
        # Determine if we're dealing with AOI or IVT data
        is_aoi = is_aoi_dataset(args.input_dir)
        
        # Check if we're using Numba JIT
        use_numba = args.nw_nj.lower() == 'true'
        if is_aoi:
            if use_numba and not NUMBA_AVAILABLE_AOI:
                logging.warning("Numba not available for AOI, falling back to non-JIT version")
                use_numba = False
        else:
            if use_numba and not NUMBA_AVAILABLE_IVT:
                logging.warning("Numba not available for IVT, falling back to non-JIT version")
                use_numba = False
            
        logging.info(f"Using {'Numba JIT' if use_numba else 'non-JIT'} version of Needleman-Wunsch")
        
        # Initialize classifier
        nw_classifier = NeedlemanWunschClassifier(
            k=args.k, 
            match=1, 
            mismatch=-1, 
            gap=-2, 
            n_jobs=args.n_jobs,
            use_numba=use_numba
        )
        
        # Train
        gc.collect()  # Clear memory before training
        start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        logging.info("Training Needleman-Wunsch classifier.")
        start_time = time.time()
        nw_classifier.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        memory_used_train = current_mem - start_mem
        
        logging.info(f"Needleman-Wunsch training completed in {train_time:.2f} seconds")
        logging.info(f"Training memory usage: {memory_used_train:.1f} MB")
        
        # Evaluate
        logging.info("Evaluating Needleman-Wunsch classifier.")
        
        # Measure memory usage during prediction
        gc.collect()  # Clear memory before prediction
        pred_start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        nw_results = evaluate_classifier(nw_classifier, X_test, y_test, label_names)
        
        pred_current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        memory_used_pred = pred_current_mem - pred_start_mem
        
        # Total memory usage is the sum of training and prediction memory
        memory_used = memory_used_train + memory_used_pred
        
        logging.info(f"Needleman-Wunsch Accuracy: {nw_results['accuracy']:.4f}")
        logging.info(f"Prediction time: {nw_results['prediction_time']:.2f} seconds")
        logging.info(f"Prediction memory usage: {memory_used_pred:.1f} MB")
        logging.info(f"Total memory usage: {memory_used:.1f} MB")
        
        # Save confusion matrix
        nw_cm_path = os.path.join(output_dir, 'nw_confusion_matrix.png')
        plot_confusion_matrix(
            nw_results['confusion_matrix'],
            [label_names[i] for i in sorted(set(y_test))],
            nw_cm_path
        )
        
        # Store results
        results["Needleman-Wunsch"] = {
            "accuracy": nw_results['accuracy'],
            "train_time": train_time,
            "prediction_time": nw_results['prediction_time'],
            "memory_mb": memory_used,
            "confusion_matrix": nw_results['confusion_matrix'].tolist(),
            "report": nw_results['report']
        }
    
    # -------------------------------------------------------------------------
    # 2. NCBI MegaBLAST Classification
    # -------------------------------------------------------------------------
    if run_mb:
        logging.info("\n" + "="*50)
        logging.info("2. NCBI MegaBLAST Classification")
        logging.info("="*50)
        
        # Determine if we're dealing with AOI or IVT data
        is_aoi = is_aoi_dataset(args.input_dir)
        
        # Initialize classifier
        mb_classifier = NCBIMegaBLASTScanpathClassifier(
            k=args.k,
            word_size=args.word_size,
            perc_identity=90,
            evalue=0.001,
            n_jobs=args.n_jobs
        )
        
        # Train
        gc.collect()  # Clear memory before training
        start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        logging.info("Training MegaBLAST classifier.")
        start_time = time.time()
        mb_classifier.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        memory_used_train = current_mem - start_mem
        
        logging.info(f"NCBI MegaBLAST training completed in {train_time:.2f} seconds")
        logging.info(f"Training memory usage: {memory_used_train:.1f} MB")
        
        # Evaluate
        logging.info("Evaluating NCBI MegaBLAST classifier.")
        
        # Measure memory usage during prediction
        gc.collect()  # Clear memory before prediction
        pred_start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        mb_results = evaluate_classifier(mb_classifier, X_test, y_test, label_names)
        
        pred_current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        memory_used_pred = pred_current_mem - pred_start_mem
        
        # Total memory usage is the sum of training and prediction memory
        memory_used = memory_used_train + memory_used_pred
        
        logging.info(f"NCBI MegaBLAST Accuracy: {mb_results['accuracy']:.4f}")
        logging.info(f"Prediction time: {mb_results['prediction_time']:.2f} seconds")
        logging.info(f"Prediction memory usage: {memory_used_pred:.1f} MB")
        logging.info(f"Total memory usage: {memory_used:.1f} MB")
        
        # Save confusion matrix
        mb_cm_path = os.path.join(output_dir, 'mb_confusion_matrix.png')
        plot_confusion_matrix(
            mb_results['confusion_matrix'],
            [label_names[i] for i in sorted(set(y_test))],
            mb_cm_path
        )
        
        # Store results
        results["NCBI MegaBLAST"] = {
            "accuracy": mb_results['accuracy'],
            "train_time": train_time,
            "prediction_time": mb_results['prediction_time'],
            "memory_mb": memory_used,
            "confusion_matrix": mb_results['confusion_matrix'].tolist(),
            "report": mb_results['report']
        }
    
    # -------------------------------------------------------------------------
    # 3. FastEyeBLAST Classification
    # -------------------------------------------------------------------------
    if run_fb:
        logging.info("\n" + "="*50)
        logging.info("3. FastEyeBLAST Classification")
        logging.info("="*50)
        
        # Determine if we're dealing with AOI or IVT data
        is_aoi = is_aoi_dataset(args.input_dir)
        
        # Initialize classifier based on data type
        if is_aoi and AOI_MODULES_AVAILABLE:
            logging.info("Using AOI version of FastEyeBLAST")
            fb_classifier = FastEyeBLAST_AOI(
                seed_length=args.seed_length,
                band_width=args.band_width,
                num_neighbors=args.k,
                max_candidates=args.max_candidates,
                use_parallel=True,
                num_processes=args.n_jobs if args.n_jobs > 0 else None
            )
        else:
            if is_aoi and not AOI_MODULES_AVAILABLE:
                logging.warning("AOI modules not available, falling back to IVT version")
            else:
                logging.info("Using IVT version of FastEyeBLAST")
                
            fb_classifier = FastEyeBLAST_IVT(
                seed_length=args.seed_length,
                band_width=args.band_width,
                num_neighbors=args.k,
                max_candidates=args.max_candidates,
                use_duration_bins=True,
                use_parallel=True,
                num_processes=args.n_jobs if args.n_jobs > 0 else None
            )
        
        # Train
        gc.collect()  # Clear memory before training
        start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        logging.info("Training FastEyeBLAST classifier.")
        start_time = time.time()
        fb_classifier.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        memory_used_train = current_mem - start_mem
        
        logging.info(f"FastEyeBLAST training completed in {train_time:.2f} seconds")
        logging.info(f"Training memory usage: {memory_used_train:.1f} MB")
        
        # Evaluate
        logging.info("Evaluating FastEyeBLAST classifier.")
        
        # Measure memory usage during prediction
        gc.collect()  # Clear memory before prediction
        pred_start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        
        fb_results = evaluate_classifier(fb_classifier, X_test, y_test, label_names)
        
        pred_current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
        memory_used_pred = pred_current_mem - pred_start_mem
        
        # Total memory usage is the sum of training and prediction memory
        memory_used = memory_used_train + memory_used_pred
        
        logging.info(f"FastEyeBLAST Accuracy: {fb_results['accuracy']:.4f}")
        logging.info(f"Prediction time: {fb_results['prediction_time']:.2f} seconds")
        logging.info(f"Prediction memory usage: {memory_used_pred:.1f} MB")
        logging.info(f"Total memory usage: {memory_used:.1f} MB")
        
        # Save confusion matrix
        fb_cm_path = os.path.join(output_dir, 'fb_confusion_matrix.png')
        plot_confusion_matrix(
            fb_results['confusion_matrix'],
            [label_names[i] for i in sorted(set(y_test))],
            fb_cm_path
        )
        
        # Store results
        results["FastEyeBLAST"] = {
            "accuracy": fb_results['accuracy'],
            "train_time": train_time,
            "prediction_time": fb_results['prediction_time'],
            "memory_mb": memory_used,
            "confusion_matrix": fb_results['confusion_matrix'].tolist(),
            "report": fb_results['report']
        }
    
    # -------------------------------------------------------------------------
    # 4. FastEyeBLAST Variable Classification
    # -------------------------------------------------------------------------
    if run_fb_var:
        logging.info("\n" + "="*50)
        logging.info("4. FastEyeBLAST Variable Classification")
        logging.info("="*50)
        
        # Determine if we're dealing with AOI or IVT data
        is_aoi = is_aoi_dataset(args.input_dir)
        
        if is_aoi:
            logging.warning("Variable word size FastEyeBLAST is currently only implemented for IVT data, skipping")
        else:
            logging.info(f"Using IVT version of FastEyeBLAST Variable with word_size={args.variable_word_size}")
            
            # Initialize classifier
            fb_var_classifier = FastEyeBLAST_IVT_Variable(
                seed_length=args.seed_length,
                band_width=args.band_width,
                num_neighbors=args.k,
                max_candidates=args.max_candidates,
                use_duration_bins=True,
                use_parallel=True,
                num_processes=args.n_jobs if args.n_jobs > 0 else None,
                word_size=args.variable_word_size
            )
            
            # Train
            gc.collect()  # Clear memory before training
            start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
            
            logging.info(f"Training FastEyeBLAST Variable (word_size={args.variable_word_size}) classifier.")
            start_time = time.time()
            fb_var_classifier.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
            memory_used_train = current_mem - start_mem
            
            logging.info(f"FastEyeBLAST Variable training completed in {train_time:.2f} seconds")
            logging.info(f"Training memory usage: {memory_used_train:.1f} MB")
            
            # Evaluate
            logging.info("Evaluating FastEyeBLAST Variable classifier.")
            
            # Measure memory usage during prediction
            gc.collect()  # Clear memory before prediction
            pred_start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
            
            fb_var_results = evaluate_classifier(fb_var_classifier, X_test, y_test, label_names)
            
            pred_current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
            memory_used_pred = pred_current_mem - pred_start_mem
            
            # Total memory usage is the sum of training and prediction memory
            memory_used = memory_used_train + memory_used_pred
            
            logging.info(f"FastEyeBLAST Variable Accuracy: {fb_var_results['accuracy']:.4f}")
            logging.info(f"Prediction time: {fb_var_results['prediction_time']:.2f} seconds")
            logging.info(f"Prediction memory usage: {memory_used_pred:.1f} MB")
            logging.info(f"Total memory usage: {memory_used:.1f} MB")
            
            # Save confusion matrix
            fb_var_cm_path = os.path.join(output_dir, 'fb_var_confusion_matrix.png')
            plot_confusion_matrix(
                fb_var_results['confusion_matrix'],
                [label_names[i] for i in sorted(set(y_test))],
                fb_var_cm_path
            )
            
            # Store results
            results[f"FastEyeBLAST Variable (w={args.variable_word_size})"] = {
                "accuracy": fb_var_results['accuracy'],
                "train_time": train_time,
                "prediction_time": fb_var_results['prediction_time'],
                "memory_mb": memory_used,
                "confusion_matrix": fb_var_results['confusion_matrix'].tolist(),
                "report": fb_var_results['report']
            }
    
    # -------------------------------------------------------------------------
    # 5. FastEyeBLAST Component Comparison
    # -------------------------------------------------------------------------
    if run_fb_component:
        logging.info("\n" + "="*50)
        logging.info("5. FastEyeBLAST Component Comparison")
        logging.info("="*50)
        
        # Determine if we're dealing with AOI or IVT data
        is_aoi = is_aoi_dataset(args.input_dir)
        
        if is_aoi:
            logging.warning("Component comparison is currently only implemented for IVT data, skipping")
        else:
            # Use tqdm for component mode iteration
            for mode in tqdm(args.component_modes, desc="Testing component modes", leave=True, ncols=100):
                logging.info(f"\nTesting FastEyeBLAST component mode: {mode}")
                
                # Initialize classifier with the specific mode
                fb_comp_classifier = FastEyeBLAST_Compare(
                    mode=mode,
                    seed_length=args.seed_length,
                    band_width=args.band_width,
                    num_neighbors=args.k,
                    max_candidates=args.max_candidates,
                    use_duration_bins=True,
                    use_parallel=True,
                    num_processes=args.n_jobs if args.n_jobs > 0 else None
                )
                
                # Train
                gc.collect()  # Clear memory before training
                start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
                
                logging.info(f"Training FastEyeBLAST ({mode} mode) classifier.")
                start_time = time.time()
                fb_comp_classifier.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
                memory_used_train = current_mem - start_mem
                
                logging.info(f"FastEyeBLAST ({mode} mode) training completed in {train_time:.2f} seconds")
                logging.info(f"Training memory usage: {memory_used_train:.1f} MB")
                
                # Evaluate
                logging.info(f"Evaluating FastEyeBLAST ({mode} mode) classifier.")
                
                # Measure memory usage during prediction
                gc.collect()  # Clear memory before prediction
                pred_start_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
                
                fb_comp_results = evaluate_classifier(fb_comp_classifier, X_test, y_test, label_names)
                
                pred_current_mem = psutil.Process().memory_info().rss / (1024*1024)  # MB
                memory_used_pred = pred_current_mem - pred_start_mem
                
                # Total memory usage is the sum of training and prediction memory
                memory_used = memory_used_train + memory_used_pred
                
                logging.info(f"FastEyeBLAST ({mode} mode) Accuracy: {fb_comp_results['accuracy']:.4f}")
                logging.info(f"Prediction time: {fb_comp_results['prediction_time']:.2f} seconds")
                logging.info(f"Prediction memory usage: {memory_used_pred:.1f} MB")
                logging.info(f"Total memory usage: {memory_used:.1f} MB")
                
                # Save confusion matrix
                fb_comp_cm_path = os.path.join(output_dir, f'fb_comp_{mode}_confusion_matrix.png')
                plot_confusion_matrix(
                    fb_comp_results['confusion_matrix'],
                    [label_names[i] for i in sorted(set(y_test))],
                    fb_comp_cm_path
                )
                
                # Store results
                results[f"FastEyeBLAST ({mode} mode)"] = {
                    "accuracy": fb_comp_results['accuracy'],
                    "train_time": train_time,
                    "prediction_time": fb_comp_results['prediction_time'],
                    "memory_mb": memory_used,
                    "confusion_matrix": fb_comp_results['confusion_matrix'].tolist(),
                    "report": fb_comp_results['report']
                }
            
            # Create a bar chart comparing the accuracies of the different modes
            if len(args.component_modes) > 1:
                logging.info("Creating component comparison bar chart.")
                plt.figure(figsize=(10, 6))
                mode_names = [f"FastEyeBLAST ({mode} mode)" for mode in args.component_modes]
                accuracies = [results[name]["accuracy"] for name in mode_names]
                plt.bar(mode_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'][:len(mode_names)])
                plt.title('FastEyeBLAST Component Comparison - Accuracy')
                plt.ylabel('Accuracy')
                plt.ylim(0, 1)
                for i, v in enumerate(accuracies):
                    plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
                plt.tight_layout(rect=[0, 0.1, 1, 1])
                comp_chart_path = os.path.join(output_dir, 'component_comparison_accuracy.png')
                plt.savefig(comp_chart_path)
                plt.close()
                logging.info(f"Component comparison chart saved to {comp_chart_path}")
    
    # -------------------------------------------------------------------------
    # Plot comparison results
    # -------------------------------------------------------------------------
    if len(results) > 1:  # Only create comparison if we have at least 2 algorithms
        logging.info("\n" + "="*50)
        logging.info("Plotting comparison results")
        logging.info("="*50)
        
        comparison_path = os.path.join(output_dir, 'performance_comparison.png')
        logging.info(f"Creating performance comparison chart at {comparison_path}.")
        plot_performance_comparison(results, comparison_path)
    else:
        logging.info("\n" + "="*50)
        logging.info("Single algorithm results")
        logging.info("="*50)
        
        # Display the results of the single algorithm that was run
        for algo_name, result in results.items():
            logging.info(f"{algo_name} Accuracy: {result['accuracy']:.4f}")
            logging.info(f"{algo_name} Train time: {result['train_time']:.2f} seconds")
            logging.info(f"{algo_name} Prediction time: {result['prediction_time']:.2f} seconds")
            logging.info(f"{algo_name} Memory usage: {result['memory_mb']:.2f} MB")
    
    # Save detailed results to JSON
    json_path = os.path.join(output_dir, 'comparison_results.json')
    logging.info(f"Saving detailed results to {json_path}.")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {output_dir}/")
    if len(results) > 1:
        logging.info(f"Performance comparison plot saved to {comparison_path}")
    logging.info(f"Run parameters saved to {params_file}")
    
    # Restore original stdout
    sys.stdout = original_stdout
    logging.info("Comparison study completed.")

if __name__ == "__main__":
    main()
