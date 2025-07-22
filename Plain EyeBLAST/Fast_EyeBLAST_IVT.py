

import os
import glob
import json
import time
import random
import argparse
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import multiprocessing as mp
from functools import partial

# Constants
MASK64 = (1 << 64) - 1
DEFAULT_SEED_LENGTH = 3
DEFAULT_NUM_NEIGHBORS = 5
DEFAULT_BAND_WIDTH = 5
DEFAULT_NUM_HASH_TABLES = 8
DEFAULT_NUM_HASH_BITS = 16
DEFAULT_MAX_CANDIDATES = 20
DEFAULT_LSH_HASH_SIZE = 64
DEFAULT_DURATION_BINS = 8  # For duration binning

class FastEyeBLAST:

    
    def __init__(self, seed_length=DEFAULT_SEED_LENGTH, band_width=DEFAULT_BAND_WIDTH,
                 num_neighbors=DEFAULT_NUM_NEIGHBORS, num_hash_tables=DEFAULT_NUM_HASH_TABLES,
                 num_hash_bits=DEFAULT_NUM_HASH_BITS, max_candidates=DEFAULT_MAX_CANDIDATES,
                 lsh_hash_size=DEFAULT_LSH_HASH_SIZE, use_duration_bins=True, 
                 duration_bins=DEFAULT_DURATION_BINS, use_parallel=True, num_processes=None):
        """
        Initialize FastEyeBLAST with optimized parameters
        
        Parameters:
        -----------
        seed_length : int
            Length of seed words for initial matching
        band_width : int
            Width of band for banded dynamic programming
        num_neighbors : int
            Number of neighbors for k-NN classification
        num_hash_tables : int
            Number of hash tables for LSH
        num_hash_bits : int
            Number of bits per hash table slice
        max_candidates : int
            Maximum number of candidates for fine-grained similarity
        lsh_hash_size : int
            Size of SimHash sketches in bits
        use_duration_bins : bool
            Whether to bin durations for token encoding
        duration_bins : int
            Number of bins for duration quantization
        use_parallel : bool
            Whether to use parallel processing
        num_processes : int or None
            Number of processes for parallel processing
        """
        self.seed_length = seed_length
        self.band_width = band_width
        self.num_neighbors = num_neighbors
        self.num_hash_tables = num_hash_tables
        self.num_hash_bits = num_hash_bits
        self.max_candidates = max_candidates
        self.lsh_hash_size = lsh_hash_size
        self.use_duration_bins = use_duration_bins
        self.duration_bins = duration_bins
        
        # Tokenization and encoding
        self.token_to_id = {}  # Maps tokens to integer IDs
        self.id_to_token = {}  # Maps integer IDs back to tokens
        self.next_token_id = 0  # Next available token ID
        
        # Bit-packing parameters
        self.bits_per_token = None
        self.tokens_per_word = None
        self.base = None  # For rolling hash
        self.pow_base = None  # Precomputed powers for rolling hash
        
        # LSH parameters
        self.lsh_tables = None  # LSH hash tables
        self.id_bitmasks = None  # Random bitmasks for SimHash
        
        # Training data
        self.encoded_scanpaths = None
        self.packed_scanpaths = None
        self.simhash_sketches = None
        self.labels = None
        
        # Parallel processing
        self.use_parallel = use_parallel
        if num_processes is None:
            self.num_processes = mp.cpu_count()
        else:
            self.num_processes = num_processes
    
    def _token_to_integer(self, token):
        """
        Convert token to integer ID with automatic registration of new tokens
        
        Parameters:
        -----------
        token : str
            Token string (e.g., 'F250' for fixation with duration 250ms)
            
        Returns:
        --------
        int
            Integer ID for token
        """
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_token_id
            self.id_to_token[self.next_token_id] = token
            self.next_token_id += 1
        
        return self.token_to_id[token]
    
    def _bin_duration(self, token):
        """
        Extract token type and bin duration
        
        Parameters:
        -----------
        token : str
            Token string (e.g., 'F250' for fixation with duration 250ms)
            
        Returns:
        --------
        str
            New token with binned duration (e.g., 'F3' for bin 3)
        """
        if not self.use_duration_bins:
            return token
        
        token_type = token[0]
        try:
            duration = int(token[1:]) if len(token) > 1 and token[1:].isdigit() else 0
            
            # Simple log-scale binning
            if duration == 0:
                bin_idx = 0
            else:
                # Apply log binning with clip to max bin
                bin_idx = min(int(np.log2(duration + 1)), self.duration_bins - 1)
                
            return f"{token_type}{bin_idx}"
        except (ValueError, IndexError):
            # If there's an error, return the original token
            return token
    
    def _encode_scanpath(self, scanpath):
        """
        Encode scanpath tokens as integer IDs
        
        Parameters:
        -----------
        scanpath : list
            List of token strings
            
        Returns:
        --------
        list
            List of integer IDs
        """
        # First bin durations if enabled
        if self.use_duration_bins:
            scanpath = [self._bin_duration(token) for token in scanpath]
        
        # Then convert to integer IDs
        return [self._token_to_integer(token) for token in scanpath]
    
    def _initialize_bit_packing(self):
        """Initialize bit-packing parameters based on alphabet size"""
        alphabet_size = len(self.token_to_id)
        self.bits_per_token = max(1, (alphabet_size - 1).bit_length())
        self.tokens_per_word = 64 // self.bits_per_token
        
        print(f"Alphabet size: {alphabet_size}")
        print(f"Bits per token: {self.bits_per_token}")
        print(f"Tokens per 64-bit word: {self.tokens_per_word}")
        
        # Initialize rolling hash parameters
        self.base = 31  # A prime number for rolling hash
        self.pow_base = [1]
        for i in range(1, max(self.seed_length, 100)):
            self.pow_base.append((self.pow_base[-1] * self.base) & MASK64)
    
    def _pack_ids_into_words(self, id_list):
        """
        Pack integer IDs into 64-bit words
        
        Parameters:
        -----------
        id_list : list
            List of integer IDs
            
        Returns:
        --------
        list
            List of packed 64-bit words
        """
        if not id_list:
            return []
        
        # Calculate how many tokens fit in one word
        tokens_per_word = self.tokens_per_word
        bits_per_token = self.bits_per_token
        
        # Calculate number of words needed
        num_words = (len(id_list) + tokens_per_word - 1) // tokens_per_word
        
        # Pack tokens into words
        packed_words = []
        for i in range(num_words):
            word = 0
            for j in range(tokens_per_word):
                idx = i * tokens_per_word + j
                if idx < len(id_list):
                    word |= (id_list[idx] & ((1 << bits_per_token) - 1)) << (j * bits_per_token)
            packed_words.append(word)
        
        return packed_words
    
    def _unpack_word(self, word, num_tokens=None):
        """
        Unpack a 64-bit word back into integer IDs
        
        Parameters:
        -----------
        word : int
            Packed 64-bit word
        num_tokens : int or None
            Number of tokens to unpack (or all if None)
            
        Returns:
        --------
        list
            List of unpacked integer IDs
        """
        if num_tokens is None:
            num_tokens = self.tokens_per_word
        
        bits_per_token = self.bits_per_token
        mask = (1 << bits_per_token) - 1
        
        # Unpack word
        ids = []
        for i in range(num_tokens):
            token_id = (word >> (i * bits_per_token)) & mask
            ids.append(token_id)
        
        return ids
    
    def _calculate_rolling_hash(self, id_list):
        """
        Calculate rolling 64-bit hash for all k-mers in the sequence
        
        Parameters:
        -----------
        id_list : list
            List of integer IDs
            
        Returns:
        --------
        list
            List of hash values for each k-mer
        """
        if len(id_list) < self.seed_length:
            return []
        
        k = self.seed_length
        base = self.base
        pow_base = self.pow_base
        
        # Initialize hash for first k-mer
        h = 0
        for i in range(k):
            h = (h * base + id_list[i]) & MASK64
        
        hashes = [h]
        
        # Calculate rolling hash for remaining k-mers
        for i in range(k, len(id_list)):
            # Remove contribution of the oldest character
            h = (h - id_list[i-k] * pow_base[k-1]) & MASK64
            # Multiply by base
            h = (h * base) & MASK64
            # Add contribution of the new character
            h = (h + id_list[i]) & MASK64
            hashes.append(h)
        
        return hashes
    
    def _initialize_simhash(self):
        """Initialize random bitmasks for SimHash"""
        # Create random bitmasks for each token ID
        alphabet_size = len(self.token_to_id)
        self.id_bitmasks = []
        
        # Use a fixed seed for reproducibility
        rng = random.Random(42)
        
        for _ in range(alphabet_size):
            # Generate a random 64-bit mask
            mask = 0
            for j in range(self.lsh_hash_size):
                if rng.random() < 0.5:
                    mask |= (1 << j)
            self.id_bitmasks.append(mask)
    
    def _compute_simhash(self, id_list):
        """
        Compute SimHash sketch for a scanpath
        
        Parameters:
        -----------
        id_list : list
            List of integer IDs
            
        Returns:
        --------
        int
            64-bit SimHash sketch
        """
        if not self.id_bitmasks:
            self._initialize_simhash()
        
        # Initialize counters for each bit position
        counters = [0] * self.lsh_hash_size
        
        # Accumulate counts
        for token_id in id_list:
            if 0 <= token_id < len(self.id_bitmasks):
                bitmask = self.id_bitmasks[token_id]
                for i in range(self.lsh_hash_size):
                    if bitmask & (1 << i):
                        counters[i] += 1
                    else:
                        counters[i] -= 1
        
        # Generate fingerprint based on counters
        fingerprint = 0
        for i in range(self.lsh_hash_size):
            if counters[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def _build_lsh_index(self, simhash_sketches):
        """
        Build LSH index with multiple hash tables
        
        Parameters:
        -----------
        simhash_sketches : list
            List of SimHash sketches
            
        Returns:
        --------
        list
            List of hash tables (dicts)
        """
        num_tables = self.num_hash_tables
        bits_per_table = self.num_hash_bits
        
        # Initialize hash tables
        lsh_tables = [{} for _ in range(num_tables)]
        
        # Compute bit ranges for each table
        bits_per_slice = self.lsh_hash_size // num_tables
        bit_ranges = [(i * bits_per_slice, (i + 1) * bits_per_slice) for i in range(num_tables)]
        
        # Add scanpaths to hash tables
        for idx, sketch in enumerate(simhash_sketches):
            for table_idx, (start_bit, end_bit) in enumerate(bit_ranges):
                # Extract bit slice
                bit_slice = (sketch >> start_bit) & ((1 << (end_bit - start_bit)) - 1)
                
                # Add to hash table
                if bit_slice not in lsh_tables[table_idx]:
                    lsh_tables[table_idx][bit_slice] = []
                lsh_tables[table_idx][bit_slice].append(idx)
        
        return lsh_tables
    
    def _query_lsh_index(self, query_sketch):
        """
        Query LSH index for candidate scanpaths
        
        Parameters:
        -----------
        query_sketch : int
            SimHash sketch of query scanpath
            
        Returns:
        --------
        list
            List of candidate indices
        """
        candidates = set()
        
        # Compute bit ranges for each table
        bits_per_slice = self.lsh_hash_size // self.num_hash_tables
        bit_ranges = [(i * bits_per_slice, (i + 1) * bits_per_slice) for i in range(self.num_hash_tables)]
        
        # Query each hash table
        for table_idx, (start_bit, end_bit) in enumerate(bit_ranges):
            # Extract bit slice
            bit_slice = (query_sketch >> start_bit) & ((1 << (end_bit - start_bit)) - 1)
            
            # Get candidates from hash table
            if bit_slice in self.lsh_tables[table_idx]:
                candidates.update(self.lsh_tables[table_idx][bit_slice])
        
        return list(candidates)
    
    def _hamming_distance(self, a, b):
        """
        Calculate Hamming distance between two integers
        
        Parameters:
        -----------
        a, b : int
            Integers to compare
            
        Returns:
        --------
        int
            Hamming distance (number of differing bits)
        """
        return bin(a ^ b).count('1')
    
    def _extract_seeds(self, packed1, packed2):
        """
        Extract seeds using bit-parallel popcount
        
        Parameters:
        -----------
        packed1, packed2 : list
            Lists of packed 64-bit words
            
        Returns:
        --------
        list
            List of (pos1, pos2) tuples for seed anchors
        """
        seeds = []
        max_mismatches = 1  # Allow up to 1 mismatch per word
        
        min_len = min(len(packed1), len(packed2))
        for i in range(min_len):
            # XOR words and count bits
            xor_result = packed1[i] ^ packed2[i]
            # Count mismatches (set bits)
            mismatches = bin(xor_result).count('1')
            
            if mismatches <= max_mismatches * self.bits_per_token:
                # This word is a seed anchor
                # Calculate starting position in tokens
                pos1 = i * self.tokens_per_word
                pos2 = i * self.tokens_per_word
                seeds.append((pos1, pos2))
        
        return seeds
    
    def _greedy_chain_seeds(self, seeds, len1, len2):
        """
        Chain seeds using greedy diagonal algorithm
        
        Parameters:
        -----------
        seeds : list
            List of (pos1, pos2) tuples for seed anchors
        len1, len2 : int
            Lengths of the two scanpaths
            
        Returns:
        --------
        list
            List of (pos1, pos2) tuples for aligned positions
        """
        if not seeds:
            return []
        
        # Calculate diagonal for each seed (pos2 - pos1)
        # Group seeds by diagonal
        diagonals = defaultdict(list)
        for pos1, pos2 in seeds:
            diag = pos2 - pos1
            diagonals[diag].append((pos1, pos2))
        
        # Find the diagonal with the most seeds
        best_diag, best_seeds = max(diagonals.items(), key=lambda x: len(x[1]))
        
        # Sort seeds on this diagonal by position
        best_seeds.sort()
        
        # Build chain greedily
        chain = [best_seeds[0]]
        for seed in best_seeds[1:]:
            # If this seed extends the chain, add it
            if seed[0] > chain[-1][0] and seed[1] > chain[-1][1]:
                chain.append(seed)
        
        return chain
    
    def _banded_levenshtein(self, seq1, seq2, band_width, chain):
        """
        Compute banded Levenshtein distance with bit-parallelism
        
        Parameters:
        -----------
        seq1, seq2 : list
            Lists of integer IDs
        band_width : int
            Width of band for dynamic programming
        chain : list
            List of (pos1, pos2) tuples for aligned positions
            
        Returns:
        --------
        int
            Edit distance
        list
            Aligned positions
        """
        if not chain:
            # No chain, fall back to SimHash distance
            return len(seq1) + len(seq2), []
        
        # Extract diagonal from chain
        diag = chain[0][1] - chain[0][0]
        
        # Adjust band width to be symmetric around diagonal
        half_band = band_width // 2
        
        # Initialize dynamic programming matrix
        m, n = len(seq1), len(seq2)
        
        # Use integer-only scoring
        match_score = 2
        mismatch_penalty = -1
        gap_penalty = -1
        
        # Preallocate DP matrix with band
        dp = np.zeros((m + 1, band_width), dtype=np.int32)
        
        # Initialize first row and column
        for i in range(band_width):
            dp[0, i] = i * gap_penalty
        
        for i in range(1, m + 1):
            # Calculate column range for this row
            col_start = max(1, i + diag - half_band)
            col_end = min(n + 1, i + diag + half_band + 1)
            band_start = 0
            
            # Initialize left edge of band if needed
            if col_start > 1:
                dp[i, 0] = dp[i-1, 0] + gap_penalty
            else:
                dp[i, 0] = i * gap_penalty
            
            # Fill the band for this row
            for j in range(col_start, col_end):
                band_j = j - col_start + band_start
                
                if band_j >= band_width:
                    break
                
                # Calculate scores
                if j <= n:
                    match = (seq1[i-1] == seq2[j-1])
                    diag_score = dp[i-1, band_j-1 if band_j > 0 else 0] + (match_score if match else mismatch_penalty)
                    up_score = dp[i-1, band_j] + gap_penalty
                    left_score = dp[i, band_j-1] if band_j > 0 else float('-inf')
                    
                    # Choose best score
                    dp[i, band_j] = max(diag_score, up_score, left_score)
        
        # Extract score from bottom-right cell
        score = dp[m, band_width-1]
        
        # Simple alignment reconstruction (just use the chain)
        alignment = chain
        
        return score, alignment
    
    def _similarity_score(self, seq1, seq2):
        """
        Calculate similarity score between two scanpaths
        
        Parameters:
        -----------
        seq1, seq2 : list
            Lists of integer IDs
            
        Returns:
        --------
        float
            Similarity score between 0 and 1
        """
        # Pack sequences into words
        packed1 = self._pack_ids_into_words(seq1)
        packed2 = self._pack_ids_into_words(seq2)
        
        # Extract seeds
        seeds = self._extract_seeds(packed1, packed2)
        
        # Chain seeds
        chain = self._greedy_chain_seeds(seeds, len(seq1), len(seq2))
        
        # Calculate banded Levenshtein distance
        score, _ = self._banded_levenshtein(seq1, seq2, self.band_width, chain)
        
        # Normalize score
        max_possible_score = max(len(seq1), len(seq2)) * 2  # Maximum possible score
        normalized_score = max(0, score) / max_possible_score
        
        return normalized_score
    
    def fit(self, scanpaths, labels):
        """
        Train the classifier
        
        Parameters:
        -----------
        scanpaths : list
            List of scanpath token lists
        labels : list
            List of class labels
            
        Returns:
        --------
        self
        """
        # Store labels
        self.labels = labels
        
        # Encode scanpaths
        print("Encoding scanpaths...")
        self.encoded_scanpaths = [self._encode_scanpath(sp) for sp in scanpaths]
        
        # Initialize bit-packing
        self._initialize_bit_packing()
        
        # Pack scanpaths into words
        print("Packing scanpaths into words...")
        self.packed_scanpaths = [self._pack_ids_into_words(sp) for sp in self.encoded_scanpaths]
        
        # Compute SimHash sketches
        print("Computing SimHash sketches...")
        self.simhash_sketches = [self._compute_simhash(sp) for sp in self.encoded_scanpaths]
        
        # Build LSH index
        print("Building LSH index...")
        self.lsh_tables = self._build_lsh_index(self.simhash_sketches)
        
        return self
    
    def _compute_pair(self, args):
        """
        Compute similarity for a pair of scanpaths.
        
        Parameters:
        -----------
        args : tuple
            (i, j, seq1, seq2) where seq1 and seq2 are token lists
            
        Returns:
        --------
        tuple
            (i, j, sim) where sim is the normalized similarity between [0,1]
        """
        i, j, seq1, seq2 = args
        
        # Encode both scanpaths using existing token_to_id mapping
        encoded1 = [self._token_to_integer(token) if token in self.token_to_id else self.token_to_id.setdefault(token, self.next_token_id) or self.id_to_token.setdefault(self.next_token_id, token) or self.next_token_id for token in seq1]  # ensure registration if new
        encoded2 = [self._token_to_integer(token) if token in self.token_to_id else self.token_to_id.setdefault(token, self.next_token_id) or self.id_to_token.setdefault(self.next_token_id, token) or self.next_token_id for token in seq2]
        
        # We should not expand token_to_id beyond initial; so instead:
        # For proper behavior, assume all tokens already registered during fit.
        # Thus simply:
        encoded1 = [self.token_to_id[token] for token in seq1]
        encoded2 = [self.token_to_id[token] for token in seq2]
        
        # Compute similarity score via the same pipeline
        sim = self._similarity_score(encoded1, encoded2)
        
        return i, j, sim

    def _process_batch(self, batch):
        """
        Process a batch of queries in parallel
        
        Parameters:
        -----------
        batch : tuple
            (start_idx, end_idx, test_scanpaths)
            
        Returns:
        --------
        list
            List of predicted labels
        """
        start_idx, end_idx, test_scanpaths = batch
        predictions = []
        
        for i in range(start_idx, end_idx):
            scanpath = test_scanpaths[i]
            
            # Encode scanpath
            encoded = self._encode_scanpath(scanpath)
            
            # Compute SimHash sketch
            sketch = self._compute_simhash(encoded)
            
            # Query LSH index for candidates
            candidates = self._query_lsh_index(sketch)
            
            # If not enough candidates, use random sampling
            if len(candidates) < self.max_candidates:
                # Add random indices that aren't already candidates
                all_indices = set(range(len(self.encoded_scanpaths)))
                available = list(all_indices - set(candidates))
                if available:
                    additional = min(self.max_candidates - len(candidates), len(available))
                    candidates.extend(random.sample(available, additional))
            
            # Sort candidates by Hamming distance
            candidates = sorted(candidates, 
                               key=lambda x: self._hamming_distance(sketch, self.simhash_sketches[x]))
            
            # Take top candidates
            top_candidates = candidates[:self.max_candidates]
            
            # Calculate full similarity for top candidates
            similarities = []
            for candidate_idx in top_candidates:
                sim = self._similarity_score(encoded, self.encoded_scanpaths[candidate_idx])
                similarities.append((candidate_idx, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Take top k
            top_k = similarities[:self.num_neighbors]
            
            # Get labels of neighbors
            neighbor_labels = [self.labels[idx] for idx, _ in top_k]
            
            # Majority vote (with weights based on similarity)
            votes = defaultdict(float)
            for (idx, sim), label in zip(top_k, neighbor_labels):
                votes[label] += sim
            
            # Get label with highest vote
            if votes:
                prediction = max(votes.items(), key=lambda x: x[1])[0]
            else:
                # Fallback: use most common label in training set
                prediction = Counter(self.labels).most_common(1)[0][0]
            
            predictions.append(prediction)
        
        return predictions
    
    def predict(self, scanpaths):
        """
        Predict labels for new scanpaths
        
        Parameters:
        -----------
        scanpaths : list
            List of scanpath token lists
            
        Returns:
        --------
        list
            Predicted labels
        """
        if self.use_parallel and len(scanpaths) > 10:
            # Parallel processing
            num_processes = min(self.num_processes, len(scanpaths))
            batch_size = len(scanpaths) // num_processes
            
            batches = []
            for i in range(num_processes):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size if i < num_processes - 1 else len(scanpaths)
                batches.append((start_idx, end_idx, scanpaths))
            
            # Process batches in parallel
            with mp.Pool(num_processes) as pool:
                results = pool.map(self._process_batch, batches)
            
            # Combine results
            predictions = []
            for batch_result in results:
                predictions.extend(batch_result)
            
            return predictions
        else:
            # Sequential processing
            return self._process_batch((0, len(scanpaths), scanpaths))


def load_scanpaths(input_dir, sample_percentage=100):
    """
    Load tokenized scanpaths from JSON files
    
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
    """
    files = glob.glob(os.path.join(input_dir, "*.json"))
    
    # Sample files if requested
    if sample_percentage < 100:
        num_files = int(len(files) * sample_percentage / 100)
        files = random.sample(files, num_files)
    
    scanpaths = []
    labels = []
    
    print(f"Loading {len(files)} scanpath files...")
    for file_path in files:
        filename = os.path.basename(file_path)
        # Parse filename: S_XXXX_SY_Z_TTT_tokens.json
        # XXXX: subject ID, Y: session, Z: task number, TTT: task code
        parts = filename.split('_')
        task_num = int(parts[3])
        
        with open(file_path, 'r') as f:
            scanpath = json.load(f)
            
        scanpaths.append(scanpath)
        labels.append(task_num)
    
    return scanpaths, labels


def evaluate_classifier(y_true, y_pred):
    """
    Evaluate classifier performance
    
    Parameters:
    -----------
    y_true : list
        True labels
    y_pred : list
        Predicted labels
        
    Returns:
    --------
    dict
        Performance metrics
    """
    # Calculate accuracy
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    
    # Calculate confusion matrix
    classes = sorted(set(y_true))
    cm = [[0 for _ in classes] for _ in classes]
    
    for true, pred in zip(y_true, y_pred):
        true_idx = classes.index(true)
        pred_idx = classes.index(pred)
        cm[true_idx][pred_idx] += 1
    
    # Calculate per-class metrics
    class_metrics = {}
    for cls_idx, cls in enumerate(classes):
        # True positives, false positives, false negatives
        tp = cm[cls_idx][cls_idx]
        fp = sum(cm[i][cls_idx] for i in range(len(classes)) if i != cls_idx)
        fn = sum(cm[cls_idx][i] for i in range(len(classes)) if i != cls_idx)
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(cm[cls_idx])
        }
    
    # Calculate macro average
    macro_precision = sum(m["precision"] for m in class_metrics.values()) / len(class_metrics)
    macro_recall = sum(m["recall"] for m in class_metrics.values()) / len(class_metrics)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if macro_precision + macro_recall > 0 else 0
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "class_metrics": class_metrics,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="FastEyeBLAST: High-performance scanpath classification"
    )
    parser.add_argument("--input_dir", default="GazeBaseVR_IVT",
                       help="Directory containing tokenized scanpath JSON files")
    parser.add_argument("--output_dir", default="FastEyeBLAST_Results",
                       help="Directory to save results")
    parser.add_argument("--sample_percentage", type=int, default=100,
                       help="Percentage of data to use (randomly sampled)")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Proportion of data to use for testing")
    parser.add_argument("--seed_length", type=int, default=DEFAULT_SEED_LENGTH,
                       help="Length of seeds for matching")
    parser.add_argument("--band_width", type=int, default=DEFAULT_BAND_WIDTH,
                       help="Width of band for banded dynamic programming")
    parser.add_argument("--num_neighbors", type=int, default=DEFAULT_NUM_NEIGHBORS,
                       help="Number of neighbors for k-NN classification")
    parser.add_argument("--max_candidates", type=int, default=DEFAULT_MAX_CANDIDATES,
                       help="Maximum number of candidates for similarity calculation")
    parser.add_argument("--use_duration_bins", action="store_true", default=True,
                       help="Whether to bin durations")
    parser.add_argument("--duration_bins", type=int, default=DEFAULT_DURATION_BINS,
                       help="Number of bins for duration quantization")
    parser.add_argument("--use_parallel", action="store_true", default=True,
                       help="Whether to use parallel processing")
    parser.add_argument("--num_processes", type=int, default=None,
                       help="Number of processes for parallel processing")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading scanpaths...")
    scanpaths, labels = load_scanpaths(args.input_dir, args.sample_percentage)
    print(f"Loaded {len(scanpaths)} scanpaths")
    
    # Split data into train and test sets
    test_size = args.test_size
    test_count = int(len(scanpaths) * test_size)
    
    # Create indices and shuffle
    indices = list(range(len(scanpaths)))
    random.shuffle(indices)
    
    # Split indices
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    # Create train/test split
    train_scanpaths = [scanpaths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_scanpaths = [scanpaths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    print(f"Training set: {len(train_scanpaths)} scanpaths")
    print(f"Test set: {len(test_scanpaths)} scanpaths")
    
    # Initialize classifier
    classifier = FastEyeBLAST(
        seed_length=args.seed_length,
        band_width=args.band_width,
        num_neighbors=args.num_neighbors,
        max_candidates=args.max_candidates,
        use_duration_bins=args.use_duration_bins,
        duration_bins=args.duration_bins,
        use_parallel=args.use_parallel,
        num_processes=args.num_processes
    )
    
    # Train classifier
    print("Training classifier...")
    start_time = time.time()
    classifier.fit(train_scanpaths, train_labels)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Make predictions
    print("Making predictions...")
    start_time = time.time()
    predictions = classifier.predict(test_scanpaths)
    predict_time = time.time() - start_time
    print(f"Prediction completed in {predict_time:.2f} seconds")
    
    # Evaluate performance
    print("Evaluating performance...")
    results = evaluate_classifier(test_labels, predictions)
    
    # Print results
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro-avg F1: {results['macro_avg']['f1']:.4f}")
    
    # Save results
    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "accuracy": results['accuracy'],
            "macro_avg": results['macro_avg'],
            "class_metrics": results['class_metrics'],
            "confusion_matrix": results['confusion_matrix'],
            "train_time": train_time,
            "predict_time": predict_time,
            "params": vars(args)
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Save predictions
    pred_file = os.path.join(args.output_dir, "predictions.json")
    with open(pred_file, 'w') as f:
        json.dump({
            "true": test_labels,
            "pred": predictions
        }, f)
    
    print(f"Predictions saved to {pred_file}")


if __name__ == "__main__":
    main()
