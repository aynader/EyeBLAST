"""
FastEyeBLAST: High-performance scanpath similarity calculation and classification
using bit-level operations, integer encodings, and efficient hashing techniques.

DEBUG VERSION: This implementation includes comprehensive debug output to show
exactly what happens at each stage of the algorithm.

This implementation focuses on pure Python optimizations without external dependencies,
achieving significant speedups compared to the original algorithm.
"""

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
import logging
from datetime import datetime

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

# Global logging setup
class DebugLogger:
    """Manages multiple log files for different stages"""
    
    def __init__(self, output_dir, debug_enabled=True):
        self.output_dir = output_dir
        self.debug_enabled = debug_enabled
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize loggers for each stage
        self.loggers = {}
        self.log_files = [
            "1_initialization_log",
            "2_data_loading_log", 
            "3_duration_binning_log",
            "4_token_encoding_log",
            "5_bit_packing_log",
            "6_rolling_hash_log",
            "7_simhash_log",
            "8_lsh_index_log",
            "9_seed_extraction_log",
            "10_seed_chaining_log",
            "11_banded_levenshtein_log",
            "12_similarity_calculation_log",
            "13_training_log",
            "14_prediction_log",
            "15_evaluation_log",
            "16_comprehensive"
        ]
        
        # Setup loggers
        for log_name in self.log_files:
            self._setup_logger(log_name)
    
    def _setup_logger(self, log_name):
        """Setup individual logger with file handler"""
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create file handler
        log_file = os.path.join(self.output_dir, f"{log_name}_{self.timestamp}.txt")
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Also add console handler for comprehensive log
        if log_name == "16_comprehensive":
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        self.loggers[log_name] = logger
    
    def log(self, stage, message):
        """Log message to specific stage and comprehensive log"""
        if not self.debug_enabled:
            return
            
        # Log to specific stage
        if stage in self.loggers:
            self.loggers[stage].info(message)
        
        # Also log to comprehensive log
        self.loggers["16_comprehensive"].info(f"[{stage}] {message}")
    
    def log_header(self, stage, title, level=1):
        """Log a formatted header"""
        separator = "=" * 80 if level == 1 else "-" * 60
        message = f"\n{separator}\nDEBUG: {title}\n{separator}"
        self.log(stage, message)
    
    def log_data(self, stage, title, data, max_items=50, max_tokens=50):
        """Log debug data with truncation"""
        message = f"\n--- {title} ---"
        
        if isinstance(data, (list, tuple)):
            message += f"\nType: {type(data).__name__}, Length: {len(data)}"
            for i, item in enumerate(data[:max_items]):
                if isinstance(item, (list, tuple)):
                    # Truncate sequences to max_tokens
                    truncated_item = item[:max_tokens]
                    if len(item) > max_tokens:
                        truncated_item_str = f"{truncated_item}... (showing {max_tokens}/{len(item)} tokens)"
                    else:
                        truncated_item_str = str(truncated_item)
                    message += f"\n  [{i}]: {truncated_item_str}"
                else:
                    message += f"\n  [{i}]: {item}"
            if len(data) > max_items:
                message += f"\n  ... and {len(data) - max_items} more items"
                
        elif isinstance(data, dict):
            message += f"\nType: {type(data).__name__}, Length: {len(data)}"
            for i, (key, value) in enumerate(list(data.items())[:max_items]):
                if isinstance(value, (list, tuple)) and len(value) > max_tokens:
                    value_str = f"{value[:max_tokens]}... (showing {max_tokens}/{len(value)} items)"
                else:
                    value_str = str(value)
                message += f"\n  {key}: {value_str}"
            if len(data) > max_items:
                message += f"\n  ... and {len(data) - max_items} more items"
                
        elif isinstance(data, np.ndarray):
            message += f"\nType: numpy array, Shape: {data.shape}, Dtype: {data.dtype}"
            message += f"\nFirst {min(max_items, len(data))} elements:"
            message += f"\n{data[:max_items]}"
        else:
            message += f"\nType: {type(data).__name__}, Value: {data}"
        
        self.log(stage, message)

# Global logger instance
debug_logger = None

def print_debug_header(title, level=1):
    """Print a formatted debug header"""
    if debug_logger:
        debug_logger.log_header("16_comprehensive", title, level)

def print_debug_data(title, data, max_items=50):
    """Print debug data with truncation"""
    if debug_logger:
        debug_logger.log_data("16_comprehensive", title, data, max_items)

class FastEyeBLAST:
    """
    Optimized implementation of EyeBLAST using bit-level operations,
    integer encodings, and efficient hashing techniques.
    
    DEBUG VERSION with comprehensive logging.
    """
    
    def __init__(self, seed_length=DEFAULT_SEED_LENGTH, band_width=DEFAULT_BAND_WIDTH,
                 num_neighbors=DEFAULT_NUM_NEIGHBORS, num_hash_tables=DEFAULT_NUM_HASH_TABLES,
                 num_hash_bits=DEFAULT_NUM_HASH_BITS, max_candidates=DEFAULT_MAX_CANDIDATES,
                 lsh_hash_size=DEFAULT_LSH_HASH_SIZE, use_duration_bins=True, 
                 duration_bins=DEFAULT_DURATION_BINS, use_parallel=True, num_processes=None,
                 debug=True):
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
        debug : bool
            Whether to enable debug output
        """
        self.debug = debug
        
        if self.debug:
            debug_logger.log_header("1_initialization_log", "INITIALIZING FASTEYEBLAST")
            debug_logger.log("1_initialization_log", f"Parameters:")
            debug_logger.log("1_initialization_log", f"  seed_length: {seed_length}")
            debug_logger.log("1_initialization_log", f"  band_width: {band_width}")
            debug_logger.log("1_initialization_log", f"  num_neighbors: {num_neighbors}")
            debug_logger.log("1_initialization_log", f"  num_hash_tables: {num_hash_tables}")
            debug_logger.log("1_initialization_log", f"  num_hash_bits: {num_hash_bits}")
            debug_logger.log("1_initialization_log", f"  max_candidates: {max_candidates}")
            debug_logger.log("1_initialization_log", f"  lsh_hash_size: {lsh_hash_size}")
            debug_logger.log("1_initialization_log", f"  use_duration_bins: {use_duration_bins}")
            debug_logger.log("1_initialization_log", f"  duration_bins: {duration_bins}")
            debug_logger.log("1_initialization_log", f"  use_parallel: {use_parallel}")
            debug_logger.log("1_initialization_log", f"  num_processes: {num_processes}")
        
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
            
            if self.debug and self.next_token_id < 50:  # Show first 50 for more detail
                debug_logger.log("4_token_encoding_log", f"    New token registered: '{token}' -> ID {self.next_token_id}")
            
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
            
            binned_token = f"{token_type}{bin_idx}"
            
            if self.debug and hasattr(self, '_binning_examples_shown') and self._binning_examples_shown < 50:
                debug_logger.log("3_duration_binning_log", f"    Duration binning: '{token}' (duration={duration}) -> '{binned_token}' (bin={bin_idx})")
                self._binning_examples_shown += 1
                
            return binned_token
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
        if self.debug and not hasattr(self, '_encoding_examples_shown'):
            self._encoding_examples_shown = 0
        
        # Track current scanpath for detailed logging
        if self.debug:
            self._current_scanpath_idx = getattr(self, '_current_scanpath_idx', 0)
            self._tokens_shown_this_scanpath = 0
        
        # First bin durations if enabled
        if self.use_duration_bins:
            if self.debug and self._encoding_examples_shown == 0:
                debug_logger.log_header("3_duration_binning_log", "DURATION BINNING EXAMPLES", level=2)
                debug_logger.log("3_duration_binning_log", f"Showing duration binning for first 50 scanpaths:")
                debug_logger.log("3_duration_binning_log", f"Log-scale binning: bin = min(log2(duration+1), {self.duration_bins-1})")
            
            if self.debug and self._current_scanpath_idx < 50:
                debug_logger.log("3_duration_binning_log", f"\n  Scanpath {self._current_scanpath_idx}:")
                debug_logger.log("3_duration_binning_log", f"    Original (first 50): {scanpath[:50]}")
            
            original_scanpath = scanpath.copy()  # Keep original for comparison
            scanpath = [self._bin_duration(token) for token in scanpath]
            
            if self.debug and self._current_scanpath_idx < 50:
                debug_logger.log("3_duration_binning_log", f"    Binned (first 50): {scanpath[:50]}")
                debug_logger.log("3_duration_binning_log", f"    Duration binning examples (first 10 tokens):")
                for i, (orig, binned) in enumerate(zip(original_scanpath[:10], scanpath[:10])):
                    if orig != binned:
                        # Extract duration and show calculation
                        try:
                            duration = int(orig[1:]) if len(orig) > 1 and orig[1:].isdigit() else 0
                            bin_idx = min(int(np.log2(duration + 1)), self.duration_bins - 1) if duration > 0 else 0
                            debug_logger.log("3_duration_binning_log", f"      '{orig}' (duration={duration}) -> '{binned}' (bin={bin_idx})")
                        except:
                            debug_logger.log("3_duration_binning_log", f"      '{orig}' -> '{binned}'")
        
        # Then convert to integer IDs
        if self.debug and self._encoding_examples_shown == 0:
            debug_logger.log_header("4_token_encoding_log", "TOKEN TO INTEGER ENCODING EXAMPLES", level=2)
            debug_logger.log("4_token_encoding_log", f"Converting tokens to integer IDs for first 50 scanpaths...")
            debug_logger.log("4_token_encoding_log", f"Each unique token gets assigned a sequential integer ID.")
            
        encoded = [self._token_to_integer(token) for token in scanpath]
        
        if self.debug and self._current_scanpath_idx < 50:
            debug_logger.log("4_token_encoding_log", f"\n  Scanpath {self._current_scanpath_idx}:")
            debug_logger.log("4_token_encoding_log", f"    Tokens (first 50): {scanpath[:50]}")
            debug_logger.log("4_token_encoding_log", f"    Encoded IDs (first 50): {encoded[:50]}")
            debug_logger.log("4_token_encoding_log", f"    Token-to-ID mappings (first 10 tokens):")
            for i, (token, token_id) in enumerate(zip(scanpath[:10], encoded[:10])):
                debug_logger.log("4_token_encoding_log", f"      '{token}' -> {token_id}")
        
        if self.debug:
            self._current_scanpath_idx += 1
            if self._current_scanpath_idx == 1:
                self._encoding_examples_shown = 1
        
        return encoded
    
    def _initialize_bit_packing(self):
        """Initialize bit-packing parameters based on alphabet size"""
        if self.debug:
            debug_logger.log_header("5_bit_packing_log", "INITIALIZING BIT-PACKING")
        
        alphabet_size = len(self.token_to_id)
        self.bits_per_token = max(1, (alphabet_size - 1).bit_length())
        self.tokens_per_word = 64 // self.bits_per_token
        
        if self.debug:
            debug_logger.log("5_bit_packing_log", f"Alphabet size: {alphabet_size}")
            debug_logger.log("5_bit_packing_log", f"Bits per token: {self.bits_per_token}")
            debug_logger.log("5_bit_packing_log", f"Tokens per 64-bit word: {self.tokens_per_word}")
            debug_logger.log("5_bit_packing_log", f"Complete token to ID mapping:")
            for token, token_id in list(self.token_to_id.items()):
                binary_repr = format(token_id, f'0{self.bits_per_token}b')
                debug_logger.log("5_bit_packing_log", f"  '{token}' -> {token_id} (binary: {binary_repr})")
        
        # Initialize rolling hash parameters
        self.base = 31  # A prime number for rolling hash
        self.pow_base = [1]
        for i in range(1, max(self.seed_length, 100)):
            self.pow_base.append((self.pow_base[-1] * self.base) & MASK64)
        
        if self.debug:
            debug_logger.log("6_rolling_hash_log", f"Rolling hash base: {self.base}")
            debug_logger.log("6_rolling_hash_log", f"Precomputed powers (first 20): {self.pow_base[:20]}")
    
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
        
        if self.debug and not hasattr(self, '_packing_examples_shown'):
            self._packing_examples_shown = 0
        
        # Calculate how many tokens fit in one word
        tokens_per_word = self.tokens_per_word
        bits_per_token = self.bits_per_token
        
        # Calculate number of words needed
        num_words = (len(id_list) + tokens_per_word - 1) // tokens_per_word
        
        if self.debug and self._packing_examples_shown == 0:
            debug_logger.log_header("5_bit_packing_log", "BIT-PACKING EXAMPLES", level=2)
            debug_logger.log("5_bit_packing_log", f"Showing bit-packing for first 50 scanpaths...")
        
        # Pack tokens into words
        packed_words = []
        for i in range(num_words):
            word = 0
            word_tokens = []
            for j in range(tokens_per_word):
                idx = i * tokens_per_word + j
                if idx < len(id_list):
                    token_id = id_list[idx] & ((1 << bits_per_token) - 1)
                    word |= token_id << (j * bits_per_token)
                    word_tokens.append(token_id)
            
            packed_words.append(word)
        
        if self.debug and self._packing_examples_shown < 50:
            debug_logger.log("5_bit_packing_log", f"Scanpath {self._packing_examples_shown}:")
            debug_logger.log("5_bit_packing_log", f"  Input IDs (first 50): {id_list[:50]}")
            debug_logger.log("5_bit_packing_log", f"  Packing {len(id_list)} token IDs into {num_words} 64-bit words")
            
            # Show up to 10 words
            words_to_show = min(10, num_words)
            for i in range(words_to_show):
                word = packed_words[i]
                # Unpack to show what tokens are in this word
                word_tokens = []
                for j in range(tokens_per_word):
                    idx = i * tokens_per_word + j
                    if idx < len(id_list):
                        word_tokens.append(id_list[idx])
                
                debug_logger.log("5_bit_packing_log", f"    Word {i}: tokens {word_tokens}")
                debug_logger.log("5_bit_packing_log", f"      Packed value: {word} (hex: 0x{word:016x})")
                debug_logger.log("5_bit_packing_log", f"      Binary: {format(word, '064b')}")
            
            if num_words > words_to_show:
                debug_logger.log("5_bit_packing_log", f"    ... and {num_words - words_to_show} more words")
            
            self._packing_examples_shown += 1
        
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
        
        if self.debug and not hasattr(self, '_rolling_hash_examples_shown'):
            self._rolling_hash_examples_shown = 0
        
        k = self.seed_length
        base = self.base
        pow_base = self.pow_base
        
        if self.debug and self._rolling_hash_examples_shown == 0:
            debug_logger.log_header("6_rolling_hash_log", "ROLLING HASH CALCULATION EXAMPLES", level=2)
            debug_logger.log("6_rolling_hash_log", f"Rolling hash computes hash values for all k-mers (subsequences of length {k}) in a sequence.")
            debug_logger.log("6_rolling_hash_log", f"This enables efficient sequence matching by comparing hash values instead of sequences.")
            debug_logger.log("6_rolling_hash_log", f"Base value: {base}, k-mer length: {k}")
            debug_logger.log("6_rolling_hash_log", f"Showing rolling hash calculation for first 50 scanpaths...")
        
        # Initialize hash for first k-mer
        h = 0
        first_kmer = id_list[:k]
        for i in range(k):
            h = (h * base + id_list[i]) & MASK64
        
        if self.debug and self._rolling_hash_examples_shown < 50:
            debug_logger.log("6_rolling_hash_log", f"\nScanpath {self._rolling_hash_examples_shown}:")
            debug_logger.log("6_rolling_hash_log", f"  Input sequence (first 20 IDs): {id_list[:20]}")
            debug_logger.log("6_rolling_hash_log", f"  First k-mer: {first_kmer}")
            debug_logger.log("6_rolling_hash_log", f"  Hash calculation for first k-mer:")
            temp_h = 0
            for i in range(k):
                temp_h = (temp_h * base + id_list[i]) & MASK64
                debug_logger.log("6_rolling_hash_log", f"    Step {i+1}: h = ({temp_h // base if i > 0 else 0} * {base} + {id_list[i]}) & MASK64 = {temp_h}")
            debug_logger.log("6_rolling_hash_log", f"  First hash: {h}")
        
        hashes = [h]
        
        # Calculate rolling hash for remaining k-mers
        for i in range(k, len(id_list)):
            # Remove contribution of the oldest character
            old_h = h
            h = (h - id_list[i-k] * pow_base[k-1]) & MASK64
            # Multiply by base
            h = (h * base) & MASK64
            # Add contribution of the new character
            h = (h + id_list[i]) & MASK64
            hashes.append(h)
            
            if self.debug and self._rolling_hash_examples_shown < 50 and i - k < 5:
                kmer = id_list[i-k+1:i+1]
                debug_logger.log("6_rolling_hash_log", f"  k-mer {i-k+1}: {kmer}")
                debug_logger.log("6_rolling_hash_log", f"    Remove old: {old_h} - {id_list[i-k]} * {pow_base[k-1]} = {(old_h - id_list[i-k] * pow_base[k-1]) & MASK64}")
                debug_logger.log("6_rolling_hash_log", f"    Shift: ({(old_h - id_list[i-k] * pow_base[k-1]) & MASK64} * {base}) & MASK64 = {((old_h - id_list[i-k] * pow_base[k-1]) * base) & MASK64}")
                debug_logger.log("6_rolling_hash_log", f"    Add new: {((old_h - id_list[i-k] * pow_base[k-1]) * base) & MASK64} + {id_list[i]} = {h}")
        
        if self.debug and self._rolling_hash_examples_shown < 50:
            debug_logger.log("6_rolling_hash_log", f"  Total k-mers: {len(hashes)}")
            debug_logger.log("6_rolling_hash_log", f"  All hashes (first 20): {hashes[:20]}")
            if len(hashes) > 20:
                debug_logger.log("6_rolling_hash_log", f"  ... and {len(hashes) - 20} more hashes")
            self._rolling_hash_examples_shown += 1
        
        return hashes
    
    def _initialize_simhash(self):
        """Initialize random bitmasks for SimHash"""
        if self.debug:
            debug_logger.log_header("7_simhash_log", "INITIALIZING SIMHASH")
        
        # Create random bitmasks for each token ID
        alphabet_size = len(self.token_to_id)
        self.id_bitmasks = []
        
        # Use a fixed seed for reproducibility
        rng = random.Random(42)
        
        if self.debug:
            debug_logger.log("7_simhash_log", f"Creating random bitmasks for {alphabet_size} tokens")
            debug_logger.log("7_simhash_log", f"Hash size: {self.lsh_hash_size} bits")
        
        for token_id in range(alphabet_size):
            # Generate a random 64-bit mask
            mask = 0
            for j in range(self.lsh_hash_size):
                if rng.random() < 0.5:
                    mask |= (1 << j)
            self.id_bitmasks.append(mask)
            
            if self.debug and token_id < 10:  # Show first 10 masks
                token = self.id_to_token[token_id]
                debug_logger.log("7_simhash_log", f"  Token '{token}' (ID {token_id}): mask = {mask} (hex: 0x{mask:016x})")
                debug_logger.log("7_simhash_log", f"    Binary: {format(mask, f'0{self.lsh_hash_size}b')}")
    
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
        
        if self.debug and not hasattr(self, '_simhash_examples_shown'):
            self._simhash_examples_shown = 0
        
        # Initialize counters for each bit position
        counters = [0] * self.lsh_hash_size
        
        if self.debug and self._simhash_examples_shown == 0:
            debug_logger.log_header("7_simhash_log", "SIMHASH CALCULATION EXAMPLES", level=2)
            debug_logger.log("7_simhash_log", f"SimHash creates a compact fingerprint of a sequence by:")
            debug_logger.log("7_simhash_log", f"1. Each token has a random {self.lsh_hash_size}-bit mask")
            debug_logger.log("7_simhash_log", f"2. For each bit position, count +1 if token's mask has 1, -1 if 0")
            debug_logger.log("7_simhash_log", f"3. Final fingerprint has 1 where count > 0, 0 otherwise")
            debug_logger.log("7_simhash_log", f"Showing SimHash calculation for first 50 scanpaths...")
        
        if self.debug and self._simhash_examples_shown < 50:
            debug_logger.log("7_simhash_log", f"\nScanpath {self._simhash_examples_shown}:")
            debug_logger.log("7_simhash_log", f"  Input sequence (first 50 IDs): {id_list[:50]}")
            debug_logger.log("7_simhash_log", f"  Hash size: {self.lsh_hash_size} bits")
            debug_logger.log("7_simhash_log", f"  Initial counters: all zeros")
        
        # Accumulate counts
        for idx, token_id in enumerate(id_list):
            if 0 <= token_id < len(self.id_bitmasks):
                bitmask = self.id_bitmasks[token_id]
                for i in range(self.lsh_hash_size):
                    if bitmask & (1 << i):
                        counters[i] += 1
                    else:
                        counters[i] -= 1
                
                if self.debug and self._simhash_examples_shown < 50 and idx < 10:
                    token = self.id_to_token[token_id]
                    debug_logger.log("7_simhash_log", f"    Token {idx}: '{token}' (ID {token_id})")
                    debug_logger.log("7_simhash_log", f"      Bitmask: {format(bitmask, f'0{self.lsh_hash_size}b')}")
                    debug_logger.log("7_simhash_log", f"      Counters after (first 10): {counters[:10]}")
        
        # Generate fingerprint based on counters
        fingerprint = 0
        for i in range(self.lsh_hash_size):
            if counters[i] > 0:
                fingerprint |= (1 << i)
        
        if self.debug and self._simhash_examples_shown < 50:
            debug_logger.log("7_simhash_log", f"  Final counters (first 10): {counters[:10]}")
            debug_logger.log("7_simhash_log", f"  Final fingerprint: {fingerprint} (hex: 0x{fingerprint:016x})")
            debug_logger.log("7_simhash_log", f"  Binary: {format(fingerprint, f'0{self.lsh_hash_size}b')}")
            self._simhash_examples_shown += 1
        
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
        if self.debug:
            debug_logger.log_header("8_lsh_index_log", "BUILDING LSH INDEX")
        
        num_tables = self.num_hash_tables
        bits_per_table = self.num_hash_bits
        
        # Initialize hash tables
        lsh_tables = [{} for _ in range(num_tables)]
        
        # Compute bit ranges for each table
        bits_per_slice = self.lsh_hash_size // num_tables
        bit_ranges = [(i * bits_per_slice, (i + 1) * bits_per_slice) for i in range(num_tables)]
        
        if self.debug:
            debug_logger.log("8_lsh_index_log", f"LSH Index Construction:")
            debug_logger.log("8_lsh_index_log", f"  Number of hash tables: {num_tables}")
            debug_logger.log("8_lsh_index_log", f"  Hash size: {self.lsh_hash_size} bits")
            debug_logger.log("8_lsh_index_log", f"  Bits per slice: {bits_per_slice}")
            debug_logger.log("8_lsh_index_log", f"  Bit ranges for tables: {bit_ranges}")
            debug_logger.log("8_lsh_index_log", f"  Number of sketches to index: {len(simhash_sketches)}")
        
        # Add scanpaths to hash tables
        for idx, sketch in enumerate(simhash_sketches):
            if self.debug and idx < 10:  # Show first 10 sketches
                debug_logger.log("8_lsh_index_log", f"\nIndexing sketch {idx}: {sketch} (hex: 0x{sketch:016x})")
                debug_logger.log("8_lsh_index_log", f"  Binary: {format(sketch, f'0{self.lsh_hash_size}b')}")
            
            for table_idx, (start_bit, end_bit) in enumerate(bit_ranges):
                # Extract bit slice
                bit_slice = (sketch >> start_bit) & ((1 << (end_bit - start_bit)) - 1)
                
                if self.debug and idx < 10:
                    slice_binary = format(bit_slice, f'0{end_bit - start_bit}b')
                    debug_logger.log("8_lsh_index_log", f"    Table {table_idx} (bits {start_bit}-{end_bit-1}): {bit_slice} (binary: {slice_binary})")
                
                # Add to hash table
                if bit_slice not in lsh_tables[table_idx]:
                    lsh_tables[table_idx][bit_slice] = []
                lsh_tables[table_idx][bit_slice].append(idx)
        
        if self.debug:
            debug_logger.log("8_lsh_index_log", f"\n{'='*80}")
            debug_logger.log("8_lsh_index_log", f"LSH INDEX STATISTICS")
            debug_logger.log("8_lsh_index_log", f"{'='*80}")
            
            for table_idx, table in enumerate(lsh_tables):
                debug_logger.log("8_lsh_index_log", f"\nTable {table_idx}:")
                debug_logger.log("8_lsh_index_log", f"  Total unique hash values: {len(table)}")
                bucket_sizes = [len(bucket) for bucket in table.values()]
                if bucket_sizes:
                    debug_logger.log("8_lsh_index_log", f"  Bucket sizes - Min: {min(bucket_sizes)}, Max: {max(bucket_sizes)}, Avg: {sum(bucket_sizes)/len(bucket_sizes):.2f}")
                
                # Create actual table view
                debug_logger.log("8_lsh_index_log", f"\n  Hash Table Contents (first 20 buckets):")
                debug_logger.log("8_lsh_index_log", f"  {'Hash Value':<12} | {'Binary':<16} | {'Count':<6} | {'Indices'}")
                debug_logger.log("8_lsh_index_log", f"  {'-'*12}-+-{'-'*16}-+-{'-'*6}-+-{'-'*40}")
                
                for i, (hash_val, indices) in enumerate(list(table.items())[:20]):
                    binary_repr = format(hash_val, f'0{bits_per_slice}b')
                    indices_str = str(indices[:10])  # Show first 10 indices
                    if len(indices) > 10:
                        indices_str = indices_str[:-1] + f"...+{len(indices)-10}]"
                    debug_logger.log("8_lsh_index_log", f"  {hash_val:<12} | {binary_repr:<16} | {len(indices):<6} | {indices_str}")
                
                if len(table) > 20:
                    debug_logger.log("8_lsh_index_log", f"  ... and {len(table) - 20} more hash values")
        
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
        if self.debug and not hasattr(self, '_lsh_query_examples_shown'):
            self._lsh_query_examples_shown = 0
        
        candidates = set()
        
        # Compute bit ranges for each table
        bits_per_slice = self.lsh_hash_size // self.num_hash_tables
        bit_ranges = [(i * bits_per_slice, (i + 1) * bits_per_slice) for i in range(self.num_hash_tables)]
        
        if self.debug and debug_logger is not None and self._lsh_query_examples_shown < 3:
            debug_logger.log_header("8_lsh_index_log", "LSH QUERY EXAMPLE", level=2)
            debug_logger.log("8_lsh_index_log", f"Query sketch: {query_sketch} (hex: 0x{query_sketch:016x})")
            debug_logger.log("8_lsh_index_log", f"Binary: {format(query_sketch, f'0{self.lsh_hash_size}b')}")
        
        # Query each hash table
        for table_idx, (start_bit, end_bit) in enumerate(bit_ranges):
            # Extract bit slice
            bit_slice = (query_sketch >> start_bit) & ((1 << (end_bit - start_bit)) - 1)
            
            # Get candidates from hash table
            table_candidates = []
            if bit_slice in self.lsh_tables[table_idx]:
                table_candidates = self.lsh_tables[table_idx][bit_slice]
                candidates.update(table_candidates)
            
            if self.debug and debug_logger is not None and self._lsh_query_examples_shown < 3:
                slice_binary = format(bit_slice, f'0{end_bit - start_bit}b')
                debug_logger.log("8_lsh_index_log", f"  Table {table_idx} (bits {start_bit}-{end_bit-1}):")
                debug_logger.log("8_lsh_index_log", f"    Bit slice: {bit_slice} (binary: {slice_binary})")
                debug_logger.log("8_lsh_index_log", f"    Candidates found: {len(table_candidates)}")
                if table_candidates:
                    debug_logger.log("8_lsh_index_log", f"    Candidate indices: {table_candidates[:10]}{'...' if len(table_candidates) > 10 else ''}")
        
        candidate_list = list(candidates)
        
        if self.debug and debug_logger is not None and self._lsh_query_examples_shown < 3:
            debug_logger.log("8_lsh_index_log", f"Total unique candidates: {len(candidate_list)}")
            debug_logger.log("8_lsh_index_log", f"Candidate indices: {candidate_list[:20]}{'...' if len(candidate_list) > 20 else ''}")
            self._lsh_query_examples_shown += 1
        
        return candidate_list
    
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
        if self.debug and not hasattr(self, '_seed_extraction_examples_shown'):
            self._seed_extraction_examples_shown = 0
        
        seeds = []
        max_mismatches = 15  # Allow up to 15 mismatch bits per word (was too restrictive at 1)
        
        if self.debug and self._seed_extraction_examples_shown < 50:
            if self._seed_extraction_examples_shown == 0:
                debug_logger.log_header("9_seed_extraction_log", "SEED EXTRACTION EXAMPLES", level=2)
                debug_logger.log("9_seed_extraction_log", f"Seed extraction finds matching regions between two sequences by:")
                debug_logger.log("9_seed_extraction_log", f"1. XORing corresponding packed words")
                debug_logger.log("9_seed_extraction_log", f"2. Counting mismatches (set bits in XOR result)")
                debug_logger.log("9_seed_extraction_log", f"3. Accepting words with ≤ {max_mismatches} mismatches per {self.bits_per_token}-bit token")
                debug_logger.log("9_seed_extraction_log", f"Showing seed extraction for first 50 similarity calculations...")
            
            debug_logger.log("9_seed_extraction_log", f"\nSimilarity calculation {self._seed_extraction_examples_shown}:")
            debug_logger.log("9_seed_extraction_log", f"  Packed sequence 1 (first 10 words): {packed1[:10]}")
            debug_logger.log("9_seed_extraction_log", f"  Packed sequence 2 (first 10 words): {packed2[:10]}")
            debug_logger.log("9_seed_extraction_log", f"  Max mismatches per word: {max_mismatches}")
            debug_logger.log("9_seed_extraction_log", f"  Bits per token: {self.bits_per_token}")
            debug_logger.log("9_seed_extraction_log", f"  Tokens per word: {self.tokens_per_word}")
        
        min_len = min(len(packed1), len(packed2))
        for i in range(min_len):
            # XOR words and count bits
            xor_result = packed1[i] ^ packed2[i]
            # Count mismatches (set bits)
            mismatches = bin(xor_result).count('1')
            
            if self.debug and self._seed_extraction_examples_shown < 50 and i < 10:
                debug_logger.log("9_seed_extraction_log", f"    Word {i}:")
                debug_logger.log("9_seed_extraction_log", f"      Word1: {packed1[i]} (hex: 0x{packed1[i]:016x})")
                debug_logger.log("9_seed_extraction_log", f"      Word2: {packed2[i]} (hex: 0x{packed2[i]:016x})")
                debug_logger.log("9_seed_extraction_log", f"      XOR:   {xor_result} (hex: 0x{xor_result:016x})")
                debug_logger.log("9_seed_extraction_log", f"      XOR binary: {format(xor_result, '064b')}")
                debug_logger.log("9_seed_extraction_log", f"      Mismatches: {mismatches}")
            
            if mismatches <= max_mismatches:
                # This word is a seed anchor
                # Calculate starting position in tokens
                pos1 = i * self.tokens_per_word
                pos2 = i * self.tokens_per_word
                seeds.append((pos1, pos2))
                
                if self.debug and self._seed_extraction_examples_shown < 50:
                    debug_logger.log("9_seed_extraction_log", f"      -> SEED FOUND at positions ({pos1}, {pos2})")
        
        if self.debug and self._seed_extraction_examples_shown < 50:
            debug_logger.log("9_seed_extraction_log", f"  Total seeds found: {len(seeds)}")
            debug_logger.log("9_seed_extraction_log", f"  Seed positions: {seeds}")
            self._seed_extraction_examples_shown += 1
        
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
        
        if self.debug and not hasattr(self, '_seed_chaining_examples_shown'):
            self._seed_chaining_examples_shown = 0
        
        if self.debug and self._seed_chaining_examples_shown < 50:
            if self._seed_chaining_examples_shown == 0:
                debug_logger.log_header("10_seed_chaining_log", "GREEDY SEED CHAINING EXAMPLES", level=2)
                debug_logger.log("10_seed_chaining_log", f"Seed chaining connects compatible seeds to form alignment chains:")
                debug_logger.log("10_seed_chaining_log", f"1. Group seeds by diagonal (pos2 - pos1)")
                debug_logger.log("10_seed_chaining_log", f"2. Find diagonal with most seeds")
                debug_logger.log("10_seed_chaining_log", f"3. Sort seeds on best diagonal by position")
                debug_logger.log("10_seed_chaining_log", f"4. Greedily chain seeds that extend monotonically")
                debug_logger.log("10_seed_chaining_log", f"Showing seed chaining for first 50 similarity calculations...")
            
            debug_logger.log("10_seed_chaining_log", f"\nSimilarity calculation {self._seed_chaining_examples_shown}:")
            debug_logger.log("10_seed_chaining_log", f"  Input seeds: {seeds}")
            debug_logger.log("10_seed_chaining_log", f"  Sequence lengths: {len1}, {len2}")
        
        # Calculate diagonal for each seed (pos2 - pos1)
        # Group seeds by diagonal
        diagonals = defaultdict(list)
        for pos1, pos2 in seeds:
            diag = pos2 - pos1
            diagonals[diag].append((pos1, pos2))
        
        if self.debug and self._seed_chaining_examples_shown < 50:
            debug_logger.log("10_seed_chaining_log", f"  Seeds grouped by diagonal:")
            for diag, diagonal_seeds in diagonals.items():
                debug_logger.log("10_seed_chaining_log", f"    Diagonal {diag}: {diagonal_seeds}")
        
        # Find the diagonal with the most seeds
        best_diag, best_seeds = max(diagonals.items(), key=lambda x: len(x[1]))
        
        if self.debug and self._seed_chaining_examples_shown < 50:
            debug_logger.log("10_seed_chaining_log", f"  Best diagonal: {best_diag} with {len(best_seeds)} seeds")
            debug_logger.log("10_seed_chaining_log", f"  Seeds on best diagonal: {best_seeds}")
        
        # Sort seeds on this diagonal by position
        best_seeds.sort()
        
        if self.debug and self._seed_chaining_examples_shown < 50:
            debug_logger.log("10_seed_chaining_log", f"  Seeds sorted by position: {best_seeds}")
        
        # Build chain greedily
        chain = [best_seeds[0]]
        for seed in best_seeds[1:]:
            # If this seed extends the chain, add it
            if seed[0] > chain[-1][0] and seed[1] > chain[-1][1]:
                chain.append(seed)
                if self.debug and self._seed_chaining_examples_shown < 50:
                    debug_logger.log("10_seed_chaining_log", f"    Added seed {seed} to chain")
            else:
                if self.debug and self._seed_chaining_examples_shown < 50:
                    debug_logger.log("10_seed_chaining_log", f"    Skipped seed {seed} (doesn't extend chain)")
        
        if self.debug and self._seed_chaining_examples_shown < 50:
            debug_logger.log("10_seed_chaining_log", f"  Final chain: {chain}")
            self._seed_chaining_examples_shown += 1
        
        return chain
    
    def _banded_levenshtein(self, seq1, seq2):
        """
        Compute banded Levenshtein distance for efficiency
        
        Parameters:
        -----------
        seq1, seq2 : list
            Sequences to compare
            
        Returns:
        --------
        int
            Edit distance
        """
        if self.debug and not hasattr(self, '_levenshtein_examples_shown'):
            self._levenshtein_examples_shown = 0
        
        m, n = len(seq1), len(seq2)
        
        if self.debug and debug_logger is not None and self._levenshtein_examples_shown < 3:
            debug_logger.log_header("11_banded_levenshtein_log", "BANDED LEVENSHTEIN EXAMPLE", level=2)
            debug_logger.log("11_banded_levenshtein_log", f"Sequence 1 length: {m}")
            debug_logger.log("11_banded_levenshtein_log", f"Sequence 2 length: {n}")
            debug_logger.log("11_banded_levenshtein_log", f"Band width: {self.band_width}")
            debug_logger.log("11_banded_levenshtein_log", f"Sequence 1 (first 10): {seq1[:10]}")
            debug_logger.log("11_banded_levenshtein_log", f"Sequence 2 (first 10): {seq2[:10]}")
        
        # Create DP table with band constraint
        band_size = min(self.band_width, max(m, n))
        dp = {}
        
        # Initialize
        for i in range(min(m + 1, band_size + 1)):
            dp[(i, 0)] = i
        for j in range(min(n + 1, band_size + 1)):
            dp[(0, j)] = j
        
        # Fill DP table within band
        for i in range(1, m + 1):
            start_j = max(1, i - band_size)
            end_j = min(n + 1, i + band_size + 1)
            
            for j in range(start_j, end_j):
                if seq1[i-1] == seq2[j-1]:
                    cost = 0
                else:
                    cost = 1
                
                candidates = []
                if (i-1, j) in dp:  # deletion
                    candidates.append(dp[(i-1, j)] + 1)
                if (i, j-1) in dp:  # insertion
                    candidates.append(dp[(i, j-1)] + 1)
                if (i-1, j-1) in dp:  # substitution
                    candidates.append(dp[(i-1, j-1)] + cost)
                
                if candidates:
                    dp[(i, j)] = min(candidates)
                else:
                    # If no valid predecessor, use a large value
                    dp[(i, j)] = max(m, n)
        
        # Get final distance
        distance = dp.get((m, n), max(m, n))
        
        if self.debug and debug_logger is not None and self._levenshtein_examples_shown < 3:
            debug_logger.log("11_banded_levenshtein_log", f"DP table size: {len(dp)} entries")
            debug_logger.log("11_banded_levenshtein_log", f"Final distance: {distance}")
            
            # Show a small portion of the DP table
            debug_logger.log("11_banded_levenshtein_log", "DP matrix (first 10x10):")
            for i in range(min(10, m + 1)):
                row = []
                for j in range(min(10, n + 1)):
                    if (i, j) in dp:
                        row.append(f"{dp[(i, j)]:3d}")
                    else:
                        row.append("  -")
                debug_logger.log("11_banded_levenshtein_log", f"  Row {i:2d}: " + " ".join(row))
            
            self._levenshtein_examples_shown += 1
        
        return distance
    
    def _similarity_score(self, scanpath1, scanpath2):
        """
        Calculate similarity score between two scanpaths using Banded Levenshtein distance
        
        Parameters:
        -----------
        scanpath1 : list
            First scanpath (encoded as integers)
        scanpath2 : list
            Second scanpath (encoded as integers)
            
        Returns:
        --------
        float
            Similarity score (0 to 1, higher is more similar)
        """
        if self.debug and not hasattr(self, '_similarity_examples_shown'):
            self._similarity_examples_shown = 0
        
        if self.debug and debug_logger is not None and self._similarity_examples_shown < 3:
            debug_logger.log_header("12_similarity_calculation_log", "SIMILARITY CALCULATION EXAMPLE", level=2)
            debug_logger.log("12_similarity_calculation_log", f"Scanpath 1 length: {len(scanpath1)}")
            debug_logger.log("12_similarity_calculation_log", f"Scanpath 2 length: {len(scanpath2)}")
            debug_logger.log("12_similarity_calculation_log", f"Scanpath 1 (first 20): {scanpath1[:20]}")
            debug_logger.log("12_similarity_calculation_log", f"Scanpath 2 (first 20): {scanpath2[:20]}")
        
        # Use Banded Levenshtein distance for efficiency
        distance = self._banded_levenshtein(scanpath1, scanpath2)
        max_len = max(len(scanpath1), len(scanpath2))
        
        if max_len == 0:
            similarity = 1.0
        else:
            similarity = 1.0 - (distance / max_len)
        
        if self.debug and debug_logger is not None and self._similarity_examples_shown < 3:
            debug_logger.log("12_similarity_calculation_log", f"Levenshtein distance: {distance}")
            debug_logger.log("12_similarity_calculation_log", f"Maximum length: {max_len}")
            debug_logger.log("12_similarity_calculation_log", f"Similarity score: {similarity:.4f}")
            self._similarity_examples_shown += 1
        
        return similarity

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
        if self.debug:
            debug_logger.log_header("13_training_log", "TRAINING FASTEYEBLAST CLASSIFIER")
            debug_logger.log("13_training_log", f"Number of training scanpaths: {len(scanpaths)}")
            debug_logger.log("13_training_log", f"Number of labels: {len(labels)}")
            debug_logger.log("13_training_log", f"Unique labels: {sorted(set(labels))}")
            debug_logger.log("13_training_log", f"Label distribution: {Counter(labels)}")
            
            # Show detailed scanpath examples (first 50 scanpaths, first 50 tokens each)
            debug_logger.log("13_training_log", f"\nDetailed scanpath examples (first 50 scanpaths):")
            for i in range(min(50, len(scanpaths))):
                debug_logger.log("13_training_log", f"  Scanpath {i} (label {labels[i]}):")
                debug_logger.log("13_training_log", f"    Length: {len(scanpaths[i])}")
                debug_logger.log("13_training_log", f"    Tokens (first 50): {scanpaths[i][:50]}")
                if len(scanpaths[i]) > 50:
                    debug_logger.log("13_training_log", f"    ... and {len(scanpaths[i]) - 50} more tokens")
        
        # Store labels
        self.labels = labels
        
        # Encode scanpaths
        debug_logger.log("13_training_log", "\nEncoding scanpaths...")
        self.encoded_scanpaths = []
        for i, sp in enumerate(scanpaths):
            encoded_sp = self._encode_scanpath(sp)
            self.encoded_scanpaths.append(encoded_sp)
            
            if self.debug and i < 50:
                debug_logger.log("13_training_log", f"  Scanpath {i}: {len(sp)} tokens -> {len(encoded_sp)} encoded IDs")
        
        if self.debug:
            debug_logger.log("13_training_log", f"\nToken vocabulary summary:")
            debug_logger.log("13_training_log", f"  Total unique tokens: {len(self.token_to_id)}")
            debug_logger.log("13_training_log", f"  Complete token-to-ID mapping:")
            for token, token_id in sorted(self.token_to_id.items(), key=lambda x: x[1]):
                debug_logger.log("13_training_log", f"    '{token}' -> {token_id}")
        
        # Initialize bit-packing
        self._initialize_bit_packing()
        
        # Pack scanpaths into words
        debug_logger.log("13_training_log", "\nPacking scanpaths into words...")
        self.packed_scanpaths = []
        for i, sp in enumerate(self.encoded_scanpaths):
            packed_sp = self._pack_ids_into_words(sp)
            self.packed_scanpaths.append(packed_sp)
            
            if self.debug and i < 50:
                debug_logger.log("13_training_log", f"  Scanpath {i}: {len(sp)} IDs -> {len(packed_sp)} packed words")
        
        # Compute SimHash sketches
        debug_logger.log("13_training_log", "\nComputing SimHash sketches...")
        self.simhash_sketches = []
        for i, sp in enumerate(self.encoded_scanpaths):
            sketch = self._compute_simhash(sp)
            self.simhash_sketches.append(sketch)
            
            if self.debug and i < 50:
                debug_logger.log("13_training_log", f"  Scanpath {i}: sketch = {sketch} (hex: 0x{sketch:016x})")
        
        if self.debug:
            debug_logger.log("13_training_log", f"\nSimHash sketches summary:")
            debug_logger.log("13_training_log", f"  Total sketches computed: {len(self.simhash_sketches)}")
            debug_logger.log("13_training_log", f"  First 20 sketches: {self.simhash_sketches[:20]}")
            if len(self.simhash_sketches) > 20:
                debug_logger.log("13_training_log", f"  ... and {len(self.simhash_sketches) - 20} more sketches")
        
        # Build LSH index
        debug_logger.log("13_training_log", "\nBuilding LSH index...")
        self.lsh_tables = self._build_lsh_index(self.simhash_sketches)
        
        if self.debug:
            debug_logger.log_header("13_training_log", "TRAINING COMPLETED")
            debug_logger.log("13_training_log", f"Final training statistics:")
            debug_logger.log("13_training_log", f"  Vocabulary size: {len(self.token_to_id)}")
            debug_logger.log("13_training_log", f"  Encoded scanpaths: {len(self.encoded_scanpaths)}")
            debug_logger.log("13_training_log", f"  Packed scanpaths: {len(self.packed_scanpaths)}")
            debug_logger.log("13_training_log", f"  SimHash sketches: {len(self.simhash_sketches)}")
            debug_logger.log("13_training_log", f"  LSH tables: {len(self.lsh_tables)}")
        
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
        
        # Disable debug logging in worker processes to avoid global variable issues
        # Check if we're in the main process (debug_logger will be None in workers)
        local_debug = self.debug and debug_logger is not None and hasattr(self, '_is_main_process')
        
        if local_debug:
            debug_logger.log_header("14_prediction_log", f"PROCESSING BATCH {start_idx}-{end_idx}", level=2)
        
        for i in range(start_idx, end_idx):
            scanpath = test_scanpaths[i]
            
            if local_debug and i - start_idx < 3:
                debug_logger.log("14_prediction_log", f"\nProcessing query {i}:")
                debug_logger.log("14_prediction_log", f"  Query scanpath length: {len(scanpath)}")
                debug_logger.log("14_prediction_log", f"  Query tokens (first 10): {scanpath[:10]}")
            
            # Encode scanpath (with debug disabled for workers)
            old_debug = self.debug
            if debug_logger is None:
                self.debug = False
            
            encoded = self._encode_scanpath(scanpath)
            
            # Restore debug state
            self.debug = old_debug
            
            # Compute SimHash sketch
            sketch = self._compute_simhash(encoded)
            
            if local_debug and i - start_idx < 3:
                debug_logger.log("14_prediction_log", f"  Query sketch: {sketch} (hex: 0x{sketch:016x})")
            
            # Query LSH index for candidates
            candidates = self._query_lsh_index(sketch)
            
            # If not enough candidates, use random sampling
            if len(candidates) < self.max_candidates:
                # Add random indices that aren't already candidates
                all_indices = set(range(len(self.encoded_scanpaths)))
                available = list(all_indices - set(candidates))
                if available:
                    additional = min(self.max_candidates - len(candidates), len(available))
                    additional_candidates = random.sample(available, additional)
                    candidates.extend(additional_candidates)
                    
                    if local_debug and i - start_idx < 3:
                        debug_logger.log("14_prediction_log", f"  Added {len(additional_candidates)} random candidates")
            
            # Sort candidates by Hamming distance
            candidates_with_distances = []
            for candidate_idx in candidates:
                hamming_dist = self._hamming_distance(sketch, self.simhash_sketches[candidate_idx])
                candidates_with_distances.append((candidate_idx, hamming_dist))
            
            candidates_with_distances.sort(key=lambda x: x[1])
            
            if local_debug and i - start_idx < 3:
                debug_logger.log("14_prediction_log", f"  Candidates sorted by Hamming distance (first 10):")
                for j, (idx, dist) in enumerate(candidates_with_distances[:10]):
                    debug_logger.log("14_prediction_log", f"    {j}: candidate {idx}, Hamming distance {dist}")
            
            # Take top candidates
            top_candidates = [idx for idx, _ in candidates_with_distances[:self.max_candidates]]
            
            # Calculate full similarity for top candidates
            similarities = []
            for candidate_idx in top_candidates:
                sim = self._similarity_score(encoded, self.encoded_scanpaths[candidate_idx])
                similarities.append((candidate_idx, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            if local_debug and i - start_idx < 3:
                debug_logger.log("14_prediction_log", f"  Top similarities:")
                for j, (idx, sim) in enumerate(similarities[:min(10, len(similarities))]):
                    label = self.labels[idx]
                    debug_logger.log("14_prediction_log", f"    {j}: candidate {idx} (label {label}), similarity {sim:.4f}")
            
            # Take top k
            top_k = similarities[:self.num_neighbors]
            
            # Get labels of neighbors
            neighbor_labels = [self.labels[idx] for idx, _ in top_k]
            
            # Majority vote (with weights based on similarity)
            votes = defaultdict(float)
            for (idx, sim), label in zip(top_k, neighbor_labels):
                votes[label] += sim
            
            if local_debug and i - start_idx < 3:
                debug_logger.log("14_prediction_log", f"  Voting results:")
                for label, vote_weight in sorted(votes.items(), key=lambda x: x[1], reverse=True):
                    debug_logger.log("14_prediction_log", f"    Label {label}: {vote_weight:.4f}")
            
            # Get label with highest vote
            if votes:
                prediction = max(votes.items(), key=lambda x: x[1])[0]
            else:
                # Fallback: use most common label in training set
                prediction = Counter(self.labels).most_common(1)[0][0]
                if local_debug and i - start_idx < 3:
                    debug_logger.log("14_prediction_log", f"  Using fallback prediction: {prediction}")
            
            if local_debug and i - start_idx < 3:
                debug_logger.log("14_prediction_log", f"  Final prediction: {prediction}")
            
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
        if self.debug:
            debug_logger.log_header("14_prediction_log", "MAKING PREDICTIONS")
            debug_logger.log("14_prediction_log", f"Number of test scanpaths: {len(scanpaths)}")
            debug_logger.log("14_prediction_log", f"Using parallel processing: {self.use_parallel}")
            debug_logger.log("14_prediction_log", f"Number of processes: {self.num_processes}")
        
        # Mark this instance as main process for debug logging
        self._is_main_process = True
        
        # If debug is enabled, process a few examples in the main process first
        if self.debug and len(scanpaths) > 0:
            debug_logger.log("14_prediction_log", f"\nProcessing first 3 examples in main process for detailed debugging...")
            
            for i in range(min(3, len(scanpaths))):
                scanpath = scanpaths[i]
                
                debug_logger.log("14_prediction_log", f"\n=== DETAILED PREDICTION EXAMPLE {i+1} ===")
                debug_logger.log("14_prediction_log", f"Query scanpath length: {len(scanpath)}")
                debug_logger.log("14_prediction_log", f"Query tokens (first 20): {scanpath[:20]}")
                
                # Encode scanpath
                encoded = self._encode_scanpath(scanpath)
                debug_logger.log("14_prediction_log", f"Encoded IDs (first 20): {encoded[:20]}")
                
                # Compute SimHash sketch
                sketch = self._compute_simhash(encoded)
                debug_logger.log("14_prediction_log", f"Query sketch: {sketch} (hex: 0x{sketch:016x})")
                
                # Query LSH index for candidates
                candidates = self._query_lsh_index(sketch)
                debug_logger.log("14_prediction_log", f"LSH candidates found: {len(candidates)}")
                debug_logger.log("14_prediction_log", f"Candidate indices: {candidates[:10]}{'...' if len(candidates) > 10 else ''}")
                
                # If not enough candidates, use random sampling
                if len(candidates) < self.max_candidates:
                    all_indices = set(range(len(self.encoded_scanpaths)))
                    available = list(all_indices - set(candidates))
                    if available:
                        additional = min(self.max_candidates - len(candidates), len(available))
                        additional_candidates = random.sample(available, additional)
                        candidates.extend(additional_candidates)
                        debug_logger.log("14_prediction_log", f"Added {len(additional_candidates)} random candidates")
                
                # Sort candidates by Hamming distance
                candidates_with_distances = []
                for candidate_idx in candidates:
                    hamming_dist = self._hamming_distance(sketch, self.simhash_sketches[candidate_idx])
                    candidates_with_distances.append((candidate_idx, hamming_dist))
                
                candidates_with_distances.sort(key=lambda x: x[1])
                debug_logger.log("14_prediction_log", f"Candidates sorted by Hamming distance (first 5):")
                for j, (idx, dist) in enumerate(candidates_with_distances[:5]):
                    debug_logger.log("14_prediction_log", f"  {j}: candidate {idx}, Hamming distance {dist}")
                
                # Take top candidates
                top_candidates = [idx for idx, _ in candidates_with_distances[:self.max_candidates]]
                
                # Calculate full similarity for top candidates (this will call the missing methods)
                debug_logger.log("14_prediction_log", f"\nCalculating detailed similarities for top {min(3, len(top_candidates))} candidates:")
                similarities = []
                for j, candidate_idx in enumerate(top_candidates[:3]):  # Show details for first 3
                    debug_logger.log("14_prediction_log", f"\n--- Similarity calculation {j+1}: Query vs Candidate {candidate_idx} ---")
                    
                    candidate_seq = self.encoded_scanpaths[candidate_idx]
                    debug_logger.log("14_prediction_log", f"Candidate sequence length: {len(candidate_seq)}")
                    debug_logger.log("14_prediction_log", f"Candidate IDs (first 20): {candidate_seq[:20]}")
                    
                    # Pack both sequences to show seed extraction
                    packed_query = self._pack_ids_into_words(encoded)
                    packed_candidate = self._pack_ids_into_words(candidate_seq)
                    
                    # Extract seeds (this will populate seed extraction log)
                    seeds = self._extract_seeds(packed_query, packed_candidate)
                    
                    # Chain seeds (this will populate seed chaining log)
                    chain = self._greedy_chain_seeds(seeds, len(encoded), len(candidate_seq))
                    
                    # Calculate similarity (this will populate similarity and Levenshtein logs)
                    sim = self._similarity_score(encoded, candidate_seq)
                    similarities.append((candidate_idx, sim))
                    
                    debug_logger.log("14_prediction_log", f"Final similarity: {sim:.4f}")
                
                # Calculate similarity for remaining candidates without debug output
                temp_debug = self.debug
                self.debug = False  # Temporarily disable debug for remaining candidates
                for candidate_idx in top_candidates[3:]:
                    sim = self._similarity_score(encoded, self.encoded_scanpaths[candidate_idx])
                    similarities.append((candidate_idx, sim))
                self.debug = temp_debug
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                debug_logger.log("14_prediction_log", f"\nTop similarities:")
                for j, (idx, sim) in enumerate(similarities[:min(5, len(similarities))]):
                    label = self.labels[idx]
                    debug_logger.log("14_prediction_log", f"  {j}: candidate {idx} (label {label}), similarity {sim:.4f}")
                
                # Take top k
                top_k = similarities[:self.num_neighbors]
                
                # Get labels of neighbors
                neighbor_labels = [self.labels[idx] for idx, _ in top_k]
                
                # Majority vote (with weights based on similarity)
                votes = defaultdict(float)
                for (idx, sim), label in zip(top_k, neighbor_labels):
                    votes[label] += sim
                
                debug_logger.log("14_prediction_log", f"\nVoting results:")
                for label, vote_weight in sorted(votes.items(), key=lambda x: x[1], reverse=True):
                    debug_logger.log("14_prediction_log", f"  Label {label}: {vote_weight:.4f}")
                
                # Get label with highest vote
                if votes:
                    prediction = max(votes.items(), key=lambda x: x[1])[0]
                else:
                    prediction = Counter(self.labels).most_common(1)[0][0]
                
                debug_logger.log("14_prediction_log", f"Final prediction: {prediction}")
        
        if self.use_parallel and len(scanpaths) > 10:
            # Parallel processing for the actual predictions
            num_processes = min(self.num_processes, len(scanpaths))
            batch_size = len(scanpaths) // num_processes
            
            if self.debug:
                debug_logger.log("14_prediction_log", f"\nSwitching to parallel processing for all {len(scanpaths)} scanpaths")
                debug_logger.log("14_prediction_log", f"Creating {num_processes} batches of size ~{batch_size}")
            
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
            
            if self.debug:
                debug_logger.log("14_prediction_log", f"Parallel processing completed, got {len(predictions)} predictions")
            
            return predictions
        else:
            # Sequential processing
            if self.debug:
                debug_logger.log("14_prediction_log", "Using sequential processing for all predictions")
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
    debug_logger.log_header("2_data_loading_log", "LOADING SCANPATHS")
    
    files = glob.glob(os.path.join(input_dir, "*.json"))
    debug_logger.log("2_data_loading_log", f"Found {len(files)} JSON files in {input_dir}")
    
    # Sample files if requested
    if sample_percentage < 100:
        num_files = int(len(files) * sample_percentage / 100)
        files = random.sample(files, num_files)
        debug_logger.log("2_data_loading_log", f"Sampled {num_files} files ({sample_percentage}%)")
    
    scanpaths = []
    labels = []
    
    debug_logger.log("2_data_loading_log", f"Loading {len(files)} scanpath files...")
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        # Parse filename: S_XXXX_SY_Z_TTT_tokens.json
        # XXXX: subject ID, Y: session, Z: task number, TTT: task code
        parts = filename.split('_')
        task_num = int(parts[3])
        
        with open(file_path, 'r') as f:
            scanpath = json.load(f)
            
        scanpaths.append(scanpath)
        labels.append(task_num)
        
        if i < 50:  # Show details for first 50 files
            debug_logger.log("2_data_loading_log", f"  File {i}: {filename}")
            debug_logger.log("2_data_loading_log", f"    Task: {task_num}")
            debug_logger.log("2_data_loading_log", f"    Scanpath length: {len(scanpath)}")
            debug_logger.log("2_data_loading_log", f"    First 50 tokens: {scanpath[:50]}")
    
    debug_logger.log("2_data_loading_log", f"Loaded {len(scanpaths)} scanpaths with {len(set(labels))} unique labels")
    debug_logger.log("2_data_loading_log", f"Label distribution: {Counter(labels)}")
    
    # Show statistics
    lengths = [len(sp) for sp in scanpaths]
    debug_logger.log("2_data_loading_log", f"Scanpath length statistics:")
    debug_logger.log("2_data_loading_log", f"  Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.1f}, Std: {np.std(lengths):.1f}")
    
    # Show detailed scanpath examples (first 50 scanpaths, first 50 tokens each)
    debug_logger.log_data("2_data_loading_log", "Detailed scanpath examples (first 50 scanpaths)", scanpaths, max_items=50, max_tokens=50)
    
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
    debug_logger.log_header("15_evaluation_log", "EVALUATING CLASSIFIER PERFORMANCE")
    
    debug_logger.log("15_evaluation_log", f"Number of predictions: {len(y_pred)}")
    debug_logger.log("15_evaluation_log", f"Number of true labels: {len(y_true)}")
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)
    
    debug_logger.log("15_evaluation_log", f"Correct predictions: {correct}/{len(y_true)}")
    debug_logger.log("15_evaluation_log", f"Accuracy: {accuracy:.4f}")
    
    # Show detailed prediction examples
    debug_logger.log("15_evaluation_log", f"\nDetailed predictions (first 50):")
    for i in range(min(50, len(y_true))):
        status = "✓" if y_true[i] == y_pred[i] else "✗"
        debug_logger.log("15_evaluation_log", f"  {i}: True={y_true[i]}, Pred={y_pred[i]} {status}")
    
    # Calculate confusion matrix
    classes = sorted(set(y_true))
    cm = [[0 for _ in classes] for _ in classes]
    
    for true, pred in zip(y_true, y_pred):
        true_idx = classes.index(true)
        pred_idx = classes.index(pred)
        cm[true_idx][pred_idx] += 1
    
    debug_logger.log("15_evaluation_log", f"\nConfusion Matrix:")
    debug_logger.log("15_evaluation_log", "True\\Pred\t" + "\t".join(str(c) for c in classes))
    for i, cls in enumerate(classes):
        debug_logger.log("15_evaluation_log", f"{cls}\t\t" + "\t".join(str(cm[i][j]) for j in range(len(classes))))
    
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
        
        debug_logger.log("15_evaluation_log", f"\nClass {cls}:")
        debug_logger.log("15_evaluation_log", f"  Precision: {precision:.4f}")
        debug_logger.log("15_evaluation_log", f"  Recall: {recall:.4f}")
        debug_logger.log("15_evaluation_log", f"  F1-score: {f1:.4f}")
        debug_logger.log("15_evaluation_log", f"  Support: {sum(cm[cls_idx])}")
        debug_logger.log("15_evaluation_log", f"  True Positives: {tp}")
        debug_logger.log("15_evaluation_log", f"  False Positives: {fp}")
        debug_logger.log("15_evaluation_log", f"  False Negatives: {fn}")
    
    # Calculate macro average
    macro_precision = sum(m["precision"] for m in class_metrics.values()) / len(class_metrics)
    macro_recall = sum(m["recall"] for m in class_metrics.values()) / len(class_metrics)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if macro_precision + macro_recall > 0 else 0
    
    debug_logger.log("15_evaluation_log", f"\nMacro Averages:")
    debug_logger.log("15_evaluation_log", f"  Precision: {macro_precision:.4f}")
    debug_logger.log("15_evaluation_log", f"  Recall: {macro_recall:.4f}")
    debug_logger.log("15_evaluation_log", f"  F1-score: {macro_f1:.4f}")
    
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
    global debug_logger
    
    parser = argparse.ArgumentParser(
        description="FastEyeBLAST: High-performance scanpath classification (DEBUG VERSION)"
    )
    parser.add_argument("--input_dir", default="GazeBaseVR_IVT",
                       help="Directory containing tokenized scanpath JSON files")
    parser.add_argument("--output_dir", default="FastEyeBLAST_Debug",
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
    parser.add_argument("--debug", action="store_true", default=True,
                       help="Enable debug output")
    parser.add_argument("--no_debug", dest="debug", action="store_false",
                       help="Disable debug output")
    
    args = parser.parse_args()
    
    # Initialize debug logger
    debug_logger = DebugLogger(args.output_dir, args.debug)
    
    if args.debug:
        debug_logger.log_header("16_comprehensive", "FASTEYEBLAST DEBUG VERSION")
        debug_logger.log("16_comprehensive", "This version includes comprehensive debug output showing")
        debug_logger.log("16_comprehensive", "the internal workings of each algorithm stage.")
        debug_logger.log("16_comprehensive", f"Command line arguments: {vars(args)}")
        debug_logger.log("16_comprehensive", f"Log files will be saved to: {args.output_dir}")
        debug_logger.log("16_comprehensive", f"Timestamp: {debug_logger.timestamp}")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.debug:
        debug_logger.log("16_comprehensive", f"Random seed set to: {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.debug:
        debug_logger.log("16_comprehensive", f"Output directory: {args.output_dir}")
    
    # Load data
    debug_logger.log("16_comprehensive", "Loading scanpaths...")
    scanpaths, labels = load_scanpaths(args.input_dir, args.sample_percentage)
    debug_logger.log("16_comprehensive", f"Loaded {len(scanpaths)} scanpaths")
    
    # Split data into train and test sets
    if args.debug:
        debug_logger.log_header("16_comprehensive", "SPLITTING DATA")
        
    test_size = args.test_size
    test_count = int(len(scanpaths) * test_size)
    
    if args.debug:
        debug_logger.log("16_comprehensive", f"Test size: {test_size} ({test_count} samples)")
        debug_logger.log("16_comprehensive", f"Train size: {len(scanpaths) - test_count} samples")
    
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
    
    if args.debug:
        debug_logger.log("16_comprehensive", f"Train set distribution: {Counter(train_labels)}")
        debug_logger.log("16_comprehensive", f"Test set distribution: {Counter(test_labels)}")
    
    debug_logger.log("16_comprehensive", f"Training set: {len(train_scanpaths)} scanpaths")
    debug_logger.log("16_comprehensive", f"Test set: {len(test_scanpaths)} scanpaths")
    
    # Initialize classifier
    if args.debug:
        debug_logger.log_header("16_comprehensive", "INITIALIZING CLASSIFIER")
        
    classifier = FastEyeBLAST(
        seed_length=args.seed_length,
        band_width=args.band_width,
        num_neighbors=args.num_neighbors,
        max_candidates=args.max_candidates,
        use_duration_bins=args.use_duration_bins,
        duration_bins=args.duration_bins,
        use_parallel=args.use_parallel,
        num_processes=args.num_processes,
        debug=args.debug
    )
    
    # Train classifier
    debug_logger.log("16_comprehensive", "Training classifier...")
    start_time = time.time()
    classifier.fit(train_scanpaths, train_labels)
    train_time = time.time() - start_time
    debug_logger.log("16_comprehensive", f"Training completed in {train_time:.2f} seconds")
    
    # Make predictions
    debug_logger.log("16_comprehensive", "Making predictions...")
    start_time = time.time()
    predictions = classifier.predict(test_scanpaths)
    predict_time = time.time() - start_time
    debug_logger.log("16_comprehensive", f"Prediction completed in {predict_time:.2f} seconds")
    
    # Evaluate performance
    debug_logger.log("16_comprehensive", "Evaluating performance...")
    results = evaluate_classifier(test_labels, predictions)
    
    # Print results summary
    debug_logger.log_header("16_comprehensive", "FINAL RESULTS SUMMARY")
    debug_logger.log("16_comprehensive", f"Accuracy: {results['accuracy']:.4f}")
    debug_logger.log("16_comprehensive", f"Macro-avg F1: {results['macro_avg']['f1']:.4f}")
    debug_logger.log("16_comprehensive", f"Training time: {train_time:.2f} seconds")
    debug_logger.log("16_comprehensive", f"Prediction time: {predict_time:.2f} seconds")
    debug_logger.log("16_comprehensive", f"Average prediction time per sample: {predict_time/len(test_scanpaths)*1000:.2f} ms")
    
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
    
    debug_logger.log("16_comprehensive", f"Results saved to {output_file}")
    
    # Save predictions
    pred_file = os.path.join(args.output_dir, "predictions.json")
    with open(pred_file, 'w') as f:
        json.dump({
            "true": test_labels,
            "pred": predictions
        }, f)
    
    debug_logger.log("16_comprehensive", f"Predictions saved to {pred_file}")
    
    if args.debug:
        debug_logger.log_header("16_comprehensive", "DEBUG SESSION COMPLETED")
        debug_logger.log("16_comprehensive", "This debug version has shown you:")
        debug_logger.log("16_comprehensive", "• Token encoding and duration binning")
        debug_logger.log("16_comprehensive", "• Bit-packing of sequences")
        debug_logger.log("16_comprehensive", "• Rolling hash calculation")
        debug_logger.log("16_comprehensive", "• SimHash fingerprint computation")
        debug_logger.log("16_comprehensive", "• LSH index construction and querying")
        debug_logger.log("16_comprehensive", "• Seed extraction and chaining")
        debug_logger.log("16_comprehensive", "• Banded Levenshtein distance calculation")
        debug_logger.log("16_comprehensive", "• k-NN classification with weighted voting")
        
        debug_logger.log("16_comprehensive", f"\nGenerated log files:")
        for log_file in debug_logger.log_files:
            log_path = os.path.join(args.output_dir, f"{log_file}_{debug_logger.timestamp}.txt")
            debug_logger.log("16_comprehensive", f"  {log_path}")


if __name__ == "__main__":
    main()
