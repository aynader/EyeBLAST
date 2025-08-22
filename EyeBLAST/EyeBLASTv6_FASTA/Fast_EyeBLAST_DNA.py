import os
import glob
import time
import random
import argparse
import numpy as np
import psutil
from collections import defaultdict, Counter
import multiprocessing as mp
from tqdm import tqdm
from Bio import SeqIO  # Add BioPython import

# Constants for DNA alphabet
MASK64 = (1 << 64) - 1
DEFAULT_SEED_LENGTH = 11  # Increased from original
DEFAULT_NUM_NEIGHBORS = 6  # Increased from 3 for better accuracy
DEFAULT_BAND_WIDTH = 16  # Increased from original
DEFAULT_NUM_HASH_TABLES = 12  # Increased from original
DEFAULT_NUM_HASH_BITS = 16
DEFAULT_MAX_CANDIDATES = 30  # Reduced for faster prediction
DEFAULT_LSH_HASH_SIZE = 64

# Ambiguous nucleotide mapping
AMBIGUOUS_NUCLEOTIDES = {
    'R': ['A', 'G'],  # puRine
    'Y': ['C', 'T'],  # pYrimidine
    'S': ['G', 'C'],  # Strong interaction (3 H bonds)
    'W': ['A', 'T'],  # Weak interaction (2 H bonds)
    'K': ['G', 'T'],  # Keto
    'M': ['A', 'C'],  # aMino
    'B': ['C', 'G', 'T'],  # not A
    'D': ['A', 'G', 'T'],  # not C
    'H': ['A', 'C', 'T'],  # not G
    'V': ['A', 'C', 'G'],  # not T
    'N': ['A', 'C', 'G', 'T']  # aNy
}

# 16S rRNA conserved regions (positions are approximate)
CONSERVED_16S_REGIONS = [
    (9, 30),    # C1
    (360, 375), # C2
    (505, 520), # C3
    (690, 710), # C4
    (787, 798), # C5
    (907, 928), # C6
    (1054, 1065), # C7
    (1392, 1408)  # C8
]

class FastEyeBLASTDNA:
    """
    FastEyeBLAST variant for DNA sequences (A, C, G, T, N)
    """
    def __init__(self,
                 seed_length=DEFAULT_SEED_LENGTH,
                 band_width=DEFAULT_BAND_WIDTH,
                 num_neighbors=DEFAULT_NUM_NEIGHBORS,
                 num_hash_tables=DEFAULT_NUM_HASH_TABLES,
                 num_hash_bits=DEFAULT_NUM_HASH_BITS,
                 max_candidates=DEFAULT_MAX_CANDIDATES,
                 lsh_hash_size=DEFAULT_LSH_HASH_SIZE,
                 use_parallel=True,
                 num_processes=None):
        # Seed & DP parameters
        self.seed_length = seed_length
        self.band_width = band_width
        self.num_neighbors = num_neighbors
        self.num_hash_tables = num_hash_tables
        self.num_hash_bits = num_hash_bits
        self.max_candidates = max_candidates
        self.lsh_hash_size = lsh_hash_size

        # DNA token mapping including N for any nucleotide
        self.token_to_id = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        # Add ambiguous nucleotides with special IDs
        for i, nt in enumerate('RYSWKMBDHV'):
            self.token_to_id[nt] = 5 + i
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # Ambiguous nucleotide probabilities
        self.ambiguous_probs = {}
        for nt, options in AMBIGUOUS_NUCLEOTIDES.items():
            self.ambiguous_probs[self.token_to_id[nt]] = {self.token_to_id[opt]: 1.0/len(options) for opt in options}

        # Bit-packing parameters - need 4 bits for 15 tokens (2^4 = 16 > 15)
        self.bits_per_token = 4
        self.tokens_per_word = 64 // self.bits_per_token  # 16 tokens per 64-bit word

        # Rolling hash base
        self.base = 31
        self.pow_base = [1]
        for i in range(1, max(self.seed_length, 100)):
            self.pow_base.append((self.pow_base[-1] * self.base) & MASK64)

        # LSH structures
        self.id_bitmasks = []
        self.lsh_tables = None

        # Training data
        self.encoded_seqs = []
        self.packed_seqs = []
        self.simhash_sketches = []
        self.labels = []
        
        # Additional k-mer features
        self.kmer_size = 9  # Increased from 7 for better specificity
        self.kmer_profiles = []  # To store k-mer profiles for training data
        
        # Conserved region weights
        self.conserved_weights = {}  # Will be populated during training
        
        # Position-specific weights for 16S rRNA
        self.position_weights = None  # Will be populated during training
        
        # Performance metrics
        self.train_time = 0
        self.prediction_time = 0
        self.memory_mb = 0

        # Parallel
        self.use_parallel = use_parallel
        self.num_processes = mp.cpu_count() if num_processes is None else num_processes

    def _encode_sequence(self, seq):
        """
        Convert sequence string to list of integers with better handling of ambiguous nucleotides
        """
        encoded = []
        for nt in seq:
            nt = nt.upper()
            if nt in self.token_to_id:
                encoded.append(self.token_to_id[nt])
            else:
                # Default to N (any nucleotide) for unknown characters
                encoded.append(self.token_to_id['N'])
        return encoded

    def _pack_ids_into_words(self, id_list):
        """Pack encoded sequence into 64-bit words"""
        if not id_list:
            return []
        num_words = (len(id_list) + self.tokens_per_word - 1) // self.tokens_per_word
        packed = []
        for i in range(num_words):
            word = 0
            for j in range(self.tokens_per_word):
                idx = i * self.tokens_per_word + j
                if idx < len(id_list):
                    token_id = id_list[idx]
                    # Ensure we have enough bits for all token types
                    word |= (token_id & ((1 << self.bits_per_token) - 1)) << (j * self.bits_per_token)
            packed.append(word)
        return packed

    def _initialize_simhash(self):
        """Initialize simhash bitmasks with improved specificity"""
        rng = random.Random(42)
        self.id_bitmasks = []
        
        # Use more bits for higher accuracy
        for _ in range(max(self.token_to_id.values()) + 1):
            mask = 0
            for j in range(self.lsh_hash_size):
                # Use a biased random pattern for better separation
                if rng.random() < 0.4:  # Change bias from 0.5 to 0.4
                    mask |= (1 << j)
            self.id_bitmasks.append(mask)

    def _compute_simhash(self, id_list):
        """
        Compute simhash fingerprint with special handling for ambiguous nucleotides
        """
        if not self.id_bitmasks:
            self._initialize_simhash()
            
        counters = [0] * self.lsh_hash_size
        
        for token_id in id_list:
            # Special handling for ambiguous nucleotides
            if token_id >= 5:  # Ambiguous nucleotide
                if token_id in self.ambiguous_probs:
                    # Weight the contribution of each possible nucleotide
                    for actual_id, prob in self.ambiguous_probs[token_id].items():
                        mask = self.id_bitmasks[actual_id]
                        for i in range(self.lsh_hash_size):
                            counters[i] += prob * (1 if (mask & (1 << i)) else -1)
            else:
                # Regular nucleotide
                mask = self.id_bitmasks[token_id]
                for i in range(self.lsh_hash_size):
                    counters[i] += 1 if (mask & (1 << i)) else -1
                    
        fingerprint = 0
        for i in range(self.lsh_hash_size):
            if counters[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    def _build_lsh_index(self):
        """Build LSH index with improved bucketing strategy"""
        tables = [{} for _ in range(self.num_hash_tables)]
        bits_per_slice = self.lsh_hash_size // self.num_hash_tables
        
        print("Building LSH index...")
        for idx, sketch in enumerate(tqdm(self.simhash_sketches, desc="Indexing sequences")):
            for t in range(self.num_hash_tables):
                # Use bit rotation for better distribution
                rotated_sketch = ((sketch << t) | (sketch >> (self.lsh_hash_size - t))) & ((1 << self.lsh_hash_size) - 1)
                start = t * bits_per_slice
                slice_ = (rotated_sketch >> start) & ((1 << bits_per_slice) - 1)
                tables[t].setdefault(slice_, []).append(idx)
        self.lsh_tables = tables

    def _query_lsh(self, sketch):
        """Query LSH index with improved multi-probe strategy"""
        candidates = set()
        bits_per_slice = self.lsh_hash_size // self.num_hash_tables
        
        for t in range(self.num_hash_tables):
            # Use bit rotation for better distribution (same as in build)
            rotated_sketch = ((sketch << t) | (sketch >> (self.lsh_hash_size - t))) & ((1 << self.lsh_hash_size) - 1)
            start = t * bits_per_slice
            slice_ = (rotated_sketch >> start) & ((1 << bits_per_slice) - 1)
            
            # Add exact matches
            if slice_ in self.lsh_tables[t]:
                candidates.update(self.lsh_tables[t][slice_])
                
            # Multi-probe: also check buckets with small Hamming distance
            # Flip one bit at a time to handle small variations
            for bit in range(bits_per_slice):
                neighbor_slice = slice_ ^ (1 << bit)
                if neighbor_slice in self.lsh_tables[t]:
                    candidates.update(self.lsh_tables[t][neighbor_slice])
                    
        return list(candidates)

    def _extract_seeds(self, p1, p2):
        """Extract seed matches between packed sequences"""
        seeds = []
        # Adjust max_mismatch for 4 bits per token
        max_mismatch = self.bits_per_token * 3  # Allow more mismatches for ambiguous nucleotides
        min_len = min(len(p1), len(p2))
        for i in range(min_len):
            mismatches = bin(p1[i] ^ p2[i]).count('1')
            if mismatches <= max_mismatch:
                seeds.append((i * self.tokens_per_word, i * self.tokens_per_word))
        return seeds

    def _extract_seeds_improved(self, seq1, seq2):
        """
        Improved seed extraction that works with the raw sequences rather than packed words
        to find longer matching regions more accurately with special handling for ambiguous nucleotides
        """
        seeds = []
        min_len = min(len(seq1), len(seq2))
        
        # Find exact k-mer matches between sequences
        k = self.seed_length
        seq1_kmers = {}
        
        # Position-based weighting for 16S rRNA
        position_importance = np.ones(len(seq1))
        
        # If we have position weights from training (for 16S rRNA), apply them
        if self.position_weights is not None and len(self.position_weights) >= len(seq1):
            position_importance = self.position_weights[:len(seq1)]
        
        # Build k-mer index with position information
        for i in range(len(seq1) - k + 1):
            # Skip k-mers with too many ambiguous nucleotides
            ambiguous_count = sum(1 for j in range(i, i+k) if seq1[j] >= 5)
            if ambiguous_count > k // 3:  # No more than 1/3 of positions can be ambiguous
                continue
                
            kmer = tuple(seq1[i:i+k])
            if kmer not in seq1_kmers:
                seq1_kmers[kmer] = []
            seq1_kmers[kmer].append((i, position_importance[i]))
        
        # Find matching k-mers in seq2, allowing for ambiguous nucleotides
        for i in range(len(seq2) - k + 1):
            # Skip k-mers with too many ambiguous nucleotides
            ambiguous_count = sum(1 for j in range(i, i+k) if seq2[j] >= 5)
            if ambiguous_count > k // 3:
                continue
                
            kmer = tuple(seq2[i:i+k])
            
            # Direct match
            if kmer in seq1_kmers:
                for pos1, weight in seq1_kmers[kmer]:
                    seeds.append((pos1, i, weight))
            
            # Also check for compatible matches with ambiguous nucleotides
            if any(id >= 5 for id in kmer):  # Contains ambiguous nucleotides
                for stored_kmer, positions in seq1_kmers.items():
                    # Check if kmers are compatible considering ambiguous nucleotides
                    compatible = True
                    for j in range(k):
                        id1, id2 = stored_kmer[j], kmer[j]
                        
                        # If either is ambiguous, check compatibility
                        if id1 >= 5 or id2 >= 5:
                            if id1 >= 5 and id1 in self.ambiguous_probs:
                                if id2 < 5 and id2 not in self.ambiguous_probs[id1]:
                                    compatible = False
                                    break
                            elif id2 >= 5 and id2 in self.ambiguous_probs:
                                if id1 < 5 and id1 not in self.ambiguous_probs[id2]:
                                    compatible = False
                                    break
                        elif id1 != id2:  # Both regular nucleotides but not matching
                            compatible = False
                            break
                            
                    if compatible:
                        for pos1, weight in positions:
                            seeds.append((pos1, i, weight))
        
        # Extend seeds to find maximal exact matches
        extended_seeds = []
        for s1, s2, weight in seeds:
            # Extend forward
            e1, e2 = s1, s2
            extension_len = 0
            while e1 < len(seq1) and e2 < len(seq2):
                id1, id2 = seq1[e1], seq2[e2]
                
                # Check if nucleotides match or are compatible ambiguous nucleotides
                compatible = False
                if id1 == id2:  # Direct match
                    compatible = True
                elif id1 >= 5 and id1 in self.ambiguous_probs and id2 < 5:
                    compatible = id2 in self.ambiguous_probs[id1]
                elif id2 >= 5 and id2 in self.ambiguous_probs and id1 < 5:
                    compatible = id1 in self.ambiguous_probs[id2]
                    
                if not compatible:
                    break
                    
                e1 += 1
                e2 += 1
                extension_len += 1
            
            # Extend backward
            b1, b2 = s1 - 1, s2 - 1
            while b1 >= 0 and b2 >= 0:
                id1, id2 = seq1[b1], seq2[b2]
                
                # Check compatibility
                compatible = False
                if id1 == id2:  # Direct match
                    compatible = True
                elif id1 >= 5 and id1 in self.ambiguous_probs and id2 < 5:
                    compatible = id2 in self.ambiguous_probs[id1]
                elif id2 >= 5 and id2 in self.ambiguous_probs and id1 < 5:
                    compatible = id1 in self.ambiguous_probs[id2]
                    
                if not compatible:
                    break
                    
                b1 -= 1
                b2 -= 1
                extension_len += 1
            
            # Store extended seed with its weight and length
            length = e1 - (b1 + 1)
            
            # Calculate importance based on position and conserved regions
            importance = weight
            if self.conserved_weights:
                # Check if this seed overlaps with conserved regions
                for start, end in CONSERVED_16S_REGIONS:
                    if (b1+1 <= end and e1 >= start):  # Overlap with conserved region
                        importance *= 1.5  # Boost importance
            
            extended_seeds.append((b1 + 1, b2 + 1, length, importance))
        
        # Filter out short extensions and overlapping seeds
        min_seed_length = max(k, 15)  # Require longer seeds for better specificity
        extended_seeds = [(s1, s2, length, imp) for s1, s2, length, imp in extended_seeds if length >= min_seed_length]
        
        # Sort by importance and length
        extended_seeds.sort(key=lambda x: (x[3], x[2]), reverse=True)
        
        # Remove overlapping seeds, keep the more important ones
        filtered_seeds = []
        for s1, s2, length, imp in extended_seeds:
            overlap = False
            for fs1, fs2, _, _ in filtered_seeds:
                if (s1 <= fs1 + min_seed_length and s1 + length >= fs1) or (fs1 <= s1 + min_seed_length and fs1 + length >= s1):
                    overlap = True
                    break
            if not overlap:
                filtered_seeds.append((s1, s2, length, imp))
        
        return [(s1, s2) for s1, s2, _, _ in filtered_seeds]

    def _chain_seeds(self, seeds):
        """Chain seeds based on diagonal patterns"""
        if not seeds:
            return []
        diagonals = defaultdict(list)
        for a, b in seeds:
            diagonals[b - a].append((a, b))
        best_diag, best = max(diagonals.items(), key=lambda x: len(x[1]))
        best.sort()
        chain = [best[0]]
        for s in best[1:]:
            if s[0] > chain[-1][0] and s[1] > chain[-1][1]:
                chain.append(s)
        return chain

    def _chain_seeds_improved(self, seeds):
        """
        Improved seed chaining that considers diagonal runs, gap penalties,
        and uses dynamic programming for optimal chain selection
        """
        if not seeds:
            return []
        
        # Group seeds by diagonals with a tolerance for small shifts
        diag_tolerance = 2  # Allow small diagonal shifts
        grouped_seeds = defaultdict(list)
        
        for a, b in seeds:
            diag = b - a
            for d in range(diag - diag_tolerance, diag + diag_tolerance + 1):
                grouped_seeds[d].append((a, b, diag))
        
        # Find best chains for each diagonal group
        all_chains = []
        
        for diag, seeds_group in grouped_seeds.items():
            if len(seeds_group) == 0:
                continue
                
            # Sort seeds by position
            seeds_group.sort()
            
            # Dynamic programming to find best chain
            n = len(seeds_group)
            dp = [1] * n  # Length of chain ending at i
            prev = [-1] * n  # Previous seed in chain
            
            for i in range(1, n):
                for j in range(i):
                    a1, b1, _ = seeds_group[j]
                    a2, b2, _ = seeds_group[i]
                    
                    # Check if seeds can be chained (a2 > a1 and b2 > b1)
                    if a2 > a1 and b2 > b1:
                        # Compute gap penalty
                        gap = min(a2 - a1, b2 - b1)
                        # Diagonal shift penalty
                        diag_shift = abs((b2 - a2) - (b1 - a1))
                        
                        # Higher score for small gaps and small diagonal shifts
                        chain_score = dp[j] + 1 - 0.01 * gap - 0.05 * diag_shift
                        
                        if chain_score > dp[i]:
                            dp[i] = chain_score
                            prev[i] = j
            
            # Find best chain end
            best_end = max(range(n), key=lambda i: dp[i])
            
            # Reconstruct chain
            chain = []
            curr = best_end
            while curr != -1:
                chain.append((seeds_group[curr][0], seeds_group[curr][1]))
                curr = prev[curr]
            
            all_chains.append(list(reversed(chain)))
        
        # Select best chain based on total length covered and number of seeds
        best_chain = []
        best_score = 0
        
        for chain in all_chains:
            if not chain:
                continue
                
            # Compute chain score based on number of seeds and coverage
            coverage = chain[-1][0] - chain[0][0] + 1
            score = len(chain) + 0.001 * coverage
            
            if score > best_score:
                best_score = score
                best_chain = chain
        
            return best_chain

    def _banded_levenshtein(self, seq1, seq2, band_width, chain):
        """Banded Levenshtein distance calculation with position-specific scoring"""
        if not chain:
            return len(seq1) + len(seq2), []
        m, n = len(seq1), len(seq2)
        half = band_width // 2
        diag = chain[0][1] - chain[0][0]
        dp = np.zeros((m + 1, band_width), dtype=float)
        
        # Scoring parameters - different penalties for different ambiguous nucleotides
        gap = -1.5
        match_score = 3.0
        mismatch = -2.0
        
        # Initialize first column
        for i in range(1, m + 1):
            dp[i, 0] = dp[i - 1, 0] + gap
            
        for i in range(1, m + 1):
            col_start = max(1, i + diag - half)
            col_end = min(n + 1, i + diag + half + 1)
            # Ensure we don't process more columns than the band width
            col_end = min(col_end, col_start + band_width - 1)
            
            for j in range(col_start, col_end):
                bj = j - col_start + 1
                # Safety check to ensure bj is within bounds
                if bj >= band_width:
                    continue
                    
                # Get the nucleotide IDs
                id1, id2 = seq1[i-1], seq2[j-1]
                
                # Check for ambiguous nucleotides
                if id1 >= 5 and id1 in self.ambiguous_probs and id2 < 5:
                    # Ambiguous in seq1 matches specific in seq2
                    if id2 in self.ambiguous_probs[id1]:
                        diag_score = dp[i-1, bj-1] + match_score * 0.8  # Slightly lower match score
                    else:
                        diag_score = dp[i-1, bj-1] + mismatch
                elif id2 >= 5 and id2 in self.ambiguous_probs and id1 < 5:
                    # Ambiguous in seq2 matches specific in seq1
                    if id1 in self.ambiguous_probs[id2]:
                        diag_score = dp[i-1, bj-1] + match_score * 0.8
                    else:
                        diag_score = dp[i-1, bj-1] + mismatch
                elif id1 == id2:  # Exact match
                    diag_score = dp[i-1, bj-1] + match_score
                else:  # Mismatch
                    diag_score = dp[i-1, bj-1] + mismatch
                    
                up = dp[i-1, bj] + gap
                left = dp[i, bj-1] + gap
                dp[i, bj] = max(diag_score, up, left)
                
        # Ensure the final index is valid
        final_idx = min(n - col_start + 1, band_width - 1) if n >= col_start else 0
        return dp[m, final_idx], chain

    def _smith_waterman(self, seq1, seq2, seeds=None):
        """
        Improved alignment using Smith-Waterman algorithm for local alignment
        with affine gap penalties, special handling for ambiguous nucleotides,
        and position-specific scoring for conserved regions
        """
        # Early optimization: If sequences are very different in length, return low similarity
        len_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
        if len_ratio < 0.5:  # If one sequence is less than half the length of the other
            return 0.1 * len_ratio  # Return a small similarity score
            
        # Early optimization: Use a smaller sample for very long sequences
        max_length = 500
        if len(seq1) > max_length or len(seq2) > max_length:
            # Sample from beginning, middle and end
            if len(seq1) > max_length:
                sample_size = max_length // 3
                seq1_sample = seq1[:sample_size] + seq1[len(seq1)//2-sample_size//2:len(seq1)//2+sample_size//2] + seq1[-sample_size:]
            else:
                seq1_sample = seq1
                
            if len(seq2) > max_length:
                sample_size = max_length // 3
                seq2_sample = seq2[:sample_size] + seq2[len(seq2)//2-sample_size//2:len(seq2)//2+sample_size//2] + seq2[-sample_size:]
            else:
                seq2_sample = seq2
                
            seq1, seq2 = seq1_sample, seq2_sample
        
        m, n = len(seq1), len(seq2)
        
        # Scoring parameters
        match = 3.0
        mismatch = -2.0
        gap_open = -3.0
        gap_extend = -1.5
        
        # Position-specific weights for conserved regions
        pos_weights1 = np.ones(m)
        pos_weights2 = np.ones(n)
        
        # Apply weights for conserved regions if we have them
        if self.position_weights is not None:
            if len(self.position_weights) >= m:
                pos_weights1 = self.position_weights[:m]
            if len(self.position_weights) >= n:
                pos_weights2 = self.position_weights[:n]
        
        # Initialize matrices
        score = np.zeros((m+1, n+1), dtype=float)
        
        # Fill the matrices
        max_score = 0
        max_i, max_j = 0, 0
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                # Get the nucleotide IDs
                id1, id2 = seq1[i-1], seq2[j-1]
                
                # Get position weights
                pos_weight = (pos_weights1[i-1] + pos_weights2[j-1]) / 2.0
                
                # Calculate match/mismatch score with position weighting
                if id1 >= 5 and id1 in self.ambiguous_probs and id2 < 5:
                    # Ambiguous in seq1 matches specific in seq2
                    if id2 in self.ambiguous_probs[id1]:
                        diag_score = score[i-1, j-1] + match * 0.8 * pos_weight
                    else:
                        diag_score = score[i-1, j-1] + mismatch * pos_weight
                elif id2 >= 5 and id2 in self.ambiguous_probs and id1 < 5:
                    # Ambiguous in seq2 matches specific in seq1
                    if id1 in self.ambiguous_probs[id2]:
                        diag_score = score[i-1, j-1] + match * 0.8 * pos_weight
                    else:
                        diag_score = score[i-1, j-1] + mismatch * pos_weight
                elif id1 == id2:  # Exact match
                    diag_score = score[i-1, j-1] + match * pos_weight
                else:  # Mismatch
                    diag_score = score[i-1, j-1] + mismatch * pos_weight
                
                # Gap scores with position weighting
                up_score = max(score[i-1, j] + gap_extend * pos_weight, 
                               score[i-1, j-1] + gap_open * pos_weight)
                left_score = max(score[i, j-1] + gap_extend * pos_weight, 
                                score[i-1, j-1] + gap_open * pos_weight)
                
                # Choose best score
                score[i, j] = max(0, diag_score, up_score, left_score)
                
                # Update max score
                if score[i, j] > max_score:
                    max_score = score[i, j]
                    max_i, max_j = i, j
        
        # Trace back to find the alignment
        alignment_len = 0
        i, j = max_i, max_j
        
        while i > 0 and j > 0 and score[i, j] > 0:
            if score[i, j] == score[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch):
                # Match or mismatch
                alignment_len += 1
                i -= 1
                j -= 1
            elif i > 0 and score[i, j] == score[i-1, j] + gap_extend:
                # Gap in seq2
                alignment_len += 1
                i -= 1
            else:
                # Gap in seq1
                alignment_len += 1
                j -= 1
        
        # Normalize score considering alignment length and position weights
        avg_pos_weight = (np.mean(pos_weights1) + np.mean(pos_weights2)) / 2.0
        max_possible = min(m, n) * match * avg_pos_weight
        
        if max_possible == 0:
            return 0
            
        norm_score = max_score / max_possible
        
        # Boost score based on alignment length relative to sequence length
        coverage = alignment_len / min(m, n)
        final_score = norm_score * (0.2 + 0.8 * coverage)
        
        return min(1.0, final_score)  # Cap at 1.0

    def _similarity(self, seq1, seq2):
        """Legacy similarity function using packed sequences"""
        p1 = self._pack_ids_into_words(seq1)
        p2 = self._pack_ids_into_words(seq2)
        seeds = self._extract_seeds(p1, p2)
        chain = self._chain_seeds(seeds)
        score, _ = self._banded_levenshtein(seq1, seq2, self.band_width, chain)
        max_score = 2 * max(len(seq1), len(seq2))
        return max(0, score) / max_score

    def _similarity_improved(self, seq1, seq2):
        """
        Optimized similarity calculation that is both accurate and fast.
        Selectively applies more expensive methods only when needed.
        """
        # Quick filter: compare lengths
        len_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
        if len_ratio < 0.4:  # Strict length ratio filter
            return 0.05 * len_ratio
        
        # Quick Hamming distance check on prefix and suffix
        prefix_size = min(75, min(len(seq1), len(seq2)) // 3)
        suffix_size = min(75, min(len(seq1), len(seq2)) // 3)
        
        # Check both prefix and suffix for better discrimination
        prefix_hamming = sum(1 for i in range(prefix_size) if seq1[i] != seq2[i] and seq1[i] < 5 and seq2[i] < 5)
        suffix_hamming = sum(1 for i in range(1, suffix_size+1) 
                             if seq1[-i] != seq2[-i] and seq1[-i] < 5 and seq2[-i] < 5)
        
        # Combine prefix and suffix distances
        hamming_dist = (prefix_hamming + suffix_hamming) / (prefix_size + suffix_size)
        if hamming_dist > 0.6:  # If more than 60% different in these regions
            return 0.1 * (1 - hamming_dist)
        
        # Check conserved regions specifically (for 16S rRNA)
        if self.conserved_weights:
            conserved_matches = 0
            conserved_total = 0
            
            for start, end in CONSERVED_16S_REGIONS:
                # Only check regions that are within both sequences
                if start < min(len(seq1), len(seq2)) and end > 0:
                    actual_start = max(0, start)
                    actual_end = min(min(len(seq1), len(seq2)), end)
                    
                    if actual_end > actual_start:
                        region_length = actual_end - actual_start
                        conserved_total += region_length
                        
                        # Count matches in this conserved region
                        for i in range(actual_start, actual_end):
                            if i < len(seq1) and i < len(seq2):
                                if seq1[i] == seq2[i]:
                                    conserved_matches += 1
                                # Partial credit for ambiguous nucleotide matches
                                elif seq1[i] >= 5 and seq1[i] in self.ambiguous_probs and seq2[i] < 5:
                                    if seq2[i] in self.ambiguous_probs[seq1[i]]:
                                        conserved_matches += 0.5
                                elif seq2[i] >= 5 and seq2[i] in self.ambiguous_probs and seq1[i] < 5:
                                    if seq1[i] in self.ambiguous_probs[seq2[i]]:
                                        conserved_matches += 0.5
            
            # If we have enough conserved regions to check
            if conserved_total > 50:
                conserved_similarity = conserved_matches / conserved_total
                
                # If conserved regions don't match well, sequences are likely different
                if conserved_similarity < 0.4:
                    return 0.1 * conserved_similarity
        
        # Sample k-mers from throughout the sequence
        k = 6  # k-mer size
        min_len = min(len(seq1), len(seq2))
        
        # Adaptive sampling based on sequence length
        if min_len > 500:
            # Sample more positions for longer sequences
            num_samples = 300
            step = max(1, min_len // num_samples)
            
            # Ensure we sample from beginning, middle, and end with equal weight
            positions = (
                list(range(0, min(200, min_len), step)) +                          # Beginning
                list(range(min_len//2-100, min(min_len//2+100, min_len), step)) +  # Middle
                list(range(max(0, min_len-200), min_len, step))                    # End
            )
        else:
            # For shorter sequences, sample more densely
            step = max(1, min_len // 150)
            positions = list(range(0, min_len, step))
        
        # Count exact k-mer matches
        matches = 0
        total_positions = 0
        
        for i in positions:
            if i + k <= min_len:
                kmer1 = tuple(seq1[i:i+k])
                kmer2 = tuple(seq2[i:i+k])
                
                # Skip k-mers with too many ambiguous nucleotides
                ambiguous1 = sum(1 for id in kmer1 if id >= 5)
                ambiguous2 = sum(1 for id in kmer2 if id >= 5)
                if ambiguous1 > k // 3 or ambiguous2 > k // 3:
                    continue
                
                # Count exact matches and partial matches
                if kmer1 == kmer2:
                    matches += 1.0
                else:
                    # Count number of matching positions
                    pos_matches = sum(1 for j in range(k) if kmer1[j] == kmer2[j])
                    if pos_matches >= k * 0.7:  # If at least 70% match
                        matches += pos_matches / k
                
                # Weight matches in conserved regions more heavily
                if self.conserved_weights:
                    if i in self.conserved_weights and self.conserved_weights[i] > 1.2:
                        matches += 0.5  # Bonus for conserved region matches
                
                total_positions += 1
        
        # Calculate initial similarity score from k-mer matches
        kmer_sim = matches / total_positions if total_positions > 0 else 0
        
        # If similarity is very low, don't bother with expensive alignments
        if kmer_sim < 0.2:
            return kmer_sim
        
        # For promising matches, add alignment-based similarity
        # Banded Levenshtein alignment is much faster than Smith-Waterman
        alignment_sim = 0
        
        # For very long sequences, sample regions
        if min_len > 400:
            # Sample from important regions (beginning, middle, end, and conserved regions)
            regions = [
                (0, min(150, min_len)),                            # Beginning
                (min_len//2-75, min(min_len//2+75, min_len)),      # Middle
                (max(0, min_len-150), min_len)                     # End
            ]
            
            # Add known conserved regions if they're in range
            if self.conserved_weights:
                for start, end in CONSERVED_16S_REGIONS:
                    if start < min_len and end > 0:
                        actual_start = max(0, start)
                        actual_end = min(min_len, end)
                        if actual_end - actual_start > 10:  # Only add if region is big enough
                            regions.append((actual_start, actual_end))
            
            # Calculate alignment score for each region
            region_scores = []
            for start, end in regions:
                s1 = seq1[start:end]
                s2 = seq2[start:end]
                
                if len(s1) > 5 and len(s2) > 5:  # Only process if regions are meaningful
                    # Use simple seeds for alignment
                    seeds = [(0, 0)]
                    score, _ = self._banded_levenshtein(s1, s2, min(self.band_width, len(s1), len(s2)), seeds)
                    max_score = 2 * max(len(s1), len(s2))
                    norm_score = max(0, score) / max_score
                    region_scores.append(norm_score)
            
            if region_scores:
                alignment_sim = sum(region_scores) / len(region_scores)
        else:
            # For shorter sequences, use the whole sequence
            seeds = [(0, 0)]
            score, _ = self._banded_levenshtein(seq1, seq2, min(self.band_width, len(seq1), len(seq2)), seeds)
            max_score = 2 * max(len(seq1), len(seq2))
            alignment_sim = max(0, score) / max_score
        
        # Combined weighted score
        final_sim = 0.6 * alignment_sim + 0.4 * kmer_sim
        
        # Apply scaling for better separation between matches and non-matches
        if final_sim > 0.6:  # Lower threshold from 0.7 to 0.6
            final_sim = 0.6 + 0.4 * (final_sim - 0.6) / 0.4  # Steeper curve
        elif final_sim < 0.4:  # Higher threshold from 0.3 to 0.4
            final_sim = 0.4 * (final_sim / 0.4)  # Sharper drop-off
        
        return min(1.0, final_sim)  # Ensure we don't exceed 1.0

    def _compute_kmer_profile(self, seq):
        """Compute k-mer frequency profile for a sequence with optimizations for 16S rRNA"""
        kmers = {}
        k = self.kmer_size
        
        # Skip if sequence is too short
        if len(seq) < k:
            return kmers
            
        # Optimization: Use a sampling approach for very long sequences
        if len(seq) > 1000:
            # Sample from beginning, middle, and end
            sample_size = 350  # Increased from 300
            parts = [
                seq[:sample_size],
                seq[len(seq)//2-sample_size//2:len(seq)//2+sample_size//2],
                seq[-sample_size:]
            ]
            sample_seq = []
            for part in parts:
                sample_seq.extend(part)
            seq = sample_seq
            
        # Position weights for conserved regions in 16S rRNA
        position_weights = np.ones(len(seq))
        
        # Apply conserved region weights if available
        if self.conserved_weights:
            for i in range(len(seq)):
                if i in self.conserved_weights:
                    position_weights[i] = self.conserved_weights[i]
            
        # Count k-mers, handling ambiguous nucleotides specially
        for i in range(len(seq) - k + 1):
            kmer = tuple(seq[i:i+k])
            
            # Skip k-mers with too many ambiguous nucleotides
            ambiguous_count = sum(1 for id in kmer if id >= 5)
            if ambiguous_count > k // 3:  # No more than 1/3 of positions can be ambiguous
                continue
                
            # Apply position weighting
            weight = position_weights[i]
            
            if kmer not in kmers:
                kmers[kmer] = 0
            kmers[kmer] += weight
            
        # Normalize counts
        total = sum(kmers.values()) or 1
        for kmer in kmers:
            kmers[kmer] /= total
            
        return kmers

    def _compare_kmer_profiles(self, seq1, seq2):
        """Compare k-mer profiles of two sequences with 16S rRNA optimizations"""
        # Fast check - if sequences are very different in length
        len_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
        if len_ratio < 0.4:  # Stricter ratio
            return 0.05 * len_ratio
            
        profile1 = self._compute_kmer_profile(seq1)
        profile2 = self._compute_kmer_profile(seq2)
        
        # If either profile is empty, return 0
        if not profile1 or not profile2:
            return 0
            
        # Optimization: Only look at the most common k-mers
        # for very large profiles
        max_kmers = 150  # Increased from 100
        if len(profile1) > max_kmers:
            profile1 = dict(sorted(profile1.items(), key=lambda x: x[1], reverse=True)[:max_kmers])
        if len(profile2) > max_kmers:
            profile2 = dict(sorted(profile2.items(), key=lambda x: x[1], reverse=True)[:max_kmers])
            
        # Find common k-mers
        common_kmers = set(profile1.keys()) & set(profile2.keys())
        
        # Fast path if few common k-mers
        if len(common_kmers) < 5:
            return len(common_kmers) / min(len(profile1), len(profile2)) * 0.5
            
        # Compute cosine similarity
        dot_product = sum(profile1[k] * profile2[k] for k in common_kmers)
        mag1 = np.sqrt(sum(v**2 for v in profile1.values()))
        mag2 = np.sqrt(sum(v**2 for v in profile2.values()))
        
        if mag1 * mag2 == 0:
            return 0
            
        return dot_product / (mag1 * mag2)

    def _count_exact_kmer_matches(self, seq1, seq2):
        """Count exact k-mer matches between sequences with conserved region awareness"""
        k = self.kmer_size
        
        # Skip if sequences are too short
        if len(seq1) < k or len(seq2) < k:
            return 0
        
        # Fast check - if sequences are very different in length
        len_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
        if len_ratio < 0.4:  # Stricter ratio
            return 0.05 * len_ratio
            
        # Optimization: Sample for very long sequences
        max_length = 600  # Increased from 500
        if len(seq1) > max_length:
            # Sample from beginning, middle, and end
            sample_size = max_length // 3
            seq1 = seq1[:sample_size] + seq1[len(seq1)//2-sample_size//2:len(seq1)//2+sample_size//2] + seq1[-sample_size:]
            
        if len(seq2) > max_length:
            # Sample from beginning, middle, and end
            sample_size = max_length // 3
            seq2 = seq2[:sample_size] + seq2[len(seq2)//2-sample_size//2:len(seq2)//2+sample_size//2] + seq2[-sample_size:]
            
        # Position weights for conserved regions
        pos_weights1 = np.ones(len(seq1))
        
        # Apply conserved region weights if available
        if self.conserved_weights:
            for i in range(len(seq1)):
                if i in self.conserved_weights:
                    pos_weights1[i] = self.conserved_weights[i]
            
        # Get k-mers from seq1 with position information
        kmers1 = {}
        for i in range(len(seq1) - k + 1):
            kmer = tuple(seq1[i:i+k])
            
            # Skip k-mers with too many ambiguous nucleotides
            ambiguous_count = sum(1 for id in kmer if id >= 5)
            if ambiguous_count > k // 3:
                continue
                
            weight = pos_weights1[i]
            
            if kmer not in kmers1:
                kmers1[kmer] = []
            kmers1[kmer].append((i, weight))
        
        # Count matches in seq2
        weighted_matches = 0
        total_weights = 0
        
        # Optimization: Only check a subset of positions for large sequences
        step = 1 if len(seq2) < 1000 else max(1, len(seq2) // 600)  # Increased sampling
        
        for i in range(0, len(seq2) - k + 1, step):
            kmer = tuple(seq2[i:i+k])
            
            # Skip k-mers with too many ambiguous nucleotides
            ambiguous_count = sum(1 for id in kmer if id >= 5)
            if ambiguous_count > k // 3:
                continue
                
            if kmer in kmers1:
                # Add weighted match
                avg_weight = sum(w for _, w in kmers1[kmer]) / len(kmers1[kmer])
                weighted_matches += avg_weight
                total_weights += 1
        
        # Adjust for step size
        if step > 1:
            weighted_matches = weighted_matches * step
            total_weights = total_weights * step
            
        # Normalize by total possible matches
        total_possible = min(len(seq1), len(seq2)) - k + 1
        if total_possible <= 0 or total_weights == 0:
            return 0
            
        return weighted_matches / total_weights

    def _variable_kmer_comparison(self, seq1, seq2):
        """
        Compare sequences using variable k-mer sizes for different regions
        particularly suited for 16S rRNA with conserved and variable regions
        """
        # Skip if sequences are too short
        if len(seq1) < 10 or len(seq2) < 10:
            return 0
            
        # Fast check - if sequences are very different in length
        len_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
        if len_ratio < 0.4:
            return 0.05 * len_ratio
            
        # Define variable k-mer sizes
        # Use smaller k-mers for variable regions, larger for conserved
        small_k = 5  # For variable regions
        medium_k = 7  # For mixed regions
        large_k = 9  # For conserved regions
        
        # Determine if we're likely dealing with 16S rRNA based on sequence length
        is_16s_likely = 1000 < max(len(seq1), len(seq2)) < 2000
        
        # Map to store the best k-mer size for each position
        best_k_sizes = {}
        
        if is_16s_likely and self.conserved_weights:
            # Use conserved region information to assign k-mer sizes
            for i in range(max(len(seq1), len(seq2))):
                if i in self.conserved_weights:
                    weight = self.conserved_weights[i]
                    if weight > 1.3:  # Highly conserved
                        best_k_sizes[i] = large_k
                    elif weight > 1.0:  # Moderately conserved
                        best_k_sizes[i] = medium_k
                    else:  # Variable region
                        best_k_sizes[i] = small_k
                else:
                    best_k_sizes[i] = medium_k
        else:
            # If conserved weights aren't available, use a sliding window approach
            window_size = 50
            for seq in [seq1, seq2]:
                for i in range(0, len(seq) - window_size, window_size // 2):
                    # Calculate variability in this window
                    window = seq[i:i+window_size]
                    unique_bases = len(set(window))
                    variability = unique_bases / len(window)
                    
                    # Assign k-mer size based on variability
                    k_size = small_k
                    if variability < 0.4:  # Low variability (conserved)
                        k_size = large_k
                    elif variability < 0.6:  # Medium variability
                        k_size = medium_k
                        
                    # Set this k-mer size for all positions in window
                    for j in range(i, min(i+window_size, len(seq))):
                        if j not in best_k_sizes:
                            best_k_sizes[j] = k_size
        
        # Count matches using variable k-mer sizes
        total_matches = 0
        total_positions = 0
        
        # Sample positions for very long sequences
        max_positions = 500
        sample_step = max(1, min(len(seq1), len(seq2)) // max_positions)
        
        for i in range(0, min(len(seq1), len(seq2)) - large_k, sample_step):
            k_size = best_k_sizes.get(i, medium_k)
            
            # Get k-mers from both sequences
            if i + k_size <= len(seq1) and i + k_size <= len(seq2):
                kmer1 = tuple(seq1[i:i+k_size])
                kmer2 = tuple(seq2[i:i+k_size])
                
                # Skip k-mers with too many ambiguous nucleotides
                ambiguous1 = sum(1 for id in kmer1 if id >= 5)
                ambiguous2 = sum(1 for id in kmer2 if id >= 5)
                if ambiguous1 > k_size // 3 or ambiguous2 > k_size // 3:
                    continue
                
                # Check for exact match
                match_score = 0
                if kmer1 == kmer2:
                    match_score = 1.0
                else:
                    # Count number of matching positions
                    matches = sum(1 for j in range(k_size) if kmer1[j] == kmer2[j])
                    match_score = matches / k_size
                
                # Weight by position importance if available
                weight = self.conserved_weights.get(i, 1.0) if self.conserved_weights else 1.0
                
                total_matches += match_score * weight
                total_positions += 1
        
        if total_positions == 0:
            return 0
            
        return total_matches / total_positions

    def _compute_conserved_region_weights(self, sequences):
        """
        Analyze sequences to identify conserved regions and assign weights
        Particularly useful for 16S rRNA analysis
        """
        # Skip if too few sequences
        if len(sequences) < 5:
            return {}
            
        # Get maximum sequence length
        max_len = max(len(seq) for seq in sequences)
        
        # Initialize conservation scores
        conservation = np.zeros(max_len)
        counts = np.zeros(max_len)
        
        # Count nucleotide frequencies at each position
        for seq in sequences:
            for i, nt in enumerate(seq):
                if nt < 5:  # Only count standard nucleotides
                    conservation[i] += 1
                counts[i] += 1
        
        # Calculate conservation rate
        conservation_rate = np.zeros(max_len)
        for i in range(max_len):
            if counts[i] > 0:
                conservation_rate[i] = conservation[i] / counts[i]
        
        # Smooth conservation rate with a sliding window
        window_size = 15
        smoothed_rate = np.zeros(max_len)
        for i in range(max_len):
            start = max(0, i - window_size // 2)
            end = min(max_len, i + window_size // 2 + 1)
            window = conservation_rate[start:end]
            if len(window) > 0:
                smoothed_rate[i] = np.mean(window)
        
        # Normalize to weights centered around 1.0
        # High conservation -> higher weight
        weights = {}
        avg_rate = np.mean(smoothed_rate[smoothed_rate > 0])
        for i in range(max_len):
            if smoothed_rate[i] > 0:
                weights[i] = smoothed_rate[i] / avg_rate
        
        # Additionally, boost weights for known 16S conserved regions
        is_16s_likely = 1000 < max_len < 2000
        if is_16s_likely:
            for start, end in CONSERVED_16S_REGIONS:
                for i in range(start, end):
                    if i in weights:
                        weights[i] *= 1.5  # Boost weights for known conserved regions
        
        return weights

    def fit(self, sequences, labels):
        """Optimized training method for faster processing with enhanced accuracy"""
        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # in MB
        
        print("Encoding sequences...")
        self.labels = labels
        self.encoded_seqs = [self._encode_sequence(seq) for seq in tqdm(sequences, desc="Encoding")]
        
        # Create optimized conserved region weights for 16S rRNA
        print("Computing conserved region weights...")
        
        # Detect if we're likely dealing with 16S rRNA based on sequence length distribution
        seq_lengths = [len(seq) for seq in self.encoded_seqs]
        avg_len = sum(seq_lengths) / len(seq_lengths)
        is_16s_likely = 1300 < avg_len < 1900  # 16S rRNA is typically 1400-1800 bp
        
        if is_16s_likely:
            print("Sequences appear to be 16S rRNA - using specialized optimization")
            # Use known conserved regions directly instead of computing from data
            # This is much faster and often more accurate for 16S rRNA
            max_len = max(seq_lengths)
            self.conserved_weights = {}
            
            # Apply weights to known conserved regions
            for start, end in CONSERVED_16S_REGIONS:
                if start < max_len:
                    # Apply higher weight to conserved regions
                    for pos in range(start, min(end+1, max_len)):
                        self.conserved_weights[pos] = 1.5
                        
            # Create position-specific weights for 16S rRNA
            self.position_weights = np.ones(max_len)
            for i, weight in self.conserved_weights.items():
                if i < max_len:
                    self.position_weights[i] = weight
        else:
            # For non-16S sequences, calculate weights from the sequences
            # Sample sequences for faster processing
            sample_size = min(500, len(self.encoded_seqs))
            sampled_seqs = random.sample(self.encoded_seqs, sample_size)
            self.conserved_weights = self._compute_conserved_region_weights(sampled_seqs)
            
            # Create position-specific weights
            max_len = max(len(seq) for seq in self.encoded_seqs)
            self.position_weights = np.ones(max_len)
            for i, weight in self.conserved_weights.items():
                if i < max_len:
                    self.position_weights[i] = weight
        
        print("Packing sequences...")
        # Only pack sequences that we'll actually use for bit operations
        # This reduces memory usage significantly
        self.packed_seqs = [self._pack_ids_into_words(e) for e in tqdm(self.encoded_seqs, desc="Packing")]
        
        print("Computing simhash sketches...")
        self.simhash_sketches = [self._compute_simhash(e) for e in tqdm(self.encoded_seqs, desc="Computing hashes")]
        
        # We no longer compute k-mer profiles for all sequences, only as needed
        # This saves a lot of memory and processing time
        self.kmer_profiles = []
        
        # Build LSH index
        self._build_lsh_index()
        
        # Track performance metrics
        self.train_time = time.time() - start_time
        self.memory_mb = psutil.Process().memory_info().rss / 1024 / 1024 - memory_start
        
        print(f"Training completed in {self.train_time:.2f} seconds")
        print(f"Memory usage: {self.memory_mb:.2f} MB")
        
        return self

    def predict(self, sequences):
        """Optimized prediction with improved accuracy for 16S rRNA classification"""
        start_time = time.time()
        
        preds = []
        print("Predicting...")
        
        # Determine if we're likely dealing with 16S rRNA
        avg_length = sum(len(seq) for seq in self.encoded_seqs) / len(self.encoded_seqs)
        is_16s_likely = 1300 < avg_length < 1900
        
        if is_16s_likely:
            print("Optimizing for 16S rRNA classification")
        
        # Prepare class weights for balanced predictions
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        class_weights = {label: total_samples / (len(class_counts) * count) for label, count in class_counts.items()}
        
        # Create a list for 16S genus-specific signature k-mers
        genus_signatures = {}
        
        if is_16s_likely:
            # For each genus, find the most discriminative regions/k-mers
            for genus in set(self.labels):
                # Get all sequences for this genus
                genus_indices = [i for i, label in enumerate(self.labels) if label == genus]
                
                if len(genus_indices) >= 5:  # Need enough samples
                    # Sample some sequences (for efficiency)
                    sample_size = min(30, len(genus_indices))
                    sample_indices = random.sample(genus_indices, sample_size)
                    
                    # Find k-mers that appear in most sequences of this genus
                    kmer_counts = {}
                    k = 8  # Slightly larger k-mer size for specificity
                    
                    for idx in sample_indices:
                        seq = self.encoded_seqs[idx]
                        # Focus on conserved regions for 16S
                        for start, end in CONSERVED_16S_REGIONS:
                            if start < len(seq) and end > 0:
                                region_start = max(0, start - 10)  # Include flanking regions
                                region_end = min(len(seq), end + 10)
                                
                                # Extract k-mers from this region
                                for i in range(region_start, region_end - k + 1):
                                    kmer = tuple(seq[i:i+k])
                                    # Skip k-mers with ambiguous nucleotides
                                    if not any(nuc >= 5 for nuc in kmer):
                                        if kmer not in kmer_counts:
                                            kmer_counts[kmer] = 0
                                        kmer_counts[kmer] += 1
                    
                    # Keep k-mers that appear in at least 70% of the samples
                    threshold = 0.7 * sample_size
                    genus_kmers = [kmer for kmer, count in kmer_counts.items() if count >= threshold]
                    
                    # Check these k-mers in other genera to ensure specificity
                    other_indices = [i for i, label in enumerate(self.labels) if label != genus]
                    other_sample = random.sample(other_indices, min(30, len(other_indices)))
                    
                    specific_kmers = []
                    for kmer in genus_kmers:
                        other_count = 0
                        for idx in other_sample:
                            other_seq = self.encoded_seqs[idx]
                            # Check if this k-mer appears in the other sequence
                            for i in range(len(other_seq) - k + 1):
                                if tuple(other_seq[i:i+k]) == kmer:
                                    other_count += 1
                                    break
                        
                        # If k-mer appears in fewer than 30% of other genera, it's specific
                        if other_count < 0.3 * len(other_sample):
                            specific_kmers.append(kmer)
                    
                    # Store signature k-mers for this genus
                    if specific_kmers:
                        genus_signatures[genus] = specific_kmers
        
        # Process each query sequence
        for seq in tqdm(sequences, desc="Predicting"):
            # Encode and hash the sequence
            enc = self._encode_sequence(seq)
            sketch = self._compute_simhash(enc)
            
            # First use LSH to get candidate matches
            candidates = self._query_lsh(sketch)
            
            # If too few candidates, expand search
            if len(candidates) < 10:  # Increased from 5 to 10
                # Calculate Hamming distances to all sequences (faster than running similarity on all)
                hamming_dists = []
                for i, ref_sketch in enumerate(self.simhash_sketches):
                    # Fast Hamming distance calculation using XOR and popcount
                    hamming = bin(sketch ^ ref_sketch).count('1')
                    hamming_dists.append((i, hamming))
                
                # Sort by Hamming distance and take top candidates
                hamming_dists.sort(key=lambda x: x[1])
                candidates = [i for i, _ in hamming_dists[:min(self.max_candidates*2, len(hamming_dists))]]
            
            # Ensure we have candidates from each class if possible
            class_candidates = {}
            for i in candidates:
                cls = self.labels[i]
                if cls not in class_candidates:
                    class_candidates[cls] = []
                class_candidates[cls].append(i)
            
            # Make sure we have at least one candidate from each class
            missing_classes = set(class_counts.keys()) - set(class_candidates.keys())
            for cls in missing_classes:
                cls_indices = [i for i, label in enumerate(self.labels) if label == cls]
                if cls_indices:
                    # Add closest match by Hamming distance
                    cls_dists = [(i, bin(sketch ^ self.simhash_sketches[i]).count('1')) for i in cls_indices]
                    cls_dists.sort(key=lambda x: x[1])
                    candidates.append(cls_dists[0][0])
            
            # Limit to max_candidates
            if len(candidates) > self.max_candidates:
                # Filter by Hamming distance for efficiency
                hamming_distances = [(i, bin(sketch ^ self.simhash_sketches[i]).count('1')) 
                                     for i in candidates]
                hamming_distances.sort(key=lambda x: x[1])
                candidates = [i for i, _ in hamming_distances[:self.max_candidates]]
            
            # Compute similarities for candidates
            similarities = []
            for i in candidates:
                sim = self._similarity_improved(enc, self.encoded_seqs[i])
                similarities.append((i, sim))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Apply signature k-mer matching for 16S rRNA if available
            signature_votes = defaultdict(int)
            if is_16s_likely and genus_signatures:
                for genus, kmers in genus_signatures.items():
                    matches = 0
                    for kmer in kmers:
                        # Check if this k-mer appears in the query sequence
                        k = len(kmer)
                        for i in range(len(enc) - k + 1):
                            if tuple(enc[i:i+k]) == kmer:
                                matches += 1
                                break
                    
                    # If enough signature k-mers match, add votes for this genus
                    if matches >= max(1, len(kmers) * 0.3):
                        signature_votes[genus] = matches / len(kmers)
            
            # Get top-k nearest neighbors for weighted voting
            top_k = similarities[:min(self.num_neighbors, len(similarities))]
            
            # Apply exponentially weighted voting with class balancing
            similarity_votes = defaultdict(float)
            for idx, sim in top_k:
                # Apply exponential weighting and class balancing
                label = self.labels[idx]
                weight = class_weights.get(label, 1.0)
                
                # Higher exponent for better separation between classes
                similarity_votes[label] += np.exp(5 * sim) * weight
            
            # Combine votes from different sources
            final_votes = defaultdict(float)
            
            # Add similarity-based votes (primary method)
            for label, vote in similarity_votes.items():
                final_votes[label] += vote
            
            # Add signature k-mer votes (for 16S rRNA)
            for label, vote in signature_votes.items():
                # Signature matches are given high weight
                final_votes[label] += vote * 30  # Significant boost for signature matches
            
            # Pick the class with highest final vote
            if final_votes:
                pred_label = max(final_votes.items(), key=lambda x: x[1])[0]
                preds.append(pred_label)
            else:
                # Fallback: use most common class in training
                preds.append(class_counts.most_common(1)[0][0])
        
        self.prediction_time = time.time() - start_time
        print(f"Prediction completed in {self.prediction_time:.2f} seconds")
                
        return preds

# Helper to load FASTA sequences from directory
def load_fasta_dir(input_dir, sample_percentage=100, balanced=False):
    """
    Load FASTA sequences from a directory with improved sampling
    
    Parameters:
    -----------
    input_dir : str
        Directory containing FASTA files
    sample_percentage : int
        Percentage of sequences to sample (1-100)
    balanced : bool
        If True, ensures balanced class distribution
    
    Returns:
    --------
    list, list
        Sequences and their corresponding labels
    """
    # Recursively find all .fasta files in the input directory and its subdirectories
    files = []
    print(f"Searching for FASTA files in {input_dir}...")
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.fasta'):
                files.append(os.path.join(root, filename))
    
    if not files:
        print(f"No .fasta files found in {input_dir} or its subdirectories")
        return [], []
    
    # First, collect all sequences organized by class
    all_seqs_by_class = defaultdict(list)
    all_class_counts = {}
    
    print(f"Loading {len(files)} FASTA files...")
    for f in tqdm(files, desc="Loading FASTA files"):
        # Use the parent directory name as the label
        label = os.path.basename(os.path.dirname(f))
        
        # Parse all sequences from the FASTA file using BioPython
        for record in SeqIO.parse(f, "fasta"):
            all_seqs_by_class[label].append(str(record.seq))
    
    # Count sequences per class
    for label, seqs in all_seqs_by_class.items():
        all_class_counts[label] = len(seqs)
    
    # Now sample sequences
    seqs = []
    labels = []
    
    if balanced:
        # For balanced sampling, determine the target count per class
        if sample_percentage < 100:
            # Calculate sample size based on the smallest class
            min_class_size = min(all_class_counts.values())
            target_size = max(1, int(min_class_size * sample_percentage / 100))
        else:
            # Use all sequences from the smallest class
            target_size = min(all_class_counts.values())
            
        print(f"Balanced sampling: {target_size} sequences per class")
        
        # Sample equally from each class
        for label, class_seqs in all_seqs_by_class.items():
            if len(class_seqs) <= target_size:
                # Use all sequences if there are fewer than target
                sampled = class_seqs
            else:
                # Random sample without replacement
                sampled = random.sample(class_seqs, target_size)
            
            seqs.extend(sampled)
            labels.extend([label] * len(sampled))
    else:
        # For unbalanced sampling, just take the requested percentage
        if sample_percentage < 100:
            for label, class_seqs in all_seqs_by_class.items():
                # Calculate sample size for this class
                target_size = max(1, int(len(class_seqs) * sample_percentage / 100))
                # Random sample without replacement
                sampled = random.sample(class_seqs, target_size)
                
                seqs.extend(sampled)
                labels.extend([label] * len(sampled))
        else:
            # Use all sequences
            for label, class_seqs in all_seqs_by_class.items():
                seqs.extend(class_seqs)
                labels.extend([label] * len(class_seqs))
    
    # Report class distribution
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"Loaded {len(seqs)} sequences from {len(set(labels))} classes")
    print("Class distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count} sequences")
    
    return seqs, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastEyeBLAST DNA classifier')
    parser.add_argument('--input_dir', required=True, help='Directory of FASTA files')
    parser.add_argument('--sample_percentage', type=int, default=100)
    parser.add_argument('--balanced', action='store_true', help='Balance class distribution')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed_length', type=int, default=DEFAULT_SEED_LENGTH)
    parser.add_argument('--band_width', type=int, default=DEFAULT_BAND_WIDTH)
    parser.add_argument('--k', type=int, default=DEFAULT_NUM_NEIGHBORS)
    parser.add_argument('--max_candidates', type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    seqs, labels = load_fasta_dir(args.input_dir, args.sample_percentage, args.balanced)
    indices = list(range(len(seqs)))
    random.shuffle(indices)
    split = int(len(seqs)*(1-args.test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train = [seqs[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [seqs[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    clf = FastEyeBLASTDNA(seed_length=args.seed_length,
                          band_width=args.band_width,
                          num_neighbors=args.k,
                          max_candidates=args.max_candidates)
    print('Training...')
    clf.fit(X_train, y_train)
    print('Predicting...')
    preds = clf.predict(X_test)

    acc = sum(1 for a, b in zip(preds, y_test) if a == b) / len(y_test)
    print(f'Accuracy: {acc:.4f}')
