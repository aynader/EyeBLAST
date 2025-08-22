#!/usr/bin/env python3
"""
compare_similarity_scores_v3.py

Compute and compare pairwise similarity matrices for Fast_EyeBLAST and Needleman-Wunsch on a sampled subset
of IVT-encoded scanpath data or AOI-encoded scanpath data. Analyze matrices using Pearson and Spearman correlations,
visualize results, and save a detailed analysis report.

New in v3: Multiple normalization options including raw scores, min-max, z-score, rank-based, 
quantile, and correlation-preserving normalization.

Usage:
    python compare_similarity_scores_v3.py \
        --input_dir /path/to/token_jsons_or_encoded_sequences \
        --sample_percentage 20 \
        --k 5 \
        --nw_nj true \
        --normalization_mode all \
        --output_dir Results_v3
"""

import os
import sys
import json
import glob
import random
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, rankdata, norm
from tqdm import tqdm
from datetime import datetime

# Import IVT implementations
from Needleman_Wunsch_IVT import compute_similarity_matrix as nw_compute_similarity_matrix_ivt
from Fast_EyeBLAST_IVT import FastEyeBLAST as FastEyeBLAST_IVT

# Import AOI implementations
from Needleman_Wunsch_AOI import compute_similarity_matrix as nw_compute_similarity_matrix_aoi, load_and_prepare_data as load_aoi_data
from Fast_EyeBLAST_AOI import FastEyeBLAST_AOI

def setup_logging(output_dir):
    """
    Setup logging to both console and file.

    Parameters:
    -----------
    output_dir : str
        Directory where run.log will be created.

    Returns:
    --------
    original_stdout : object
        Original stdout to restore later.
    """
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

def generate_run_name(args):
    """
    Generate a descriptive run name based on parameters.

    Format:
        YYYYMMDD_HHMMSS_<dataset>_sample<sample_pct>_k<k>_nw<true/false>_<mode>_norm<normalization>

    Parameters:
    -----------
    args : Namespace
        Command-line arguments.

    Returns:
    --------
    run_id : str
        Generated run identifier.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = os.path.basename(os.path.normpath(args.input_dir))
    sample_pct = args.sample_percentage
    k_val = args.k
    nw_flag = args.nw_nj.lower()
    mode = "AOI" if os.path.exists(os.path.join(args.input_dir, "encoded_sequences.json")) else "IVT"
    norm_mode = args.normalization_mode
    run_id = f"{timestamp}_{dataset}_sample{sample_pct}_k{k_val}_nw{nw_flag}_{mode}_norm{norm_mode}"
    return run_id

def load_scanpaths_ivt(input_dir, sample_percentage):
    """
    Load tokenized scanpaths (lists of strings) from JSON files in input_dir (IVT).
    Sample a percentage of files uniformly at random.

    Parameters:
    -----------
    input_dir : str
        Directory containing JSON files, each a list of tokens (strings).
    sample_percentage : int
        Percentage (1-100) of files to load.

    Returns:
    --------
    scanpaths : list of list of str
        Loaded token lists.
    """
    all_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not all_files:
        logging.error(f"No JSON files found in directory: {input_dir}")
        sys.exit(1)

    N = len(all_files)
    if sample_percentage < 100:
        k = int(N * sample_percentage / 100)
        random.seed(42)
        sampled = random.sample(all_files, k)
    else:
        sampled = all_files

    scanpaths = []
    logging.info(f"Loading {len(sampled)} IVT scanpath files ({sample_percentage}% of {N})...")
    for fp in tqdm(sampled, desc="Reading IVT JSON"):
        with open(fp, 'r') as f:
            tokens = json.load(f)
        scanpaths.append(tokens)

    return scanpaths

def compute_fb_similarity_matrix_ivt(scanpaths, seed_length=3, band_width=5, num_neighbors=5,
                                     max_candidates=20, use_duration_bins=True, duration_bins=8,
                                     use_parallel=False, num_processes=None):
    """
    Compute pairwise Fast_EyeBLAST similarity matrix for IVT scanpaths.

    Parameters:
    -----------
    scanpaths : list of list of str
        Token lists.
    seed_length, band_width, num_neighbors, max_candidates, use_duration_bins, duration_bins, use_parallel, num_processes : parameters for FastEyeBLAST_IVT

    Returns:
    --------
    fb_mat : np.ndarray, shape (n, n)
        Symmetric similarity matrix with values in [0,1].
    """
    n = len(scanpaths)
    labels = [0] * n  # dummy labels

    fb = FastEyeBLAST_IVT(
        seed_length=seed_length,
        band_width=band_width,
        num_neighbors=num_neighbors,
        max_candidates=max_candidates,
        use_duration_bins=use_duration_bins,
        duration_bins=duration_bins,
        use_parallel=use_parallel,
        num_processes=num_processes
    )

    logging.info("Fitting Fast_EyeBLAST_IVT (building indices, encoding all scanpaths)...")
    fb.fit(scanpaths, labels)
    encoded = fb.encoded_scanpaths

    fb_mat = np.zeros((n, n), dtype=np.float32)
    logging.info("Computing Fast_EyeBLAST_IVT pairwise similarities...")
    for i in tqdm(range(n), desc="FB_IVT: outer loop"):
        fb_mat[i, i] = 1.0
        seq_i = encoded[i]
        for j in range(i + 1, n):
            seq_j = encoded[j]
            sim = fb._similarity_score(seq_i, seq_j)
            fb_mat[i, j] = sim
            fb_mat[j, i] = sim

    return fb_mat

def compute_nw_similarity_matrix_ivt(scanpaths, use_numba=True):
    """
    Compute Needleman-Wunsch similarity matrix (n x n) for IVT scanpaths.

    Parameters:
    -----------
    scanpaths : list of list of str
        Token lists.
    use_numba : bool
        Whether to use Numba JIT.

    Returns:
    --------
    nw_mat : np.ndarray, shape (n, n)
        Symmetric similarity matrix with values in [0,1].
    """
    logging.info("Computing Needleman-Wunsch_IVT similarity matrix (may be slow)...")
    nw_mat = nw_compute_similarity_matrix_ivt(scanpaths, n_jobs=-1, use_numba=use_numba)
    return nw_mat

def compute_fb_similarity_matrix_aoi(sequences, seed_length=3, band_width=5, num_neighbors=5,
                                     max_candidates=20, use_parallel=False, num_processes=None):
    """
    Compute pairwise Fast_EyeBLAST similarity matrix for AOI scanpaths.

    Parameters:
    -----------
    sequences : list of str
        AOI character sequences.
    seed_length, band_width, num_neighbors, max_candidates, use_parallel, num_processes : parameters for FastEyeBLAST_AOI

    Returns:
    --------
    fb_mat : np.ndarray, shape (n, n)
        Symmetric similarity matrix with values in [0,1].
    """
    n = len(sequences)
    labels = [0] * n  # dummy labels

    fb = FastEyeBLAST_AOI(
        seed_length=seed_length,
        band_width=band_width,
        num_neighbors=num_neighbors,
        max_candidates=max_candidates,
        use_parallel=use_parallel,
        num_processes=num_processes
    )

    logging.info("Fitting Fast_EyeBLAST_AOI (encoding all AOI scanpaths)...")
    fb.fit(sequences, labels)
    encoded = fb.encoded_scanpaths

    fb_mat = np.zeros((n, n), dtype=np.float32)
    logging.info("Computing Fast_EyeBLAST_AOI pairwise similarities...")
    for i in tqdm(range(n), desc="FB_AOI: outer loop"):
        fb_mat[i, i] = 1.0
        seq_i = encoded[i]
        for j in range(i + 1, n):
            seq_j = encoded[j]
            sim = fb._similarity_score(seq_i, seq_j)
            fb_mat[i, j] = sim
            fb_mat[j, i] = sim

    return fb_mat

def compute_nw_similarity_matrix_aoi(sequences, use_numba=True):
    """
    Compute Needleman-Wunsch similarity matrix (n x n) for AOI scanpaths.

    Parameters:
    -----------
    sequences : list of str
        AOI character sequences.
    use_numba : bool
        Whether to use Numba JIT.

    Returns:
    --------
    nw_mat : np.ndarray, shape (n, n)
        Symmetric similarity matrix with values in [0,1].
    """
    logging.info("Computing Needleman-Wunsch_AOI similarity matrix (may be slow)...")
    nw_mat = nw_compute_similarity_matrix_aoi(sequences,
                                              n_jobs=-1,
                                              use_numba=use_numba,
                                              max_seq_length=5000,
                                              band_width=100,
                                              batch_size=None)
    return nw_mat

def normalize_matrix_none(mat):
    """
    No normalization - return raw scores.

    Parameters:
    -----------
    mat : np.ndarray
        Input similarity matrix.

    Returns:
    --------
    mat : np.ndarray
        Unchanged matrix (raw scores).
    """
    return mat.copy()

def normalize_matrix_minmax(mat):
    """
    Linearly rescale a matrix to the [0,1] range based on its min and max values.

    Parameters:
    -----------
    mat : np.ndarray
        Input similarity matrix.

    Returns:
    --------
    mat_norm : np.ndarray
        Rescaled matrix.
    """
    mn = mat.min()
    mx = mat.max()
    if mx > mn:
        mat_norm = (mat - mn) / (mx - mn)
    else:
        mat_norm = np.zeros_like(mat, dtype=np.float32)
    return mat_norm

def normalize_matrix_zscore(mat):
    """
    Z-score normalization - standardize to mean=0, std=1.

    Parameters:
    -----------
    mat : np.ndarray
        Input similarity matrix.

    Returns:
    --------
    mat_norm : np.ndarray
        Z-score normalized matrix.
    """
    mean_val = mat.mean()
    std_val = mat.std()
    if std_val > 0:
        mat_norm = (mat - mean_val) / std_val
    else:
        mat_norm = mat - mean_val  # If std=0, just center
    return mat_norm

def normalize_matrix_rank(mat):
    """
    Rank-based normalization - convert to ranks and normalize to [0,1].

    Parameters:
    -----------
    mat : np.ndarray
        Input similarity matrix.

    Returns:
    --------
    mat_norm : np.ndarray
        Rank-normalized matrix.
    """
    # Flatten matrix, get ranks, reshape back
    flat_vals = mat.flatten()
    ranks = rankdata(flat_vals, method="average")
    # Normalize ranks to [0,1]
    min_rank = ranks.min()
    max_rank = ranks.max()
    if max_rank > min_rank:
        ranks_norm = (ranks - min_rank) / (max_rank - min_rank)
    else:
        ranks_norm = np.zeros_like(ranks)
    mat_norm = ranks_norm.reshape(mat.shape).astype(np.float32)
    return mat_norm

def normalize_matrix_quantile(mat):
    """
    Quantile normalization - map to uniform distribution on [0,1].

    Parameters:
    -----------
    mat : np.ndarray
        Input similarity matrix.

    Returns:
    --------
    mat_norm : np.ndarray
        Quantile-normalized matrix.
    """
    # Flatten matrix to get all values
    flat_vals = mat.flatten()
    # Get sorted indices and ranks
    sorted_indices = np.argsort(flat_vals)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(flat_vals))
    
    # Map ranks to uniform distribution [0,1]
    uniform_vals = ranks / (len(flat_vals) - 1)
    mat_norm = uniform_vals.reshape(mat.shape).astype(np.float32)
    return mat_norm

def normalize_matrix_correlation_preserving(nw_mat, fb_mat):
    """
    Correlation-preserving normalization using geometric mean of score ranges.

    Parameters:
    -----------
    nw_mat : np.ndarray
        Needleman-Wunsch matrix.
    fb_mat : np.ndarray
        Fast_EyeBLAST matrix.

    Returns:
    --------
    nw_norm, fb_norm : tuple of np.ndarray
        Normalized matrices with common scale.
    """
    # Get ranges
    nw_range = nw_mat.max() - nw_mat.min()
    fb_range = fb_mat.max() - fb_mat.min()
    
    # Use geometric mean of ranges as common scale
    if nw_range > 0 and fb_range > 0:
        common_scale = np.sqrt(nw_range * fb_range)
        
        # Center and scale to common range
        nw_centered = nw_mat - nw_mat.min()
        fb_centered = fb_mat - fb_mat.min()
        
        nw_norm = nw_centered * (common_scale / nw_range) if nw_range > 0 else nw_centered
        fb_norm = fb_centered * (common_scale / fb_range) if fb_range > 0 else fb_centered
    else:
        # If either range is 0, just center
        nw_norm = nw_mat - nw_mat.min()
        fb_norm = fb_mat - fb_mat.min()
    
    return nw_norm.astype(np.float32), fb_norm.astype(np.float32)

def apply_normalization(nw_mat_raw, fb_mat_raw, norm_method):
    """
    Apply specified normalization method to both matrices.

    Parameters:
    -----------
    nw_mat_raw : np.ndarray
        Raw Needleman-Wunsch matrix.
    fb_mat_raw : np.ndarray
        Raw Fast_EyeBLAST matrix.
    norm_method : str
        Normalization method: 'none', 'minmax', 'zscore', 'rank', 'quantile', 'correlation_preserving'.

    Returns:
    --------
    nw_mat_norm, fb_mat_norm : tuple of np.ndarray
        Normalized matrices.
    """
    if norm_method == "none":
        return normalize_matrix_none(nw_mat_raw), normalize_matrix_none(fb_mat_raw)
    elif norm_method == "minmax":
        return normalize_matrix_minmax(nw_mat_raw), normalize_matrix_minmax(fb_mat_raw)
    elif norm_method == "zscore":
        return normalize_matrix_zscore(nw_mat_raw), normalize_matrix_zscore(fb_mat_raw)
    elif norm_method == "rank":
        return normalize_matrix_rank(nw_mat_raw), normalize_matrix_rank(fb_mat_raw)
    elif norm_method == "quantile":
        return normalize_matrix_quantile(nw_mat_raw), normalize_matrix_quantile(fb_mat_raw)
    elif norm_method == "correlation_preserving":
        return normalize_matrix_correlation_preserving(nw_mat_raw, fb_mat_raw)
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

def analyze_and_visualize(nw_mat, fb_mat, output_dir, norm_method):
    """
    Analyze correlation between NW and FB matrices and produce visualizations:
      - Scatter plot of pairwise similarities.
      - Pearson and Spearman correlation computation.
      - Heatmaps of NW and FB matrices.
      - Overlaid histograms of similarity distributions.
      - Rank-vs-Rank scatter to visualize Spearman agreement.

    Saves plots to output_dir/norm_method/. Returns a dictionary with metrics and file paths.

    Parameters:
    -----------
    nw_mat : np.ndarray
        Normalized Needleman-Wunsch matrix.
    fb_mat : np.ndarray
        Normalized Fast_EyeBLAST matrix.
    output_dir : str
        Base output directory.
    norm_method : str
        Normalization method used.

    Returns:
    --------
    stats : dict
        Statistics and file paths.
    """
    assert nw_mat.shape == fb_mat.shape, "Matrices must have the same shape"
    n = nw_mat.shape[0]

    # Create subdirectory for this normalization method
    norm_dir = os.path.join(output_dir, f"normalization_{norm_method}")
    os.makedirs(norm_dir, exist_ok=True)

    iu = np.triu_indices(n, k=1)
    nw_vals = nw_mat[iu]
    fb_vals = fb_mat[iu]

    # Compute Pearson and Spearman correlations
    pearson_corr, pearson_p = pearsonr(nw_vals, fb_vals)
    spearman_corr, spearman_p = spearmanr(nw_vals, fb_vals)
    logging.info(f"[{norm_method}] Pearson correlation: {pearson_corr:.4f} (p-value = {pearson_p:.4e})")
    logging.info(f"[{norm_method}] Spearman correlation: {spearman_corr:.4f} (p-value = {spearman_p:.4e})")

    # Determine plot limits based on normalization method
    if norm_method == "minmax" or norm_method == "rank" or norm_method == "quantile":
        plot_lims = [0.0, 1.0]
        use_auto_lims = False
        use_equal_axes = True
    elif norm_method == "none":
        # For raw scores, use individual axis limits and no diagonal line if scales are very different
        plot_lims = None
        use_auto_lims = True
        use_equal_axes = False
    else:
        # For zscore, correlation_preserving, use auto limits with equal axes
        plot_lims = None
        use_auto_lims = True
        use_equal_axes = True

    # Scatter plot: FB vs NW
    scatter_path = os.path.join(norm_dir, "compare_similarity_scatter.png")
    plt.figure(figsize=(8, 8))
    plt.scatter(nw_vals, fb_vals, alpha=0.3, s=10, edgecolor='none')

    if not use_auto_lims:
        plt.plot(plot_lims, plot_lims, 'r--', label="y = x")
        plt.xlim(plot_lims)
        plt.ylim(plot_lims)
    else:
        if use_equal_axes:
            # Use equal axes and draw meaningful diagonal
            min_val = min(nw_vals.min(), fb_vals.min())
            max_val = max(nw_vals.max(), fb_vals.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
        else:
            # For raw scores with different scales, use separate axis limits
            # Check if scales are very different (more than 10x difference in range)
            nw_range = nw_vals.max() - nw_vals.min()
            fb_range = fb_vals.max() - fb_vals.min()
            
            if nw_range > 0 and fb_range > 0:
                scale_ratio = max(nw_range, fb_range) / min(nw_range, fb_range)
                if scale_ratio > 10:
                    # Very different scales - use individual axis limits, no diagonal
                    plt.xlim(nw_vals.min() * 1.05 - nw_vals.max() * 0.05, 
                            nw_vals.max() * 1.05 - nw_vals.min() * 0.05)
                    plt.ylim(fb_vals.min() * 1.05 - fb_vals.max() * 0.05, 
                            fb_vals.max() * 1.05 - fb_vals.min() * 0.05)
                    # Add text note instead of misleading diagonal
                    plt.text(0.05, 0.95, 'Note: Different scales\n(raw scores)', 
                            transform=plt.gca().transAxes, fontsize=10, 
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                else:
                    # Similar scales - can use diagonal
                    min_val = min(nw_vals.min(), fb_vals.min())
                    max_val = max(nw_vals.max(), fb_vals.max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
            else:
                # One or both ranges are zero - use auto limits
                pass

    plt.xlabel("Needleman-Wunsch Similarity")
    plt.ylabel("Fast_EyeBLAST Similarity")
    plt.title(f"Pairwise Similarities Scatter ({norm_method})\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    logging.info(f"[{norm_method}] Saved scatter plot to {scatter_path}")

    # Heatmap: NW matrix
    nw_heatmap_path = os.path.join(norm_dir, "compare_similarity_nw_heatmap.png")
    plt.figure(figsize=(8, 6))
    if not use_auto_lims:
        sns.heatmap(nw_mat, vmin=plot_lims[0], vmax=plot_lims[1], cmap="viridis")
    else:
        sns.heatmap(nw_mat, cmap="viridis")
    plt.title(f"Needleman-Wunsch Similarity Matrix ({norm_method})")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig(nw_heatmap_path, dpi=300)
    plt.close()
    logging.info(f"[{norm_method}] Saved NW heatmap to {nw_heatmap_path}")

    # Heatmap: FB matrix
    fb_heatmap_path = os.path.join(norm_dir, "compare_similarity_fb_heatmap.png")
    plt.figure(figsize=(8, 6))
    if not use_auto_lims:
        sns.heatmap(fb_mat, vmin=plot_lims[0], vmax=plot_lims[1], cmap="viridis")
    else:
        sns.heatmap(fb_mat, cmap="viridis")
    plt.title(f"Fast_EyeBLAST Similarity Matrix ({norm_method})")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig(fb_heatmap_path, dpi=300)
    plt.close()
    logging.info(f"[{norm_method}] Saved FB heatmap to {fb_heatmap_path}")

    # Overlaid histograms of similarity distributions
    hist_path = os.path.join(norm_dir, "compare_similarity_histogram.png")
    plt.figure(figsize=(8, 6))
    if not use_auto_lims:
        bins = np.linspace(plot_lims[0], plot_lims[1], 50)
    else:
        bins = 50
    plt.hist(nw_vals, bins=bins, alpha=0.5, label="NW", density=True)
    plt.hist(fb_vals, bins=bins, alpha=0.5, label="FB", density=True)
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.title(f"Distribution of Pairwise Similarities ({norm_method})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    logging.info(f"[{norm_method}] Saved histogram plot to {hist_path}")

    # Rank-vs-Rank plot for Spearman visualization
    # Convert similarities to ranks (1 = smallest, M = largest)
    nw_ranks = rankdata(nw_vals, method="average")
    fb_ranks = rankdata(fb_vals, method="average")
    M = len(nw_vals)  # number of distinct pairs

    rankplot_path = os.path.join(norm_dir, "compare_similarity_rank_scatter.png")
    plt.figure(figsize=(6, 6))
    plt.scatter(nw_ranks, fb_ranks, s=5, alpha=0.3, edgecolor='none')
    lims_rank = [0, M]
    plt.plot(lims_rank, lims_rank, 'r--', linewidth=1)
    plt.xlabel("Rank in NW (1 = lowest similarity)")
    plt.ylabel("Rank in FB (1 = lowest similarity)")
    plt.title(f"Rank-vs-Rank Comparison ({norm_method})\nSpearman: {spearman_corr:.3f}")
    plt.xlim(0, M)
    plt.ylim(0, M)
    plt.tight_layout()
    plt.savefig(rankplot_path, dpi=300)
    plt.close()
    logging.info(f"[{norm_method}] Saved rank-vs-rank scatter to {rankplot_path}")

    return {
        "normalization_method": norm_method,
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "scatter_path": scatter_path,
        "nw_heatmap_path": nw_heatmap_path,
        "fb_heatmap_path": fb_heatmap_path,
        "hist_path": hist_path,
        "rankplot_path": rankplot_path,
        "nw_min": nw_vals.min(),
        "nw_max": nw_vals.max(),
        "nw_mean": nw_vals.mean(),
        "nw_std": nw_vals.std(),
        "fb_min": fb_vals.min(),
        "fb_max": fb_vals.max(),
        "fb_mean": fb_vals.mean(),
        "fb_std": fb_vals.std()
    }

def generate_analysis_report(args, mode, n_samples, all_stats, output_dir):
    """
    Generate a comprehensive analysis report covering all normalization methods.

    Parameters:
    -----------
    args : Namespace
        Command-line arguments.
    mode : str
        "IVT" or "AOI".
    n_samples : int
        Number of scanpaths in the sample.
    all_stats : list of dict
        Statistics for each normalization method.
    output_dir : str
        Directory where analysis_report.log will be saved.
    """
    report_path = os.path.join(output_dir, "analysis_report.log")
    with open(report_path, 'w') as f:
        f.write("Analysis Report - Multiple Normalization Methods\n")
        f.write("================================================\n\n")
        f.write(f"Run Timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode                : {mode}\n")
        f.write(f"Input Directory     : {args.input_dir}\n")
        f.write(f"Sample Percentage   : {args.sample_percentage}%\n")
        f.write(f"Number of Samples   : {n_samples}\n")
        f.write(f"k (FB neighbors)    : {args.k}\n")
        f.write(f"Needleman-Wunsch NJ : {args.nw_nj}\n")
        f.write(f"Normalization Mode  : {args.normalization_mode}\n\n")

        # Summary table of correlations
        f.write("Correlation Summary Across Normalization Methods\n")
        f.write("=================================================\n")
        f.write(f"{'Method':<20} {'Pearson':<10} {'P-value':<12} {'Spearman':<10} {'P-value':<12}\n")
        f.write("-" * 76 + "\n")
        
        for stats in all_stats:
            norm_method = stats["normalization_method"]
            pc, pp = stats["pearson_corr"], stats["pearson_p"]
            sc, sp = stats["spearman_corr"], stats["spearman_p"]
            f.write(f"{norm_method:<20} {pc:<10.4f} {pp:<12.4e} {sc:<10.4f} {sp:<12.4e}\n")
        
        f.write("\n")

        # Detailed analysis for each method
        for stats in all_stats:
            norm_method = stats["normalization_method"]
            f.write(f"Detailed Analysis: {norm_method.upper()} Normalization\n")
            f.write("=" * 50 + "\n")
            
            pc, pp = stats["pearson_corr"], stats["pearson_p"]
            sc, sp = stats["spearman_corr"], stats["spearman_p"]
            
            f.write(f"Pearson Correlation : {pc:.4f} (p-value = {pp:.4e})\n")
            if abs(pc) > 0.7:
                f.write("  Interpretation     : Strong linear relationship between NW and FB similarity scores.\n")
            elif abs(pc) > 0.3:
                f.write("  Interpretation     : Moderate linear relationship between NW and FB similarity scores.\n")
            else:
                f.write("  Interpretation     : Weak linear relationship between NW and FB similarity scores.\n")

            f.write(f"Spearman Correlation: {sc:.4f} (p-value = {sp:.4e})\n")
            if abs(sc) > 0.7:
                f.write("  Interpretation     : Strong monotonic relationship between NW and FB similarity scores.\n")
            elif abs(sc) > 0.3:
                f.write("  Interpretation     : Moderate monotonic relationship between NW and FB similarity scores.\n")
            else:
                f.write("  Interpretation     : Weak monotonic relationship between NW and FB similarity scores.\n")

            # Data distribution statistics
            f.write("\nData Distribution Statistics:\n")
            f.write(f"  NW: min={stats['nw_min']:.4f}, max={stats['nw_max']:.4f}, mean={stats['nw_mean']:.4f}, std={stats['nw_std']:.4f}\n")
            f.write(f"  FB: min={stats['fb_min']:.4f}, max={stats['fb_max']:.4f}, mean={stats['fb_mean']:.4f}, std={stats['fb_std']:.4f}\n")

            f.write("\nVisualizations Saved:\n")
            f.write(f"  - Scatter Plot            : {stats['scatter_path']}\n")
            f.write(f"  - NW Heatmap              : {stats['nw_heatmap_path']}\n")
            f.write(f"  - FB Heatmap              : {stats['fb_heatmap_path']}\n")
            f.write(f"  - Histogram               : {stats['hist_path']}\n")
            f.write(f"  - Rank-vs-Rank Scatter    : {stats['rankplot_path']}\n\n")

        # Comparative analysis
        f.write("Comparative Analysis Across Normalization Methods\n")
        f.write("==================================================\n")
        
        # Find best correlations
        pearson_corrs = [(s["normalization_method"], s["pearson_corr"]) for s in all_stats]
        spearman_corrs = [(s["normalization_method"], s["spearman_corr"]) for s in all_stats]
        
        best_pearson = max(pearson_corrs, key=lambda x: abs(x[1]))
        best_spearman = max(spearman_corrs, key=lambda x: abs(x[1]))
        
        f.write(f"Highest Absolute Pearson Correlation : {best_pearson[0]} ({best_pearson[1]:.4f})\n")
        f.write(f"Highest Absolute Spearman Correlation: {best_spearman[0]} ({best_spearman[1]:.4f})\n\n")
        
        # Normalization method descriptions
        f.write("Normalization Method Descriptions:\n")
        f.write("- none                : Raw similarity scores (no normalization)\n")
        f.write("- minmax              : Linear scaling to [0,1] range\n")
        f.write("- zscore              : Standardization to mean=0, std=1\n")
        f.write("- rank                : Rank-based normalization to [0,1]\n")
        f.write("- quantile            : Quantile normalization to uniform distribution\n")
        f.write("- correlation_preserving : Geometric mean scaling preserving correlation structure\n\n")
        
        f.write("Please inspect the visualizations in each normalization subdirectory for detailed analysis.\n")
    
    logging.info(f"Comprehensive analysis report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare Fast_EyeBLAST and Needleman-Wunsch similarity scores with multiple normalization options (v3)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing IVT token JSON files or AOI encoded_sequences.json"
    )
    parser.add_argument(
        "--sample_percentage",
        type=int,
        default=100,
        help="Percentage of files/sequences to use (1-100)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of neighbors for Fast_EyeBLAST k-NN voting"
    )
    parser.add_argument(
        "--nw_nj",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Use Numba JIT for Needleman-Wunsch (true/false)"
    )
    parser.add_argument(
        "--normalization_mode",
        type=str,
        choices=["none", "minmax", "zscore", "rank", "quantile", "correlation_preserving", "all"],
        default="all",
        help="Normalization method to use, or 'all' to run all methods"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="EyeBLAST/Results_v3",
        help="Base directory to save results and logs"
    )

    args = parser.parse_args()

    # Determine use_numba flag
    use_numba = True if args.nw_nj.lower() == "true" else False

    # Determine if AOI mode or IVT mode
    is_aoi = os.path.exists(os.path.join(args.input_dir, "encoded_sequences.json"))
    mode = "AOI" if is_aoi else "IVT"

    # Create unique output directory with parameters
    run_id = generate_run_name(args)
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging (captures stdout into run.log)
    original_stdout = setup_logging(output_dir)

    # Log run parameters
    logging.info("Starting similarity comparison with the following parameters:")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"  {arg}: {value}")
    logging.info(f"  mode: {mode}")

    # Save run parameters to JSON
    params = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input_dir": args.input_dir,
        "sample_percentage": args.sample_percentage,
        "k": args.k,
        "nw_nj": args.nw_nj,
        "normalization_mode": args.normalization_mode,
        "mode": mode,
        "command": " ".join(["python"] + [os.path.basename(__file__)] +
                             [f"--{k.replace('_', '-')}={v}" for k, v in vars(args).items()])
    }
    params_json_path = os.path.join(output_dir, "run_parameters.json")
    with open(params_json_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    # Save run parameters to text
    params_txt_path = os.path.join(output_dir, "run_parameters.txt")
    with open(params_txt_path, 'w') as f:
        f.write(f"Similarity Comparison Run v3: {params['timestamp']}\n")
        f.write("==========================================\n\n")
        f.write(f"Command: {params['command']}\n\n")
        f.write("Parameters:\n")
        for k, v in params.items():
            if k not in ("command", "timestamp"):
                f.write(f"  {k}: {v}\n")

    # Load and sample scanpaths
    if mode == "IVT":
        scanpaths = load_scanpaths_ivt(args.input_dir, args.sample_percentage)
        n_samples = len(scanpaths)
        logging.info(f"Total IVT scanpaths loaded: {n_samples}")
    else:
        sequences, labels, subjects, file_ids = load_aoi_data(args.input_dir, args.sample_percentage, max_seq_length=None)
        scanpaths = sequences  # AOI sequences are simple strings
        n_samples = len(scanpaths)
        logging.info(f"Total AOI sequences loaded: {n_samples}")

    # Compute raw similarity matrices (only once)
    logging.info("Computing raw similarity matrices...")
    
    # Compute Needleman-Wunsch similarity matrix (raw)
    if mode == "IVT":
        nw_mat_raw = compute_nw_similarity_matrix_ivt(scanpaths, use_numba=use_numba)
    else:
        nw_mat_raw = compute_nw_similarity_matrix_aoi(scanpaths, use_numba=use_numba)
    
    # Save raw NW matrix
    nw_mat_raw_path = os.path.join(output_dir, "nw_similarity_matrix_raw.npy")
    np.save(nw_mat_raw_path, nw_mat_raw)
    logging.info(f"Saved raw NW similarity matrix to {nw_mat_raw_path}")

    # Compute Fast_EyeBLAST similarity matrix (raw)
    if mode == "IVT":
        fb_mat_raw = compute_fb_similarity_matrix_ivt(
            scanpaths,
            seed_length=3,
            band_width=5,
            num_neighbors=args.k,
            max_candidates=20,
            use_duration_bins=True,
            duration_bins=8,
            use_parallel=False,
            num_processes=None
        )
    else:
        fb_mat_raw = compute_fb_similarity_matrix_aoi(
            scanpaths,
            seed_length=3,
            band_width=5,
            num_neighbors=args.k,
            max_candidates=20,
            use_parallel=False,
            num_processes=None
        )
    
    # Save raw FB matrix
    fb_mat_raw_path = os.path.join(output_dir, "fb_similarity_matrix_raw.npy")
    np.save(fb_mat_raw_path, fb_mat_raw)
    logging.info(f"Saved raw FB similarity matrix to {fb_mat_raw_path}")

    # Determine which normalization methods to run
    if args.normalization_mode == "all":
        norm_methods = ["none", "minmax", "zscore", "rank", "quantile", "correlation_preserving"]
    else:
        norm_methods = [args.normalization_mode]

    # Apply normalizations and analyze
    all_stats = []
    for norm_method in norm_methods:
        logging.info(f"Processing normalization method: {norm_method}")
        
        # Apply normalization
        nw_mat_norm, fb_mat_norm = apply_normalization(nw_mat_raw, fb_mat_raw, norm_method)
        
        # Save normalized matrices
        nw_norm_path = os.path.join(output_dir, f"nw_similarity_matrix_{norm_method}.npy")
        fb_norm_path = os.path.join(output_dir, f"fb_similarity_matrix_{norm_method}.npy")
        np.save(nw_norm_path, nw_mat_norm)
        np.save(fb_norm_path, fb_mat_norm)
        logging.info(f"Saved {norm_method} normalized matrices")
        
        # Analyze and visualize
        stats = analyze_and_visualize(nw_mat_norm, fb_mat_norm, output_dir, norm_method)
        all_stats.append(stats)

    # Generate comprehensive analysis report
    generate_analysis_report(args, mode, n_samples, all_stats, output_dir)

    # Restore original stdout
    sys.stdout = original_stdout
    logging.info("Similarity comparison completed for all normalization methods.")

if __name__ == "__main__":
    main() 