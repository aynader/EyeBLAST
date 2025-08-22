"""
MegaBLAST for AOI Sequences: A script for creating FASTA files from AOI sequences
and using the NCBI MegaBLAST algorithm to perform sequence comparisons.

This script will:
1. Convert AOI character sequences from MPIIDPEye and GazeBaseVR datasets to FASTA format
2. Organize them in separate folders
3. Use the original MegaBLAST algorithm to perform similarity searches
"""

import os
import json
import subprocess
import time
import argparse
import glob
from pathlib import Path
import shutil
from collections import defaultdict
import pandas as pd

# Constants
MPII_INPUT_DIR = "MPIIDPEye_AOIs"
GAZEBASE_INPUT_DIR = "GazeBaseVR_AOIs"
MPII_FASTA_DIR = "MPIIDPEye_AOIs_FASTA"
GAZEBASE_FASTA_DIR = "GazeBaseVR_AOIs_FASTA"
BLAST_DB_DIR = "MegaBLAST_DB_AOI"
BLAST_RESULTS_DIR = "MegaBLAST_RESULTS_AOI"

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

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

def sequence_to_fasta(sequence, description):
    """
    Convert AOI character sequence to FASTA format string with proper nucleotide encoding
    
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

def load_encoded_sequences(input_dir, sample_percentage=100):
    """
    Load AOI-encoded sequences from encoded_sequences.json
    
    Parameters:
    -----------
    input_dir : str
        Directory containing encoded_sequences.json
    sample_percentage : int
        Percentage of sequences to sample (for testing with smaller datasets)
        
    Returns:
    --------
    dict
        Dictionary of file_id -> sequence data
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
    return data

def convert_mpii_to_fasta(sample_percentage=100):
    """Convert MPIIDPEye AOI sequences to FASTA format"""
    create_directory(MPII_FASTA_DIR)
    
    # Load encoded sequences
    sequences = load_encoded_sequences(MPII_INPUT_DIR, sample_percentage)
    
    for file_id, data in sequences.items():
        # Extract sequence
        sequence = data['sequence']
        
        # Convert to FASTA
        fasta_content = sequence_to_fasta(sequence, file_id)
        
        # Write to output file
        output_file = os.path.join(MPII_FASTA_DIR, f"{file_id}.fasta")
        with open(output_file, 'w') as f:
            f.write(fasta_content)
        
        print(f"Converted {file_id} to FASTA format")

def convert_gazebase_to_fasta(sample_percentage=100):
    """Convert GazeBaseVR AOI sequences to FASTA format"""
    create_directory(GAZEBASE_FASTA_DIR)
    
    # Load encoded sequences
    sequences = load_encoded_sequences(GAZEBASE_INPUT_DIR, sample_percentage)
    
    for file_id, data in sequences.items():
        # Extract sequence
        sequence = data['sequence']
        
        # Convert to FASTA
        fasta_content = sequence_to_fasta(sequence, file_id)
        
        # Write to output file
        output_file = os.path.join(GAZEBASE_FASTA_DIR, f"{file_id}.fasta")
        with open(output_file, 'w') as f:
            f.write(fasta_content)
        
        print(f"Converted {file_id} to FASTA format")

def create_blast_database(fasta_dir, db_name):
    """
    Create a BLAST database from FASTA files
    
    Parameters:
    -----------
    fasta_dir : str
        Directory containing FASTA files
    db_name : str
        Name of the BLAST database to create
    """
    create_directory(BLAST_DB_DIR)
    
    # Path to the output database using forward slashes even on Windows
    db_path = Path(BLAST_DB_DIR) / db_name
    
    # Concatenate all FASTA files into a single file
    concatenated_fasta = Path(BLAST_DB_DIR) / f"{db_name}_all.fasta"
    
    with open(concatenated_fasta, 'w') as outfile:
        fasta_files = glob.glob(os.path.join(fasta_dir, "*.fasta"))
        for fasta_file in fasta_files:
            try:
                with open(fasta_file, 'r') as infile:
                    outfile.write(infile.read())
            except Exception as e:
                print(f"Error reading {fasta_file}: {e}")
    
    # Convert paths to strings with forward slashes for BLAST
    db_path_str = str(db_path).replace('\\', '/')
    concatenated_fasta_str = str(concatenated_fasta).replace('\\', '/')
    
    # Create BLAST database
    cmd = [
        "makeblastdb",
        "-in", concatenated_fasta_str,
        "-dbtype", "nucl",
        "-out", db_path_str,
        "-title", db_name
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Created BLAST database: {db_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating BLAST database: {e}")
        return None
    
    return db_path_str

def run_megablast(query_file, db_path, output_file, evalue=10, word_size=28, perc_identity=90):
    """
    Run MegaBLAST on a query file against a database
    
    Parameters:
    -----------
    query_file : str
        Path to the query FASTA file
    db_path : str
        Path to the BLAST database
    output_file : str
        Path to the output file
    evalue : float
        E-value threshold for reporting matches
    word_size : int
        Word size for MegaBLAST (default: 28)
    perc_identity : float
        Percent identity threshold (default: 90)
    """
    # Ensure paths use forward slashes for consistency
    query_file = str(Path(query_file)).replace('\\', '/')
    output_file = str(Path(output_file)).replace('\\', '/')
    
    cmd = [
        "blastn",
        "-task", "megablast",  # Use MegaBLAST algorithm
        "-query", query_file,
        "-db", db_path,
        "-out", output_file,
        "-evalue", str(evalue),
        "-word_size", str(word_size),
        "-perc_identity", str(perc_identity),
        "-outfmt", "6 qseqid sseqid pident length qlen slen mismatch gapopen qstart qend sstart send evalue bitscore"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"MegaBLAST completed: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running MegaBLAST: {e}")

def run_all_vs_all_megablast(fasta_dir, db_path, output_dir, dataset_name):
    """
    Run MegaBLAST for all FASTA files in a directory against a database
    
    Parameters:
    -----------
    fasta_dir : str
        Directory containing query FASTA files
    db_path : str
        Path to the BLAST database
    output_dir : str
        Directory to store BLAST results
    dataset_name : str
        Name of the dataset (for output file naming)
    """
    create_directory(output_dir)
    
    # Get all FASTA files in the directory
    fasta_files = glob.glob(os.path.join(fasta_dir, "*.fasta"))
    
    for query_file in fasta_files:
        # Get filename without extension
        basename = os.path.basename(query_file)
        filename = os.path.splitext(basename)[0]
        
        # Define output file
        output_file = Path(output_dir) / f"{filename}_blast.tsv"
        
        # Run MegaBLAST
        run_megablast(query_file, db_path, str(output_file))

def process_blast_results(results_dir, output_file):
    """
    Process BLAST results and generate a summary
    
    Parameters:
    -----------
    results_dir : str
        Directory containing BLAST result files
    output_file : str
        Path to the output summary file
    """
    # Dictionary to store best matches for each query
    best_matches = defaultdict(list)
    
    # Get all BLAST result files
    result_files = glob.glob(os.path.join(results_dir, "*.tsv"))
    
    for result_file in result_files:
        # Get filename without extension
        basename = os.path.basename(result_file)
        query_name = basename.replace("_blast.tsv", "")
        
        # Read BLAST results
        with open(result_file, 'r') as f:
            for line in f:
                # Parse BLAST output fields
                fields = line.strip().split('\t')
                if len(fields) < 12:
                    continue
                
                query_id = fields[0]
                subject_id = fields[1]
                percent_identity = float(fields[2])
                alignment_length = int(fields[3])
                e_value = float(fields[12])
                bit_score = float(fields[13]) if len(fields) > 13 else 0.0
                
                # Skip self-matches
                if query_id == subject_id:
                    continue
                
                # Store match information
                best_matches[query_name].append({
                    'subject_id': subject_id,
                    'percent_identity': percent_identity,
                    'alignment_length': alignment_length,
                    'e_value': e_value,
                    'bit_score': bit_score
                })
    
    # Sort matches by bit score and keep top matches
    for query_name, matches in best_matches.items():
        matches.sort(key=lambda x: x['bit_score'], reverse=True)
        best_matches[query_name] = matches[:10]  # Keep top 10 matches
    
    # Write summary to file
    with open(output_file, 'w') as f:
        f.write("Query\tTop Matches\n")
        
        for query_name, matches in best_matches.items():
            match_strings = []
            
            for match in matches:
                match_str = (
                    f"{match['subject_id']}"
                    f" (Identity: {match['percent_identity']:.2f}%,"
                    f" Length: {match['alignment_length']},"
                    f" E-value: {match['e_value']:.2e},"
                    f" Bit Score: {match['bit_score']:.2f})"
                )
                match_strings.append(match_str)
            
            f.write(f"{query_name}\t{'; '.join(match_strings)}\n")
    
    print(f"Saved results summary to {output_file}")

class NCBIMegaBLASTScanpathClassifier:
    """
    Classifier using NCBI MegaBLAST for AOI scanpath comparison
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
        
    def fit(self, X_train, y_train):
        """Train the classifier"""
        self.X_train = X_train
        self.y_train = y_train
        
        # Create temporary directory for BLAST files
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory for BLAST: {self.temp_dir}")
        
        # Create FASTA files for training data
        print("Converting training scanpaths to FASTA format...")
        self.fasta_files = {}
        for i, scanpath in enumerate(X_train):
            fasta_file = os.path.join(self.temp_dir, f"train_{i}.fasta")
            with open(fasta_file, 'w') as f:
                f.write(sequence_to_fasta(scanpath, f"train_{i}"))
            self.fasta_files[i] = fasta_file
        
        # Create a concatenated FASTA file for the database
        concat_file = os.path.join(self.temp_dir, "train_all.fasta")
        with open(concat_file, 'w') as outf:
            for fasta_file in self.fasta_files.values():
                with open(fasta_file, 'r') as inf:
                    outf.write(inf.read())
        
        # Create BLAST database
        print("Creating BLAST database from training data...")
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
        print(f"Predicting labels for {len(X_test)} test samples using MegaBLAST...")
        from tqdm import tqdm
        predictions = []
        
        for test_idx, test_scanpath in enumerate(tqdm(X_test, desc="MegaBLAST predictions")):
            # Create FASTA for test scanpath
            test_fasta = os.path.join(self.temp_dir, f"test_{test_idx}.fasta")
            with open(test_fasta, 'w') as f:
                f.write(sequence_to_fasta(test_scanpath, f"test_{test_idx}"))
            
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
                from collections import Counter
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

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MegaBLAST for AOI Sequences")
    parser.add_argument("--convert", action="store_true", help="Convert AOI sequences to FASTA")
    parser.add_argument("--create-db", action="store_true", help="Create BLAST databases")
    parser.add_argument("--run-blast", action="store_true", help="Run MegaBLAST")
    parser.add_argument("--process-results", action="store_true", help="Process BLAST results")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--sample-percentage", type=int, default=100,
                        help="Percentage of data to sample (default: 100)")
    
    args = parser.parse_args()
    
    # If no arguments provided, run all steps
    if not any(vars(args).values()):
        args.all = True
    
    # Create output directories
    create_directory(BLAST_RESULTS_DIR)
    
    # Convert AOI sequences to FASTA
    if args.convert or args.all:
        print("Converting AOI sequences to FASTA format...")
        convert_mpii_to_fasta(args.sample_percentage)
        convert_gazebase_to_fasta(args.sample_percentage)
    
    # Create BLAST databases
    if args.create_db or args.all:
        print("Creating BLAST databases...")
        mpii_db = create_blast_database(MPII_FASTA_DIR, "mpii_aoi")
        if mpii_db is None:
            print("Error creating MPII database, stopping.")
            return
            
        gazebase_db = create_blast_database(GAZEBASE_FASTA_DIR, "gazebase_aoi")
        if gazebase_db is None:
            print("Error creating GazeBase database, stopping.")
            return
        
        # Create a combined database
        combined_dir = "Combined_AOI_FASTA"
        create_directory(combined_dir)
        
        # Copy all FASTA files to the combined directory
        for fasta_file in glob.glob(os.path.join(MPII_FASTA_DIR, "*.fasta")):
            shutil.copy(fasta_file, combined_dir)
        
        for fasta_file in glob.glob(os.path.join(GAZEBASE_FASTA_DIR, "*.fasta")):
            shutil.copy(fasta_file, combined_dir)
        
        combined_db = create_blast_database(combined_dir, "combined_aoi")
        if combined_db is None:
            print("Error creating combined database, stopping.")
            return
    
    # Run MegaBLAST
    if args.run_blast or args.all:
        print("Running MegaBLAST...")
        
        # Ensure all database paths use forward slashes
        mpii_db_path = str(Path(BLAST_DB_DIR) / "mpii_aoi").replace('\\', '/')
        gazebase_db_path = str(Path(BLAST_DB_DIR) / "gazebase_aoi").replace('\\', '/')
        combined_db_path = str(Path(BLAST_DB_DIR) / "combined_aoi").replace('\\', '/')
        
        # MPII vs MPII
        mpii_output_dir = str(Path(BLAST_RESULTS_DIR) / "MPII_vs_MPII")
        create_directory(mpii_output_dir)
        run_all_vs_all_megablast(MPII_FASTA_DIR, mpii_db_path, mpii_output_dir, "MPII")
        
        # GazeBase vs GazeBase
        gazebase_output_dir = str(Path(BLAST_RESULTS_DIR) / "GazeBase_vs_GazeBase")
        create_directory(gazebase_output_dir)
        run_all_vs_all_megablast(GAZEBASE_FASTA_DIR, gazebase_db_path, gazebase_output_dir, "GazeBase")
        
        # MPII vs GazeBase
        mpii_vs_gazebase_dir = str(Path(BLAST_RESULTS_DIR) / "MPII_vs_GazeBase")
        create_directory(mpii_vs_gazebase_dir)
        run_all_vs_all_megablast(MPII_FASTA_DIR, gazebase_db_path, mpii_vs_gazebase_dir, "MPII_vs_GazeBase")
        
        # GazeBase vs MPII
        gazebase_vs_mpii_dir = str(Path(BLAST_RESULTS_DIR) / "GazeBase_vs_MPII")
        create_directory(gazebase_vs_mpii_dir)
        run_all_vs_all_megablast(GAZEBASE_FASTA_DIR, mpii_db_path, gazebase_vs_mpii_dir, "GazeBase_vs_MPII")
    
    # Process BLAST results
    if args.process_results or args.all:
        print("Processing BLAST results...")
        
        # Process results for each comparison
        for comparison in ["MPII_vs_MPII", "GazeBase_vs_GazeBase", "MPII_vs_GazeBase", "GazeBase_vs_MPII"]:
            results_dir = str(Path(BLAST_RESULTS_DIR) / comparison)
            output_file = str(Path(BLAST_RESULTS_DIR) / f"{comparison}_summary.tsv")
            
            if os.path.exists(results_dir) and os.listdir(results_dir):
                process_blast_results(results_dir, output_file)
    
    print("Done!")

if __name__ == "__main__":
    main() 