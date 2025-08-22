"""
MegaBLAST for IVT Sequences: A script for creating FASTA files from IVT sequences
and using the NCBI MegaBLAST algorithm to perform sequence comparisons.

This script will:
1. Convert IVT token sequences from MPIIDPEye and GazeBaseVR datasets to FASTA format
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

# Constants
MPII_INPUT_DIR = "MPIIDPEye_IVT"
GAZEBASE_INPUT_DIR = "GazeBaseVR_IVT"
MPII_FASTA_DIR = "MPIIDPEye_IVT_FASTA"
GAZEBASE_FASTA_DIR = "GazeBaseVR_IVT_FASTA"
BLAST_DB_DIR = "MegaBLAST_DB_IVT"
BLAST_RESULTS_DIR = "MegaBLAST_RESULTS_IVT"

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def tokens_to_fasta(tokens, description):
    """
    Convert IVT tokens to FASTA format string with proper nucleotide encoding
    
    Parameters:
    -----------
    tokens : list
        List of IVT tokens (e.g., ['S13', 'M32', ...])
    description : str
        Description for the FASTA header
        
    Returns:
    --------
    str
        FASTA formatted string with encoded nucleotides
    """
    # Convert tokens to valid nucleotide sequence
    nucleotide_sequence = ""
    for token in tokens:
        token_type = token[0]  # First character (S, M, F)
        duration = int(token[1:])  # Duration value
        
        # Map token types to nucleotides
        if token_type == 'S':
            base = 'A'  # Saccades as A
        elif token_type == 'M':
            base = 'C'  # Microsaccades as C
        elif token_type == 'F':
            base = 'G'  # Fixations as G
        else:
            # Use N for unknown token types
            base = 'N'
        
        # Add the base followed by Ts for duration (limit to max 50 to prevent excessively long sequences)
        max_duration = min(duration, 50)
        nucleotide_sequence += base + ('T' * max_duration)
    
    # Create FASTA formatted string
    fasta = f">{description}\n{nucleotide_sequence}\n"
    return fasta

def convert_mpii_to_fasta():
    """Convert MPIIDPEye IVT token files to FASTA format"""
    create_directory(MPII_FASTA_DIR)
    
    # Get all JSON files in the input directory
    input_files = glob.glob(os.path.join(MPII_INPUT_DIR, "*.json"))
    
    for input_file in input_files:
        # Get filename without extension
        basename = os.path.basename(input_file)
        filename = os.path.splitext(basename)[0]
        
        # Read JSON data
        with open(input_file, 'r') as f:
            tokens = json.load(f)
        
        # Convert to FASTA
        fasta_content = tokens_to_fasta(tokens, filename)
        
        # Write to output file
        output_file = os.path.join(MPII_FASTA_DIR, f"{filename}.fasta")
        with open(output_file, 'w') as f:
            f.write(fasta_content)
        
        print(f"Converted {basename} to FASTA format")

def convert_gazebase_to_fasta():
    """Convert GazeBaseVR IVT token files to FASTA format"""
    create_directory(GAZEBASE_FASTA_DIR)
    
    # Get all JSON files in the input directory
    input_files = glob.glob(os.path.join(GAZEBASE_INPUT_DIR, "*.json"))
    
    for input_file in input_files:
        # Get filename without extension
        basename = os.path.basename(input_file)
        filename = os.path.splitext(basename)[0]
        
        # Read JSON data
        with open(input_file, 'r') as f:
            tokens = json.load(f)
        
        # Convert to FASTA
        fasta_content = tokens_to_fasta(tokens, filename)
        
        # Write to output file
        output_file = os.path.join(GAZEBASE_FASTA_DIR, f"{filename}.fasta")
        with open(output_file, 'w') as f:
            f.write(fasta_content)
        
        print(f"Converted {basename} to FASTA format")

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

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MegaBLAST for IVT Sequences")
    parser.add_argument("--convert", action="store_true", help="Convert IVT tokens to FASTA")
    parser.add_argument("--create-db", action="store_true", help="Create BLAST databases")
    parser.add_argument("--run-blast", action="store_true", help="Run MegaBLAST")
    parser.add_argument("--process-results", action="store_true", help="Process BLAST results")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    # If no arguments provided, run all steps
    if not any(vars(args).values()):
        args.all = True
    
    # Create output directories
    create_directory(BLAST_RESULTS_DIR)
    
    # Convert IVT tokens to FASTA
    if args.convert or args.all:
        print("Converting IVT tokens to FASTA format...")
        convert_mpii_to_fasta()
        convert_gazebase_to_fasta()
    
    # Create BLAST databases
    if args.create_db or args.all:
        print("Creating BLAST databases...")
        mpii_db = create_blast_database(MPII_FASTA_DIR, "mpii_ivt")
        if mpii_db is None:
            print("Error creating MPII database, stopping.")
            return
            
        gazebase_db = create_blast_database(GAZEBASE_FASTA_DIR, "gazebase_ivt")
        if gazebase_db is None:
            print("Error creating GazeBase database, stopping.")
            return
        
        # Create a combined database
        combined_dir = "Combined_IVT_FASTA"
        create_directory(combined_dir)
        
        # Copy all FASTA files to the combined directory
        for fasta_file in glob.glob(os.path.join(MPII_FASTA_DIR, "*.fasta")):
            shutil.copy(fasta_file, combined_dir)
        
        for fasta_file in glob.glob(os.path.join(GAZEBASE_FASTA_DIR, "*.fasta")):
            shutil.copy(fasta_file, combined_dir)
        
        combined_db = create_blast_database(combined_dir, "combined_ivt")
        if combined_db is None:
            print("Error creating combined database, stopping.")
            return
    
    # Run MegaBLAST
    if args.run_blast or args.all:
        print("Running MegaBLAST...")
        
        # Ensure all database paths use forward slashes
        mpii_db_path = str(Path(BLAST_DB_DIR) / "mpii_ivt").replace('\\', '/')
        gazebase_db_path = str(Path(BLAST_DB_DIR) / "gazebase_ivt").replace('\\', '/')
        combined_db_path = str(Path(BLAST_DB_DIR) / "combined_ivt").replace('\\', '/')
        
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