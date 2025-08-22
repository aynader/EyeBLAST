#!/usr/bin/env python3
# dataset_downloader_16S.py
# Usage:
#   pip install biopython tqdm
#   python dataset_downloader_16S.py --output_dir dataset16S

import os
import time
import argparse
import random
from collections import defaultdict
from Bio import Entrez, SeqIO
from tqdm import tqdm

# Constants
GENERA = [
    "Escherichia",
    "Staphylococcus",
    "Streptococcus",
    "Bacillus",
    "Pseudomonas",
    "Salmonella",
    "Clostridium",
    "Lactobacillus",
    "Neisseria",
    "Enterococcus"
]
TARGET_PER_GENUS = 1000
MIN_LEN = 1400
MAX_LEN = 1700
BATCH_SIZE = 200
SLEEP_SEC = 0.3
DEFAULT_RETM = 2000

def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def fetch_ids(genus, email, api_key, retmax=DEFAULT_RETM):
    """Search NCBI for 16S rRNA IDs for a given genus."""
    Entrez.email = email
    Entrez.api_key = api_key
    term = (
        f"16S ribosomal RNA[Title] AND {genus}[Organism] "
        f"AND {MIN_LEN}:{MAX_LEN}[Sequence Length]"
    )
    handle = Entrez.esearch(db="nucleotide", term=term, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])

def fetch_fasta(ids, email, api_key):
    """Fetch FASTA records in batches from NCBI."""
    Entrez.email = email
    Entrez.api_key = api_key
    records = []
    for batch in chunked(ids, BATCH_SIZE):
        time.sleep(SLEEP_SEC)
        handle = Entrez.efetch(
            db="nucleotide",
            id=batch,
            rettype="fasta",
            retmode="text"
        )
        records.extend(SeqIO.parse(handle, "fasta"))
        handle.close()
    return records

def filter_and_dedupe(records):
    """Filter by sequence length and remove exact duplicates."""
    filtered = [r for r in records if MIN_LEN <= len(r.seq) <= MAX_LEN]
    uniq = {}
    for rec in filtered:
        seq_str = str(rec.seq)
        if seq_str not in uniq:
            uniq[seq_str] = rec
    return list(uniq.values())

def main():
    parser = argparse.ArgumentParser(
        description="Download, clean, and balance 16S rRNA dataset by genus"
    )
    parser.add_argument(
        "--email",
        default="a.youssef.nader@gmail.com",
        help="NCBI Entrez email (required by NCBI)"
    )
    parser.add_argument(
        "--api_key",
        default="1cbde0893d10b150c5fbc717039fea98cb09",
        help="NCBI Entrez API key"
    )
    parser.add_argument(
        "--output_dir",
        default="dataset16S",
        help="Where to write genus subfolders with FASTA files"
    )
    parser.add_argument(
        "--target_per_genus",
        type=int,
        default=TARGET_PER_GENUS,
        help="How many sequences to keep per genus (undersampled)"
    )
    parser.add_argument(
        "--retmax",
        type=int,
        default=DEFAULT_RETM,
        help="Max IDs to fetch per genus from Entrez"
    )
    args = parser.parse_args()

    random.seed(42)
    summary = {}

    os.makedirs(args.output_dir, exist_ok=True)

    for genus in GENERA:
        print(f"\n=== {genus} ===")
        ids = fetch_ids(genus, args.email, args.api_key, retmax=args.retmax)
        print(f"Found {len(ids)} raw IDs")

        if not ids:
            print("  → No IDs, skipping")
            continue

        # Fetch FASTA and clean
        recs = fetch_fasta(ids, args.email, args.api_key)
        print(f"Fetched {len(recs)} FASTA records")

        recs = filter_and_dedupe(recs)
        print(f"{len(recs)} unique records after length filter & dedupe")

        # Undersample to target
        if len(recs) > args.target_per_genus:
            random.shuffle(recs)
            recs = recs[: args.target_per_genus]
        print(f"Keeping {len(recs)} sequences for {genus}")

        # Write out
        genus_dir = os.path.join(args.output_dir, genus)
        os.makedirs(genus_dir, exist_ok=True)
        out_path = os.path.join(genus_dir, f"{genus}_16S.fasta")
        with open(out_path, "w") as out_f:
            SeqIO.write(recs, out_f, "fasta")
        print(f"Wrote -> {out_path}")

        summary[genus] = len(recs)

    print("\n=== Summary ===")
    for genus, count in summary.items():
        print(f"{genus}: {count}")

if __name__ == "__main__":
    main()
