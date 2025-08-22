# count_fasta_by_genus.py
from pathlib import Path
import sys, csv

def count_fasta_records(root: Path):
    per = {}
    notes = []

    # iterate only immediate subfolders (each = one genus)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        genus = d.name
        # common FASTA extensions
        files = sorted(list(d.glob("*.fasta")) + list(d.glob("*.fa")) + list(d.glob("*.fna")))
        if not files:
            notes.append(f"[WARN] No FASTA found in {genus}/")
            continue

        n = 0
        for fp in files:
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                n += sum(1 for line in f if line.startswith(">"))
        per[genus] = n

        if len(files) != 1:
            notes.append(f"[INFO] {genus}/ has {len(files)} FASTA files; counted across all.")

    total = sum(per.values())
    return total, dict(sorted(per.items())), notes

if __name__ == "__main__":
    # Default to the directory where this script lives
    default_root = Path(__file__).resolve().parent
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default_root

    total, per, notes = count_fasta_records(root)

    print(f"Root: {root}")
    print("Per-genus counts:")
    width = max((len(g) for g in per), default=10)
    for g, n in per.items():
        print(f"  {g:<{width}} {n}")
    print(f"\nTOTAL {total}")

    if notes:
        print("\nNotes:")
        for m in notes:
            print(" ", m)

    # write a CSV beside the root folder
    csv_path = root / "fasta_counts.csv"
    with csv_path.open("w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["genus", "count"])
        for g, n in per.items():
            w.writerow([g, n])
        w.writerow(["TOTAL", total])
    print(f"\nWrote {csv_path}")
