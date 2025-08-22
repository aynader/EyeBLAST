import json
import argparse

def chunk_list(data, size):
    """
    Yield successive chunks of 'size' elements from the list 'data'.
    """
    for i in range(0, len(data), size):
        yield data[i:i + size]


def main():
    parser = argparse.ArgumentParser(
        description='Split a JSON array into multiple files with a given number of elements each.'
    )
    parser.add_argument(
        'input', type=str,
        help='Path to the input JSON file containing an array.'
    )
    parser.add_argument(
        'count', type=int,
        help='Number of elements per chunk.'
    )
    parser.add_argument(
        'output_prefix', type=str,
        help='Prefix for output files. Files will be named <prefix>_1.json, <prefix>_2.json, ...'
    )
    args = parser.parse_args()

    # Load the JSON array
    with open(args.input, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Error: JSON root is not a list.")
        return

    # Split into chunks and write out
    for idx, chunk in enumerate(chunk_list(data, args.count), start=1):
        out_file = f"{args.output_prefix}_{idx}.json"
        with open(out_file, 'w') as f:
            # Dump JSON on a single line without extra spaces
            json.dump(chunk, f, separators=(',',':'))
        print(f"Wrote {out_file} ({len(chunk)} items)")


if __name__ == '__main__':
    main()
