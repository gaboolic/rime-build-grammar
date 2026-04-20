import argparse
import math
import re
from pathlib import Path


NGRAM_LINE_PATTERN = re.compile(r"^(-?\d+\.\d+)\t(.+?)(?:\t-?\d+\.\d+)?$")


def extract_ngram_counts(arpa_file):
    ngrams_counts = {}
    with open(arpa_file, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line.startswith("ngram"):
                parts = line.split("=")
                if len(parts) == 2 and parts[0].startswith("ngram"):
                    order = int(parts[0].split()[1])
                    count = int(parts[1])
                    ngrams_counts[order] = count
            elif line.startswith("\\1-grams:"):
                break
    return ngrams_counts


def extract_ngrams(arpa_file):
    with open(arpa_file, "r", encoding="utf-8") as file:
        current_order = 0
        for raw_line in file:
            line = raw_line.strip()
            if line.startswith("\\") and "-grams:" in line:
                current_order = int(line.split("-")[0][1:])
                continue

            match = NGRAM_LINE_PATTERN.match(line)
            if match:
                logprob, ngram = match.groups()
                prob = math.exp(float(logprob))
                yield current_order, (ngram.strip(), prob)


def write_frequencies_to_file(ngrams_generator, ngrams_counts, output_dir, filename_pattern):
    current_order = 0
    current_file = None
    try:
        for order, ngram_data in ngrams_generator:
            if order != current_order:
                if current_file:
                    current_file.close()
                current_order = order
                filename = output_dir / filename_pattern.format(order)
                current_file = open(filename, "w", encoding="utf-8")
                print(f"Writing {current_order}-grams to {filename}")

            ngram, prob = ngram_data
            total_count = ngrams_counts.get(order, 1)
            freq = round(prob * total_count)
            current_file.write(f"{ngram}\t{freq}\n")
    finally:
        if current_file:
            current_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract n-gram frequencies from an ARPA file.")
    parser.add_argument("arpa_file", help="Path to the input ARPA file.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for generated ngram frequency files. Defaults to current directory.",
    )
    parser.add_argument(
        "--filename-pattern",
        default="ngram_{}_frequencies.txt",
        help="Filename pattern used for generated n-gram files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    arpa_file = Path(args.arpa_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ngrams_counts = extract_ngram_counts(arpa_file)
    ngrams_generator = extract_ngrams(arpa_file)
    write_frequencies_to_file(ngrams_generator, ngrams_counts, output_dir, args.filename_pattern)


if __name__ == "__main__":
    main()