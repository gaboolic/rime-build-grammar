import argparse
from pathlib import Path


def normalize_ngram_text(text):
    text = text.replace(" ", "")
    text = text.replace("<s>", "")
    return text.replace("</s>", "$")


def merge_ngram_files(input_dir, orders):
    word_map = {}
    for order in orders:
        file_name = input_dir / f"ngram_{order}_frequencies.txt"
        with open(file_name, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.rstrip()
                if line.startswith("#") or "\t" not in line:
                    continue

                params = line.split("\t")
                word = normalize_ngram_text(params[0])
                freq = params[2] if len(params) == 3 else params[1]

                if len(word) <= 1 or len(word) > 8:
                    continue
                word_map[word] = freq
    return word_map


def parse_args():
    parser = argparse.ArgumentParser(description="Merge extracted n-gram files into build_grammar input.")
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing ngram_*_frequencies.txt files. Defaults to current directory.",
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        type=int,
        required=True,
        help="N-gram orders to merge, for example: --orders 2 3",
    )
    parser.add_argument(
        "--output",
        help="Output file path. Defaults to merge_<orders>.txt in the input directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_dir / f"merge_{'_'.join(str(order) for order in args.orders)}.txt"
    )

    word_map = merge_ngram_files(input_dir, args.orders)
    with open(output_path, "w", encoding="utf-8") as write_file:
        for word, freq in word_map.items():
            write_file.write(f"{word}\t{freq}\n")

    print(f"Wrote {len(word_map)} entries to {output_path}")


if __name__ == "__main__":
    main()
