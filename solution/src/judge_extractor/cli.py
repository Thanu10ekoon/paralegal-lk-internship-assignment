from __future__ import annotations

import argparse
from pathlib import Path

from .extractor import process_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract bench and author_judge from all PDFs in a folder."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data"),
        help="Directory containing input PDFs (default: ../data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../output"),
        help="Directory to write JSON outputs (default: ../output)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    generated = process_folder(args.data_dir.resolve(), args.output_dir.resolve())
    print(f"Generated {len(generated)} files in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
