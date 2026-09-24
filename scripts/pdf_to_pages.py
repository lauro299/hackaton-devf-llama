"""Renders the textbook PDF into one JPEG per page, named by PRINTED page
number (resources/014.jpg == printed page 14).

build_index.py joins the printed Índice's page numbers directly to these
filenames, so the naming must match the printed numbering, not the PDF's
page index. Use --offset to shift between the two (printed = pdf_index -
offset) and check a few pages against the Índice before running the
transcription.

Requires poppler's `pdftoppm`.

Usage:
    python scripts/pdf_to_pages.py book.pdf --offset 1
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("resources"))
    parser.add_argument("--offset", type=int, default=0, help="printed page = PDF page index - offset")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["pdftoppm", "-jpeg", "-r", str(args.dpi), str(args.pdf), f"{tmp}/p"],
            check=True,
        )
        written = 0
        for page in sorted(Path(tmp).glob("p-*.jpg")):
            pdf_index = int(re.search(r"(\d+)$", page.stem).group(1))
            printed = pdf_index - args.offset
            if printed < 0:
                continue
            page.rename(args.output / f"{printed:03d}.jpg")
            written += 1
    print(f"{written} pages written to {args.output}/")


if __name__ == "__main__":
    main()
