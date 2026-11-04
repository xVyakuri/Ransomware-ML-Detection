# Ensure features consist of counts/rates, unique files/dirs, read/write ratio, and API n-grams.
"""Pipeline step 8. Implementation: mlran_parser.aggregates.

Run: uv run python src/features.py  (same as `uv run mlran aggregates`)
"""

import sys

from mlran_parser.cli import main

if __name__ == "__main__":
    sys.exit(main(["aggregates"]))
