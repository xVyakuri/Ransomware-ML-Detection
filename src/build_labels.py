# Utilize labels.csv for (benign|ransomware, family, type, year, and source)
"""Pipeline step 3. Implementation: mlran_parser.labels.

Run: uv run python src/build_labels.py  (same as `uv run mlran labels`)
"""

import sys

from mlran_parser.cli import main

if __name__ == "__main__":
    sys.exit(main(["labels"]))
