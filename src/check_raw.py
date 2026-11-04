# Verify raw snapshot exists and write/verify SHA-256 manifest
"""Pipeline step 1. Implementation: mlran_parser.snapshot.

Run: uv run python src/check_raw.py  (same as `uv run mlran snapshot`)
"""

import sys

from mlran_parser.cli import main

if __name__ == "__main__":
    sys.exit(main(["snapshot"]))
