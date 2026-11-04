# Compare a smaple of 20 random raw records to derived features.
"""Pipeline step 10. Implementation: mlran_parser.audit.

Run: uv run python src/spot_check.py  (same as `uv run mlran audit`)
"""

import sys

from mlran_parser.cli import main

if __name__ == "__main__":
    sys.exit(main(["audit"]))
