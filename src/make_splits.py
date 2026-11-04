# Utilize a grouped 70/15/15 training/validation/testing split.
# Keep prefixes of one execution together for family and duplicate groups.
"""Pipeline step 5. Implementation: mlran_parser.split.

Run: uv run python src/make_splits.py  (same as `uv run mlran split`)
"""

import sys

from mlran_parser.cli import main

if __name__ == "__main__":
    sys.exit(main(["split"]))
