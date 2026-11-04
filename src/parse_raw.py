# Raw records goes into normalized event table via (execution_id, event_order/time, category, operation)
"""Pipeline step 2. Implementation: mlran_parser.convert.

Run: uv run python src/parse_raw.py  (same as `uv run mlran convert`)
"""

import sys

from mlran_parser.cli import main

if __name__ == "__main__":
    sys.exit(main(["convert"]))
