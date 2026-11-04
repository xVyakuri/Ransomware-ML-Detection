# Choose 8 held-out families (>=3 of 4 types seeded).
"""Pipeline step 6. Implementation: mlran_parser.split.choose_holdout (seed 520).

The holdout is chosen inside `mlran split`, because it must be removed before every split
seed. This script reports the chosen families and checks the constraints.

Run: uv run python src/make_holdout.py
"""

import logging
import sys

import pandas as pd

from mlran_parser.config import Paths
from mlran_parser.leakage import check_holdout

log = logging.getLogger("make_holdout")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = Paths.default()
    if not paths.split_manifest.exists():
        log.error("no split manifest; run `uv run python src/make_splits.py` first")
        sys.exit(1)
    m = pd.read_csv(paths.split_manifest)
    h = m[m.holdout]
    for (family, type_), n in h.groupby(["family", "type"]).size().items():
        log.info("held out: %-14s %-7s %3d samples", family, type_, n)
    result = check_holdout(m)
    (log.info if result.passed else log.error)("%s", result.detail)
    sys.exit(0 if result.passed else 1)
