# Check for malformed/dropped records, missing fields, duplicate rates.
# Also ensure class balance and write per-stage log
"""Pipeline step 4. Implementation: per-stage counts in reports/stage_report.json (written by
mlran_parser.convert, labels and split) plus the checks in mlran_parser.leakage.

Run: uv run python src/quality_checks.py
"""

import json
import logging
import sys

from mlran_parser.cli import main
from mlran_parser.config import Paths

log = logging.getLogger("quality_checks")


def summarise(paths: Paths) -> None:
    report = json.loads(paths.stage_report.read_text())
    lab, conv, spl = report["labels"], report["convert"], report["split"]
    log.info("class balance: %d ransomware, %d goodware", lab["ransomware"], lab["goodware"])
    log.info("unlabeled rows dropped: %s", conv["tables"]["3_file"]["unlabeled_ids_dropped"])
    log.info("failed runs excluded: %s", conv["failed_runs"])
    log.info(
        "malformed feature names: %d",
        sum(t["malformed_names"] for t in conv["tables"].values()),
    )
    log.info("duplicate sha256 rows: %d", lab["duplicate_sha256_rows"])
    log.info("exact duplicate feature rows: %d", conv["exact_duplicate_feature_rows"])
    log.info("near-duplicate pairs: %d", spl["near_dup_pairs"])
    log.info("missing first_seen_year: %d", lab["missing_first_seen_year"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summarise(Paths.default())
    sys.exit(main(["leakcheck"]))
