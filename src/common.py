# Script for config loading, seeding, paths, and logging helpers.
"""Paths and fixed constants for the MLRan parsing pipeline.

Settings are loaded from config/config.yaml; this module exposes shared derived values.

Stage report: per-stage counts of records in/out, drops, malformed and duplicate rates.

Each stage writes its own section into reports/stage_report.json so a rerun of one stage
updates only that section.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.configuration import CONFIG

# The nine MLRan behaviour tables, in the authors' order.
TABLES = tuple(CONFIG['dataset']['tables'])

# Tables recorded while the sample ran. STRING (static strings) and SIGNATURE (Cuckoo
# post-processing) are excluded: a sample with only these never produced runtime behaviour.
DYNAMIC_TABLES = tuple(CONFIG['dataset']['dynamic_tables'])

# Feature-name prefix per table, and whether the name carries an operation field
# (PREFIX:OPERATION:TARGET) or only a target (PREFIX:TARGET).
TABLE_PREFIX = CONFIG['dataset']['table_prefix']

# Sample IDs present in tables 3-9 but absent from the metadata and tables 1-2.
# The Zenodo record lists them for removal.
KNOWN_EXTRA_IDS = frozenset(CONFIG['dataset']['known_extra_ids'])

# Upstream metadata, pinned to one commit of github.com/faithfulco/mlran.
MLRAN_COMMIT = CONFIG['dataset']['mlran_commit']

MLRAN_RAW_BASE = f"https://raw.githubusercontent.com/faithfulco/mlran/{MLRAN_COMMIT}"

METADATA_FILES = CONFIG['dataset']['metadata_files']

COLUMN_BATCH = CONFIG['parser']['column_batch']
THRIFT_LIMIT = CONFIG['parser']['thrift_limit']
NEAR_DUP_JACCARD = CONFIG['splits']['near_duplicate_jaccard']
HOLDOUT_SEED = CONFIG['holdout']['seed']
HOLDOUT_N_FAMILIES = CONFIG['holdout']['families']
HOLDOUT_MIN_TYPES = CONFIG['holdout']['minimum_types']
SPLIT_SEEDS = tuple(CONFIG['splits']['seeds'])
SPLIT_FRACTIONS = CONFIG['splits']['fractions']
AUDIT_SEED = CONFIG['audit']['seed']
AUDIT_N_SAMPLES = CONFIG['audit']['samples']


@dataclass(frozen=True)
class Paths:
    """Filesystem layout configured in YAML, with explicit roots for test fixtures."""

    root: Path
    raw_dataset: Path
    data: Path

    @classmethod
    def default(cls, root: Path | None = None) -> Paths:
        root = (root or Path.cwd()).resolve()
        raw = root / CONFIG['paths']['raw']
        data = root / CONFIG['paths']['data']
        return cls(root=root, raw_dataset=raw, data=data)

    @property
    def metadata_dir(self) -> Path:
        return self.data / "raw" / "metadata"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def reports(self) -> Path:
        return self.root / CONFIG['paths']['reports']

    @property
    def raw_manifest(self) -> Path:
        return self.reports / "raw_manifest.csv"

    @property
    def stage_report(self) -> Path:
        return self.reports / "stage_report.json"

    @property
    def metadata_csv(self) -> Path:
        return self.metadata_dir / "mlran_dataset_metadata.csv"

    # Processed artifacts
    @property
    def matrix(self) -> Path:
        return self.processed / "X.npz"

    @property
    def samples(self) -> Path:
        return self.processed / "samples.parquet"

    @property
    def catalog(self) -> Path:
        return self.processed / "feature_catalog.parquet"

    @property
    def labels(self) -> Path:
        return self.processed / "labels.csv"

    @property
    def aggregates(self) -> Path:
        return self.processed / "aggregates.parquet"

    @property
    def near_dup_pairs(self) -> Path:
        return self.processed / "near_dup_pairs.parquet"

    @property
    def split_manifest(self) -> Path:
        return self.processed / "split_manifest.csv"

    def ensure(self) -> None:
        for p in (self.metadata_dir, self.processed, self.reports):
            p.mkdir(parents=True, exist_ok=True)


log = logging.getLogger(__name__)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_stage(path: Path, stage: str, payload: dict[str, Any]) -> None:
    """Replace one stage's section in the report file."""
    report: dict[str, Any] = {}
    if path.exists():
        report = json.loads(path.read_text())
    payload = {"completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
               "resolved_config": CONFIG, **payload}
    report[stage] = _jsonable(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    log.info("stage report updated: %s", stage)


def read_stage(path: Path, stage: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get(stage, {})


SPLITS = tuple(SPLIT_FRACTIONS)

HOLDOUT = "holdout"

EXCLUDED = "excluded"


class PipelineError(RuntimeError):
    """Expected pipeline failure; entry points log it and exit with code 2."""


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_stage(stage: Callable[[Paths], object], verbose: bool = False) -> int:
    """Run one pipeline stage with default paths. Exit code: 0 ok, 1 check failed, 2 error."""
    setup_logging(verbose)
    try:
        result = stage(Paths.default())
    except PipelineError as e:
        logging.getLogger("pipeline").error("%s", e)
        return 2
    return 1 if result is False else 0
