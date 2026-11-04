"""Paths and fixed constants for the MLRan parsing pipeline.

Every value that affects the output (seeds, thresholds, pinned commit) lives here so a
clean clone reproduces the same artifacts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The nine MLRan behaviour tables, in the authors' order.
TABLES: tuple[str, ...] = (
    "1_api",
    "2_reg",
    "3_file",
    "4_dir",
    "5_str",
    "6_net",
    "7_sys",
    "8_drop",
    "9_sig",
)

# Tables recorded while the sample ran. STRING (static strings) and SIGNATURE (Cuckoo
# post-processing) are excluded: a sample with only these never produced runtime behaviour.
DYNAMIC_TABLES: tuple[str, ...] = ("1_api", "2_reg", "3_file", "4_dir", "6_net", "7_sys", "8_drop")

# Feature-name prefix per table, and whether the name carries an operation field
# (PREFIX:OPERATION:TARGET) or only a target (PREFIX:TARGET).
TABLE_PREFIX: dict[str, tuple[str, bool]] = {
    "1_api": ("API", False),
    "2_reg": ("REG", True),
    "3_file": ("FILE", True),
    "4_dir": ("DIRECTORY", True),
    "5_str": ("STRING", False),
    "6_net": ("NETWORK", True),
    "7_sys": ("SYSTEM", True),
    "8_drop": ("DROP", True),
    "9_sig": ("SIGNATURE", False),
}

# Sample IDs present in tables 3-9 but absent from the metadata and tables 1-2.
# The Zenodo record lists them for removal.
KNOWN_EXTRA_IDS: frozenset[int] = frozenset({11898, 11899, 12057, 12107})

# Upstream metadata, pinned to one commit of github.com/faithfulco/mlran.
MLRAN_COMMIT = "273fcac5af82ad60d60fa2bff427ebcb1db61273"
MLRAN_RAW_BASE = f"https://raw.githubusercontent.com/faithfulco/mlran/{MLRAN_COMMIT}"
METADATA_FILES: dict[str, str] = {
    "mlran_dataset_metadata.csv": "2_collected_samples_metadata/mlran_dataset_metadata.csv",
    "ransomware_samples_families_description.csv": (
        "2_collected_samples_metadata/ransomware_samples_families_description.csv"
    ),
    "goodware_samples_description.csv": (
        "2_collected_samples_metadata/goodware_samples_description.csv"
    ),
}

# Grouping and split settings.
NEAR_DUP_JACCARD = 0.9  # samples at or above this feature-set Jaccard share a group
HOLDOUT_SEED = 520  # fixed: the same 8 families are held out for every split seed
HOLDOUT_N_FAMILIES = 8
HOLDOUT_MIN_TYPES = 3
SPLIT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SPLIT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
AUDIT_SEED = 620
AUDIT_N_SAMPLES = 20

# Parquet reading: the wide MLRan files exceed pyarrow's default thrift metadata limits.
THRIFT_LIMIT = 2**31 - 1
COLUMN_BATCH = 5000


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. Override the roots with MLRAN_RAW_DIR / MLRAN_DATA_DIR."""

    root: Path
    raw_dataset: Path
    data: Path

    @classmethod
    def default(cls, root: Path | None = None) -> Paths:
        root = (root or Path.cwd()).resolve()
        raw = Path(os.environ.get("MLRAN_RAW_DIR", root / "data" / "raw" / "MLRan_full_dataset"))
        data = Path(os.environ.get("MLRAN_DATA_DIR", root / "data"))
        return cls(root=root, raw_dataset=raw, data=data)

    @property
    def metadata_dir(self) -> Path:
        return self.data / "raw" / "metadata"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

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
