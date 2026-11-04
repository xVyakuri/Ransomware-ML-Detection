# Utilize labels.csv for (benign|ransomware, family, type, year, and source)
"""Traceable labels: one row per execution, built only from the authors' metadata CSV.

A sample without a metadata row is unlabeled. It is excluded, never assumed benign.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from src.check_raw import verify
from src.common import Paths, PipelineError, run_stage, write_stage

log = logging.getLogger(__name__)

LABEL_COLUMNS = [
    "sample_id",
    "sha256",
    "label",
    "family",
    "type",
    "first_seen_year",
    "metadata_year",
    "source",
]

RANSOMWARE_ID_MIN = 10001

GOODWARE_ID_MIN = 13000


class LabelError(PipelineError, ValueError):
    pass


def build_labels(paths: Paths) -> tuple[pd.DataFrame, dict]:
    md = pd.read_csv(paths.metadata_csv)
    n_in = len(md)

    required = ["sample_id", "sha256", "sample_type", "ransomware_family", "ransomware_type"]
    missing_cols = [c for c in required if c not in md.columns]
    if missing_cols:
        raise LabelError(f"metadata missing columns: {missing_cols}")

    if md.sample_id.duplicated().any():
        raise LabelError("duplicate sample_id in metadata")
    if not md.sample_type.isin([0, 1]).all():
        raise LabelError("sample_type outside {0,1}")

    labels = pd.DataFrame(
        {
            "sample_id": md.sample_id.astype("int64"),
            "sha256": md.sha256.str.lower(),
            "label": md.sample_type.astype("int8"),  # 1 = ransomware, 0 = goodware
            "family": md.ransomware_family.astype("string"),
            "type": md.ransomware_type.astype("string"),
            # VirusTotal first-submission year; the paper's definition of first-seen.
            "first_seen_year": md.first_submission_date_year.astype("Int64"),
            # Second year column from the authors (collection timestamp); kept for audit.
            "metadata_year": md.Year.astype("Int64"),
            "source": md.source.astype("string"),
        }
    ).sort_values("sample_id", ignore_index=True)

    # Consistency checks between the label and the authors' ID ranges.
    is_r = labels.label == 1
    id_mismatch = int(
        (is_r & (labels.sample_id >= GOODWARE_ID_MIN)).sum()
        + (~is_r & (labels.sample_id < GOODWARE_ID_MIN)).sum()
    )
    if id_mismatch:
        raise LabelError(f"{id_mismatch} labels disagree with the sample_id ranges")
    if labels.loc[is_r, "family"].isna().any():
        raise LabelError("ransomware sample without family")
    fam_types = labels[is_r].groupby("family")["type"].nunique()
    if (fam_types > 1).any():
        raise LabelError(f"families with >1 type: {list(fam_types[fam_types > 1].index)}")

    dup_sha = labels[labels.sha256.duplicated(keep=False)].sort_values(["sha256", "sample_id"])
    stats = {
        "metadata_rows_in": n_in,
        "labels_out": len(labels),
        "ransomware": int(is_r.sum()),
        "goodware": int((~is_r).sum()),
        "ransomware_share": round(float(is_r.mean()), 4),
        "families": int(labels.loc[is_r, "family"].nunique()),
        "type_counts": labels.loc[is_r, "type"].value_counts().to_dict(),
        "family_size": labels.loc[is_r, "family"]
        .value_counts()
        .describe()[["min", "50%", "max"]]
        .to_dict(),
        "missing_first_seen_year": int(labels.first_seen_year.isna().sum()),
        "year_columns_disagree": int((labels.first_seen_year != labels.metadata_year).sum()),
        "first_seen_year_median": {
            "ransomware": float(labels.loc[is_r, "first_seen_year"].median()),
            "goodware": float(labels.loc[~is_r, "first_seen_year"].median()),
        },
        "duplicate_sha256_rows": len(dup_sha),
        "duplicate_sha256_pairs": dup_sha.groupby("sha256")["sample_id"].apply(list).tolist(),
    }
    log.info(
        "labels: %d ransomware, %d goodware, %d families",
        stats["ransomware"],
        stats["goodware"],
        stats["families"],
    )
    return labels, stats


def read_labels(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(
        paths.labels,
        dtype={"family": "string", "type": "string", "source": "string", "sha256": "string"},
    )


def write_labels(paths: Paths) -> None:
    """Rebuild labels.csv only (parse_raw.run also does this)."""
    verify(paths)
    paths.ensure()
    frame, stats = build_labels(paths)
    frame.to_csv(paths.labels, index=False)
    write_stage(paths.stage_report, "labels", stats)


if __name__ == "__main__":
    sys.exit(run_stage(write_labels))
