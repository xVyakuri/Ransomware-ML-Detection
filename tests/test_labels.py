# Every execution has a label and unlabeled does not mean benign.
from __future__ import annotations

import pandas as pd
import pytest

from src import quality_checks
from src.build_labels import LabelError, build_labels, read_labels
from src.common import Paths

from .conftest import REAL, build_fixture, real_data


def test_every_execution_has_a_label(built: Paths):
    c = quality_checks.check_labels_traceable(pd.read_csv(built.split_manifest), read_labels(built))
    assert c.passed, c.detail


def test_unlabeled_rows_are_dropped_not_benign(built: Paths):
    samples = pd.read_parquet(built.samples)
    assert 11898 not in set(samples.sample_id)  # present in raw tables, absent from metadata


def test_labels_reject_id_range_mismatch(tmp_path):
    paths = build_fixture(tmp_path)
    md = pd.read_csv(paths.metadata_csv)
    md.loc[md.sample_id == 13000, "sample_type"] = 1
    md.loc[md.sample_id == 13000, "ransomware_family"] = "fam00"
    md.to_csv(paths.metadata_csv, index=False)
    with pytest.raises(LabelError):
        build_labels(paths)


@real_data
def test_real_labels_traceable():
    c = quality_checks.check_labels_traceable(pd.read_csv(REAL.split_manifest), read_labels(REAL))
    assert c.passed, c.detail
