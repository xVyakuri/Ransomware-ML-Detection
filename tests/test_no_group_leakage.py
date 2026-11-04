# No family/duplicate group spans train/validation/test.
from __future__ import annotations

import pandas as pd
import pytest

from mlran_parser import leakage
from mlran_parser.config import Paths
from mlran_parser.labels import read_labels

from .conftest import REAL, real_data


def _checks(paths: Paths):
    m = pd.read_csv(paths.split_manifest)
    labels = read_labels(paths)
    pairs = pd.read_parquet(paths.near_dup_pairs)
    return [
        leakage.check_groups_disjoint(m),
        leakage.check_sha256_disjoint(m, labels),
        leakage.check_near_dups_disjoint(m, pairs),
    ]


def test_groups_never_span_splits(built: Paths):
    failed = [(c.name, c.detail) for c in _checks(built) if not c.passed]
    assert not failed


def test_duplicate_binary_and_near_dup_share_group(built: Paths):
    m = pd.read_csv(built.split_manifest).set_index("sample_id")
    assert m.at[10001, "group_id"] == m.at[10002, "group_id"]  # same sha256
    assert m.at[13000, "group_id"] == m.at[13001, "group_id"]  # identical features


@real_data
@pytest.mark.parametrize("check", range(3))
def test_real_groups_never_span_splits(check):
    c = _checks(REAL)[check]
    assert c.passed, c.detail
