# 8 families absent from training/validation for type coverage.
from __future__ import annotations

import pandas as pd

from mlran_parser import leakage
from mlran_parser.config import HOLDOUT_MIN_TYPES, HOLDOUT_N_FAMILIES, Paths

from .conftest import REAL, real_data


def _assert_holdout(paths: Paths) -> None:
    m = pd.read_csv(paths.split_manifest)
    c = leakage.check_holdout(m)
    assert c.passed, c.detail
    h = m[m.holdout]
    assert h.family.nunique() == HOLDOUT_N_FAMILIES
    assert h.type.nunique() >= HOLDOUT_MIN_TYPES


def test_holdout_families_unseen(built: Paths):
    _assert_holdout(built)


@real_data
def test_real_holdout_families_unseen():
    _assert_holdout(REAL)
