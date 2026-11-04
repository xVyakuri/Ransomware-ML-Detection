# 70/15/15 training/validation/test split and fixed-seed determinism.
from __future__ import annotations

import pandas as pd

from src import quality_checks
from src.common import Paths

from .conftest import REAL, real_data


def test_split_is_deterministic(built: Paths):
    c = quality_checks.check_deterministic(pd.read_csv(built.split_manifest))
    assert c.passed, c.detail


@real_data
def test_real_split_is_deterministic():
    c = quality_checks.check_deterministic(pd.read_csv(REAL.split_manifest))
    assert c.passed, c.detail


@real_data
def test_real_split_ratios_near_70_15_15():
    # The synthetic fixture is too small for the +-0.05 tolerance, so this runs on real data.
    c = quality_checks.check_fractions(pd.read_csv(REAL.split_manifest))
    assert c.passed, c.detail
