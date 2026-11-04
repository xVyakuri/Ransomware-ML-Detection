# Scaler/imputer/selecter never see validation/test.
from __future__ import annotations

import numpy as np

from src import quality_checks
from src.common import Paths
from src.features import load

from .conftest import REAL, real_data


def test_vocabulary_counted_on_train_only(built: Paths):
    b = load(seed=0, min_df=3, paths=built)
    assert (np.asarray((b.train.X > 0).sum(axis=0)).ravel() >= 3).all()
    assert b.val.X.shape[1] == b.train.X.shape[1] == len(b.features)
    assert set(b.train.y) == {0, 1}
    assert len(b.holdout) and (b.holdout.y == 1).all()


def test_vocab_check_passes(built: Paths):
    c = quality_checks.check_vocab_train_only(built, min_df=3)
    assert c.passed, c.detail


@real_data
def test_real_vocab_check_passes():
    c = quality_checks.check_vocab_train_only(REAL)
    assert c.passed, c.detail
