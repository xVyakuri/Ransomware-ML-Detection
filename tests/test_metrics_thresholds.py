import numpy as np
import pytest

from src.thresholds import select_threshold


def test_ties_require_conservative_threshold():
    scores = np.ones(100)
    t = select_threshold(scores, np.zeros(100))
    assert t > 1
    assert (scores >= t).sum() == 0


def test_threshold_ignores_malicious_validation_scores():
    benign = np.arange(100, dtype=float)
    a = select_threshold(np.r_[benign, 1000], np.r_[np.zeros(100), 1])
    b = select_threshold(np.r_[benign, -1000], np.r_[np.zeros(100), 1])
    assert a == b == 99


def test_requires_benign_validation():
    with pytest.raises(ValueError):
        select_threshold([1], [1])
