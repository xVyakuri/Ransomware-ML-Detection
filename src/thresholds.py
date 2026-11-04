# Select validation-selected threshold at FPR 1%.
import numpy as np


def select_threshold(scores, labels, target=None):
    from src.configuration import CONFIG
    target = CONFIG['threshold']['target_fpr'] if target is None else target
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('scores must be finite and aligned with labels')
    benign = scores[labels == 0]
    if not len(benign) or not 0 <= target <= 1:
        raise ValueError('benign validation rows and a valid FPR target are required')
    candidates = np.r_[np.unique(benign), np.nextafter(benign.max(), np.inf)]
    return float(next(t for t in candidates if np.mean(benign >= t) <= target))
