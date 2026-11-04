# Choose 8 held-out families (>=3 of 4 types seeded).
"""Unseen-family holdout: whole pure-ransomware groups totalling exactly 8 families
spanning at least 3 types, drawn once with HOLDOUT_SEED.

`python -m src.make_splits` calls choose_holdout so the holdout is removed before every
split seed. Running this module reports the chosen families and checks the constraints.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from src.common import (
    EXCLUDED,
    HOLDOUT_MIN_TYPES,
    HOLDOUT_N_FAMILIES,
    HOLDOUT_SEED,
    Paths,
    PipelineError,
    run_stage,
)

log = logging.getLogger(__name__)


class SplitError(PipelineError):
    pass


def choose_holdout(frame: pd.DataFrame, seed: int = HOLDOUT_SEED) -> list[str]:
    """Pick whole pure-ransomware groups totalling exactly N families spanning >= K types."""
    pool = frame[frame.group_id != EXCLUDED]
    by_group = pool.groupby("group_id").agg(labels=("label", "nunique"), label=("label", "first"))
    units = sorted(by_group[(by_group.labels == 1) & (by_group.label == 1)].index)
    fam_of = {g: pool.loc[pool.group_id == g, "family"].unique().tolist() for g in units}
    type_of = pool[pool.label == 1].groupby("family")["type"].first().to_dict()

    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        chosen, fams = [], []
        for g in rng.permutation(units):
            if len(fams) + len(fam_of[g]) <= HOLDOUT_N_FAMILIES:
                chosen.append(g)
                fams.extend(fam_of[g])
            if len(fams) == HOLDOUT_N_FAMILIES:
                break
        if len(fams) == HOLDOUT_N_FAMILIES and len({type_of[f] for f in fams}) >= HOLDOUT_MIN_TYPES:
            return sorted(chosen)
    raise SplitError("could not find a holdout satisfying the family/type constraints")


def report(paths: Paths) -> bool:
    """Log the held-out families and check the constraints on the current split manifest."""
    from src.quality_checks import check_holdout

    if not paths.split_manifest.exists():
        raise SplitError("no split manifest; run `python -m src.make_splits` first")
    m = pd.read_csv(paths.split_manifest)
    h = m[m.holdout]
    for (family, type_), n in h.groupby(["family", "type"]).size().items():
        log.info("held out: %-14s %-7s %3d samples", family, type_, n)
    result = check_holdout(m)
    (log.info if result.passed else log.error)("%s", result.detail)
    return result.passed


if __name__ == "__main__":
    sys.exit(run_stage(report))
