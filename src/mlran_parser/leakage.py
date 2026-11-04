"""Automated leakage checks over the processed artifacts. Any failure blocks hand-off.

Run with `mlran leakcheck`; the pytest suite runs the same checks when artifacts exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from .config import (
    HOLDOUT_MIN_TYPES,
    HOLDOUT_N_FAMILIES,
    SPLIT_FRACTIONS,
    SPLIT_SEEDS,
    Paths,
)
from .labels import read_labels
from .report import write_stage
from .split import EXCLUDED, HOLDOUT, SPLITS, assign_split, choose_holdout

log = logging.getLogger(__name__)

FRACTION_TOLERANCE = 0.05
FORBIDDEN_INPUTS = ("MI_selected", "RFE_selected", "MLRan_X_train", "MLRan_X_test")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _manifest(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.split_manifest, dtype={"family": "string", "type": "string"})


def _cols() -> list[str]:
    return [f"split_seed{s}" for s in SPLIT_SEEDS]


def check_groups_disjoint(m: pd.DataFrame) -> Check:
    bad = {}
    for c in _cols():
        a = m[m[c] != EXCLUDED]
        n = a.groupby("group_id")[c].nunique()
        if (n > 1).any():
            bad[c] = list(n[n > 1].index[:5])
    return Check("groups_disjoint", not bad, str(bad) if bad else "every group in one split")


def check_sha256_disjoint(m: pd.DataFrame, labels: pd.DataFrame) -> Check:
    sha = m.sample_id.map(labels.set_index("sample_id").sha256)
    bad = {}
    for c in _cols():
        keep = m[c] != EXCLUDED
        n = pd.DataFrame({"sha": sha[keep], "s": m.loc[keep, c]}).groupby("sha").s.nunique()
        if (n > 1).any():
            bad[c] = int((n > 1).sum())
    return Check("sha256_disjoint", not bad, str(bad) if bad else "no binary in two splits")


def check_near_dups_disjoint(m: pd.DataFrame, pairs: pd.DataFrame) -> Check:
    bad = {}
    idx = m.set_index("sample_id")
    for c in _cols():
        a = pairs.sample_id_a.map(idx[c])
        b = pairs.sample_id_b.map(idx[c])
        n = int(((a != b) & (a != EXCLUDED) & (b != EXCLUDED)).sum())
        if n:
            bad[c] = n
    detail = str(bad) if bad else f"{len(pairs)} near-dup pairs, none split apart"
    return Check("near_dups_disjoint", not bad, detail)


def check_holdout(m: pd.DataFrame) -> Check:
    h = m[m.holdout]
    fams = set(h.family.dropna())
    types = set(h.type.dropna())
    leaks = {c: int(m[m[c].isin(SPLITS)].family.isin(fams).sum()) for c in _cols()}
    leaks = {k: v for k, v in leaks.items() if v}
    not_holdout = {c: int((h[c] != HOLDOUT).sum()) for c in _cols()}
    not_holdout = {k: v for k, v in not_holdout.items() if v}
    ok = (
        len(fams) == HOLDOUT_N_FAMILIES
        and len(types) >= HOLDOUT_MIN_TYPES
        and (h.label == 1).all()
        and not leaks
        and not not_holdout
    )
    detail = (
        f"{len(fams)} families, {len(types)} types, {len(h)} samples; "
        f"families seen in train/val/test: {leaks or 'none'}; mislabelled: {not_holdout or 'none'}"
    )
    return Check("holdout_unseen", ok, detail)


def check_failed_excluded(m: pd.DataFrame) -> Check:
    bad = {}
    for c in _cols():
        wrong = int(((m[c] == EXCLUDED) != m.failed_run).sum())
        if wrong:
            bad[c] = wrong
    detail = str(bad) if bad else f"{int(m.failed_run.sum())} failed runs excluded"
    return Check("failed_runs_excluded", not bad, detail)


def check_deterministic(m: pd.DataFrame) -> Check:
    base = m[["sample_id", "label", "family", "type", "group_id", "failed_run"]]
    holdout = choose_holdout(base)
    expected = sorted(m.loc[m.holdout, "group_id"].unique())
    diffs = {}
    if holdout != expected:
        diffs["holdout"] = "differs"
    for s in SPLIT_SEEDS:
        again = assign_split(base, holdout, s)
        n = int((again.to_numpy() != m[f"split_seed{s}"].to_numpy()).sum())
        if n:
            diffs[f"seed{s}"] = n
    return Check("deterministic", not diffs, str(diffs) if diffs else "same seed, same split")


def check_fractions(m: pd.DataFrame) -> Check:
    off = {}
    for c in _cols():
        a = m[m[c].isin(SPLITS)]
        for label in (0, 1):
            frac = a[a.label == label][c].value_counts(normalize=True)
            for s, target in SPLIT_FRACTIONS.items():
                got = float(frac.get(s, 0.0))
                if abs(got - target) > FRACTION_TOLERANCE:
                    off[f"{c}/label{label}/{s}"] = round(got, 3)
    return Check("split_fractions", not off, str(off) if off else "within ±0.05 of 70/15/15")


def check_labels_traceable(m: pd.DataFrame, labels: pd.DataFrame) -> Check:
    j = m.merge(labels, on="sample_id", how="left", suffixes=("", "_lab"), indicator=True)
    missing = int((j._merge != "both").sum())
    mismatch = int((j.label != j.label_lab).sum())
    ok = missing == 0 and mismatch == 0 and len(m) == len(labels)
    return Check(
        "labels_traceable", ok, f"missing={missing}, label mismatches={mismatch}, rows={len(m)}"
    )


def check_vocab_train_only(paths: Paths, seed: int = 0, min_df: int = 5) -> Check:
    from .loader import load

    b = load(seed=seed, min_df=min_df, paths=paths)
    train_counts = np.asarray((b.train.X > 0).sum(axis=0)).ravel()
    cat = pq.read_table(paths.catalog, columns=["leak_hash"]).column(0).to_numpy()
    X = sp.load_npz(paths.matrix).tocsr()
    m = _manifest(paths)
    train_rows = np.flatnonzero(m[f"split_seed{seed}"].to_numpy() == "train")
    expected = int(((np.asarray(X[train_rows].sum(axis=0)).ravel() >= min_df) & ~cat).sum())
    ok = bool((train_counts >= min_df).all()) and b.train.X.shape[1] == expected
    detail = f"{b.train.X.shape[1]} features kept (expected {expected}) from train counts only"
    return Check("vocab_fit_on_train", ok, detail)


def check_no_own_id_features(paths: Paths) -> Check:
    """No non-zero cell may sit in a column whose name contains that row's own sample_id."""
    from .convert import id_tokens

    cat = pq.read_table(paths.catalog, columns=["col", "target", "id_token"]).to_pandas()
    cols = cat[cat.id_token]
    X = sp.load_npz(paths.matrix).tocsc()
    sid = pd.read_parquet(paths.samples, columns=["sample_id"]).sample_id.to_numpy()
    ids = set(sid.tolist())
    bad = 0
    for col, target in zip(cols.col.to_numpy(), cols.target.to_numpy(), strict=True):
        rows = X.indices[X.indptr[col] : X.indptr[col + 1]]
        if len(rows):
            bad += int(np.isin(sid[rows], list(id_tokens(target, ids))).sum())
    return Check(
        "no_own_sample_id_features",
        bad == 0,
        f"{bad} cells name their own sample_id" if bad else "own IDs replaced by <SAMPLE>",
    )


def check_no_preselected_inputs() -> Check:
    src = Path(__file__).parent
    hits = []
    for f in src.glob("*.py"):
        if f.name == Path(__file__).name:
            continue
        text = f.read_text()
        hits += [f"{f.name}:{tok}" for tok in FORBIDDEN_INPUTS if tok in text]
    return Check(
        "no_author_preselected_features",
        not hits,
        str(hits) if hits else "authors' MI/RFE/X_train files never read",
    )


def run_all(paths: Paths, write: bool = True) -> list[Check]:
    m = _manifest(paths)
    labels = read_labels(paths)
    pairs = pd.read_parquet(paths.near_dup_pairs)
    checks = [
        check_labels_traceable(m, labels),
        check_groups_disjoint(m),
        check_sha256_disjoint(m, labels),
        check_near_dups_disjoint(m, pairs),
        check_holdout(m),
        check_failed_excluded(m),
        check_deterministic(m),
        check_fractions(m),
        check_vocab_train_only(paths),
        check_no_own_id_features(paths),
        check_no_preselected_inputs(),
    ]
    for c in checks:
        (log.info if c.passed else log.error)(
            "%-32s %s  %s", c.name, "PASS" if c.passed else "FAIL", c.detail
        )
    if write:
        write_stage(
            paths.stage_report,
            "leakage",
            {
                "passed": all(c.passed for c in checks),
                "checks": {c.name: {"passed": c.passed, "detail": c.detail} for c in checks},
            },
        )
    return checks
