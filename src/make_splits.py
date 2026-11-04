# Utilize a grouped 70/15/15 training/validation/testing split.
"""Leakage-safe grouping, unseen-family holdout, and grouped 70/15/15 splits.

Groups (union-find, so any link merges whole groups):
  * every ransomware sample of one family
  * samples sharing a binary sha256
  * samples whose feature sets have Jaccard >= NEAR_DUP_JACCARD (near duplicates)
A group never spans two splits.

Failed runs (no runtime behaviour) are excluded from every split and counted in the report.
The 8 held-out families are chosen once (HOLDOUT_SEED) and removed before every split seed.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from src.build_labels import read_labels
from src.common import (
    EXCLUDED,
    HOLDOUT,
    HOLDOUT_SEED,
    NEAR_DUP_JACCARD,
    SPLIT_FRACTIONS,
    SPLIT_SEEDS,
    SPLITS,
    Paths,
    run_stage,
    write_stage,
)
from src.make_holdout import SplitError, choose_holdout

log = logging.getLogger(__name__)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        p = self.parent
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:
            p[i], i = root, p[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(X: sp.csr_matrix, threshold: float) -> pd.DataFrame:
    """All row pairs (i < j) with Jaccard(feature set i, feature set j) >= threshold. Exact."""
    Xi = X.astype(np.int32)
    sizes = np.asarray(Xi.sum(axis=1)).ravel()
    inter = sp.triu(Xi @ Xi.T, k=1).tocoo()
    union = sizes[inter.row] + sizes[inter.col] - inter.data
    jac = np.divide(inter.data, union, out=np.zeros(len(union)), where=union > 0)
    keep = jac >= threshold
    return pd.DataFrame({"i": inter.row[keep], "j": inter.col[keep], "jaccard": jac[keep]})


def build_groups(
    labels: pd.DataFrame,
    failed: np.ndarray,
    X: sp.csr_matrix,
    threshold: float = NEAR_DUP_JACCARD,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return group_id per row of `labels` and the near-duplicate pair table."""
    n = len(labels)
    uf = UnionFind(n)
    pool = np.flatnonzero(~failed)

    fam = labels.family.to_numpy(dtype=object)
    is_r = labels.label.to_numpy() == 1
    for _, idx in pd.Series(pool[is_r[pool]]).groupby(fam[pool[is_r[pool]]]):
        first = idx.iloc[0]
        for k in idx.iloc[1:]:
            uf.union(first, k)
    for _, idx in pd.Series(pool).groupby(labels.sha256.to_numpy()[pool]):
        first = idx.iloc[0]
        for k in idx.iloc[1:]:
            uf.union(first, k)

    pairs = near_duplicate_pairs(X[pool], threshold)
    pairs["i"] = pool[pairs.i.to_numpy()]
    pairs["j"] = pool[pairs.j.to_numpy()]
    for a, b in zip(pairs.i.to_numpy(), pairs.j.to_numpy(), strict=True):
        uf.union(int(a), int(b))

    roots = np.array([uf.find(i) for i in range(n)])
    sid = labels.sample_id.to_numpy()
    group = np.empty(n, dtype=object)
    for root in np.unique(roots[pool]):
        members = pool[roots[pool] == root]
        fams = sorted({fam[m] for m in members if is_r[m]})
        group[members] = "fam:" + "+".join(fams) if fams else f"good:{sid[members].min()}"
    group[failed] = EXCLUDED

    pairs = pd.DataFrame(
        {
            "sample_id_a": sid[pairs.i],
            "sample_id_b": sid[pairs.j],
            "jaccard": pairs.jaccard.round(4),
            "label_a": labels.label.to_numpy()[pairs.i],
            "label_b": labels.label.to_numpy()[pairs.j],
            "family_a": fam[pairs.i],
            "family_b": fam[pairs.j],
        }
    )
    return pd.Series(group, index=labels.index, name="group_id"), pairs


def assign_split(frame: pd.DataFrame, holdout_groups: list[str], seed: int) -> pd.Series:
    """Grouped split stratified by label; each group goes wholly to the split with most deficit."""
    out = pd.Series(EXCLUDED, index=frame.index, dtype=object)
    out[frame.group_id.isin(holdout_groups)] = HOLDOUT
    active = frame[(out != HOLDOUT) & (frame.group_id != EXCLUDED)]

    groups = active.groupby("group_id").agg(size=("label", "size"), label=("label", "mean"))
    groups["stratum"] = (groups.label >= 0.5).astype(int)
    rng = np.random.default_rng(seed)
    fractions = np.array([SPLIT_FRACTIONS[s] for s in SPLITS])
    for stratum in (0, 1):
        g = groups[groups.stratum == stratum]
        target = fractions * g["size"].sum()
        # A group bigger than half the smallest split target would wreck val/test balance
        # wherever it landed, so oversized groups go to train first; the rest are random.
        oversized = g.index[g["size"] > 0.5 * target.min()]
        rest = rng.permutation(np.asarray(g.index.difference(oversized)))
        order = [*sorted(oversized), *rest]
        filled = np.zeros(len(SPLITS))
        for gid in order:
            k = int(np.argmax(target - filled))
            filled[k] += g.at[gid, "size"]
            out[active.index[active.group_id == gid]] = SPLITS[k]
    return out


def run(paths: Paths) -> pd.DataFrame:
    labels = read_labels(paths)
    samples = pd.read_parquet(paths.samples)
    if not (samples.sample_id.to_numpy() == labels.sample_id.to_numpy()).all():
        raise SplitError("samples.parquet and labels.csv row order differ; rerun parse_raw")
    X = sp.load_npz(paths.matrix).tocsr()
    leak = pq.read_table(paths.catalog, columns=["leak_hash"]).column(0).to_numpy()
    X = X[:, np.flatnonzero(~leak)]

    failed = samples.failed_run.to_numpy()
    group, pairs = build_groups(labels, failed, X)
    pairs.to_parquet(paths.near_dup_pairs, index=False)

    frame = labels[["sample_id", "label", "family", "type", "first_seen_year"]].copy()
    frame["group_id"] = group
    frame["failed_run"] = failed
    holdout = choose_holdout(frame)
    frame["holdout"] = frame.group_id.isin(holdout)
    for seed in SPLIT_SEEDS:
        frame[f"split_seed{seed}"] = assign_split(frame, holdout, seed)
    frame.to_csv(paths.split_manifest, index=False)

    active = frame[frame.group_id != EXCLUDED]
    gsize = active.groupby("group_id").size()
    gmix = active.groupby("group_id").label.nunique()
    holdout_fams = sorted(frame.loc[frame.holdout, "family"].unique())
    per_seed = {}
    for seed in SPLIT_SEEDS:
        col = f"split_seed{seed}"
        ct = pd.crosstab(frame[col], frame.label).rename(columns={0: "goodware", 1: "ransomware"})
        in_split = frame[frame[col].isin(SPLITS)]
        per_seed[seed] = {
            "counts": ct.to_dict(orient="index"),
            "fraction": (in_split[col].value_counts(normalize=True).round(4)).to_dict(),
            "ransomware_families_per_split": in_split[in_split.label == 1]
            .groupby(col)
            .family.nunique()
            .to_dict(),
        }
    write_stage(
        paths.stage_report,
        "split",
        {
            "near_dup_threshold": NEAR_DUP_JACCARD,
            "near_dup_pairs": len(pairs),
            "near_dup_pairs_cross_label": int((pairs.label_a != pairs.label_b).sum()),
            "near_dup_pairs_cross_family": int(
                (
                    (pairs.label_a == 1) & (pairs.label_b == 1) & (pairs.family_a != pairs.family_b)
                ).sum()
            ),
            "excluded_failed_runs": int(failed.sum()),
            "groups": int(len(gsize)),
            "group_size": {"max": int(gsize.max()), "singletons": int((gsize == 1).sum())},
            "groups_mixing_labels": int((gmix > 1).sum()),
            "merged_family_groups": sorted(g for g in gsize.index if "+" in g),
            "holdout_seed": HOLDOUT_SEED,
            "holdout_groups": holdout,
            "holdout_families": holdout_fams,
            "holdout_types": sorted(frame.loc[frame.holdout, "type"].unique()),
            "holdout_samples": int(frame.holdout.sum()),
            "seeds": per_seed,
        },
    )
    log.info("split: %d groups, holdout families %s", len(gsize), holdout_fams)
    return frame


if __name__ == "__main__":
    sys.exit(run_stage(run))
