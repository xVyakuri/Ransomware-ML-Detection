# Ensure features consist of counts/rates, unique files/dirs, read/write ratio, and API n-grams.
"""Per-sample count and breadth features derived from the feature catalog.

MLRan stores presence only, so every count is "number of distinct features of this kind the
sample showed" (e.g. distinct files written), never an event count or a rate. Nothing here is
fitted on data, so the table is safe to compute once over all samples.

Team-facing loader: model-ready matrices for one split seed.

    from src.features import load
    b = load(seed=0, min_df=5)            # all 9 tables
    b.train.X, b.train.y                  # scipy CSR + labels (1 = ransomware)
    b.val, b.test, b.holdout              # same features as train
    b.features                            # DataFrame describing every column of X

Leakage rules enforced here:
  * the min_df vocabulary is counted on the train split only
  * features naming a sample's own hash (leak_hash) are always dropped
  * failed runs are never returned (they have split "excluded")
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from src.common import SPLIT_SEEDS, TABLES, Paths, run_stage, write_stage

log = logging.getLogger(__name__)

_EXT = re.compile(r"\.([a-z0-9_\-]{1,10})$")


def _parent(path: str) -> str:
    p = path.lower().rstrip("\\")
    return p.rsplit("\\", 1)[0] if "\\" in p else ""


def _ext(path: str) -> str:
    base = path.lower().rsplit("\\", 1)[-1]
    m = _EXT.search(base)
    return m.group(1) if m else ""


def _distinct_count(X: sp.csr_matrix, cols: np.ndarray, keys: list[str]) -> np.ndarray:
    """Per row: number of distinct keys among the present columns `cols`."""
    if len(cols) == 0:
        return np.zeros(X.shape[0], dtype=np.int64)
    codes, uniq = pd.factorize(pd.Series(keys))
    M = sp.csr_matrix(
        (np.ones(len(cols), dtype=np.int32), (np.arange(len(cols)), codes)),
        shape=(len(cols), len(uniq)),
    )
    hits = X[:, cols].astype(np.int32) @ M
    return hits.getnnz(axis=1).astype(np.int64)


def build(X: sp.csr_matrix, catalog: pd.DataFrame, sample_ids: np.ndarray) -> pd.DataFrame:
    X = X.tocsc()
    out: dict[str, np.ndarray] = {"sample_id": sample_ids}

    leak = catalog.leak_hash.to_numpy()
    for table in TABLES:
        cols = catalog.col.to_numpy()[(catalog.table == table).to_numpy() & ~leak]
        out[f"{table.split('_', 1)[1]}_n"] = X[:, cols].getnnz(axis=1)

    # Counts per (category, operation) for tables that record an operation.
    ops = catalog[(catalog.operation != "") & (catalog.operation != "CALL") & ~catalog.leak_hash]
    for (cat, op), grp in ops.groupby(["category", "operation"], sort=True):
        out[f"{cat.lower()}_{op.lower()}_n"] = X[:, grp.col.to_numpy()].getnnz(axis=1)

    # Breadth of file activity: distinct parent directories and extensions touched.
    Xr = X.tocsr()
    file_ops = catalog[(catalog.category == "FILE") & ~catalog.leak_hash]
    for op in ("WRITTEN", "CREATED", "DELETED"):
        sub = file_ops[file_ops.operation == op]
        cols = sub.col.to_numpy()
        tg = sub.target.tolist()
        out[f"file_{op.lower()}_distinct_dirs"] = _distinct_count(
            Xr, cols, [_parent(t) for t in tg]
        )
        out[f"file_{op.lower()}_distinct_ext"] = _distinct_count(Xr, cols, [_ext(t) for t in tg])

    # Columns the ratios and the team rely on exist even if an operation never occurs.
    for op in ("WRITTEN", "READ", "CREATED", "DELETED"):
        out.setdefault(f"file_{op.lower()}_n", np.zeros(X.shape[0], dtype=np.int64))
    df = pd.DataFrame(out)
    # Ratios with +1 smoothing so samples with no reads/creates stay finite.
    df["file_write_read_ratio"] = df.file_written_n / (df.file_read_n + 1)
    df["file_delete_create_ratio"] = df.file_deleted_n / (df.file_created_n + 1)
    df["file_written_per_dir"] = df.file_written_n / (df.file_written_distinct_dirs + 1)
    return df


def run(paths: Paths) -> pd.DataFrame:
    X = sp.load_npz(paths.matrix).tocsr()
    catalog = pq.read_table(paths.catalog).to_pandas()
    samples = pd.read_parquet(paths.samples)
    df = build(X, catalog, samples.sample_id.to_numpy())
    df.to_parquet(paths.aggregates, index=False)
    write_stage(
        paths.stage_report,
        "aggregates",
        {
            "rows": len(df),
            "columns": [c for c in df.columns if c != "sample_id"],
            "null_cells": int(df.isna().sum().sum()),
        },
    )
    log.info("aggregates: %d rows x %d features", len(df), df.shape[1] - 1)
    return df


AGG_PREFIX_TABLE = {
    "api": "1_api",
    "reg": "2_reg",
    "file": "3_file",
    "dir": "4_dir",
    "directory": "4_dir",
    "str": "5_str",
    "net": "6_net",
    "network": "6_net",
    "sys": "7_sys",
    "system": "7_sys",
    "drop": "8_drop",
    "sig": "9_sig",
}


def aggregate_table(column: str) -> str:
    return AGG_PREFIX_TABLE[column.split("_", 1)[0]]


@dataclass
class SplitData:
    X: sp.csr_matrix
    y: np.ndarray
    sample_id: np.ndarray
    group_id: np.ndarray
    family: np.ndarray
    type: np.ndarray
    first_seen_year: np.ndarray

    def __len__(self) -> int:
        return self.X.shape[0]


@dataclass
class Bundle:
    seed: int
    train: SplitData
    val: SplitData
    test: SplitData
    holdout: SplitData
    features: pd.DataFrame  # one row per column of X: name, table, category, operation, target

    @property
    def feature_names(self) -> list[str]:
        return self.features.name.tolist()


def load(
    seed: int = 0,
    tables: list[str] | tuple[str, ...] | None = None,
    min_df: int = 5,
    sparse_features: bool = True,
    aggregates: bool = False,
    sample_tokens: bool = True,
    paths: Paths | None = None,
) -> Bundle:
    """Build matrices for one split seed.

    tables          subset of TABLES (feature-family ablation: drop one table at a time)
    min_df          keep a sparse feature only if >= min_df TRAIN samples show it
    sparse_features include the presence features (False = aggregates only)
    aggregates      append the count/breadth features from aggregates.parquet
    sample_tokens   keep features where the sample's own file name became <SAMPLE>; they still
                    carry the submitted file's extension (.dll, .tmp, ...), a collection artifact
    """
    if seed not in SPLIT_SEEDS:
        raise ValueError(f"seed must be one of {SPLIT_SEEDS}")
    tables = tuple(tables or TABLES)
    unknown = set(tables) - set(TABLES)
    if unknown:
        raise ValueError(f"unknown tables: {sorted(unknown)}")
    if not (sparse_features or aggregates):
        raise ValueError("select sparse_features and/or aggregates")
    paths = paths or Paths.default()

    manifest = pd.read_csv(paths.split_manifest, dtype={"family": "string", "type": "string"})
    samples = pd.read_parquet(paths.samples, columns=["sample_id", "row"])
    if not (manifest.sample_id.to_numpy() == samples.sample_id.to_numpy()).all():
        raise RuntimeError("split manifest and samples.parquet row order differ")
    split_col = manifest[f"split_seed{seed}"].to_numpy()
    rows = {s: np.flatnonzero(split_col == s) for s in ("train", "val", "test", "holdout")}

    blocks: list[sp.csr_matrix] = []
    feature_frames: list[pd.DataFrame] = []

    if sparse_features:
        cat = pq.read_table(
            paths.catalog, columns=["table", "leak_hash", "sample_token"]
        ).to_pandas()
        allowed = cat.table.isin(tables).to_numpy() & ~cat.leak_hash.to_numpy()
        if not sample_tokens:
            allowed &= ~cat.sample_token.to_numpy()
        X = sp.load_npz(paths.matrix).tocsr()
        train_df = np.asarray(X[rows["train"]].sum(axis=0)).ravel()
        keep = np.flatnonzero(allowed & (train_df >= min_df))
        blocks.append(X[:, keep].tocsr())
        full = pq.read_table(
            paths.catalog, columns=["table", "category", "operation", "target"]
        ).take(keep)
        f = full.to_pandas()
        f["name"] = np.where(
            f.operation.isin(["", "CALL"]),
            f.category + ":" + f.target,
            f.category + ":" + f.operation + ":" + f.target,
        )
        f["kind"] = "presence"
        feature_frames.append(f)

    if aggregates:
        agg = pd.read_parquet(paths.aggregates)
        if not (agg.sample_id.to_numpy() == manifest.sample_id.to_numpy()).all():
            raise RuntimeError("aggregates.parquet row order differs; rerun features")
        cols = [c for c in agg.columns if c != "sample_id" and aggregate_table(c) in tables]
        blocks.append(sp.csr_matrix(agg[cols].to_numpy(dtype=np.float32)))
        feature_frames.append(
            pd.DataFrame(
                {
                    "table": [aggregate_table(c) for c in cols],
                    "category": "AGGREGATE",
                    "operation": "",
                    "target": cols,
                    "name": cols,
                    "kind": "aggregate",
                }
            )
        )

    X_all = sp.hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
    if aggregates:
        X_all = X_all.astype(np.float32)

    def take(split: str) -> SplitData:
        r = rows[split]
        m = manifest.iloc[r]
        return SplitData(
            X=X_all[r],
            y=m.label.to_numpy(dtype=np.int8),
            sample_id=m.sample_id.to_numpy(),
            group_id=m.group_id.to_numpy(dtype=object),
            family=m.family.to_numpy(dtype=object),
            type=m.type.to_numpy(dtype=object),
            first_seen_year=m.first_seen_year.to_numpy(),
        )

    features = pd.concat(feature_frames, ignore_index=True)[
        ["name", "kind", "table", "category", "operation", "target"]
    ]
    return Bundle(
        seed=seed,
        train=take("train"),
        val=take("val"),
        test=take("test"),
        holdout=take("holdout"),
        features=features,
    )


if __name__ == "__main__":
    sys.exit(run_stage(run))
