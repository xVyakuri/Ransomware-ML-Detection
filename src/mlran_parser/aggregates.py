"""Per-sample count and breadth features derived from the feature catalog.

MLRan stores presence only, so every count is "number of distinct features of this kind the
sample showed" (e.g. distinct files written), never an event count or a rate. Nothing here is
fitted on data, so the table is safe to compute once over all samples.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from .config import TABLES, Paths
from .report import write_stage

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
