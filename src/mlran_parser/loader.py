"""Team-facing loader: model-ready matrices for one split seed.

    from mlran_parser.loader import load
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

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from .config import SPLIT_SEEDS, TABLES, Paths

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
            raise RuntimeError("aggregates.parquet row order differs; rerun aggregates")
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
