"""Convert the nine wide MLRan Parquet tables into one sparse matrix plus a feature catalog.

Output (data/processed/):
  X.npz                   CSR uint8, rows = samples.parquet order, cols = feature_catalog order
  samples.parquet         sample_id, row, per-table feature counts, failed_run flag
  feature_catalog.parquet col, feature_id, table, category, operation, target, leak_hash
  labels.csv              see labels.py

The raw tables store 0/1 presence as dense int64 (~250 GB if loaded). Reading them in column
batches and keeping only non-zero cells gives ~17M entries (~100 MB).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from src.build_labels import build_labels
from src.check_raw import verify
from src.common import (
    COLUMN_BATCH,
    DYNAMIC_TABLES,
    KNOWN_EXTRA_IDS,
    TABLE_PREFIX,
    TABLES,
    THRIFT_LIMIT,
    Paths,
    PipelineError,
    run_stage,
    write_stage,
)

log = logging.getLogger(__name__)

CATALOG_SCHEMA = pa.schema(
    [
        ("col", pa.int32()),
        ("feature_id", pa.int64()),
        ("table", pa.string()),
        ("category", pa.string()),
        ("operation", pa.string()),
        ("target", pa.string()),
        ("leak_hash", pa.bool_()),
        ("id_token", pa.bool_()),
        ("sample_token", pa.bool_()),
    ]
)

_HEX = re.compile(r"[0-9a-f]{32,64}")

# A 5-digit number in the MLRan sample_id range, not part of a longer token.
_ID = re.compile(r"(?<![0-9a-z])(1[0-5][0-9]{3})(?![0-9])")

SAMPLE_TOKEN = "<SAMPLE>"


class ConvertError(PipelineError):
    pass


def open_parquet(path) -> pq.ParquetFile:
    return pq.ParquetFile(
        path, thrift_string_size_limit=THRIFT_LIMIT, thrift_container_size_limit=THRIFT_LIMIT
    )


def parse_name(table: str, name: str) -> tuple[str, str, str, bool]:
    """Split a feature name into (category, operation, target, well_formed).

    API:NtWriteFile               -> ("API", "CALL", "NtWriteFile")
    FILE:WRITTEN:c:\\x\\a.txt      -> ("FILE", "WRITTEN", "c:\\x\\a.txt")
    STRING:c:\\windows            -> ("STRING", "", "c:\\windows")   (colons kept in target)
    """
    prefix, has_op = TABLE_PREFIX[table]
    head = prefix + ":"
    if not name.startswith(head):
        return prefix, "", name, False
    rest = name[len(head) :]
    if not has_op:
        op = "CALL" if table == "1_api" else ""
        return prefix, op, rest, True
    op, sep, target = rest.partition(":")
    return prefix, op, target, bool(sep)


def _safe_strings(values: list[str]) -> tuple[pa.Array, int]:
    """Arrow rejects lone UTF-16 surrogates found in some extracted strings; escape them."""
    try:
        return pa.array(values, type=pa.string()), 0
    except (UnicodeEncodeError, pa.ArrowInvalid, pa.ArrowTypeError):
        fixed = [v.encode("utf-8", "backslashreplace").decode("utf-8") for v in values]
        n_changed = sum(a != b for a, b in zip(values, fixed, strict=True))
        return pa.array(fixed, type=pa.string()), n_changed


def _hash_set(paths: Paths) -> set[str]:
    md = pd.read_csv(paths.metadata_csv, usecols=["sha256", "sha1", "md5"])
    return {h.lower() for c in md.columns for h in md[c].dropna()}


def id_tokens(text: str, ids: set[int]) -> set[int]:
    """Sample IDs named inside a feature string (e.g. temp\\11846.exe -> {11846})."""
    return {int(m) for m in _ID.findall(text.lower())} & ids


def _convert_table(
    paths: Paths,
    table: str,
    row_ids: np.ndarray,
    col_offset: int,
    hashes: set[str],
    writer,
    candidates: list[tuple[int, str, str, str, str]],
) -> tuple[sp.csc_matrix, dict]:
    t0 = time.time()
    pf = open_parquet(paths.raw_dataset / f"{table}_dataset.parquet")
    names_all = pf.schema_arrow.names
    if "sample_id" not in names_all:
        raise ConvertError(f"{table}: no sample_id column")
    cols = [c for c in names_all if c != "sample_id"]

    sid = pf.read(columns=["sample_id"]).column(0).to_numpy()
    if len(np.unique(sid)) != len(sid):
        raise ConvertError(f"{table}: duplicate sample_id")
    idx = pd.Index(sid).get_indexer(row_ids)
    if (idx < 0).any():
        raise ConvertError(f"{table}: {int((idx < 0).sum())} labeled samples missing")
    extra = sorted(set(sid.tolist()) - set(row_ids.tolist()))
    unexpected_extra = sorted(set(extra) - KNOWN_EXTRA_IDS)
    if unexpected_extra:
        log.warning("%s: unlabeled sample_ids dropped: %s", table, unexpected_extra)

    blocks: list[sp.csc_matrix] = []
    non_binary = 0
    for i in range(0, len(cols), COLUMN_BATCH):
        batch = pf.read(columns=cols[i : i + COLUMN_BATCH])
        dense = np.column_stack([c.to_numpy() for c in batch.columns])[idx]
        non_binary += int(((dense != 0) & (dense != 1)).sum())
        blocks.append(sp.csc_matrix((dense != 0).astype(np.uint8)))
        del batch, dense
    if non_binary:
        raise ConvertError(f"{table}: {non_binary} cells outside {{0,1}}")
    X = sp.hstack(blocks, format="csc")

    # Feature catalog for this table.
    with open(paths.raw_dataset / f"{table}_feature_names_dic.json") as f:
        dic: dict[str, str] = json.load(f)
    dict_missing = [c for c in cols if c not in dic]
    if dict_missing:
        raise ConvertError(f"{table}: {len(dict_missing)} columns missing from name dictionary")
    dict_unused = len(set(dic) - set(cols))

    parsed = [parse_name(table, dic[c]) for c in cols]
    malformed = sum(not p[3] for p in parsed)
    targets = [p[2] for p in parsed]
    leak = [any(m in hashes for m in _HEX.findall(dic[c].lower())) for c in cols]
    id_set = set(row_ids.tolist())
    has_id = [bool(id_tokens(t, id_set)) for t in targets]
    for j in np.flatnonzero(has_id):
        candidates.append((col_offset + int(j), table, *parsed[j][:3]))
    target_arr, n_sanitized = _safe_strings(targets)
    writer.write_table(
        pa.table(
            {
                "col": pa.array(np.arange(col_offset, col_offset + len(cols)), pa.int32()),
                "feature_id": pa.array([int(c) for c in cols], pa.int64()),
                "table": pa.array([table] * len(cols), pa.string()),
                "category": pa.array([p[0] for p in parsed], pa.string()),
                "operation": pa.array([p[1] for p in parsed], pa.string()),
                "target": target_arr,
                "leak_hash": pa.array(leak, pa.bool_()),
                "id_token": pa.array(has_id, pa.bool_()),
                "sample_token": pa.array([False] * len(cols), pa.bool_()),
            },
            schema=CATALOG_SCHEMA,
        )
    )
    del dic, parsed, targets

    col_df = X.getnnz(axis=0)
    stats = {
        "rows_in": len(sid),
        "rows_out": len(row_ids),
        "unlabeled_ids_dropped": extra,
        "features": len(cols),
        "nonzero_cells": int(X.nnz),
        "density": round(X.nnz / max(1, X.shape[0] * X.shape[1]), 7),
        "features_in_one_sample": int((col_df == 1).sum()),
        "features_in_no_kept_sample": int((col_df == 0).sum()),
        "malformed_names": malformed,
        "names_sanitized": n_sanitized,
        "dictionary_entries_unused": dict_unused,
        "leak_hash_features": int(sum(leak)),
        "id_token_features": int(sum(has_id)),
        "seconds": round(time.time() - t0, 1),
    }
    log.info("%s: %d features, %d non-zero (%.1fs)", table, len(cols), X.nnz, stats["seconds"])
    return X, stats


def normalize_sample_tokens(
    X: sp.csr_matrix,
    candidates: list[tuple[int, str, str, str, str]],
    row_ids: np.ndarray,
) -> tuple[sp.csr_matrix, pa.Table, dict]:
    """Move cells whose feature names the row's OWN sample ID into a shared <SAMPLE> feature.

    Cuckoo named each submission after its sample_id, so a sample writing itself to a Run key
    shows up as ...\\Run\\11846 and only ever in row 11846. Sample IDs encode the label
    (ransomware < 13000), so these features are label proxies and never generalise. Rewriting
    the own ID to <SAMPLE> keeps the behaviour and removes the identity. Cells naming a
    different sample's ID (coincidental numbers) are left untouched.
    """
    Xc = X.tocsc()
    n_rows, n_cols = X.shape
    new_cols: dict[tuple[str, str, str, str], int] = {}
    add_r: list[int] = []
    add_c: list[int] = []
    del_r: list[int] = []
    del_c: list[int] = []
    for col, table, category, op, target in candidates:
        rows = Xc.indices[Xc.indptr[col] : Xc.indptr[col + 1]]
        tokens = id_tokens(target, set(row_ids[rows].tolist()))
        for r in rows:
            sid = int(row_ids[r])
            if sid not in tokens:
                continue
            norm = re.sub(rf"(?<![0-9a-z]){sid}(?![0-9])", SAMPLE_TOKEN, target)
            key = (table, category, op, norm)
            if key not in new_cols:
                new_cols[key] = n_cols + len(new_cols)
            add_r.append(int(r))
            add_c.append(new_cols[key] - n_cols)
            del_r.append(int(r))
            del_c.append(col)

    removed = sp.csr_matrix(
        (np.ones(len(del_r), np.uint8), (del_r, del_c)), shape=X.shape, dtype=np.uint8
    )
    kept = (X - removed).tocsr()
    kept.eliminate_zeros()
    added = sp.csr_matrix(
        (np.ones(len(add_r), np.int32), (add_r, add_c)), shape=(n_rows, len(new_cols))
    )
    added.data = np.minimum(added.data, 1)
    X_new = sp.hstack([kept, added.astype(np.uint8)], format="csr")

    keys = list(new_cols)
    k = len(keys)
    cat = pa.table(
        {
            "col": pa.array(np.arange(n_cols, n_cols + k), pa.int32()),
            "feature_id": pa.array(-np.arange(1, k + 1), pa.int64()),  # derived: negative ids
            "table": pa.array([t[0] for t in keys], pa.string()),
            "category": pa.array([t[1] for t in keys], pa.string()),
            "operation": pa.array([t[2] for t in keys], pa.string()),
            "target": _safe_strings([t[3] for t in keys])[0],
            "leak_hash": pa.array([False] * k, pa.bool_()),
            "id_token": pa.array([False] * k, pa.bool_()),
            "sample_token": pa.array([True] * k, pa.bool_()),
        },
        schema=CATALOG_SCHEMA,
    )
    stats = {
        "id_token_features": len(candidates),
        "own_id_cells_moved": len(del_r),
        "sample_token_features_created": k,
        "samples_affected": len(set(del_r)),
    }
    return X_new, cat, stats


def run(paths: Paths) -> None:
    verify(paths)
    paths.ensure()

    labels, label_stats = build_labels(paths)
    labels.to_csv(paths.labels, index=False)
    write_stage(paths.stage_report, "labels", label_stats)

    row_ids = labels.sample_id.to_numpy()
    hashes = _hash_set(paths)

    blocks: list[sp.csc_matrix] = []
    per_table: dict[str, dict] = {}
    col_table: list[np.ndarray] = []
    candidates: list[tuple[int, str, str, str, str]] = []
    offset = 0
    tmp_catalog = paths.catalog.with_suffix(".parquet.part")
    with pq.ParquetWriter(tmp_catalog, CATALOG_SCHEMA) as writer:
        for table in TABLES:
            X_t, stats = _convert_table(paths, table, row_ids, offset, hashes, writer, candidates)
            blocks.append(X_t)
            per_table[table] = stats
            col_table.append(np.full(X_t.shape[1], table, dtype=object))
            offset += X_t.shape[1]
        X = sp.hstack(blocks, format="csr")
        del blocks
        X, token_catalog, token_stats = normalize_sample_tokens(X, candidates, row_ids)
        writer.write_table(token_catalog)
        col_table.append(token_catalog.column("table").to_numpy(zero_copy_only=False))
    tmp_catalog.rename(paths.catalog)
    X.sort_indices()
    col_table_arr = np.concatenate(col_table)
    table_nnz = {t: X[:, np.flatnonzero(col_table_arr == t)].getnnz(axis=1) for t in TABLES}
    sp.save_npz(paths.matrix, X, compressed=True)

    samples = pd.DataFrame({"sample_id": row_ids, "row": np.arange(len(row_ids))})
    for table in TABLES:
        samples[f"n_{table}"] = table_nnz[table]
    samples["n_dynamic"] = sum(table_nnz[t] for t in DYNAMIC_TABLES)
    # No runtime behaviour recorded: the sample most likely never executed in the sandbox.
    samples["failed_run"] = samples.n_dynamic == 0
    samples.to_parquet(paths.samples, index=False)

    # Exact duplicate feature rows (all features).
    digests = pd.Series(
        [
            hashlib.sha1(X.indices[X.indptr[i] : X.indptr[i + 1]].tobytes()).hexdigest()
            for i in range(X.shape[0])
        ]
    )
    dup_mask = digests.duplicated(keep=False)
    lab = labels.label.to_numpy()

    write_stage(
        paths.stage_report,
        "convert",
        {
            "samples": X.shape[0],
            "features": X.shape[1],
            "nonzero_cells": int(X.nnz),
            "failed_runs": {
                "ransomware": int((samples.failed_run & (lab == 1)).sum()),
                "goodware": int((samples.failed_run & (lab == 0)).sum()),
            },
            "exact_duplicate_feature_rows": int(dup_mask.sum()),
            "exact_duplicate_feature_groups": int(digests[dup_mask].nunique()),
            "exact_duplicate_rate": round(float(dup_mask.mean()), 5),
            "sample_id_normalisation": token_stats,
            "tables": per_table,
        },
    )
    log.info("convert done: %s, %d non-zero", X.shape, X.nnz)


if __name__ == "__main__":
    sys.exit(run_stage(run))
