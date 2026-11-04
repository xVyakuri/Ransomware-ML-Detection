# Compare a smaple of 20 random raw records to derived features.
"""Trace 20 random samples from the raw files to every derived artifact.

For each sample: one present and one absent feature are re-read from the raw Parquet cell and
the raw name dictionary, and compared with X.npz and feature_catalog.parquet. The label is
re-read from the raw metadata CSV, and two aggregates are recomputed from raw columns.
Writes reports/audit_trace.md.
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from src.build_labels import read_labels
from src.common import AUDIT_N_SAMPLES, AUDIT_SEED, TABLES, Paths, run_stage, write_stage
from src.parse_raw import open_parquet, parse_name

log = logging.getLogger(__name__)


def run(paths: Paths) -> bool:
    X = sp.load_npz(paths.matrix).tocsr()
    cat = pq.read_table(paths.catalog).to_pandas()
    samples = pd.read_parquet(paths.samples)
    labels = read_labels(paths)
    agg = pd.read_parquet(paths.aggregates).set_index("sample_id")
    raw_md = pd.read_csv(paths.metadata_csv).set_index("sample_id")
    table_of_col = cat.table.to_numpy()
    # Raw-traceable columns map 1:1 to a raw Parquet column. Columns naming a sample ID were
    # rewritten by parse_raw.normalize_sample_tokens and are traced separately below.
    traceable = ~(cat.id_token.to_numpy() | cat.sample_token.to_numpy())

    rng = np.random.default_rng(AUDIT_SEED)
    rows = np.sort(rng.choice(len(samples), size=AUDIT_N_SAMPLES, replace=False))

    checks: list[dict] = []
    for r in rows:
        sid = int(samples.sample_id.iat[r])
        present_all = X.indices[X.indptr[r] : X.indptr[r + 1]]
        present = present_all[traceable[present_all]]
        tables_here = sorted(set(table_of_col[present])) if len(present) else []
        picks: list[tuple[int, int]] = []
        if tables_here:
            t = tables_here[rng.integers(len(tables_here))]
            cand = present[table_of_col[present] == t]
            picks.append((int(cand[rng.integers(len(cand))]), 1))
        t_abs = TABLES[rng.integers(len(TABLES))]
        tcols = cat.col.to_numpy()[(table_of_col == t_abs) & traceable]
        absent = np.setdiff1d(tcols, present_all, assume_unique=True)
        picks.append((int(absent[rng.integers(len(absent))]), 0))
        for col, expect in picks:
            checks.append({"sample_id": sid, "col": col, "expect": expect})

    # Re-read raw cells and names, grouped by table so each file opens once.
    by_table: dict[str, list[dict]] = {}
    for c in checks:
        by_table.setdefault(table_of_col[c["col"]], []).append(c)
    for table, items in by_table.items():
        pf = open_parquet(paths.raw_dataset / f"{table}_dataset.parquet")
        with open(paths.raw_dataset / f"{table}_feature_names_dic.json") as f:
            dic = json.load(f)
        for c in items:
            fid = int(cat.feature_id.iat[c["col"]])
            raw = (
                pf.read(columns=["sample_id", str(fid)])
                .to_pandas(ignore_metadata=True)
                .set_index("sample_id")
            )
            c["table"] = table
            c["feature_id"] = fid
            c["raw_value"] = int(raw.at[c["sample_id"], str(fid)])
            c["X_value"] = int(
                X[int(samples.index[samples.sample_id == c["sample_id"]][0]), c["col"]]
            )
            category, op, target, _ = parse_name(table, dic[str(fid)])
            row = cat.iloc[c["col"]]
            c["name_matches"] = (category, op) == (row.category, row.operation) and (
                target == row.target
                or target.encode("utf-8", "backslashreplace").decode() == row.target
            )
            c["raw_name"] = dic[str(fid)][:80]
            c["ok"] = c["raw_value"] == c["X_value"] == c["expect"] and c["name_matches"]
        del dic

    # <SAMPLE> features: the raw row must hold an original feature naming its own ID.
    token_rows = []
    id_cols = cat[cat.id_token]
    for r in rows:
        sid = int(samples.sample_id.iat[r])
        present_all = X.indices[X.indptr[r] : X.indptr[r + 1]]
        derived = present_all[cat.sample_token.to_numpy()[present_all]]
        if not len(derived):
            continue
        d = int(derived[rng.integers(len(derived))])
        row = cat.iloc[d]
        same = id_cols[(id_cols.table == row.table) & (id_cols.operation == row.operation)]
        restored = row.target.replace("<SAMPLE>", str(sid))
        src = same[same.target == restored]
        raw_value = 0
        if len(src):
            fid = str(int(src.feature_id.iat[0]))
            pf = open_parquet(paths.raw_dataset / f"{row.table}_dataset.parquet")
            raw = pf.read(columns=["sample_id", fid]).to_pandas(ignore_metadata=True)
            raw_value = int(raw.set_index("sample_id").at[sid, fid])
        token_rows.append(
            {
                "sample_id": sid,
                "table": row.table,
                "derived_name": f"{row.category}:{row.operation}:{row.target}"[:80],
                "raw_source_found": bool(len(src)),
                "raw_value": raw_value,
                "ok": bool(len(src)) and raw_value == 1,
            }
        )
    token_df = pd.DataFrame(
        token_rows,
        columns=["sample_id", "table", "derived_name", "raw_source_found", "raw_value", "ok"],
    )

    # Labels and aggregates against raw.
    label_rows = []
    api = (
        open_parquet(paths.raw_dataset / "1_api_dataset.parquet")
        .read()
        .to_pandas(ignore_metadata=True)
    )
    api = api.set_index("sample_id")
    sig = (
        open_parquet(paths.raw_dataset / "9_sig_dataset.parquet")
        .read()
        .to_pandas(ignore_metadata=True)
    )
    sig = sig.set_index("sample_id")
    lab = labels.set_index("sample_id")
    for r in rows:
        sid = int(samples.sample_id.iat[r])
        raw_label = int(raw_md.at[sid, "sample_type"])
        raw_family = str(raw_md.at[sid, "ransomware_family"])
        api_raw = int(api.loc[sid].sum())
        sig_raw = int(sig.loc[sid].sum())
        ok = (
            raw_label == int(lab.at[sid, "label"])
            and raw_family == str(lab.at[sid, "family"])
            and api_raw == int(agg.at[sid, "api_n"])
            and sig_raw == int(agg.at[sid, "sig_n"])
        )
        label_rows.append(
            {
                "sample_id": sid,
                "raw_label": raw_label,
                "label": int(lab.at[sid, "label"]),
                "raw_family": raw_family,
                "family": str(lab.at[sid, "family"]),
                "raw_api_n": api_raw,
                "agg_api_n": int(agg.at[sid, "api_n"]),
                "raw_sig_n": sig_raw,
                "agg_sig_n": int(agg.at[sid, "sig_n"]),
                "ok": ok,
            }
        )

    feat_df = pd.DataFrame(checks)[
        [
            "sample_id",
            "table",
            "feature_id",
            "raw_name",
            "expect",
            "raw_value",
            "X_value",
            "name_matches",
            "ok",
        ]
    ]
    lab_df = pd.DataFrame(label_rows)
    passed = bool(feat_df.ok.all() and lab_df.ok.all() and token_df.ok.all())

    def md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, row in df.iterrows():
            cells = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in cols]
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out)

    md = [
        "# Audit trace: raw records vs derived artifacts",
        "",
        f"Seed {AUDIT_SEED}, {AUDIT_N_SAMPLES} random samples. Result: "
        f"**{'PASS' if passed else 'FAIL'}**.",
        "",
        "## Feature cells (raw Parquet cell and name dictionary vs X.npz and feature catalog)",
        "",
        md_table(feat_df),
        "",
        "## <SAMPLE>-normalised features (derived cell vs raw cell naming the own sample ID)",
        "",
        md_table(token_df) if len(token_df) else "No audited sample has <SAMPLE> features.",
        "",
        "## Labels and aggregates (raw metadata and raw columns vs labels.csv and aggregates)",
        "",
        md_table(lab_df),
        "",
    ]
    (paths.reports / "audit_trace.md").write_text("\n".join(md))
    write_stage(
        paths.stage_report,
        "audit",
        {
            "passed": passed,
            "samples": AUDIT_N_SAMPLES,
            "feature_cells_checked": len(feat_df),
            "feature_cells_failed": int((~feat_df.ok).sum()),
            "label_rows_failed": int((~lab_df.ok).sum()),
            "sample_token_cells_checked": len(token_df),
            "sample_token_cells_failed": int((~token_df.ok).sum()),
        },
    )
    (log.info if passed else log.error)("audit %s", "PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    sys.exit(run_stage(run))
