"""Synthetic mini-MLRan that reproduces the real dataset's quirks.

* sample_id is the first column in tables 1-2 and the last in tables 3-9
* tables 3-9 carry one extra, unlabeled sample (11898)
* some samples have no runtime behaviour (failed runs)
* one ransomware binary appears twice (shared sha256)
* two goodware samples are near duplicates
* feature names that contain the sample's own ID and its own hash
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlran_parser.config import TABLE_PREFIX, TABLES, Paths

N_FAMILIES = 20
PER_FAMILY = 5
N_GOOD = 100
TYPES = ["crypto", "locker", "raas", "modern"]
FEATURES_PER_TABLE = 30
EXTRA_ID = 11898
FAILED = {10003, 13005}


def _ids() -> tuple[list[int], list[int]]:
    ransom = list(range(10001, 10001 + N_FAMILIES * PER_FAMILY))
    good = list(range(13000, 13000 + N_GOOD))
    return ransom, good


def build_fixture(root: Path) -> Paths:
    rng = np.random.default_rng(0)
    raw = root / "MLRan_full_dataset"
    raw.mkdir(parents=True)
    paths = Paths(root=root, raw_dataset=raw, data=root / "data")
    paths.metadata_dir.mkdir(parents=True)

    ransom, good = _ids()
    ids = ransom + good
    sha = {s: f"{s:064x}" for s in ids}
    sha[10002] = sha[10001]  # duplicate binary, same family

    fam = {s: f"fam{(s - 10001) // PER_FAMILY:02d}" for s in ransom}
    md = pd.DataFrame(
        {
            "sample_id": ids,
            "sha256": [sha[s] for s in ids],
            "sample_type": [1] * len(ransom) + [0] * len(good),
            "ransomware_family": [fam[s] for s in ransom] + ["goodware"] * len(good),
            "family_label": [int(fam[s][3:]) + 1 for s in ransom] + [0] * len(good),
            "ransomware_type": [TYPES[int(fam[s][3:]) % 4] for s in ransom]
            + ["goodware"] * len(good),
            "type_label": [0] * len(ids),
            "sha1": [f"{s:040x}" for s in ids],
            "md5": [f"{s:032x}" for s in ids],
            "source": ["test"] * len(ids),
            "Year": [2015] * len(ids),
            "first_submission_date_year": [2016] * len(ids),
        }
    )
    md.to_csv(paths.metadata_csv, index=False)
    for name in ("ransomware_samples_families_description.csv", "goodware_samples_description.csv"):
        (paths.metadata_dir / name).write_text("x\n")

    col = 1
    for t in TABLES:
        table_ids = ids if t in ("1_api", "2_reg") else ids + [EXTRA_ID]
        names = {}
        prefix, has_op = TABLE_PREFIX[t]
        for _ in range(FEATURES_PER_TABLE):
            op = ("WRITTEN", "READ", "CREATED", "DELETED")[col % 4] if t == "3_file" else "OP"
            target = f"c:\\d{col % 3}\\{t}_{col}.txt"
            names[str(col)] = f"{prefix}:{op}:{target}" if has_op else f"{prefix}:{t}_{col}"
            col += 1
        if t == "3_file":
            # Own-ID file names: two samples each "see" their own submission file.
            names[str(col)] = "FILE:EXISTS:c:\\temp\\10010.exe"
            names[str(col + 1)] = "FILE:EXISTS:c:\\temp\\10011.exe"
            col += 2
        if t == "5_str":
            names[str(col)] = f"STRING:{sha[10020]}.exe"
            col += 1
        cols = list(names)
        data = (rng.random((len(table_ids), len(cols))) < 0.3).astype("int64")
        df = pd.DataFrame(data, columns=cols)
        df.insert(0, "sample_id", table_ids)
        df = df.set_index("sample_id")
        # Wire the special features.
        for c, n in names.items():
            if n.endswith("10010.exe"):
                df[c] = 0
                df.loc[10010, c] = 1
            elif n.endswith("10011.exe"):
                df[c] = 0
                df.loc[10011, c] = 1
            elif sha[10020] in n:
                df[c] = 0
                df.loc[10020, c] = 1
        # Failed runs: nothing in dynamic tables.
        if t not in ("5_str", "9_sig"):
            df.loc[sorted(FAILED)] = 0
        # Near duplicates: 13001 copies 13000.
        df.loc[13001] = df.loc[13000]
        df = df.reset_index()
        if t not in ("1_api", "2_reg"):
            df = df[[*cols, "sample_id"]]
        df.to_parquet(raw / f"{t}_dataset.parquet", index=False)
        (raw / f"{t}_feature_names_dic.json").write_text(json.dumps(names))
    return paths


@pytest.fixture(scope="session")
def fixture_paths(tmp_path_factory) -> Paths:
    return build_fixture(tmp_path_factory.mktemp("mlran"))


@pytest.fixture(scope="session")
def built(fixture_paths: Paths) -> Paths:
    from mlran_parser import aggregates, convert, snapshot, split

    snapshot.build_manifest(fixture_paths)
    convert.run(fixture_paths)
    aggregates.run(fixture_paths)
    split.run(fixture_paths)
    return fixture_paths


REAL = Paths.default(Path(__file__).resolve().parents[1])
real_data = pytest.mark.skipif(
    not REAL.split_manifest.exists(), reason="real processed artifacts not built"
)
