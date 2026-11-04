from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from src import check_raw, quality_checks
from src.common import Paths
from src.features import load
from src.parse_raw import parse_name

from .conftest import EXTRA_ID, FAILED, REAL, build_fixture, real_data


@pytest.mark.parametrize(
    ("table", "name", "expected"),
    [
        ("1_api", "API:NtWriteFile", ("API", "CALL", "NtWriteFile", True)),
        ("3_file", "FILE:WRITTEN:c:\\x\\a.txt", ("FILE", "WRITTEN", "c:\\x\\a.txt", True)),
        ("5_str", "STRING:c:\\windows:x", ("STRING", "", "c:\\windows:x", True)),
        ("6_net", "NETWORK:CONNECTS_HOST:", ("NETWORK", "CONNECTS_HOST", "", True)),
        ("3_file", "REG:READ:HKLM", ("FILE", "", "REG:READ:HKLM", False)),
        ("2_reg", "REG:NOOP", ("REG", "NOOP", "", False)),
    ],
)
def test_parse_name(table, name, expected):
    assert parse_name(table, name) == expected


def test_matrix_matches_raw_cells(built: Paths):
    X = sp.load_npz(built.matrix).tocsr()
    cat = pq.read_table(built.catalog).to_pandas()
    samples = pd.read_parquet(built.samples)
    assert EXTRA_ID not in set(samples.sample_id)
    raw = pd.read_parquet(built.raw_dataset / "3_file_dataset.parquet").set_index("sample_id")
    file_cols = cat[(cat.table == "3_file") & ~cat.id_token & ~cat.sample_token]
    for c, fid in zip(file_cols.col, file_cols.feature_id, strict=True):
        expect = raw.loc[samples.sample_id, str(fid)].to_numpy()
        assert (X[:, c].toarray().ravel() == expect).all()


def test_failed_runs_flagged(built: Paths):
    s = pd.read_parquet(built.samples).set_index("sample_id")
    assert set(s.index[s.failed_run]) == FAILED


def test_own_sample_id_normalised(built: Paths):
    cat = pq.read_table(built.catalog).to_pandas()
    X = sp.load_npz(built.matrix).tocsc()
    tok = cat[cat.sample_token]
    assert tok.target.tolist() == ["c:\\temp\\<SAMPLE>.exe"]
    assert X[:, tok.col.iat[0]].nnz == 2  # 10010 and 10011 now share one feature
    for c in cat[cat.id_token].col:
        assert X[:, c].nnz == 0  # own-ID cells moved out


def test_hash_leak_flagged(built: Paths):
    cat = pq.read_table(built.catalog).to_pandas()
    assert cat.leak_hash.sum() == 1


def test_loader_ablation_and_aggregates(built: Paths):
    full = load(seed=0, min_df=1, paths=built)
    no_str = load(
        seed=0,
        min_df=1,
        paths=built,
        tables=[
            t for t in ("1_api", "2_reg", "3_file", "4_dir", "6_net", "7_sys", "8_drop", "9_sig")
        ],
    )
    assert "5_str" not in set(no_str.features.table)
    assert no_str.train.X.shape[1] < full.train.X.shape[1]
    agg = load(seed=0, sparse_features=False, aggregates=True, paths=built)
    assert set(agg.features.kind) == {"aggregate"}
    no_tok = load(seed=0, min_df=1, sample_tokens=False, paths=built)
    assert not no_tok.features.target.str.contains("<SAMPLE>").any()


def test_snapshot_detects_changed_raw_file(tmp_path):
    paths = build_fixture(tmp_path)
    check_raw.build_manifest(paths)
    check_raw.verify(paths, full_hash=True)
    target = paths.raw_dataset / "9_sig_feature_names_dic.json"
    target.write_text(target.read_text() + " ")
    with pytest.raises(check_raw.SnapshotError):
        check_raw.verify(paths)


@real_data
def test_real_artifacts_pass_leakage_checks():
    checks = quality_checks.run_all(REAL, write=False)
    assert all(c.passed for c in checks), [(c.name, c.detail) for c in checks if not c.passed]


def test_own_id_and_preselected_checks(built: Paths):
    assert quality_checks.check_no_own_id_features(built).passed
    assert quality_checks.check_no_preselected_inputs().passed


@real_data
def test_real_shapes():
    s = pd.read_parquet(REAL.samples)
    assert len(s) == 4880
    m = pd.read_csv(REAL.split_manifest)
    assert int(m.failed_run.sum()) == 382
    assert m.holdout.sum() > 0
