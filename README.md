# Ransomware-ML-Detection

Machine learning models that detect Windows ransomware from sandbox behaviour, trained and evaluated on the MLRan dataset (CSEC 520/620 Project 3).

The parser (`src/mlran_parser`) turns the raw MLRan release into leakage-safe, model-ready data. It runs before modelling and metrics. See [docs/PARSER.md](docs/PARSER.md) for the data contract, design decisions and known limitations.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Download `MLRan_full_dataset.zip` from [Zenodo record 23403681](https://zenodo.org/records/23403681) and unzip it into the repository root as `MLRan_full_dataset/`.
3. Create the environment:

```bash
uv sync --extra dev
```

## Build the data

```bash
uv run mlran all
```

The command runs these stages in order, and stops at the first failure:

| Stage | Output |
|---|---|
| `snapshot` | Downloads the pinned metadata and hashes every raw file into `reports/raw_manifest.csv` |
| `convert` | `data/processed/X.npz`, `samples.parquet`, `feature_catalog.parquet`, `labels.csv` |
| `aggregates` | `data/processed/aggregates.parquet` |
| `split` | `data/processed/split_manifest.csv`, `near_dup_pairs.parquet` |
| `leakcheck` | 11 leakage checks; a non-zero exit code means at least one failed |
| `audit` | `reports/audit_trace.md`: 20 random samples traced from raw cells to derived data |

A full build takes about 6 minutes. `convert` peaks at about 15 GB RAM, because the Parquet footer of the 3.6M-column strings table alone takes about 12 GB, in both pyarrow and DuckDB. Run the build once on a machine with 16 GB or more and share `data/processed/` (160 MB). Loading a split with `load()` peaks at about 1.8 GB. Every stage writes its counts to `reports/stage_report.json`. Run a single stage with `uv run mlran <stage>`, and check the raw files with `uv run mlran verify --full`.

## Use the data

```python
from mlran_parser.loader import load

b = load(seed=0, min_df=5)          # seeds 0-4 for the variability test
b.train.X, b.train.y                # scipy CSR, y: 1 = ransomware, 0 = goodware
b.val, b.test, b.holdout            # holdout = 8 unseen families (ransomware only)
b.features                          # one row per column: name, table, operation, target
```

Common options:

```python
load(seed=0, tables=[...])                          # feature-family ablation: leave one table out
load(seed=0, aggregates=True)                       # append 41 count/breadth features
load(seed=0, sparse_features=False, aggregates=True)  # aggregates only (threshold baseline)
load(seed=0, sample_tokens=False)                   # drop <SAMPLE> features (artifact ablation)
```

For the anomaly detectors, train on benign rows only: `b.train.X[b.train.y == 0]`.

## Test

```bash
uv run pytest
```
