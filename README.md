# Early Ransomware/Intrusion Detection on Windows with Machine Learning

## What are we addressing?
Can a defender flag a Windows ransomware execution early enough to isolate the host before most files are lost, without flooding analysts with false alerts on backups, builds, and archive extraction?: Here, we will be addressing this with a creation of an anomaly detector utilizing machine learning models to detect and classify several Ransomware attack families against regular benign behavior.

## Why use Machine Learning over a rule?
A single rule (e.g., "many file writes per second") usually backfires on regular day-to-day organizatoin behavior, such as backups, archive extraction, software builds, and legitimate encryption. Machine Learning can instead combine operation rates, breadth of files touched, read/write relationships, and operation order to properly distinguish between malicious and benign behavior.

## Reference and Citation
If referencing this project, cite this github repository with the following:
Andrew Xie, Robert Wingert, Lucas Brown, Joshua Waltman, Gabriel Gradski, Update November 2026, "Early Ransomware/Intrusion Detection on Windows with Machine Learning", URL=https://github.com/xVyakuri/Ransomware-ML-Detection/

## Quick start
Requirements: Linux or macOS, and installed python3 (3.10+)
```bash
git clone https://github.com/xVyakuri/Ransomware-ML-Detection/ && cd Ransomware-ML-Detection

# 1. Download MLRan's public dataset yourself and put it in data/ (see data/README.md)
# 2. Fill in the TODO fields in config/config.yaml for raw format and field names.
make setup
make reproduce
```

## Dataset
We are using the public dataset MLRan, created by Onwuegbuche, Olaoluwa, Jurcut, and Pasquale). This dataset contains ransomware and goodware samples from 2006 to 2024 executed in a sandbox, containing: 
- 4,800+ samples
- 64 ransomware families
- 4 ransomware types (locker, crypto, RaaS, and modern)
- 9 behavior categories (API calls, registry keys, file/directory operations, strings, network, processes, dropped files, and signatures)
- About 6.4 million raw features.
See data/README.md for setup and verification as the dataset will NOT be in this repository.

## Github Structure
### Pipeline
The pipeline for this anomaly detector follows this step framework:
| Step number | Stage |
| ----------- | ----- |
| 1 | Verify raw snapshot & record manifest |
| 2 | Parse new records to normalized events table |
| 3 | Build traceable labels |
| 4 | Quality checks and cleaning |
| 5 | Grouped 70/15/15 training/validation/testing split |
| 6 | Hold out 8 whole families |
| 7 | Prefix windows (10/25/50/100%) |
| 8 | Feature extraction |
| 9 | Train only on scaling/imputation/selection |
| 10 | Spot check between 20 random raw records and derived features |
| 11 | Train models
| 12 | Pick FPR 1% thresholds on validation |
| 13 | Evaluate |
| 14 | Experiments |
| 15 | Final figures and tables for reporting |

### Normalized Event Table
Each row per observed event is ordered within an execution. Early-detection analysis requires the raw data to preserve event order or time. The following is the normalized event table for this dataset:
| Column | Description |
| ------ | ----------- |
| `execution_id` | Unique run (from `sample_id`) |
| `event_order` | Position within the execution |
| `event_time` | Time within the execution, if available |
| `category` | Behavior category (API call, file/directory operation, registry, etc.) |
| `operation` | Operation or API name |
| `target` | File, directory, key, or host the operation touched |

### Labels
Each row per observed event tracable to the raw data. Unlabeled events are never treated as benign, where they are dropped and logged, and class balance is measured before training following the labels.csv file:
`execution_id, label (benign | ransomware), family, type, first_seen_year, source`

### Leakage-safe split
For our split, we are using a fixed seed with a grouped 70/15/15 training/validation/testing split. Scaling, imputation, and feature selection are fitted on training data only. Moreover, we are using separate groups, where all prefixes of a single execution always stay together, of:
- Ransomware family
- Goodware
- Exact and near duplicates

### Features
| Feature family | Examples |
| -------------- | -------- |
| `op_counts_rates` | Operation counts and rates |
| `unique_targets` | Unique files and directories touched |
| `read_write_ratio` | Read/write relationship |
| `api_ngrams` | API call n-grams (order information) |

### Models
| Role | Method | Notes |
| ---- | ------ | ----- |
| Operational baseline | Threshold on 1 behavior count/rate | Threshold tuned on validation |
| Traditional ML baseline | Logistic Regression | Class-weighted |
| Primary model | Random Forest | Handles mixed, noisy count features |
| Anomaly comparisons | One-Class SVM and/or distance to nearest K-means centroid | Trained on benign training records only |
Alert thresholds are selected on the validation set to hit a 1% false-positive rate, then applied unchanged to test.

### Experiments & Stres tests
| Test | What changes |
| ---- | ------------ |
| Generalization | Hold out 8 whole families (at least 3 of 4 types, seeded) from train and validation and evaluate all models on them	|
| Data/feature | Remove one feature family at a time (same model and split)	|
| Model/decision | Random Forest class-weighted vs. unweighted, and same FPR 1% threshold rule |
| Timeliness | Prefix length 10/25/50/100%; recall vs. delay trade-off |
| Variability | 5 seeds with whole groups resampled	|

### Limitations
- MLRan's Public Dataset covers 2006 to 2024 Windows samples ran in a sandbox. It does not represent current EDR telemetry, evasive samples, or production prevalence. Results show relative model behavior on this dataset, not real-world alert volumes.
- Goodware in MLRan is a curated benign set, and it may not cover backup tools, builds, indexing, and archive extraction as thoroughly as the research question imples.

## Github Repository
```text
ransomware-early-detection/
├── Makefile
├── README.md
├── .gitignore                  # ignores data/*, results/*, .venv/, __pycache__
├── config/
│   └── config.yaml             # seed(s), split ratios, FPR target, prefix fractions,
│                               # held-out family count, model hyperparameters
├── data/                       # NOT tracked (only .gitkeep + README.md committed)
│   ├── README.md               # where to put MLRan, expected layout, expected checksum
│   ├── raw/                    # USER places MLRan here manually (treated as read-only)
│   ├── interim/                # normalized event table, labels.csv
│   └── processed/              # splits, prefix windows, feature matrices
├── src/
│   ├── __init__.py
│   ├── common.py               # config loading, seeding, paths, logging helpers
│   ├── check_raw.py            # verify raw snapshot exists, write/verify SHA-256 manifest
│   ├── parse_raw.py            # raw records -> normalized event table (execution_id, event_order/time, category, operation, target)
│   ├── build_labels.py         # labels.csv (benign/ransomware, family, type, year, source)
│   ├── quality_checks.py       # malformed/dropped records, missing fields, duplicate rates, class balance; writes per-stage log
│   ├── spot_check.py           # sample 20 random raw records, compare to derived features
│   ├── make_splits.py          # grouped 70/15/15 split (family + duplicate groups, prefixes of one execution kept together)
│   ├── make_holdout.py         # choose 8 held-out families (>=3 of 4 types, seeded)
│   ├── build_prefixes.py       # prefix windows at 10/25/50/100%
│   ├── features.py             # counts/rates, unique files/dirs, read/write ratio, API n-grams, defines feature families for ablation
│   ├── preprocess.py           # scaling, imputation, feature selection (fit on train only)
│   ├── models.py               # model factory: threshold baseline, LogReg, RF, One-Class SVM, K-means distance
│   ├── train.py                # train all models on a given split/seed
│   ├── thresholds.py           # validation-selected threshold at FPR 1%
│   ├── evaluate.py             # metrics on val/test: recall, FPR, PR/ROC, etc.
│   ├── exp_generalization.py   # held-out families test
│   ├── exp_ablation.py         # remove one feature family at a time
│   ├── exp_class_weight.py     # RF class-weighted vs. none
│   ├── exp_timeliness.py       # prefix 10/25/50/100%, recall vs. delay
│   ├── exp_variability.py      # 5 seeds, whole groups resampled
│   └── make_report.py          # final figures and tables from results/
├── tests/
│   ├── test_no_group_leakage.py     # no family/duplicate group spans train/val/test
│   ├── test_prefix_integrity.py     # all prefixes of one execution share a split
│   ├── test_holdout_families.py     # 8 families absent from train/val; type coverage >= 3
│   ├── test_fit_on_train_only.py    # scaler/imputer/selector never see val/test
│   ├── test_labels.py               # every execution has a label, and unlabeled does not mean benign
│   ├── test_split_ratios.py         # ~70/15/15 and fixed-seed determinism
│   └── test_threshold_rule.py       # threshold chosen on validation only
├── results/                    # NOT tracked, generated
│   ├── logs/                   # per-stage logs (dropped/malformed/missing/duplicates)
│   ├── metrics/                # CSV/JSON per model and experiment
│   ├── figures/                # PNGs/PDFs for the write-up
│   └── tables/                 # CSV/LaTeX/Markdown tables
└── requirements.txt
```

## References
Mayur Rele, John Samuel, Dipti Patil, Udaya Krishnan, etc., Published 2025, Accessed October 2026, “Exploring Ransomware Detection Based on Artificial Intelligence and Machine Learning”, https://www.sciencedirect.com/science/article/pii/S1877050925000146
Indra Chaudhary, Suyash Adhikari, Published August 2024, Accessed October 2026, “Ransomware Detection using Machine Learning Techniques”, https://www.researchgate.net/publication/382931458_Ransomware_Detection_Using_Machine_Learning_Techniques
Onwuegbuche, Olaoluwa, Jurcut, Pasquale, Updated October 2026, Accessed October 2026, “A behavioural dataset for ransomware analysis and detection. Journal of Network and Computer Applications, 104475.”, https://github.com/faithfulco/mlran/tree/main
Onwuegbuche, Olaoluwa, Jurcut, Pasquale, Updated October 2026, Accessed October 2026, “A behavioural dataset for ransomware analysis and detection. Journal of Network and Computer Applications, 104475.”, https://zenodo.org/records/23403681

## Parser (implemented)
Stages 1-6 and 10 of the pipeline are implemented in the `mlran_parser` package (`src/mlran_parser/`). The outline scripts in `src/` call into it. Data contract, design decisions and known limitations are in [docs/PARSER.md](docs/PARSER.md).

### Setup
1. Install [uv](https://docs.astral.sh/uv/).
2. Download `MLRan_full_dataset.zip` from [Zenodo record 23403681](https://zenodo.org/records/23403681) and unzip it to `data/raw/MLRan_full_dataset/`.
3. Create the environment with `uv sync --extra dev`.

### Build the data
```bash
uv run mlran all
```

| Stage | Outline script | Output |
|---|---|---|
| `snapshot` | `src/check_raw.py` | Pinned metadata in `data/raw/metadata/`, SHA-256 manifest in `reports/raw_manifest.csv` |
| `convert` | `src/parse_raw.py`, `src/build_labels.py` | `data/processed/X.npz`, `samples.parquet`, `feature_catalog.parquet`, `labels.csv` |
| `aggregates` | `src/features.py` | `data/processed/aggregates.parquet` |
| `split` | `src/make_splits.py`, `src/make_holdout.py` | `data/processed/split_manifest.csv`, `near_dup_pairs.parquet` |
| `leakcheck` | `src/quality_checks.py` | 11 leakage checks; non-zero exit code on failure |
| `audit` | `src/spot_check.py` | `reports/audit_trace.md` |

Each outline script runs its stage, for example `uv run python src/parse_raw.py`. Every stage writes its counts to `reports/stage_report.json`.

A full build takes about 6 minutes. `convert` peaks at about 15 GB RAM, because the Parquet footer of the 3.6M-column strings table alone takes about 12 GB. Run the build once on a machine with 16 GB or more and share `data/processed/` (160 MB). Loading a split peaks at about 1.8 GB.

### Use the data
```python
from mlran_parser.loader import load

b = load(seed=0, min_df=5)          # seeds 0-4 for the variability test
b.train.X, b.train.y                # scipy CSR, y: 1 = ransomware, 0 = goodware
b.val, b.test, b.holdout            # holdout = 8 unseen families (ransomware only)
b.features                          # one row per column: name, table, operation, target

load(seed=0, tables=[...])                            # feature-family ablation
load(seed=0, aggregates=True)                         # append 41 count/breadth features
load(seed=0, sparse_features=False, aggregates=True)  # aggregates only (threshold baseline)
load(seed=0, sample_tokens=False)                     # drop <SAMPLE> features (artifact ablation)
```
For the anomaly detectors, train on benign rows only: `b.train.X[b.train.y == 0]`.

### Test
```bash
uv run pytest
```
