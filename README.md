# Ransomware detection on Windows with machine learning

This project compares models for distinguishing ransomware from benign software
using MLRan sandbox behavior features. It evaluates detection quality, false alarms,
and generalization to ransomware families absent from training.

MLRan supplies binary feature presence, without event order or timestamps. This
project cannot measure detection delay or whether an alert arrives before files are
lost. Models predict ransomware versus benign, not the ransomware family.

## Setup and full run

Requirements: Linux or macOS, Python 3.11+, and Make. Setup uses Python's built-in
virtual environment support and pip; uv and conda are not required.
Parsing can use about 15 GB RAM; use a machine with at least 16 GB.

1. Download and extract MLRan as described in [data/README.md](data/README.md).
2. Review [config/config.yaml](config/config.yaml).
3. Run from the project root:

```bash
make setup
make data
make reproduce
```

`make data` verifies the raw snapshot, builds features and grouped splits, and runs
quality and raw-data spot checks. `make reproduce` verifies prepared data, trains
configured experiments, evaluates them, and generates reports. Raw files are read-only.

## Configuration and commands

Change settings in [config/config.yaml](config/config.yaml). The
[configuration guide](docs/CONFIGURATION.md) explains which stages to rerun after
changing settings. `make setup` installs `requirements.txt` into `.venv`.
Run `make help` to see targets and options.

Use `CONFIG` to select a YAML file and `OUTPUT_DIR` to override its results location:

```bash
make reproduce CONFIG=config/config.yaml OUTPUT_DIR=results-comparison
```

| Command | Purpose |
|---|---|
| `make help` | List commands and configuration options |
| `make setup` | Create `.venv` and install requirements using pip |
| `make data` | Build processed data, splits, quality checks, and audit |
| `make verify` | Verify raw-file SHA-256 hashes against the manifest |
| `make train` | Train enabled models on the configured training seed |
| `make experiments` | Run configured seeds and comparison experiments |
| `make evaluate` | Evaluate completed runs compatible with the configuration |
| `make report` | Generate the results README, tables, and charts |
| `make test` | Run automated tests |
| `make lint` | Byte-compile all source and test modules, including experiment scripts |
| `make clean` | Remove generated results; preserve raw/processed data and unrecognized files |

Individual data stages are available as `make snapshot`, `make parse`,
`make features`, `make splits`, `make quality`, and `make spot-check`.

## Models and evaluation

The default comparison includes a file-write-feature count rule, Logistic Regression,
Random Forest, and benign-only K-means distance. Supervised models use presence
features and aggregates; K-means uses aggregates.

The default split is grouped 70/15/15 training/validation/test. Ransomware families,
shared binaries, and near-duplicates stay together. Eight additional families form
a fixed holdout. Failed runs are excluded. Vocabulary filtering and preprocessing
are fitted only on the appropriate training cohort. See [docs/PARSER.md](docs/PARSER.md)
for the data contract and leakage checks.

Thresholds target empirical FPR of at most 1% on benign validation samples, then stay
fixed for evaluation. Test FPR can differ. Both ordinary test and fixed holdout contain
unseen ransomware families; holdout FPR reuses the ordinary test benign samples.

Experiments compare feature-table removals, removal of sample-token presence
features, Random Forest class weighting, and variability across split seeds.

## Reading results

Start with **`results/README.md`** after generating reports. It explains findings,
caught/missed sample counts, experiment meanings, metrics, and artifact locations.
Models, predictions, metrics, tables, and figures are generated under `results/`
and ignored by Git.

Data provenance and quality evidence are in `reports/raw_manifest.csv`,
`reports/stage_report.json`, and `reports/audit_trace.md`.

## Limitations

- Curated Windows sandbox samples do not represent production EDR telemetry.
- Benign coverage may not adequately represent backups, builds, and archive extraction.
- Collection artifacts and different sample eras may influence performance.
- Precision depends on the evaluation cohort's ransomware prevalence.
- Seed variation is not an independent confidence interval.
- Presence features describe distinct behaviors, not event frequencies or rates.

## Citation

Andrew Xie, Robert Wingert, Lucas Brown, Joshua Waltman, Gabriel Gradski,
November 2026, "Early Ransomware/Intrusion Detection on Windows with Machine Learning",
https://github.com/xVyakuri/Ransomware-ML-Detection/

## References
Mayur Rele, John Samuel, Dipti Patil, Udaya Krishnan, etc., Published 2025, Accessed October 2026, “Exploring Ransomware Detection Based on Artificial Intelligence and Machine Learning”, https://www.sciencedirect.com/science/article/pii/S1877050925000146
Indra Chaudhary, Suyash Adhikari, Published August 2024, Accessed October 2026, “Ransomware Detection using Machine Learning Techniques”, https://www.researchgate.net/publication/382931458_Ransomware_Detection_Using_Machine_Learning_Techniques
Onwuegbuche, Olaoluwa, Jurcut, Pasquale, Updated October 2026, Accessed October 2026, “A behavioural dataset for ransomware analysis and detection. Journal of Network and Computer Applications, 104475.”, https://github.com/faithfulco/mlran/tree/main
Onwuegbuche, Olaoluwa, Jurcut, Pasquale, Updated October 2026, Accessed October 2026, “A behavioural dataset for ransomware analysis and detection. Journal of Network and Computer Applications, 104475.”, https://zenodo.org/records/23403681
