# Configuration and results guide

Edit `config/config.yaml` before running commands. Every module loads it once at
startup. Make commands accept `CONFIG=path/to/settings.yaml` and optionally
`OUTPUT_DIR=path/to/results`. Module commands support the equivalent `--config`
and `--output-dir` arguments. These explicit command-line selections are the
only overrides to the YAML configuration.
Paths are relative to the project root. Avoid changing data contracts under
`dataset` unless deliberately migrating to a different raw release.

| Settings | Meaning | Required action |
|---|---|---|
| paths | Input/output locations | Ensure inputs exist at the new locations |
| dataset | Tables, mappings, metadata revision, excluded IDs | Rebuild snapshot and data |
| parser | Batch sizes and Parquet limits | Reparse to apply resource settings |
| splits, holdout | Grouping, seeds, fractions, unseen families | Rebuild splits and run quality checks; retrain |
| audit | Spot-check seed and sample count | Rerun spot check |
| features | Training vocabulary minimum and artifact/aggregate choices | Retrain |
| preprocessing, models | Imputation, scaling and model parameters | Retrain |
| training | Reference training seed and enabled models | Run training or experiments |
| threshold | Benign validation FPR target | Retrain/recalibrate |
| experiments | Enabled comparisons and ablation model | Run experiments and evaluation |
| report | Reference seed and image formats | Regenerate report only |
| baseline | Count-rule feature name | Retrain the count baseline |

For example, change `models.rf.n_estimators` to change forest size, or
`threshold.target_fpr` to change the tolerated validation false-positive rate.
The latter changes the trade-off between missed ransomware and false alarms.
The `training.seed` must exist in `splits.seeds`.

Use `make experiments`, `make evaluate`, and `make report` after changing training
settings. Use `make splits` and `make quality` first after changing splits.
Use `make setup` once to install requirements with venv and pip, then `make data`
to prepare the dataset. `make reproduce` runs training through reporting against
that prepared data. Use `make help` for the command list. `make clean` removes
generated outputs from the selected results directory while preserving data and
unrecognized files.

Each completed model stores a resolved configuration in `complete.json` alongside
input fingerprints and code/dependency provenance. Evaluation excludes incompatible
configurations. Report-only changes do not require retraining. Algorithmic constants
such as label encodings, sparse-matrix indexing and the score comparison operator
are implementation semantics rather than user tuning parameters.

Start reading results at `results/README.md`. Detailed CSVs
are supporting evidence, not the recommended first stop.
