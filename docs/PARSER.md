# MLRan parser: data contract and decisions


## What MLRan provides

MLRan ([paper](https://arxiv.org/abs/2505.18613), [data](https://zenodo.org/records/23403681), [code](https://github.com/faithfulco/mlran)) is 4,880 Windows samples (2,330 ransomware in 64 families, 2,550 goodware) run in Cuckoo Sandbox 2.0.7 on Windows 7 with a 120 s timeout.

The release is nine wide Parquet tables, one per behaviour category. Every cell is 0/1 presence of one feature for one sample. There are no counts, timestamps or event order.

| Table | Features | Example feature name |
|---|---|---|
| 1_api | 313 | `API:NtWriteFile` |
| 2_reg | 525,505 | `REG:WRITTEN:HKEY_LOCAL_MACHINE\...\Run\x` |
| 3_file | 2,078,287 | `FILE:DELETED:c:\users\...\a.docx` |
| 4_dir | 158,123 | `DIRECTORY:ENUMERATED:c:\users\*` |
| 5_str | 3,632,119 | `STRING:vssadmin delete shadows` |
| 6_net | 4,814 | `NETWORK:CONNECTS_HOST:1.2.3.4` |
| 7_sys | 16,911 | `SYSTEM:MUTEX:x` |
| 8_drop | 51,651 | `DROP:EXTENSION:locked` |
| 9_sig | 203 | `SIGNATURE:ransomware_file_modifications` |

### Experiments this supports

Without event order there are no prefix windows, rates or API n-grams, so the **timeliness test (prefix 10/25/50/100%) is not run**. The report states this as a dataset limitation. All other experiments run on MLRan:

- **Unseen-family generalisation:** 8 held-out families.
- **Feature-family ablation:** leave one of the 9 tables out.
- **Class weighting at FPR 1%:** compared on the same split.
- **Variability:** 5 split seeds.

## Outputs (`data/processed/`, not committed)

| File | Content |
|---|---|
| `X.npz` | CSR `uint8`, 4,880 rows × 6,484,506 columns, 17.0M non-zero cells. Rows follow `samples.parquet`, columns follow `feature_catalog.parquet`. |
| `samples.parquet` | `sample_id`, `row`, non-zero count per table, `n_dynamic`, `failed_run` |
| `feature_catalog.parquet` | `col`, `feature_id` (negative = derived), `table`, `category`, `operation`, `target`, `leak_hash`, `id_token`, `sample_token` |
| `labels.csv` | `sample_id, sha256, label, family, type, first_seen_year, metadata_year, source` |
| `aggregates.parquet` | 41 per-sample count and breadth features |
| `split_manifest.csv` | `sample_id, label, family, type, first_seen_year, group_id, failed_run, holdout, split_seed0..4` |
| `near_dup_pairs.parquet` | Every sample pair with feature-set Jaccard ≥ 0.9 |

Committed reports: `reports/raw_manifest.csv`, `reports/stage_report.json`, `reports/audit_trace.md`.

## Decisions

### Snapshot

The raw files are never modified. `reports/raw_manifest.csv` records size and SHA-256 for the 18 Zenodo files and 3 metadata CSVs, which are pinned to commit `273fcac` of the authors' repository. Every stage verifies the snapshot first.

### Labels

Labels come only from `mlran_dataset_metadata.csv`. A sample with no metadata row is excluded, not treated as goodware. `first_seen_year` is the VirusTotal first-submission year, which is the paper's definition of first-seen. The second year column (`Year`) disagrees on 3,266 rows and is kept as `metadata_year` for audit only.

### Data quality

| Problem | Count | Handling |
|---|---|---|
| Extra rows in tables 3–9 with no metadata (11898, 11899, 12057, 12107) | 4 | Dropped. The Zenodo record lists them for removal. |
| Failed runs: no runtime behaviour, only strings and signatures | 196 ransomware, 186 goodware | Excluded from every split, split value `excluded` |
| The same binary submitted twice (`motif` and `curated` sources) | 4 pairs | Same group |
| Identical feature rows | 47 rows in 19 groups | Same group |
| Feature names containing a sample's own hash | 11 | Flagged `leak_hash`; the loader always drops them |
| Feature names containing the sample's own ID | 38,337 cells, 3,645 samples | Rewritten to `<SAMPLE>` (see below) |
| Features seen in exactly one sample | about 6.0M of 6.47M | Removed by the train-only `min_df` filter |

### Own sample ID is rewritten to `<SAMPLE>`

Cuckoo named each submitted file after its `sample_id`. A sample that copies itself to a Run key therefore produces `REG:WRITTEN:...\Run\11846`, which only that sample can ever have. Sample IDs also encode the label: ransomware IDs are below 13000. The converter rewrites the sample's own ID in these names to `<SAMPLE>`. This turns 38,337 one-off cells into 16,580 shared features, such as `...\Run\<SAMPLE>`, which keep the behaviour without the identity. Numbers that match some *other* sample's ID are left unchanged. The `no_own_sample_id_features` check verifies this.

### Grouping

Union-find merges samples into one group when any of these holds:

- They are ransomware of the same family.
- They share a SHA-256.
- Their feature sets have Jaccard ≥ 0.9.

The Jaccard pairs are computed exactly, with no approximation. A group never spans two splits.

Near-duplicate behaviour links 11 older crypto families into one group of 616 samples. These are critroni, cryptolocker, cryptowall, delshad, locker, magniber, matsnu, reveton, revil, trojan-ransom and vhd. The linked samples have large feature sets (median about 3,100 dynamic features) that are more than 90% identical. This points to shared builders or inconsistent AV family names. Smaller merges: gimemo+mbrlock, kollah+kovter, nefilim+nemty.

### Unseen-family holdout

8 families are drawn with seed 520 and removed before every split: 8base, avoslocker, bianlian, dharma, locky, philadelphia, royal and shade. They span crypto, modern and RaaS (188 samples). Only pure-ransomware groups can be drawn, so a merged group is held out whole or not at all. Locker is not covered; the holdout requires at least 3 of the 4 types.

The holdout contains no goodware. For FPR on the holdout, pair it with the goodware in `test`.

### Split

The split is grouped 70/15/15 and stratified by label, with seeds 0–4. Each group goes to the split furthest below its target. A group larger than half the smallest split target goes to train first, because it would distort val or test wherever it landed. In practice this is only the 616-sample group. Achieved fractions are within ±0.05 of the target for both classes and all seeds.

### Train-only fitting

`load()` counts `min_df` on the train rows only. Scaling, imputation and feature selection in the models must also be fitted on `b.train` only. The authors' pre-selected feature sets (`MI_selected…`, `RFE_selected…`, `MLRan_X_train/test`) are never read, because the authors selected them on their own split.

## Leakage checks (`make quality`)

`labels_traceable`, `groups_disjoint`, `sha256_disjoint`, `near_dups_disjoint`, `holdout_unseen`, `failed_runs_excluded`, `deterministic`, `split_fractions`, `vocab_fit_on_train`, `no_own_sample_id_features`, `no_author_preselected_features`. All pass on the current build.

## Resource use

| Step | Time | Peak RAM |
|---|---|---|
| `make data` (one-time build) | about 6 min | about 15 GB (Parquet footer of `5_str_dataset.parquet`: about 12 GB) |
| `load(seed, min_df=5, aggregates=True)` | under 1 s | about 1.8 GB |

The processed artifacts take 160 MB. Teammates with 8 GB machines use the shared `data/processed/` and never run `make parse`.

## Known limitations (for the report)

- **No event order.** There is no timeliness or early-detection result.
- **Submission file type survives in `<SAMPLE>` features.** `<SAMPLE>.dll` appears only in ransomware (195 samples). `<SAMPLE>.tmp` (installer unpacking) is almost only in goodware (386 vs 4). 370 of 416 `.zip` submissions are goodware. Run the `sample_tokens=False` ablation and report the difference.
- **Era skew.** The median first-seen year is 2016 for goodware and 2021 for ransomware. Some features may track the era rather than the behaviour. The split is not temporal.
- **Sandbox only.** All samples ran on Windows 7 with a 120 s timeout, so the data does not represent current EDR telemetry or evasive samples.
- **Presence only.** "Read/write ratio" and the other aggregates count distinct files or keys, not events.
