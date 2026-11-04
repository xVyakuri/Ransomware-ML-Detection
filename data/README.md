# Data
MLRan is not stored in this repository. It is far too large for Github, so you must download it yourself locally and place it in `data/raw/` after cloning this repo and before running `make reproduce`.

## 1. Get MLRan
Check the dataset's license and terms of use before using or sharing derived files.
- Dataset is available via the following URL:  https://zenodo.org/records/23403681
- Paper: Onwuegbuche, Olaoluwa, Jurcut, Pasquale, Updated October 2026, Accessed October 2026, “A behavioural dataset for ransomware analysis and detection. Journal of Network and Computer Applications, 104475.”

## 2. Place dataset into data/raw/
Copy or extract the downloaded dataset into `data/raw/`, keeping its original file names and folder structure. Do not edit, rename, or re-save anything there. `data/raw/` is treated as an immutable snapshot, and the pipeline only ever reads from it.

```text
data/
├── README.md
├── raw/
│   ├── MLRan_full_dataset/   # extracted Parquet tables and feature dictionaries
│   └── metadata/            # pinned label metadata downloaded by the pipeline
└── processed/               # generated matrices, labels, aggregates, and splits
```

## 3. Verify and build
Run `make setup`, `make snapshot`, `make verify`, then `make data`.
The parser expects `data/raw/MLRan_full_dataset/` and downloads pinned label metadata
into `data/raw/metadata/`. Settings are in `config/config.yaml`.
Actual outputs are described in `docs/PARSER.md`. This release has presence features,
not ordered events or prefix windows.

