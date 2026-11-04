# Data
MLRan is not stored in this repository. It is far too large for Github, so you must download it yourself locally and place it in `data/raw/` after cloning this repo and before running `make reproduce`.

## 1. Get MLRan
Check the dataset's license and terms of use before using or sharing derived files.
- Dataset is available via the following URL:  https://zenodo.org/records/23403681
- Paper: Onwuegbuche, Olaoluwa, Jurcut, Pasquale, Updated October 2026, Accessed October 2026, “A behavioural dataset for ransomware analysis and detection. Journal of Network and Computer Applications, 104475.”

## 2. Place dataset into data/raw/
Copy or extract the downloaded dataset into `data/raw/`, keeping its original file names and folder structure. Do not edit, rename, or re-save anything there. `data/raw/` is treated as an immutable snapshot, and the pipeline only ever reads from it.

data/
├── README.md        (this file, tracked)
├── raw/             <- YOU put MLRan here (git-ignored)
├── interim/         generated: parsed events, labels (labels.csv), cleaned events
└── processed/       generated: splits, prefix windows, features, preprocessor

## 3. Tell the pipeline what the files look like
Open `config/config.yaml` and fill in the fields.

