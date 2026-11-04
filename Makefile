# Pipeline entry points. Run from the repository root.
PY := uv run python -m

.PHONY: setup data snapshot verify labels parse features splits holdout quality spot-check test reproduce

setup:            ## create .venv with all dependencies
	uv sync --extra dev

data: snapshot parse features splits quality spot-check   ## steps 1-6 and 10: raw MLRan -> model-ready data

snapshot:         ## step 1: hash raw files into reports/raw_manifest.csv
	$(PY) src.check_raw

verify:           ## re-hash raw files against the manifest
	$(PY) src.check_raw --verify --full

labels:           ## step 3 only (parse also builds labels)
	$(PY) src.build_labels

parse:            ## steps 2-3: sparse matrix, feature catalog, labels
	$(PY) src.parse_raw

features:         ## step 8: per-sample count/breadth aggregates
	$(PY) src.features

splits:           ## steps 5-6: groups, 8-family holdout, 70/15/15 for 5 seeds
	$(PY) src.make_splits

holdout:          ## report held-out families and check constraints
	$(PY) src.make_holdout

quality:          ## step 4: quality summary + leakage checks
	$(PY) src.quality_checks

spot-check:       ## step 10: trace 20 random samples raw -> derived
	$(PY) src.spot_check

test:
	uv run pytest

reproduce: data   ## full pipeline; model, evaluation and report steps go here
