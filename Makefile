# One-command reproducibility. Run make help to see targets.
# Uses a plain Python venv and pip; uv is not required.
.DEFAULT_GOAL := help
.PHONY: help setup reproduce test lint clean data snapshot verify labels parse features splits holdout quality spot-check train evaluate experiments report

VENV := .venv
PY := $(VENV)/bin/python
CONFIG ?= config/config.yaml
OUTPUT_DIR ?=
ARGS = --config "$(CONFIG)" $(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",)
.NOTPARALLEL: data

help:
	@echo "make setup      - create .venv and install requirements.txt"
	@echo "make data       - verify and process manually downloaded MLRan data"
	@echo "make reproduce  - train experiments, evaluate, and report from prepared data"
	@echo "make test       - run automated tests"
	@echo "make lint       - byte-compile src/ and tests/ (stdlib only)"
	@echo "make clean      - remove generated results; preserve data and unrecognized files"
	@echo "Options: CONFIG=path/to/config.yaml OUTPUT_DIR=path/to/results"

setup:
	@test -x "$(PY)" || python3 -m venv "$(VENV)"
	"$(PY)" -m ensurepip --upgrade
	"$(PY)" -m pip install --upgrade pip
	"$(PY)" -m pip install -r requirements.txt

data: snapshot parse features splits quality spot-check

snapshot:
	"$(PY)" -m src.check_raw $(ARGS)
verify:
	"$(PY)" -m src.check_raw --verify --full $(ARGS)
labels:
	"$(PY)" -m src.build_labels $(ARGS)
parse:
	"$(PY)" -m src.parse_raw $(ARGS)
features:
	"$(PY)" -m src.features $(ARGS)
splits:
	"$(PY)" -m src.make_splits $(ARGS)
holdout:
	"$(PY)" -m src.make_holdout $(ARGS)
quality:
	"$(PY)" -m src.quality_checks $(ARGS)
spot-check:
	"$(PY)" -m src.spot_check $(ARGS)

reproduce:
	"$(PY)" -m src.reproduce $(ARGS)
test:
	"$(PY)" -m pytest -q
lint:
	"$(PY)" -m compileall -q src tests
clean:
	"$(PY)" -m src.clean_results $(ARGS)
train:
	"$(PY)" -m src.train $(ARGS)
evaluate:
	"$(PY)" -m src.evaluate $(ARGS)
experiments:
	"$(PY)" -m src.exp.run_all $(ARGS)
report:
	"$(PY)" -m src.make_report $(ARGS)
