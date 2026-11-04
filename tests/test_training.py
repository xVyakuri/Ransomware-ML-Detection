import json

import pandas as pd
import pytest

from src import evaluate, train
from src.features import load


@pytest.mark.parametrize('name', ['count', 'logreg', 'rf', 'kmeans'])
def test_training_and_saved_metrics(built, tmp_path, monkeypatch, name):
    bundle = load(seed=0, min_df=1, aggregates=True,
                  sparse_features=name not in ('count', 'kmeans'), paths=built)
    monkeypatch.setattr(train, 'load', lambda **kwargs: bundle)
    from src.common import Paths
    monkeypatch.setattr(Paths, 'default', classmethod(lambda cls: built))
    monkeypatch.chdir(tmp_path)
    # Git provenance is exercised separately in real runs; fixture lives outside the repo.
    check_output = train.subprocess.check_output
    monkeypatch.setattr(train.subprocess, 'check_output', lambda *a, **k: 'fixture')
    train.run(names=(name,))
    evaluate.run()
    predictions = pd.read_csv(f'results/predictions/baseline/seed0/{name}/test.csv')
    saved = json.loads((tmp_path / 'results/metrics/summary.json').read_text())
    test = next(row for row in saved if row['cohort'] == 'test')
    assert test['n'] == len(bundle.test)
    assert test['tp'] == int(((predictions.label == 1) & (predictions.prediction == 1)).sum())
    val = pd.read_csv(f'results/predictions/baseline/seed0/{name}/val.csv')
    assert val.loc[val.label == 0, 'prediction'].mean() <= .01
    if name == 'rf':
        monkeypatch.setattr(train.subprocess, 'check_output', check_output)
        from src.make_report import run as report
        report()
        text = (tmp_path / 'results/tables/report.md').read_text()
        assert 'Missing:' in text
        assert 'Timeliness: unsupported' in text
        assert (tmp_path / 'results/figures/roc.png').stat().st_size > 0


def test_anomaly_preprocessing_fits_benign_only(built, tmp_path, monkeypatch):
    import joblib
    import numpy as np

    bundle = load(seed=0, aggregates=True, sparse_features=False, paths=built)
    monkeypatch.setattr(train, 'load', lambda **kwargs: bundle)
    from src.common import Paths
    monkeypatch.setattr(Paths, 'default', classmethod(lambda cls: built))
    monkeypatch.setattr(train.subprocess, 'check_output', lambda *a, **k: 'fixture')
    monkeypatch.chdir(tmp_path)
    train.run(names=('kmeans',))
    artifact = joblib.load('results/models/baseline/seed0/kmeans/model.joblib')
    imputer = artifact['preprocessing'].named_transformers_['aggregate'].steps[0][1]
    expected = np.median(bundle.train.X[bundle.train.y == 0].toarray(), axis=0)
    np.testing.assert_allclose(imputer.statistics_, expected)


def test_partial_runs_are_not_evaluated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'results/predictions/baseline/seed0/rf'
    path.mkdir(parents=True)
    (path / 'test.csv').write_text('invalid incomplete data')
    evaluate.run()
    assert json.loads((tmp_path / 'results/metrics/summary.json').read_text()) == []
