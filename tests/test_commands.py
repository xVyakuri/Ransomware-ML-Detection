import subprocess
import sys

import pytest

from src import reproduce
from src.clean_results import main as clean
from src.configuration import CONFIG


def test_make_help_is_default_and_has_no_grade():
    text = subprocess.check_output(['make'], text=True)
    for name in ('setup', 'data', 'reproduce', 'test', 'lint', 'clean'):
        assert f'make {name}' in text
    assert 'make grade' not in text


def test_reproduce_runs_checks_before_training(monkeypatch):
    calls = []
    monkeypatch.setattr(reproduce.check_raw, 'verify', lambda *a, **k: calls.append('verify'))
    monkeypatch.setattr(reproduce.quality_checks, 'run_all', lambda *a, **k: [])
    monkeypatch.setattr(reproduce, 'experiments', lambda: calls.append('experiments'))
    monkeypatch.setattr(reproduce.evaluate, 'run', lambda: calls.append('evaluate'))
    monkeypatch.setattr(reproduce.make_report, 'run', lambda: calls.append('report'))
    reproduce.main()
    assert calls == ['verify', 'experiments', 'evaluate', 'report']


def test_clean_preserves_unrecognized_files(tmp_path, monkeypatch):
    monkeypatch.setitem(CONFIG['paths'], 'results', str(tmp_path))
    for folder in ('models', 'metrics', 'notes'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'keep.txt').write_text('example')
    (tmp_path / 'notes.txt').write_text('keep')
    clean()
    assert not (tmp_path / 'models').exists()
    assert not (tmp_path / 'metrics').exists()
    assert (tmp_path / 'notes/keep.txt').exists()
    assert (tmp_path / 'notes.txt').exists()


def test_output_directory_override(tmp_path):
    output = subprocess.check_output([sys.executable, '-c',
        'from src.configuration import results_root; print(results_root())',
        '--output-dir', str(tmp_path)], text=True)
    assert output.strip() == str(tmp_path)


def test_clean_refuses_data_directory(monkeypatch):
    monkeypatch.setitem(CONFIG['paths'], 'results', CONFIG['paths']['data'])
    with pytest.raises(ValueError, match='Refusing to clean'):
        clean()
