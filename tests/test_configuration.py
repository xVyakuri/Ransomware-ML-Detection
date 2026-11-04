import copy
import json
import subprocess
import sys

import pytest
import yaml

from src import configuration
from src.models import create


def test_model_parameters_follow_configuration(monkeypatch):
    monkeypatch.setitem(configuration.CONFIG['models']['rf'], 'n_estimators', 17)
    assert create('rf').n_estimators == 17


@pytest.mark.parametrize('section,key,value', [
    ('threshold', 'target_fpr', 2), ('features', 'min_df', 0),
    ('training', 'seed', 999), ('parser', 'column_batch', -2),
    ('splits', 'typo', 1),
    ('features', 'sample_tokens', 'false'),
    ('report', 'reference_seed', 999),
    ('paths', 'results', ''),
])
def test_invalid_configuration_fails(tmp_path, monkeypatch, section, key, value):
    cfg = copy.deepcopy(configuration.CONFIG)
    cfg[section][key] = value
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(configuration, 'CONFIG_PATH', path)
    with pytest.raises(ValueError):
        configuration.load_config()


def test_alternate_yaml_propagates_to_entry_points(tmp_path):
    cfg = copy.deepcopy(configuration.CONFIG)
    cfg['training']['seed'] = 2
    cfg['training']['models'] = ['rf']
    cfg['models']['rf']['n_estimators'] = 17
    cfg['threshold']['target_fpr'] = .02
    cfg['paths']['results'] = str(tmp_path / 'custom-results')
    path = tmp_path / 'alternate.yaml'
    path.write_text(yaml.safe_dump(cfg))
    code = ('import json; from src.configuration import CONFIG,results_root; '
            'from src.models import create; '
            'print(json.dumps([create("rf").n_estimators,create("rf").random_state,'
            'CONFIG["threshold"]["target_fpr"],str(results_root())]))')
    values = json.loads(subprocess.check_output([sys.executable, '-c', code,
                                               '--config', str(path)], text=True))
    assert values == [17, 2, .02, str(tmp_path / 'custom-results')]


def test_report_changes_do_not_invalidate_computation():
    cfg = copy.deepcopy(configuration.CONFIG)
    cfg['report']['reference_seed'] = 1
    original = configuration.computational_config(configuration.CONFIG)
    assert configuration.computational_config(cfg) == original
    cfg['models']['rf']['n_estimators'] = 17
    assert configuration.computational_config(cfg) != original
