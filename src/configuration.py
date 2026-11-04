"""Load and validate the authoritative pipeline YAML configuration."""
import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config/config.yaml'
if '--config' in sys.argv:
    position = sys.argv.index('--config')
    if position + 1 >= len(sys.argv):
        raise ValueError('--config requires a YAML file path')
    CONFIG_PATH = Path(sys.argv[position + 1]).resolve()
    del sys.argv[position:position + 2]


def load_config():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    required = {'paths', 'splits', 'holdout', 'parser', 'audit', 'features',
                'training', 'threshold', 'models', 'preprocessing', 'experiments', 'report', 'baseline', 'dataset'}
    if set(cfg) != required:
        raise ValueError('config sections must be: ' + ', '.join(sorted(required)))
    if abs(sum(cfg['splits']['fractions'].values()) - 1) > 1e-9:
        raise ValueError('split fractions must sum to one')
    if not 0 <= cfg['threshold']['target_fpr'] < 1:
        raise ValueError('target_fpr must be in [0, 1)')
    if cfg['training']['seed'] not in cfg['splits']['seeds']:
        raise ValueError('training seed must have a configured split')
    if set(cfg['training']['models']) - {'count', 'logreg', 'rf', 'kmeans'}:
        raise ValueError('unknown enabled model')
    if cfg['features']['min_df'] < 1:
        raise ValueError('min_df must be positive')
    expected = {
        'paths': {'raw', 'data', 'results', 'reports'},
        'splits': {'seeds', 'fractions', 'near_duplicate_jaccard'},
        'holdout': {'seed', 'families', 'minimum_types'},
        'parser': {'column_batch', 'thrift_limit'}, 'audit': {'seed', 'samples'},
        'features': {'min_df', 'sample_tokens', 'supervised_aggregates'}, 'training': {'seed', 'models'},
        'preprocessing': {'imputation', 'scale_aggregates'},
        'experiments': {'variability', 'ablation', 'sample_tokens', 'class_weight', 'ablation_model'},
        'report': {'formats', 'reference_seed'}, 'baseline': {'feature'},
        'dataset': {'tables', 'dynamic_tables', 'table_prefix', 'known_extra_ids', 'mlran_commit', 'metadata_files'},
        'threshold': {'target_fpr'}, 'models': {'logreg', 'rf', 'kmeans'}}
    for section, keys in expected.items():
        if set(cfg[section]) != keys:
            raise ValueError(f'invalid keys in {section}: expected {sorted(keys)}')
    if set(cfg['splits']['fractions']) != {'train', 'val', 'test'}:
        raise ValueError('fractions require train, val and test')
    if any(v <= 0 for v in cfg['splits']['fractions'].values()):
        raise ValueError('split fractions must be positive')
    if len(set(cfg['splits']['seeds'])) != len(cfg['splits']['seeds']):
        raise ValueError('split seeds must be unique')
    for section, fields in {'parser': ['column_batch', 'thrift_limit'],
                            'holdout': ['families', 'minimum_types'],
                            'audit': ['samples']}.items():
        if any(type(cfg[section][key]) is not int or cfg[section][key] < 1 for key in fields):
            raise ValueError(f'{section} sizes must be positive integers')
    if not 0 < cfg['splits']['near_duplicate_jaccard'] <= 1:
        raise ValueError('near_duplicate_jaccard must be in (0,1]')
    if cfg['report']['reference_seed'] not in cfg['splits']['seeds']:
        raise ValueError('reference seed must have a configured split')
    if not set(cfg['report']['formats']) <= {'png', 'pdf'} or 'png' not in cfg['report']['formats']:
        raise ValueError('report formats require png and optionally pdf')
    if cfg['experiments']['ablation_model'] not in ('rf', 'logreg'):
        raise ValueError('ablation model must be rf or logreg')
    if cfg['preprocessing']['imputation'] not in ('median', 'mean', 'most_frequent', 'constant'):
        raise ValueError('unsupported imputation strategy')
    for section, fields in {'features': ['sample_tokens', 'supervised_aggregates'],
                            'preprocessing': ['scale_aggregates'],
                            'experiments': ['variability', 'ablation', 'sample_tokens', 'class_weight']}.items():
        if any(type(cfg[section][key]) is not bool for key in fields):
            raise ValueError(f'{section} switches must be YAML booleans')
    if any(type(v) is not int for v in cfg['splits']['seeds']):
        raise ValueError('split seeds must be integers')
    for name in ('raw', 'data', 'results', 'reports'):
        if not isinstance(cfg['paths'][name], str) or not cfg['paths'][name]:
            raise ValueError(f'paths.{name} must be a nonempty path string')
    import inspect

    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    for name, model in [('logreg', LogisticRegression), ('rf', RandomForestClassifier), ('kmeans', KMeans)]:
        if set(cfg['models'][name]) - set(inspect.signature(model).parameters):
            raise ValueError(f'unknown parameters in models.{name}')
        if 'random_state' in cfg['models'][name]:
            raise ValueError('random_state is derived from the configured split seed')
    for name, keys in {'rf': ['n_estimators', 'min_samples_leaf'],
                       'kmeans': ['n_clusters', 'n_init'], 'logreg': ['C', 'max_iter']}.items():
        if any(not isinstance(cfg['models'][name].get(key), (int, float))
               or cfg['models'][name][key] <= 0 for key in keys):
            raise ValueError(f'models.{name} sizes must be positive numbers')
    return cfg


CONFIG = load_config()
if '--output-dir' in sys.argv:
    position = sys.argv.index('--output-dir')
    if position + 1 >= len(sys.argv) or not sys.argv[position + 1]:
        raise ValueError('--output-dir requires a directory')
    CONFIG['paths']['results'] = sys.argv[position + 1]
    del sys.argv[position:position + 2]


def results_root():
    return Path(CONFIG['paths']['results'])


def computational_config(cfg):
    """Report presentation must not invalidate computational artifacts."""
    return {k: v for k, v in cfg.items() if k != 'report'}
