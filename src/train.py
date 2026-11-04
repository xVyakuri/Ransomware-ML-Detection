# Train all models on a given split/seed.
"""Reproducible seed-level training entry point."""
import argparse
import hashlib
import json
import subprocess
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

from src import models, preprocess
from src.configuration import CONFIG, computational_config, results_root
from src.features import load
from src.thresholds import select_threshold


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def run(seed=None, names=None, tables=None, sample_tokens=None, balanced=True,
        experiment='baseline'):
    from src.common import Paths
    paths = Paths.default()
    from src.common import read_stage
    recorded = read_stage(paths.stage_report, 'split').get('resolved_config')
    if recorded is None or any(recorded.get(k) != CONFIG[k] for k in ('splits', 'holdout', 'dataset')):
        raise ValueError('Processed split settings are missing or changed; rebuild splits before training')
    converted = read_stage(paths.stage_report, 'convert').get('resolved_config')
    if converted is None or converted.get('dataset') != CONFIG['dataset']:
        raise ValueError('Processed dataset settings are missing or changed; rebuild data before training')
    seed = CONFIG['training']['seed'] if seed is None else seed
    names = CONFIG['training']['models'] if names is None else names
    sample_tokens = CONFIG['features']['sample_tokens'] if sample_tokens is None else sample_tokens
    provenance = dict(
        resolved_config=CONFIG,
        config_hash=hashlib.sha256(json.dumps(computational_config(CONFIG), sort_keys=True).encode()).hexdigest(),
        inputs={str(p): fingerprint(p) for p in
                (paths.matrix, paths.catalog, paths.samples, paths.labels,
                 paths.aggregates, paths.split_manifest)},
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        dirty=subprocess.check_output(['git', 'status', '--porcelain'], text=True),
        source={str(p): fingerprint(p) for p in Path('src').rglob('*.py')},
        versions={p: version(p) for p in ('numpy', 'scipy', 'pandas', 'scikit-learn')})
    for name in names:
        key = f'{experiment}/seed{seed}/{name}'
        dest = results_root() / 'models' / key
        dest.mkdir(parents=True, exist_ok=True)
        complete = dest / 'complete.json'
        configuration = dict(seed=seed, model=name, tables=tables,
                             sample_tokens=sample_tokens, balanced=balanced)
        if complete.exists() and (dest / 'config.json').exists():
            old = json.loads((dest / 'config.json').read_text())
            predictions = results_root() / 'predictions' / key
            previous = json.loads(complete.read_text())
            previous['resolved_config'] = computational_config(previous.get('resolved_config', {}))
            comparison = dict(provenance, resolved_config=computational_config(CONFIG))
            if (previous == comparison
                    and all(old.get(k) == v for k, v in configuration.items())
                    and (dest / 'model.joblib').exists()
                    and all((predictions / f'{s}.csv').exists() for s in ('val', 'test', 'holdout'))):
                print(f'Reusing {key}', flush=True)
                continue
        complete.unlink(missing_ok=True)
        b = load(seed=seed, tables=tables, sample_tokens=sample_tokens,
                 min_df=CONFIG['features']['min_df'],
                 sparse_features=name not in ('count', 'kmeans'),
                 aggregates=True if name in ('count', 'kmeans') else CONFIG['features']['supervised_aggregates'])
        if not b.train.X.shape[1]:
            raise ValueError('no features available')
        for split in (b.train, b.val, b.test, b.holdout):
            if np.isinf(split.X.data).any():
                raise ValueError('infinite feature values')
        transformer = model = None
        if name == 'count':
            index = b.features.index[b.features.name == CONFIG['baseline']['feature']][0]
            def score(split, index=index):
                return split.X[:, index].toarray().ravel()
        else:
            fit = b.train.X[b.train.y == 0] if name == 'kmeans' else b.train.X
            if not fit.shape[0]:
                raise ValueError('no fitting rows')
            transformer = preprocess.create(b.features)
            x = transformer.fit_transform(fit)
            model = models.create(name, seed, balanced, fit.shape[0])
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(x, None if name == 'kmeans' else b.train.y)

            def score(split, transformer=transformer, model=model, name=name):
                x = transformer.transform(split.X)
                return (model.transform(x).min(axis=1) if name == 'kmeans'
                        else model.predict_proba(x)[:, 1])
        threshold = select_threshold(score(b.val), b.val.y, CONFIG['threshold']['target_fpr'])
        benign_scores = score(b.val)[b.val.y == 0]
        joblib.dump(dict(model=model, preprocessing=transformer, features=b.features,
                         threshold=threshold, seed=seed), dest / 'model.joblib.tmp')
        (dest / 'model.joblib.tmp').replace(dest / 'model.joblib')
        for split_name in ('val', 'test', 'holdout'):
            split = getattr(b, split_name)
            scores = score(split)
            if not np.isfinite(scores).all():
                raise ValueError('non-finite model scores')
            frame = pd.DataFrame(dict(sample_id=split.sample_id, label=split.y,
                                      family=split.family, type=split.type,
                                      group_id=split.group_id, score=scores,
                                      prediction=(scores >= threshold).astype(int)))
            path = results_root() / 'predictions' / key
            path.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path / f'{split_name}.csv.tmp', index=False)
            (path / f'{split_name}.csv.tmp').replace(path / f'{split_name}.csv')
        (dest / 'config.json').write_text(json.dumps(dict(seed=seed, model=name,
            tables=tables, sample_tokens=sample_tokens, balanced=balanced,
            threshold=threshold, benign_validation_count=len(benign_scores),
            validation_fp=int((benign_scores >= threshold).sum()),
            validation_fpr=float((benign_scores >= threshold).mean()),
            fpr_resolution=1 / len(benign_scores)), indent=2))
        temporary = complete.with_suffix('.tmp')
        (dest / 'resolved_config.yaml').write_text(yaml.safe_dump(CONFIG, sort_keys=False))
        temporary.write_text(json.dumps(provenance, indent=2))
        temporary.replace(complete)
        print(f'Completed {key}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.parse_args()
    run()
