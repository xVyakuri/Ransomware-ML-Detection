# Create metrics on validation/test for Recall, FPR, PR/ROC, etc.
import argparse
import json

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.configuration import CONFIG, computational_config, results_root


def metrics(frame):
    y, pred = frame.label, frame.prediction
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    def ratio(a, b):
        return a / b if b else None
    both = y.nunique() == 2
    return dict(n=len(y), tn=tn, fp=fp, fn=fn, tp=tp, recall=ratio(tp, tp+fn),
                fpr=ratio(fp, fp+tn), precision=ratio(tp, tp+fp),
                f1=ratio(2*tp, 2*tp+fp+fn),
                average_precision=float(average_precision_score(y, frame.score)) if both else None,
                roc_auc=float(roc_auc_score(y, frame.score)) if both else None,
                undefined_reason=None if both else 'cohort contains only one class')


def run(seed=None, names=None):
    rows = []
    root = results_root()
    for path in sorted((root / 'predictions').rglob('test.csv')):
        key = path.parent.relative_to(root / 'predictions')
        if seed is not None and key.parts[-2] != f'seed{seed}':
            continue
        if names is not None and key.parts[-1] not in names:
            continue
        complete = root / 'models' / key / 'complete.json'
        if not complete.exists():
            continue
        if computational_config(json.loads(complete.read_text()).get('resolved_config', {})) != computational_config(CONFIG):
            continue
        test = pd.read_csv(path)
        for cohort in ('val', 'test', 'holdout'):
            frame = pd.read_csv(path.with_name(f'{cohort}.csv'))
            if cohort == 'holdout':
                frame = pd.concat([frame, test[test.label == 0]], ignore_index=True)
            rows.append(dict(run=str(key),
                             cohort=cohort, **metrics(frame)))
            if frame.label.nunique() == 2:
                fpr, tpr, _ = roc_curve(frame.label, frame.score)
                precision, recall, _ = precision_recall_curve(frame.label, frame.score)
                curves = root / 'metrics' / key
                curves.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(dict(fpr=fpr, tpr=tpr)).to_csv(curves / f'{cohort}_roc.csv', index=False)
                pd.DataFrame(dict(precision=precision, recall=recall)).to_csv(
                    curves / f'{cohort}_pr.csv', index=False)
            if cohort == 'holdout':
                for field in ('family', 'type'):
                    groups = frame[frame.label == 1].groupby(field).prediction.agg(['count', 'mean'])
                    groups.to_csv(root / 'metrics' / key / f'holdout_{field}.csv')
    out = root / 'metrics'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    pd.DataFrame(rows).to_csv(out / 'summary.csv', index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.parse_args()
    run()
