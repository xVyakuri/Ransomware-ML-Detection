# Script to create final figures and tables from results/
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.configuration import CONFIG, results_root

matplotlib.use('Agg')

def run():
    root = results_root()
    metrics = pd.read_csv(root / 'metrics/summary.csv')
    tables = root / 'tables'
    figures = root / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / 'comparison.csv', index=False)
    base = metrics[metrics.run.str.startswith('baseline/') & (metrics.cohort == 'test')].copy()
    base['model'] = base.run.str.split('/').str[-1]
    summary = base.groupby('model')[['recall', 'fpr', 'precision', 'roc_auc']].agg(['mean', 'std'])
    summary.to_csv(tables / 'variability.csv')
    for kind, x, y in (('roc', 'fpr', 'tpr'), ('pr', 'recall', 'precision')):
        fig, ax = plt.subplots()
        for path in sorted((root / 'metrics/baseline' / f"seed{CONFIG['report']['reference_seed']}").glob(f'*/test_{kind}.csv')):
            curve = pd.read_csv(path)
            ax.plot(curve[x], curve[y], label=path.parent.name)
        ax.set(xlabel=x, ylabel=y, xlim=(0, 1), ylim=(0, 1))
        if ax.lines:
            ax.legend()
        fig.tight_layout()
        for extension in CONFIG['report']['formats']:
            fig.savefig(figures / f'{kind}.{extension}')
        plt.close(fig)
    for cohort in ('test', 'holdout'):
        subset = metrics[metrics.cohort == cohort]
        fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(subset) * .35)), sharey=True)
        for ax, metric in zip(axes, ('recall', 'fpr'), strict=True):
            subset.set_index('run')[metric].plot.barh(ax=ax)
            ax.set_xlim(0, 1 if metric == 'recall' else max(.02, subset.fpr.max() * 1.1))
            ax.set_title('Ransomware caught' if metric == 'recall' else 'Benign falsely flagged')
            ax.set_xlabel('Fraction of samples')
        plt.tight_layout()
        for extension in CONFIG['report']['formats']:
            plt.savefig(figures / f'{cohort}_comparison.{extension}')
        plt.close()
    seeds = CONFIG['splits']['seeds'] if CONFIG['experiments']['variability'] else [CONFIG['training']['seed']]
    expected = {f'baseline/seed{s}/{m}' for s in seeds
                for m in CONFIG['training']['models']}
    from src.common import TABLES
    seed = CONFIG['training']['seed']
    model = CONFIG['experiments']['ablation_model']
    if CONFIG['experiments']['ablation']:
        expected |= {f'without_{t}/seed{seed}/{model}' for t in TABLES}
    if CONFIG['experiments']['sample_tokens']:
        expected.add(f'without_sample_tokens/seed{seed}/{model}')
    if CONFIG['experiments']['class_weight']:
        expected.add(f'unweighted/seed{seed}/rf')
    missing = sorted(expected - set(metrics.run))
    lines = ['# Detection results', '',
             f'Experiment coverage: {len(expected)-len(missing)}/{len(expected)} expected runs.',
             'Missing: ' + (', '.join(missing) or 'none'), '',
             'Timeliness: unsupported; MLRan contains presence only, without event order.',
             'Counts describe distinct features, not operation frequency or rates.',
             'Ordinary test and holdout both assess unseen ransomware families.',
             'Holdout FPR reuses ordinary test benign samples.',
             'Precision reflects evaluation prevalence, not production alert volume.',
             'Seed standard deviations describe split variability, not confidence intervals.',
             'Limitations: Windows 7 sandbox, era skew, collection artifacts, and excluded failed runs.',
             'The sample-token ablation removes presence features; aggregates retain their definition.', '',
             '| Run | Cohort | Recall | FPR |', '|---|---|---:|---:|']
    for row in metrics.itertuples():
        lines.append(f'| {row.run} | {row.cohort} | {row.recall:.4f} | {row.fpr:.4f} |')
    (tables / 'report.md').write_text('\n'.join(lines) + '\n')
    from src.results_guide import run as guide
    guide()


if __name__ == '__main__':
    run()
