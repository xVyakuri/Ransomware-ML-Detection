"""Generate offline team-facing results entry points."""
import pandas as pd

from src.configuration import CONFIG, results_root


def run():
    root = results_root()
    metrics = pd.read_csv(root / 'metrics/summary.csv')
    baseline = metrics[(metrics.cohort == 'test') & metrics.run.str.startswith('baseline/')].copy()
    baseline['model'] = baseline.run.str.split('/').str[-1]
    names = {'count': 'File-write count rule', 'logreg': 'Logistic Regression',
             'rf': 'Random Forest', 'kmeans': 'K-means anomaly detector'}
    summary = baseline.groupby('model')[['recall', 'fpr']].mean().sort_values('recall', ascending=False)
    lines = ['# Start here: understanding detection results', '',
             '## What was measured', '',
             'Each sample is classified as ransomware or benign. Recall is the fraction of ransomware caught; '
             'false-positive rate (FPR) is the fraction of benign samples incorrectly flagged.', '',
             'Thresholds were chosen on validation data. Test results measure performance on families absent '
             'from training. They do not measure detection speed or production alert volumes.', '',
             '## Main comparison', '',
             'Averages across available baseline split seeds; these are not pooled sample counts.', '',
             '| Model | Ransomware caught (recall) | Benign falsely flagged (FPR) |',
             '|---|---:|---:|']
    for name, row in summary.iterrows():
        lines.append(f'| {names.get(name, name)} | {row.recall:.2%} | {row.fpr:.2%} |')
    if len(summary):
        lines += ['', f'{names.get(summary.index[0], summary.index[0])} has the highest mean recall '
                  'among the evaluated baseline models. This ranking applies to these runs only.']
    lines += ['', '## Concrete counts for the reference split', '',
              '| Model | Caught | Missed | False alarms | Correctly accepted benign |',
              '|---|---:|---:|---:|---:|']
    seed = CONFIG['report']['reference_seed']
    for row in baseline[baseline.run.str.contains(f'/seed{seed}/')].itertuples():
        lines.append(f'| {names.get(row.model, row.model)} | {row.tp} | {row.fn} | {row.fp} | {row.tn} |')
    lines += ['', '## How to read the experiments', '',
              '- **Baseline:** compare different models on the same split.',
              '- **Variability:** change the split seed to check dependence on which families are tested.',
              f"- **Holdout:** evaluate the fixed {CONFIG['holdout']['families']}-family ransomware cohort; benign samples are reused from test.",
              '- **without_*:** remove a feature table and retrain to see how performance changes.',
              '- **without_sample_tokens:** remove sample-token presence features; aggregates retain their definitions.',
              '- **unweighted:** remove Random Forest class balancing.', '',
              'Ablation differences are initial evidence, not causal proof. Each model has its own validation threshold.', '',
              '## Glossary and limitations', '',
              '- **Precision:** fraction of flagged samples that are ransomware; depends on dataset prevalence.',
              '- **ROC AUC:** score ranking across thresholds; it is not the recall at the selected threshold.',
              '- **Threshold:** score cutoff for an alert. A validation FPR target does not guarantee the same test FPR.',
              '- **Standard deviation:** variation across split seeds, not a confidence interval.',
              '- **Timeliness:** unsupported because the dataset has no timestamps or event ordering.',
              '- **Scope:** Windows sandbox samples, era skew, excluded failed runs, and collection artifacts limit generalization.', '',
              '## Where to go next', '',
              '- [Full report and experiment coverage](tables/report.md)',
              '- [All cohort metrics (CSV)](metrics/summary.csv)',
              '- [Seed variability (CSV)](tables/variability.csv)',
              '- [ROC plot](figures/roc.png) and [precision–recall plot](figures/pr.png)',
              '- `models/`: fitted models, preprocessing, configuration, and provenance; use for reproducibility.',
              '- `predictions/`: individual sample scores and decisions; use to investigate errors.',
              '- `metrics/<experiment>/seed<N>/<model>/`: curves and per-family/type holdout tables.', '',
              '## Provenance', '']
    lines += ['Each completed model records its resolved YAML configuration and input fingerprints. '
              'Use that snapshot to interpret a run rather than inferring its settings from today’s YAML.', '',
              'Change settings in `config/config.yaml`. To regenerate explanations from saved metrics: '
              '`make report`. To retrain: `make experiments`, then `make evaluate` and `make report`.']
    text = '\n'.join(lines) + '\n'
    coverage = (root / 'tables/report.md').read_text().splitlines()[2:4]
    text = text.replace('## What was measured', '\n'.join(coverage) + '\n\n## What was measured')
    (root / 'README.md').write_text(text)



if __name__ == '__main__':
    run()
