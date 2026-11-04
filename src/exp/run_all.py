"""Run the fixed experiment suite sequentially."""
from src.common import TABLES
from src.configuration import CONFIG
from src.train import run


def main(only=None):
    seeds = CONFIG['splits']['seeds'] if CONFIG['experiments']['variability'] else [CONFIG['training']['seed']]
    for seed in seeds if only in (None, 'variability') else []:
        run(seed=seed, names=CONFIG['training']['models'])
    for table in TABLES if CONFIG['experiments']['ablation'] and only in (None, 'ablation') else []:
        run(names=(CONFIG['experiments']['ablation_model'],), tables=[t for t in TABLES if t != table],
            experiment=f'without_{table}')
    if CONFIG['experiments']['sample_tokens'] and only in (None, 'ablation'):
        run(names=(CONFIG['experiments']['ablation_model'],), sample_tokens=False, experiment='without_sample_tokens')
    if CONFIG['experiments']['class_weight'] and only in (None, 'class_weight'):
        run(names=('rf',), balanced=False, experiment='unweighted')


if __name__ == '__main__':
    main()
