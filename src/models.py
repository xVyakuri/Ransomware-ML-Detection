"""Factories for the configured supervised and K-means comparison models."""
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.configuration import CONFIG

NAMES = ('count', 'logreg', 'rf', 'kmeans')


def create(name, seed=None, balanced=True, n_benign=None):
    seed = CONFIG['training']['seed'] if seed is None else seed
    if name == 'logreg':
        return LogisticRegression(**CONFIG['models']['logreg'], random_state=seed)
    if name == 'rf':
        settings = dict(CONFIG['models']['rf'])
        if not balanced:
            settings['class_weight'] = None
        return RandomForestClassifier(**settings, random_state=seed)
    if name == 'kmeans':
        settings = dict(CONFIG['models']['kmeans'])
        if n_benign is not None:
            settings['n_clusters'] = min(settings['n_clusters'], n_benign)
        return KMeans(**settings, random_state=seed)
    raise ValueError(name)
