"""Training-fitted aggregate imputation and scaling."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.configuration import CONFIG


def create(features):
    aggregate = list(features.index[features.kind == 'aggregate'])
    return ColumnTransformer(
        [('aggregate', make_pipeline(SimpleImputer(strategy=CONFIG['preprocessing']['imputation'], keep_empty_features=True),
                                     StandardScaler(with_mean=False, with_std=CONFIG['preprocessing']['scale_aggregates'])), aggregate)],
        remainder='passthrough', sparse_threshold=1.0)
