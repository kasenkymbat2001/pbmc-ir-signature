"""sklearn-compatible transformers used inside the model pipeline.

* ``WithinSampleRank`` converts each sample to gene-rank percentiles. It uses only that
  sample, so it cannot leak information between samples, and it removes platform-specific
  scale (microarray intensity vs RNA-seq CPM) and monotone batch effects.
* ``TopKWelch`` keeps the k genes with the largest |Welch t| between classes. Because it
  sits inside the pipeline, it is re-fitted on training folds only.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class WithinSampleRank(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        r = stats.rankdata(X, axis=1, method="average")
        return (r - 0.5) / X.shape[1]


class TopKWelch(SelectorMixin, BaseEstimator):
    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("TopKWelch needs exactly two classes")
        a, b = X[y == classes[1]], X[y == classes[0]]
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # constant genes -> NaN t, scored 0
            t, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
        self.scores_ = np.nan_to_num(np.abs(t), nan=0.0)
        self.n_features_in_ = X.shape[1]
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        k = min(self.k, len(self.scores_))
        mask = np.zeros(len(self.scores_), dtype=bool)
        mask[np.argsort(-self.scores_, kind="stable")[:k]] = True
        return mask
