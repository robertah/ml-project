from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
import numpy as np

##############################################################################
# PROJECT 1 - AGE PREDICTION
##############################################################################


class NonZeroSelection(BaseEstimator, TransformerMixin):
    """Select non-zero voxels"""
    def fit(self, X, y=None):
        X = check_array(X)
        self.nonzero = X.sum(axis=0) > 0

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["nonzero"])
        X = check_array(X)
        return X[:, self.nonzero]


class RandomSelection(BaseEstimator, TransformerMixin):
    """Random Selection of features"""
    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
                            n_features,
                            self.n_components,
                            random_state=random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new


##############################################################################
# PROJECT 2 - DEMENTIA CLASSIFICATION
##############################################################################


class RegionsOfInterest(BaseEstimator, TransformerMixin):
    def __init__(self, hipp_bins=35, ventr_bins=35, cortex_bins=35):
        self.h_bins = hipp_bins
        self.v_bins = ventr_bins
        self.c_bins = cortex_bins

    def fit(self, X, y=None):
        self.hl = [36, 86, 60, 135, 35, 85]
        self.hr = [90, 140, 60, 135, 35, 85]
        self.v = [53, 121, 65, 160, 70, 115]
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["hl", "hr", "v"])
        X = np.reshape(X, (-1, 176, 208, 176))
        n_sample, _, _, _ = X.shape

        tot_bins = 2 * self.h_bins + self.v_bins
        features = np.zeros((n_sample, tot_bins))
        for i in range(n_sample):
            hipp_left = np.histogram(X[i][
                                     self.hl[0]:self.hl[1],
                                     self.hl[2]:self.hl[3],
                                     self.hl[4]:self.hl[5]],
                                     bins=self.h_bins)[0]
            hipp_right = np.histogram(X[i][
                                      self.hr[0]:self.hr[1],
                                      self.hr[2]:self.hr[3],
                                      self.hr[4]:self.hr[5]],
                                      bins=self.h_bins)[0]
            ventr = np.histogram(X[i][
                                 self.v[0]:self.v[1],
                                 self.v[2]:self.v[3],
                                 self.v[4]:self.v[5]],
                                 bins=self.v_bins)[0]
            features[i] = np.concatenate((hipp_left, hipp_right,
                                          ventr)).flatten()
        return features
