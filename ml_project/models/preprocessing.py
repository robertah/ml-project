import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans

##############################################################################
# PROJECT 1 - AGE PREDICTION
##############################################################################


class BordersCropping(BaseEstimator, TransformerMixin):
    def __init__(self, x_size=176, y_size=208, z_size=176):
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = np.reshape(X, (-1, self.x_size, self.y_size, self.z_size))
        n_samples, _, _, _ = X.shape
        X_resized = []
        i1, i2, j1, j2, k1, k2 = 12, 16, 12, 14, 3, 20
        for i in range(n_samples):
            X_resized.append(X[i][i1:-i2, j1:-j2, k1:-k2])
        X_resized = np.array(X_resized)
        print("###### MODULE: preprocessing.py - BordersCropping", n_samples,
              "samples resized, new shape:",
              X_resized.shape)
        return X_resized


##############################################################################
# PROJECT 2 - DEMENTIA CLASSIFICATION
##############################################################################


class Normalization(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = check_array(X)
        self.mean = np.mean(X.ravel())
        self.var = np.var(X.ravel())
        return self

    def transform(self, X, y=None):
        X_norm = (X - self.mean) / self.var
        X_norm = np.reshape(X_norm.T, (-1, 176, 208, 176))
        return X_norm


class ImageSegmentation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = check_array(X)
        self.kcenters = np.array([[4], [720], [1250]])
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["kcenters"])
        n_samples, _ = X.shape
        k_labels = np.zeros(X.shape)
        for i in range(n_samples):
            k_labels[i] = KMeans(n_clusters=3, init=self.kcenters).fit(
                X[i].reshape((-1, 1))).labels_
        return k_labels
