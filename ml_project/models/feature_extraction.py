import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from pywt import wavedec
from math import floor
from re import findall
from biosppy.signals import ecg


##############################################################################
# PROJECT 1 - AGE PREDICTION
##############################################################################


class HistogramsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, partition_size=9, bins=45):
        self.size = partition_size
        self.bins = bins

    def fit(self, X, y=None):
        n_sample, x_size, y_size, z_size = X.shape
        self.partitions = np.array(
            [x_size / self.size, y_size / self.size, z_size / self.size])
        self.partitions = np.rint(self.partitions).astype(int)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["partitions"])
        n_sample, x_size, y_size, z_size = X.shape
        max_voxel_value = 4418  # to speed up the running
        features = np.zeros((n_sample, self.partitions[0], self.partitions[1],
                             self.partitions[2], self.bins))
        for i in range(n_sample):
            for x in range(self.partitions[0]):
                for y in range(self.partitions[1]):
                    for z in range(self.partitions[2]):
                        partition = X[i][
                                    x * self.size: (x + 1) * self.size,
                                    y * self.size: (y + 1) * self.size,
                                    z * self.size: (z + 1) * self.size]
                        features[i][x][y][z] = \
                            np.histogram(partition, bins=self.bins,
                                         range=(0, max_voxel_value))[0]
        features = np.reshape(features,
                              (-1, np.prod(self.partitions) * self.bins))
        print("####features ", features.shape, features is None)
        return features


##############################################################################
# PROJECT 3 - ECG CLASSIFICATION
##############################################################################

class HeartBeatDWTExtractor(BaseEstimator, ClassifierMixin):
    def __init__(self, sampling_rate=300.0, dec_level=3, wavelet="db4"):
        self.sampling_rate = sampling_rate
        self.dec_level = dec_level
        self.wavelet = wavelet

    def fit(self, X, y=None):
        X = check_array(X)
        _, n_features = X.shape
        n = int(findall('\d+', self.wavelet)[0])
        self.coeff_len = n_features
        for i in range(self.dec_level):
            self.coeff_len = floor((self.coeff_len - 1) / 2) + n
        self.wave_len = 23 if self.wavelet == "db1" else 28 \
            if self.wavelet == "db4" else 35
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["coeff_len"])
        n_samples, _ = X.shape
        features = np.zeros((n_samples, self.wave_len + 3))
        for i in range(n_samples):
            _, filtered, rpeaks, _, beats, _, _ = ecg.ecg(
                np.trim_zeros(X[i]),
                sampling_rate=self.sampling_rate,
                show=False)
            beats_dwt = wavedec(np.mean(beats, axis=0), self.wavelet,
                                level=self.dec_level)[0]
            rpeaks_magnitude = np.take(filtered, rpeaks)
            r_peaks_var = np.var(rpeaks_magnitude)
            rr_interval_mean = np.mean(np.ediff1d(rpeaks))
            rr_interval_var = np.var(np.ediff1d(rpeaks))
            features[i] = np.hstack((beats_dwt, r_peaks_var,
                                     rr_interval_mean, rr_interval_var))
        print("#######FEAT", features.shape)
        return features
