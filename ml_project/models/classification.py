import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
from math import floor


##############################################################################
# PROJECT 2 - DEMENTIA CLASSIFICATION
##############################################################################


class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""

    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))


class VoteClassifier(BaseEstimator, ClassifierMixin):
    def generate_ordered_labels(self, values, n_labels):
        new_label = ''
        for i in range(n_labels):
            new_label += chr(65 + i) * int(floor(values[i] / 0.08))
        return ''.join(sorted(new_label))

    def fit(self, X, y):
        n_samples, n_labels = y.shape
        self.y_new = ['' for s in range(n_samples)]
        for i in range(n_samples):
            self.y_new[i] = self.generate_ordered_labels(y[i], n_labels)
        self.svc = SVC(probability=True)
        self.logreg = LogisticRegression()
        self.forest = RandomForestClassifier(300)
        self.votingclass = VotingClassifier(estimators=[('svc', self.svc),
                                                        ('lr', self.logreg),
                                                        ('rf', self.forest)],
                                            voting='soft',
                                            weights=[1, 1, 1])
        self.votingclass.fit(X, self.y_new)
        return self

    def merge_proba(self, classes, probas):
        merged_proba = np.zeros(4)
        for i in range(classes.shape[0]):
            for j in range(4):
                merged_proba[j] += (''.join(classes[i])).count(chr(
                    65 + j)) * probas[i]
        return merged_proba

    def predict_proba(self, X):
        check_is_fitted(self, ["forest"])
        result = self.votingclass.predict_proba(X)
        labels = set(''.join(self.votingclass.classes_))  # 4 labels
        y = np.zeros((result.shape[0], len(labels)))
        n_labels = np.zeros(len(labels))  # count labels
        for i in range(len(labels)):
            n_labels[i] = (''.join(self.votingclass.classes_)).count(chr(
                65 + i))
        for i in range(y.shape[0]):
            y[i] = self.merge_proba(self.votingclass.classes_, result[i])
            y[i] /= sum(y[i])
        return y

    def score(self, X, y, sample_weight=None):
        n_samples, _ = y.shape
        return np.mean([spearmanr(y[i], self.predict_proba(X)[i])[0] for i in
                        range(n_samples)])


##############################################################################
# PROJECT 3 - ECG CLASSIFICATION
##############################################################################

class RandomForest(RandomForestClassifier, ClassifierMixin):
    def __init__(self, n_estimators=1000):
        super().__init__(n_estimators=n_estimators, random_state=2017,
                         class_weight="balanced")

    def fit(self, X, y):
        return super().fit(X, y)

    def predict(self, X):
        return super().predict(X).astype(int)

    def score(self, X, y, sample_weight=None):
        score = f1_score(y, self.predict(X), average='micro')
        print("f1_score:", score)
        return score
