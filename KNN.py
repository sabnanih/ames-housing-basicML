from sklearn.base import BaseEstimator
import math
import numpy as np

class KNN(BaseEstimator):

    def __init__(self, k=1):
        self.k = k

    # just save training data, everything else is done at test time
    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        y = []
        for x in X:
            y.append(self._bestOutput(x, self.k))
        return y

    def _distance(self, x):
        diff_array = self.X - x
        dist = [math.sqrt(sum(d ** 2)) for d in diff_array]
        return dist

    def _bestOutput(self, x, k):
        dist = self._distance(x)
        sorted_idx = np.argsort(dist)
        y = self.y[sorted_idx]
        return np.mean(y[0:k])


