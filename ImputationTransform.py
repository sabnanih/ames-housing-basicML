from sklearn.base import BaseEstimator
import numpy as np

class ImputationTransform(BaseEstimator):

    def fit(self, X, y):
        self.col_mean = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        # Find indices that need to replaced
        indices = np.where(np.isnan(X))

        # update X only at indices by taking column mean only at positions given by indices
        X[indices] = np.take(self.col_mean, indices[1])

        return X