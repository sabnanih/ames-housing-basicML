from sklearn.base import BaseEstimator
import numpy as np

class ImputationTransform(BaseEstimator):

    # do nothing
    def fit(self, X, y):
        return self

    def transform(self, X):
        col_mean = np.nanmean(X, axis=0)

        # Find indices that need to replaced
        indices = np.where(np.isnan(X))

        # update X only at indices by taking column mean only at positions given by indices
        X[indices] = np.take(col_mean, indices[1])

        return X