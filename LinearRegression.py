from sklearn.base import BaseEstimator
import numpy as np
import time

class LinearRegression(BaseEstimator):

    def __init__(self, learning_rate=0.001, reg_strength=0, regularization="Ridge", max_iter=1000, gd_threshold=0.01, iteration_threshold=100):
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.regularization = regularization
        self.gd_threshold = gd_threshold
        self.max_iter = max_iter
        self.iteration_threshold = iteration_threshold

    def fit(self, X, y):
        """
        fits Linear Regression using SGD on labeled dataset <X, y>
        :param X: feature matrix
        :param y: labeled value
        :return: self
        """
        start_time = time.time()

        num_examples = X.shape[0]
        num_features = X.shape[1]
        theta = np.zeros((1, num_features))
        bias = 0
        h = bias + np.dot(X,theta.T)
        prev_cost = -1
        curr_cost = np.sqrt((1/(2*num_examples)) * sum((h - y)**2))

        iter = 0
        cost = curr_cost.copy()
        iterations = np.array(iter)
        while self.gd_threshold is None or (prev_cost - curr_cost) > self.gd_threshold:
            #print(iter, theta, curr_cost)
            prev_cost = curr_cost
            theta = theta - (1/num_examples) * self.learning_rate * (sum((h - y)*X) + self.reg_strength*theta)
            bias = bias - (1/num_examples) * self.learning_rate * sum((h - y))
            h = bias + np.dot(X,theta.T)
            curr_cost = np.sqrt((1/(2*num_examples)) * sum((h - y)**2))

            iter += 1
            if iter % self.iteration_threshold == 0:
                cost = np.append(cost, [curr_cost])
                iterations = np.append(iterations, [iter])

            if iter >= self.max_iter:
                break


        end_time = time.time()

        self.training_time = end_time - start_time
        self.cost_by_iteration = cost
        self.iterations = iterations
        self.final_cost = curr_cost
        self._intercept = bias
        self._coef = theta

        return self

    def predict(self, X):
        y = self._intercept + np.dot(X, self._coef.T)
        return y
