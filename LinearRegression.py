from sklearn.base import BaseEstimator
import numpy as np
import time

class LinearRegression(BaseEstimator):

    def __init__(self, learning_rate=0.001, reg_strength=0, regularization="Ridge", max_iter=1000, cost_threshold=None,
                 iteration_threshold=100, method="GD", minibatch_size=1, learning_rate_decay=False):
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.regularization = regularization
        self.cost_threshold = cost_threshold
        self.max_iter = max_iter
        self.iteration_threshold = iteration_threshold
        self.method = method
        self.minibatch_size = minibatch_size
        self.learning_rate_decay = learning_rate_decay

    def fit(self, X, y):
        """
        fits Linear Regression using GD on labeled dataset <X, y>
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
        curr_cost = np.sqrt((1/num_examples) * sum((h - y)**2))

        iter = 0
        cost = curr_cost.copy()
        iterations = np.array(iter)
        lr = self.learning_rate
        if self.method == "GD":
            while self.cost_threshold is None or prev_cost < 0 or (prev_cost - curr_cost) > self.cost_threshold:
                prev_cost = curr_cost
                theta = theta - (1/num_examples) * lr * (sum((h - y)*X) + self.reg_strength*theta)
                bias = bias - (1/num_examples) * lr * sum((h - y))
                h = bias + np.dot(X,theta.T)
                curr_cost = np.sqrt((1/num_examples) * sum((h - y)**2))

                iter += 1
                if iter % self.iteration_threshold == 0:
                    cost = np.append(cost, [curr_cost])
                    iterations = np.append(iterations, [iter])

                if self.learning_rate_decay:
                    lr = (1/(1+(0.001*iter))) * self.learning_rate

                if self.cost_threshold is None and iter >= self.max_iter:
                    break

        elif self.method == "SGD" or self.method == "Minibatch":
            if self.method == "SGD":
                self.minibatch_size = 1
            while self.cost_threshold is None or prev_cost < 0 or (prev_cost - curr_cost) > self.cost_threshold:
                # shuffle at each epoch
                data_idx = [i for i in range(0,num_examples)]
                np.random.shuffle(data_idx)
                X = X[data_idx,]
                y = y[data_idx,]
                h = h[data_idx,]
                prev_cost = curr_cost
                for i in range(0,num_examples,self.minibatch_size):
                    theta = theta - lr * (1/self.minibatch_size) * sum(((h - y)*X)[i:i+self.minibatch_size,]) - \
                            (1/num_examples) * lr * self.reg_strength * theta
                    bias = bias - lr * (1/self.minibatch_size) * sum((h - y)[i:i+self.minibatch_size,])
                    h = bias + np.dot(X,theta.T)
                curr_cost = np.sqrt((1/num_examples) * sum((h - y)**2))

                iter += 1
                if iter % self.iteration_threshold == 0:
                    cost = np.append(cost, [curr_cost])
                    iterations = np.append(iterations, [iter])

                if self.learning_rate_decay:
                    lr = (1/(1+(0.001*iter))) * self.learning_rate

                if self.cost_threshold is None and iter >= self.max_iter:
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

