import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from LinearRegression import LinearRegression

def plot_curve(Xlist, Ylist, title, xlabel, ylabel, plotlabels):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    cnt = 0
    for X in Xlist:
        Y = Ylist[cnt]
        label = plotlabels[cnt]
        plt.plot(X, Y, 'o-', label=label)
        cnt += 1
    plt.legend(loc="best")
    plt.show()
    plt.clf()

def load_dataset():
    train_initial = pd.read_csv('datasets/train.csv')
    test_initial = pd.read_csv('datasets/test.csv')
    return train_initial, test_initial


def preprocess(train):
    # ignore Id; ignore MasVnrArea and GarageYrBlt too for now as no easy way of handling missing values
    train = train.drop(['Id', 'MasVnrArea', 'GarageYrBlt'], axis=1)

    # transform year variables to be, 'years since 2018'
    train[['YearBuilt', 'YearRemodAdd', 'YrSold']] = 2018 - train[['YearBuilt', 'YearRemodAdd', 'YrSold']]

    numeric_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                        'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
                        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                        'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold', 'SalePrice']

    # get non-numeric columns
    non_numeric_columns = np.setdiff1d(train.columns.values, numeric_features)

    # create dummy variables for categorical variables
    # TODO: ideally would want to create these based on a list of possible feature values,
    # otherwise will have to combine with test data and then create dummy variables
    # as some feature values might be present in one set but not the other sample
    train = pd.get_dummies(train, columns=non_numeric_columns, drop_first=True, dummy_na=True)

    # interaction variable for MiscVal based on type of MiscFeature
    miscfeatures = [col for col in train.columns.values if col.startswith('MiscFeature')]
    train[miscfeatures] = train[miscfeatures].multiply(train['MiscVal'], axis="index")

    # replace missing values with mean. ideally we will want to do this separately in each training fold separately
    # but here we are focusing on training only and not validation / leakage
    train = train.fillna(train.mean())
    train_X = train.drop(['SalePrice'], axis=1).values
    train_y = train[['SalePrice']].values
    # scaling y variable
    train_y = train_y / 100000
    return train_X, train_y

def get_multiple_estimates(X, y, learning_rate=[0.0000000001], max_iter=1000, iteration_threshold=100, plotlabels=None,
                           reg_strength=0, regularization="Ridge", method="GD", minibatch_size=1, plot_by_lr=True, plot_by_mb=False):
    data_points = math.floor(max_iter / iteration_threshold) + 1

    cost_by_lr = np.empty((0, data_points))
    iterations = np.empty((0, data_points))
    if plotlabels is None:
        plotlabels = []

    if plot_by_lr:
        for lr in learning_rate:
            estimator_linReg = LinearRegression(learning_rate=lr, reg_strength=reg_strength, regularization=regularization,
                                                max_iter=max_iter, iteration_threshold=iteration_threshold, method=method)
            estimator_linReg.fit(X, y)

            cost_by_lr = np.vstack((cost_by_lr, estimator_linReg.cost_by_iteration))
            iterations = np.vstack((iterations, estimator_linReg.iterations))
            plotlabels.append("Learning rate = " + str(lr))
    elif plot_by_mb:
        for mb in minibatch_size:
            estimator_linReg = LinearRegression(learning_rate=learning_rate, reg_strength=reg_strength, regularization=regularization,
                                                max_iter=max_iter, iteration_threshold=iteration_threshold,
                                                method=method, minibatch_size=mb)
            estimator_linReg.fit(X, y)

            cost_by_lr = np.vstack((cost_by_lr, estimator_linReg.cost_by_iteration))
            iterations = np.vstack((iterations, estimator_linReg.iterations))
            plotlabels.append("Minibatch size = " + str(mb))
    else:
        cnt = 0
        for pl in plotlabels:
            estimator_linReg = LinearRegression(learning_rate=learning_rate[cnt], reg_strength=reg_strength,
                                                regularization=regularization,
                                                max_iter=max_iter, iteration_threshold=iteration_threshold,
                                                method=pl, minibatch_size=minibatch_size)
            estimator_linReg.fit(X, y)

            cost_by_lr = np.vstack((cost_by_lr, estimator_linReg.cost_by_iteration))
            iterations = np.vstack((iterations, estimator_linReg.iterations))
            cnt += 1

    return cost_by_lr, iterations, plotlabels