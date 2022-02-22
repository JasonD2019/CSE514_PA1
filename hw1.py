# summarize the data
from pandas import read_excel
import numpy as np
import pandas as pd
import xlrd
import random
import matplotlib.pyplot as plt
import statistics
import copy

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
data = read_excel(url)
features = data.columns

columns = data.columns
train = np.array(data[:900])
test = np.array(data[900:])
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

# A) Uni-variate linear regression
# Uni-variate linear: f(x) = mx + b
def MSE_loss_function(x, y, m, b, n):
    return 1 / n * np.sum((y - (m*x + b)) ** 2)

def univariateLinearRegression(feature, x_train, y_train, x_test, y_test, learning_rate):

    # init
    m = 0
    b = 0
    old_mse = float("inf")
    iteration = 0
    mse = MSE_loss_function(x_train, y_train, m, b, 900)

    # when to stop, start training
    while old_mse - mse > 0.00001 and iteration < 1000000:
        y_pred = m * x_train + b
        m = m - learning_rate / 900 * np.sum(-2 * x_train * (y_train - y_pred))
        b = b - learning_rate / 900 * np.sum(-2 * y_train - y_pred)
        old_mse = mse
        mse = MSE_loss_function(x_train, y_train, m, b, 900)
        iteration += 1

    # print result
    print('%s' %(feature))
    print("Stop after %s iteration" %(iteration))
    print('Final m: %s' %(m))
    print('Final b: %s' %(b))
    print(f"Final Regression algorithm: f(x) = { round(m, 3) } * x + {round(b, 3)}" )

    print('Variance explained of your models on the testing data points: %s' %(1-MSE_loss_function(x_test, y_test, m, b, 130)/statistics.variance(y_test)))
    print('Variance explained of your models on the training dataset: %s' %(1-MSE_loss_function(x_train, y_train, m, b, 900)/statistics.variance(y_train)))
    print("----------------------------------------------------------------")


    # plot
    plt.figure()
    plt.scatter(x_train, y_train, s=5)
    x_range = np.arange(np.min(x_train), np.max(x_train), 0.1)
    y_range = m * x_range + b
    plt.plot(x_range, y_range)
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel(features[-1])
    plt.show()
    plt.close()

for i, feature in enumerate(features[:-1]):
    x_train = X_train[:, i]
    x_test = X_test[:, i]
    univariateLinearRegression(feature, x_train, y_train, x_test, y_test, 0.000000001)

# B) Multi-variate linear regression
# Multi-variate: f(x) = (m . x)
def multi_loss_function(x, y, a, n):
    return 1/n * np.sum((np.dot(x, a) - y) ** 2)

def multivariateLinearRegression(x_train, y_train, x_test, y_test):
    x_train = np.hstack((np.ones(900)[:, np.newaxis], x_train))
    x_test = np.hstack((np.ones(130)[:, np.newaxis], x_test))
    a = np.zeros(x_train.shape[1])
    prev_mse = float("inf")
    iteration = 0
    mse = multi_loss_function(x_train, y_train, a, 900)
    while prev_mse - mse > 0.00001 and iteration < 1000000:
        y_pred = np.dot(x_train, a.reshape(-1, 1))
        iteration += 1
        a = a - 0.0000001 * (2/900 * np.dot(y_pred.reshape(1, -1) - y_train.reshape(1, -1), x_train).reshape(-1,))
        prev_mse = mse
        mse = multi_loss_function(x_train, y_train, a, 900)

    print("When alpha = 0.0000001:")
    print("Stop after %s iteration" %(iteration))
    # print(a)
    for i in range(0,9):
        print(f"m{i} : {a[i]}")

    print('Variance explained of your models on the testing data points: %s' %(1-multi_loss_function(x_test, y_test, a, 130)/statistics.variance(y_test)))
    print('Variance explained of your models on the training dataset: %s' %(1-multi_loss_function(x_train, y_train, a, 900)/statistics.variance(y_train)))

# C) Optional extension 1 – Multi-variate polynomial regression
# Multi-variate polynomial regression
def poly_loss_function(x, y, a, n):
    return 1/n * np.sum((np.dot(x, a) - y) ** 2)

def multiVariatePolynomialRegression(x_train, y_train, x_test, y_test, learning_rate):
    newX_train = np.zeros((900,45))
    newcols=0
    for i in range(8):
        for j in range(i, 8):
            for k in range(900):
                newX_train[k][newcols] = copy.deepcopy(x_train[k][i] * x_train[k][j])
                for n in range(36, 44):
                    newX_train[k][n] = copy.deepcopy(x_train[k][n-36])
            newcols +=1;

    newX_test = np.zeros((130,45))
    newcols=0
    for i in range(8):
        for j in range(i, 8):
            for k in range(130):
                newX_test[k][newcols] = copy.deepcopy(x_test[k][i] * x_test[k][j])
                for n in range(36, 44):
                    newX_test[k][n] = copy.deepcopy(x_test[k][n-36])
            newcols +=1;

    a = np.zeros(45)
    prev_mse = float("inf")
    iteration = 0
    mse = poly_loss_function(newX_train, y_train, a, 900)
    while abs(prev_mse - mse) > 0.0000000000000000000001 and iteration < 1000000:
        y_pred = np.dot(newX_train, a.reshape(-1, 1))
        iteration += 1
        a = a - 0.0000000000000000001 * (2/900 * np.dot(y_pred.reshape(1, -1) - y_train.reshape(1, -1), newX_train).reshape(-1,))
        prev_mse = mse
        mse = poly_loss_function(newX_train, y_train, a, 900)
        # print("prev_mse: %s" %(prev_mse))
        # print("mse: %s" %(mse))

    print("When alpha = 0.0000000000000000001:")
    print("Stop after %s iteration" %(iteration))
    print(a)
    print('Variance explained of your models on the testing data points: %s' %(1-poly_loss_function(newX_test, y_test, a, 130)/statistics.variance(y_test)))
    print('Variance explained of your models on the training dataset: %s' %(1-poly_loss_function(newX_train, y_train, a, 900)/statistics.variance(y_train)))

# run Question A
for i, feature in enumerate(features[:-1]):
    x_train = X_train[:, i]
    x_test = X_test[:, i]
    univariateLinearRegression(feature, x_train, y_train, x_test, y_test, 0.000000001)

# run Question B
multivariateLinearsRegression(X_train, y_train, X_test, y_test)

# run Question C
multiVariatePolynomialRegression(X_train, y_train, X_test, y_test, 0.00001)


# D) Optional extension 2 – Data pre-processing
