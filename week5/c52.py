import numpy as np
import pandas as pd
from numpy import argmin
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# Dataset contains properties of some molecules and possibility of reaction between them
# Comparison of Gradient Boosting and Random Forest estimators


def get_loss(clf, X, y):  # log_loss method extention
    loss = []
    for y_pred in clf.staged_decision_function(X):
        form = (1.0 / (1.0 + np.exp(-y_pred)))
        loss.append(log_loss(y, form))
    min_loss = loss[argmin(loss)]
    return loss, argmin(loss), min_loss  # I want to return iteration num too

# Loading data
data = pd.read_csv('gbm-data.csv').values
# Splitting data to train and test set
X_train, X_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0], test_size=0.8, random_state=241)

# Gradient Boosting Classifier training with variable learning rate
for lr in [1, 0.5, 0.3, 0.2, 0.1]:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=lr)
    clf.fit(X_train, y_train)
    train_loss, train_min_iter, train_min_loss = get_loss(clf, X_train, y_train)
    test_loss, test_min_iter, test_min_loss = get_loss(clf, X_test, y_test)

    plot = False
    if plot:
        plt.figure()
        plt.plot(test_loss, 'r')
        plt.plot(train_loss, 'b')
        plt.legend(['test', 'train'])
        plt.show()

    # find iteration num for minimal log-loss value at lr = 0.2
    if lr == 0.2:
        gb_min_loss = test_min_loss
        gb_min_iter = test_min_iter

print('\nGradient boosting minimal log-loss: {0} at iteration num: {1}'
      .format(np.round(gb_min_loss, 2), gb_min_iter))

# Random Forest Classifier training
# n_estimators = min iteration for gradient boosting
clf = RandomForestClassifier(n_estimators=gb_min_iter, random_state=241)
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)[:, 1]
rf_loss = log_loss(y_test, y_proba)

print('Random forest minimal log-loss: {0}'.format(np.round(rf_loss, 2)))