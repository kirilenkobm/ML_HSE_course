import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import math as m
# Logistic regression task
# Evaluation of the usefulness of the regularization

# Dataset loading
dataset = np.loadtxt('data-logistic.csv', delimiter=",")
y = np.array(dataset[:, 0])
X = np.array(dataset[:, 1:3])

# Realization of logistic regression without regularization
clf = LogisticRegression(intercept_scaling=0.1, tol=0.00001)
clf.fit(X, y)
print(clf.coef_)
pred = clf.predict_proba(X)[:, 1]
print(roc_auc_score(y, pred))

# Realization of logistic regression with L2-regularization
clf2 = LogisticRegression(C=10, penalty='l2', intercept_scaling=0.1, tol=0.00001)
clf2.fit(X, y)
print(clf2.coef_)
pred = clf2.predict_proba(X)[:, 1]
print(roc_auc_score(y, pred))

