import pandas as pd
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
# Validation task

# Loading of data
classification = np.loadtxt('classification.csv', delimiter=",", skiprows=1)
act = classification[:, 0]
pred = classification[:, 1]

# True/False Positive/Negative results determination
TP, FP, FN, TN = 0, 0, 0, 0
for i in range(200):
    if act[i] == pred[i]:
        if act[i] == 1:
            TP += 1
        else:
            TN += 1
    else:
        if act[i] == 0:
            FP += 1
        else:
            FN += 1

print('First part:')
print('TP: {0}, FP: {1}, FN: {2}, TN: {3}'.format(TP, FP, FN, TN))
# Calculation of accuracy, precision, recall and F-value
accuracy = accuracy_score(act, pred)
precision = precision_score(act, pred)
recall = recall_score(act, pred)
f1 = f1_score(act, pred)

print('Accuracy: ', np.round((accuracy), decimals=2))
print('Precision: ', np.round((precision), decimals=2))
print('Recall: ', np.round((recall), decimals=2))
print('F-value: ', np.round((f1), decimals=2), '\n')

# Loading of second dataset; prediction scores for different classifiers
task2 = np.loadtxt('scores.csv', delimiter=",", skiprows=1)
actual = task2[:, 0]
logreg = task2[:, 1]
svm = task2[:, 2]
knn = task2[:, 3]
tree = task2[:, 4]

# ROC-AUC score for every classifier
logregauc = roc_auc_score(actual, logreg)
svnauc = roc_auc_score(actual, svm)
knnauc = roc_auc_score(actual, knn)
treeauc = roc_auc_score(actual, tree)

print('Second part: ')
print('Logistic regression: {0}\nSVN: {1}\nkNN: {2}\nDecision Tree: {3}'
      .format(logregauc, svnauc, knnauc, treeauc))

"""
Какой классификатор достигает наибольшей точности
(Precision) при полноте (Recall) не менее 70% ?
Чтобы получить ответ на этот вопрос, найдите все точки'
precision-recall-кривой с помощью функции
sklearn.metrics.precision_recall_curve.
"""

# Which classifier reaches most accuracy with recall <= 70%?
preclog, relog, threlog = precision_recall_curve(actual, logreg)
precsvn, resvn, thresvn = precision_recall_curve(actual, svm)
precknn, reknn, threknn = precision_recall_curve(actual, knn)
prectree, retree, thretree = precision_recall_curve(actual, tree)

if False:
    plt.hist(preclog, threlog)
    plt.hist(precsvn, thresvn)
    plt.hist(precknn, threknn)
    plt.hist(prectree, thretree)
    plt.show()
