from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime as dt
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression

"""
Final project at machine learning course (HSE & Yandex), Coursera
"""
__author__ = 'kirilenko_bm'

GB = False
LR1 = False
LR2 = False
Bags = False

"""
GB - Gradient Boosting, LR1 - Logistic Regression with whole data,
LR2 - Logistic Regression without categorial features
Bags - Logistic Regression + Bag of words
"""

# Dataset loading
features = read_csv('features.csv', index_col='match_id')

# Dropping features with data from game results
data_from_future = ['duration', 'tower_status_radiant', 'tower_status_dire',
                    'barracks_status_radiant', 'barracks_status_dire']
st = ['start_time']  # this variable not matter

features.drop(data_from_future+st, axis=1, inplace=True)  # 97230 x 102

# Part 1, Gradient Boosting, Q 1-4 ############################################

# Which features contains NA values? #1
missed = features.count()[features.count() != len(features)]
print('Features with missed values: \n', ', '.join(list(missed.index)), '\n')
features.fillna(0, inplace=True)

# Extracting the target value from the set of features #2
rw = ['radiant_win']
y = features.radiant_win.values
features.drop(rw, axis=1, inplace=True)
X = features.ix[:, :]

# How long have held the cross-validation for GB with 30 trees? # 3 # 4
# Performing the gradient boosting over 5 folds with the cross-validation using ROC-AUC scoring
folds = KFold(n=features.shape[0], n_folds=5, shuffle=True, random_state=241)

if GB:
    print('Gradient Boosting:')
    for trees in [1, 10, 20, 30, 40]:
        start_time = dt.now()
        clf = GradientBoostingClassifier(n_estimators=trees)
        score = cross_val_score(
            estimator=clf,
            X=X, y=y,
            scoring='roc_auc',
            cv=folds
        ).mean()
        elapsed = dt.now() - start_time
        print('Trees: {0}, ROC-AUC score: {1:.6f},'
              ' Elapsed time: {2} seconds'.format(trees, score, elapsed.seconds))

# Part 2, Logistic Regression, Q 5-9 ##########################################

# Features scaling
X_scaled = scale(X)

# Find the best C-value for the Linear Regression: #5
best_score, best_C, best_time = 0, 0, 0

if LR1:
    print('\nLogistic regression:')
    for C in np.logspace(-3, 1, num=10):
        start_time = dt.now()
        score = cross_val_score(
            LogisticRegression(penalty='l2', C=C),
            X=X_scaled,
            y=y,
            scoring='roc_auc',
            cv=folds
        ).mean()
        elapsed = dt.now() - start_time

        if score > best_score:
            best_score = score
            best_C = C
            best_time = elapsed

if LR1:
    print('Best ROC-AUC score: {0:.6f}, Best C: {1:.4f},'
          ' Best time: {2} seconds'.format(best_score, best_C, best_time.seconds), '\n')
C = 0.01

# Removing categorial features and re-running the regression #6
heroes = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
          'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
lt = ['lobby_type']

# Load the dataset again for convenience in the future
features2 = read_csv('features.csv', index_col='match_id')

features2.drop(heroes+lt+data_from_future+st+rw, axis=1, inplace=True)
features2.fillna(0, inplace=True)
X2 = features2.ix[:, :]
X2_scaled = scale(X2)

# Measuring the quality after categorial features removing
if LR2:
    print('Logistic regression without categorial features:')
    start_time = dt.now()
    score = cross_val_score(LogisticRegression(penalty='l2', C=C),
                            X=X2_scaled, y=y,
                            scoring='roc_auc',
                            cv=folds
                            ).mean()
    elapsed = dt.now() - start_time
    print('C: {0:.4f}, ROC-AUC Score: {1:.6f},'
          ' Elapsed time: {2} seconds'.format(C, score, elapsed.seconds), '\n')

# How many heroes exists in the game? #7
hero_data = features.ix[:, heroes]
unique_count = len(pd.Series(hero_data.values.ravel()).unique())
print('Unique heroes count: {0}'.format(unique_count), '\n')
N_words = pd.Series(hero_data.values.ravel()).unique().max()

# Adding new features, bag-of-words for heroes #8
X_pick = np.zeros((features.shape[0], N_words))
for i, match_id in enumerate(features.index):
    for p in range(5):
        X_pick[i, features.ix[match_id, 'r%d_hero' % (p + 1)]-1] = 1
        X_pick[i, features.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
X_bags = np.concatenate([X2_scaled, X_pick], axis=1)

clf = LogisticRegression(penalty='l2', C=C)
if Bags:
    clf.fit(X_bags, y)
    score = cross_val_score(estimator=clf, X=X_bags, y=y,
                            cv=folds, scoring='roc_auc').mean()
    print('Logistic Regression with bag-of-words, ROC-AUC Score: {0:.6f}'.format(score), '\n')

# What is the minimum\maximum value of the forecast on
# the test sample came from the best of the algorithms?

if not Bags:
    clf.fit(X_bags, y)

features_test = read_csv('features_test.csv', index_col='match_id')
features_test_raw = read_csv('features_test.csv', index_col='match_id')
features_test.drop(heroes+lt+st, inplace=True, axis=1)
features_test.fillna(0, inplace=True)
X_test_no_bag = features_test.ix[:, :]
X_test_no_bag = scale(X_test_no_bag)

# Bag-of-words for the heroes of the test data:
X_pick = np.zeros((len(features_test_raw), N_words))
for i, match_id in enumerate(features_test_raw.index):
    for p in range(5):
        X_pick[i, features_test_raw.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick[i, features_test_raw.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
X_test = np.concatenate([X_test_no_bag, X_pick], axis=1)

# Min/max values:
proba = clf.predict_proba(X_test)[:, 1]
pmax, pmin = np.amax(proba), np.amin(proba)
print('Max proba: {0:.6f},\nMin proba: {1:.6f}'.format(pmax, pmin))
