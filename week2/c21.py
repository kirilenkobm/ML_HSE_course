from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_score, KFold
import numpy as np

# The calculation of the mean square error
# at the classic dataset example

# Loading and preprocessing data
boston = load_boston()
data = boston.data
X = preprocessing.scale(data)
y = boston.target

# Best accuracy score calculation
best_p, best_score = 0, -float('inf')
kf = KFold(len(y), n_folds=5,
           shuffle=True, random_state=241)

# For variable power parameter for Minkowski metric
for p in np.linspace(1, 10, num=200):
    knr = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
    score = max(cross_val_score(knr, X, y, cv=kf, scoring='mean_squared_error'))
    if score > best_score:
        best_score = score
        best_p = p

# Output
best_score = np.round(best_score, decimals=4)
best_p = np.round(best_p, decimals=4)
print('Best p: {0}, best score: {1}'.format(best_p, best_score))

