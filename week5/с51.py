import pandas as pd
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score
# Predicting the age of seashells using RandomForest Estimator

df = read_csv('abalone.csv')
#print(df.head())

# Transformation Sex features to digital form
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
# Division to X and y
y = df.as_matrix()
y = y[:, 8]
X = df.as_matrix()
X = X[:, 0:8]

# Initiation of the Random Forest with the different estimators count
# Calculation of the R2-score
# Determine at what minimum amount of trees classifier shows quality at cross-validation above 0.52
for i in range(1, 51):
    rfg = RandomForestRegressor(n_estimators=i)
    rfg.fit(X, y)
    y_pre = rfg.predict(X)
    cv = KFold(n=4177, n_folds=5, random_state=1, shuffle=True)
    array = cross_val_score(X=X, y=y, estimator=rfg, cv=cv, scoring='r2')
    m = array.mean()
    print('Quality: {0}, N_estimators: {1}'.format(m, i))
