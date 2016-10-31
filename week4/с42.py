import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.decomposition import PCA
from numpy import corrcoef
# DJI dataset task

# Loading of data
X = read_csv('close_prices.csv')
X = X.drop('date', 1)

# How many the components enough to explain 90% of the variance?
pca = PCA(n_components=1)
pca.fit(X)
companies = pca.components_
print(companies)

# Pearson corellation between prediction and real DJI
from scipy.stats import pearsonr
X_trans = pca.transform(X)
df = read_csv('djia_index.csv')
df = df.drop('date', 1)
y_trans = df.as_matrix()
pear = pearsonr(X_trans, y_trans)
print('Pearson correlation: ', np.round(pear[0], decimals=2))

# Which company has the the greatest weight in the first component?
print('Greatest weight: ', companies.max())
wh = pd.Series(pca.components_[0]).sort_values(ascending=False).head(1).index[0]
company = X.columns[wh]
print('Company: ', company)
