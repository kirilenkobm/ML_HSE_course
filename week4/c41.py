import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
# Prediction of annual salary by description

# Dataset loading
df = pd.read_csv('salary-train.csv')
testdf = pd.read_csv('salary-test-mini.csv')

# Text should be lowercased
df['FullDescription'].str.lower()
testdf['FullDescription'].str.lower()
# ...and purified from special characters
train1 = df['FullDescription'].str.lower()
train2 = train1.replace('[^a-zA-Z0-9]', ' ', regex = True)
test1 = testdf['FullDescription'].str.lower()
test2 = test1.replace('[^a-zA-Z0-9]', ' ', regex = True)

# TF-IDF
vectorizer = TfidfVectorizer(min_df=5)
X_block1 = vectorizer.fit_transform(train2)
X_test1 = vectorizer.transform(test2)
# Filling of NAN values
LocTrain = df['LocationNormalized'].fillna('nan', inplace=True)
ContrTime =  df['ContractTime'].fillna('nan', inplace=True)
# One-hot coding for LocationNormalized and ContractTime features
enc = DictVectorizer()
X_block2 = enc.fit_transform(df[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test2 = enc.transform(testdf[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Merge of the fragments
X_train = hstack([X_block1, X_block2])
X_test = hstack([X_test1, X_test2])
y_train = df['SalaryNormalized']

# Prediction
y_train = df['SalaryNormalized']
clf = Ridge(alpha=1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('Predicted salary: ', np.round(pred, decimals=2))
