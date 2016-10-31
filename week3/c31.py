from sklearn.svm import SVC
import numpy as np

# loading data from 'svm-data.csv'
data = np.loadtxt('svm-data.csv', delimiter=",")
y_data = np.array(data[:, 0])
X_data = np.array(data[:, 1:3])

# Initiation and training C-Support Vector classifier
clf = SVC(C=100000, random_state=241, kernel='linear')
clf.fit(X_data, y_data)

# Find support vectors
print('support', clf.support_, '\n')
print('support vectors,', clf.support_vectors_, '\n')
print('n_support', clf.n_support_, '\n')