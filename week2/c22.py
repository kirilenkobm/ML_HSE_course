import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# Task of perceptron classifier training

# Loading and preparation of data
train = np.loadtxt('perceptron-train.csv', delimiter=",")
test = np.loadtxt('perceptron-test.csv', delimiter=",")
y_train = np.array(train[:, 0])
X_train = np.array(train[:, 1:3])
y_test = np.array(test[:, 0])
X_test = np.array(test[:, 1:3])

# Initiation and fitting classifier
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Accuracy calculation
acc = accuracy_score(y_test, predictions)
print('Accuracy score:', acc)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Accuracy calculation after scaling
clf.fit(X_train_scaled, y_train)
predictions_scaled = clf.predict(X_test_scaled)
acc_scaled = accuracy_score(y_test, predictions_scaled)
print('Accuracy score after scaling: ',acc_scaled)

# Difference between quality on the test sample after normalization and quality before it
output = (np.round((acc_scaled - acc), decimals=3))
print('Quality difference: ', output)