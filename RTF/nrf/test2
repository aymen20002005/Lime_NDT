import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from RTF.nrf.Neural_Decision_Tree import *

dataset_length = 100
D = 2

X = np.random.randn(dataset_length, D)*0.1
X[0:dataset_length//2, 0] += 0.1
X[0:dataset_length//2, 0] += 0.2
Y = np.ones(dataset_length)
Y[0:dataset_length//2] *= 0

X_test = np.random.randn(dataset_length, D)*0.1
X_test[0:dataset_length//2, 0] += 0.1
X_test[0:dataset_length//2, 0] += 0.2
Y_test = np.ones(dataset_length)
Y_test[0:dataset_length//2] *= 0
# Train a Tree
clf = DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X, Y)

a = NDTClassifier(D=2, gammas=[100, 1], tree_id=0)
a.compute_matrices_and_biases(clf)
a.to_keras("binary_crossentropy", dropouts=[0., 0., 0.])
print("FLOPs before:", a.count_ops)
a.fit(X, Y, verbose=1)
print("FLOPs after:", a.count_ops)

dataset_length = 100
D = 2

X = np.random.randn(dataset_length, D)*0.1
Y = X[:,0]**2+3*X[:, 1]+2

X_test = np.random.randn(dataset_length, D)*0.1
Y_test = X[:,0]**2+3*X[:, 1]+2
# Train a Tree
clf = DecisionTreeRegressor(max_depth=5)
clf = clf.fit(X, Y)

b = NDTRegressor(D=2, gammas=[100, 1], tree_id=0)
b.compute_matrices_and_biases(clf)
b.to_keras('mean_squared_error', dropouts=[0., 0., 0.],
            kernel_regularizer=[None, None, None])
print("FLOPs before:", b.count_ops)
b.fit(X, Y, verbose=1)
print("FLOPs after:", b.count_ops)