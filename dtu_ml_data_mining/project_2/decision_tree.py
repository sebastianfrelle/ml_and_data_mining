# exercise 6.1.1

from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show
from scipy.io import loadmat
from sklearn import model_selection, tree
import numpy as np

from dtu_ml_data_mining.project_2.project_2 import *

import graphviz

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)


# Encode grades into classes for pass/fail
y = grades[:, 2]
for i in range(y.shape[0]):  # iterate using no. of columns in grades
    e = y[i, :]
    if e < 10:
        y[i, :] = 0
    else:
        y[i, :] = 1

print(y.shape)
print(X_k.shape)

y = y.A.ravel()
print(y.shape)

# Simple holdout-set crossvalidation
test_proportion = 0.5
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_k, y, test_size=test_proportion)

# Initialize variables
Error_train = np.empty((len(tc), 1))
Error_test = np.empty((len(tc), 1))

for i, t in enumerate(tc):
    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
    dtc = dtc.fit(X_train, y_train)

    # Evaluate classifier's misclassification rate over train/test data
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)
    misclass_rate_test = sum(np.abs(y_est_test - y_test)
                             ) / float(len(y_est_test))
    misclass_rate_train = sum(
        np.abs(y_est_train - y_train)) / float(len(y_est_train))
    Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train

f = figure()
plot(tc, Error_train)
plot(tc, Error_test)
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate)')
legend(['Error_train', 'Error_test'])

show()

tc = np.arange(2, 21, 1)
t = Error_test[0]
k = 2
for i, j in enumerate(tc):
    v = Error_test[i]
    if v < t:
        t = v
        k = j

print(f'lowest error for depth {k}')

dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=k)
dtc = dtc.fit(X_k, y)

out = tree.export_graphviz(
    dtc, out_file='tree_project_2.gvz', feature_names=k_encoded_attr_names)
# graphviz.render('dot','png','tree_gini',quiet=False)
src = graphviz.Source.from_file('tree_project_2.gvz')

print('Ran Exercise 6.1.1')
