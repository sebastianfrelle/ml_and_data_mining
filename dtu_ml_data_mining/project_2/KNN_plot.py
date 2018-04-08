from matplotlib.pyplot import figure, plot, xlabel, ylabel, show, legend, title

import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

from project_2 import *


# Maximum number of KNN
L = 35

# Multiply the number of neigbors
M = 1

# K-nearest neighbors
KNN = [e * M for e in range(1, L + 1)]

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist = list(range(1, 3))

# K-fold crossvalid
CV = model_selection.KFold(n_splits=K, shuffle=True)

# Error matrixs
errors = np.empty((2, K, L))

# Start two layer crossvalidation
k = 0
for train_index, test_index in CV.split(X_k):
    print('Crossvalidation fold: {0}/{1}'.format(k + 1, K))

    # extract training and test set for current outer CV fold
    X_train = X_k[train_index, :]
    y_train = y[train_index]
    X_test = X_k[test_index, :]
    y_test = y[test_index]

    # Train and test data for every model
    for knn in KNN:
        for d in dist:
            knclassifier = KNeighborsClassifier(n_neighbors=knn, p=d)
            knclassifier.fit(X_train, y_train)
            y_est = knclassifier.predict(X_test)
            # knclassifier.validation()

            # Calculate validation error * X_val / X_par
            errors[(d - 1), k, int(knn / M - 1)] = np.sum(y_est != y_test, dtype=float)
    k += 1


# Plot the classification error rate
figure()
plot(KNN, 100 * sum(errors[0]) / len(X_k), label='Distance: 0')
plot(KNN, 100 * sum(errors[1]) / len(X_k), label='Distance: 1')
legend()
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
title('Classificarion Error Rate for KNN')
show()
