
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import model_selection

from project_2 import *

# K-nearest neighbors
KNN = [e * 1 for e in range(1, 11)]

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist = list(range(1, 3))

# Outer K-fold crossvalidation
K_1 = 10
K_2 = 10
OCV = model_selection.KFold(n_splits=K_1, shuffle=True)
ICV = model_selection.KFold(n_splits=K_2, shuffle=True)

error_val = np.empty((10, 2, 10))
error_gen = np.empty((10, 2))
errors = np.zeros(K_1)

k_1 = 0
for par_index, test_index in OCV.split(X_k):
    print('Outer Crossvalidation fold: {0}/{1}'.format(k_1 + 1, K_1))

    # extract training and test set for current outer CV fold
    X_par = X_k[par_index, :]
    y_par = y[par_index]
    X_test = X_k[test_index, :]
    y_test = y[test_index]

    # Inner K-fold crossvalidation
    k_2 = 0
    for train_index, val_index in ICV.split(X_par):
        print('  Inner Crossvalidation fold: {0}/{1}'.format(k_2 + 1, K_2))

        # extract training and test set for current inner CV fold
        X_train = X_par[train_index, :]
        y_train = y_par[train_index]
        X_val = X_par[val_index, :]
        y_val = y_par[val_index]

        # Train and test data
        for knn in KNN:
            for d in dist:
                knclassifier = KNeighborsClassifier(n_neighbors=knn, p=d)
                knclassifier.fit(X_train, y_train)
                y_est = knclassifier.predict(X_val)

                error_val[int(knn / 1 - 1)][(d - 1)][k_2] = np.sum(y_est != y_val, dtype=float) * len(X_val) / len(X_par)

        k_2 += 1

    for knn in KNN:
        for d in dist:
            error_gen[int(knn / 1 - 1)][d - 1] = np.sum(error_val[int(knn / 1 - 1)][d - 1], dtype=float)

    print(error_gen)
    # find model with min error
    d_min_index = 0
    knn_min_index = 0
    min_error = 1000
    for knn in KNN:
        min_temp_index = np.argmin(error_gen[int(knn / 1 - 1)])

        if min_error > error_gen[int(knn / 1 - 1)][min_temp_index]:
            min_error = error_gen[int(knn / 1 - 1)][min_temp_index]
            d_min_index = min_temp_index
            knn_min_index = knn

    knclassifier = KNeighborsClassifier(n_neighbors=knn_min_index, p=d_min_index + 1)
    knclassifier.fit(X_par, y_par)
    y_est = knclassifier.predict(X_test)

    errors[k_1] = np.sum(y_est != y_test, dtype=float) * len(X_test) / len(X_k)

    k_1 += 1


# Plot the classification error rate
print('Error rate: {0}%'.format(100 * np.sum(errors)))
