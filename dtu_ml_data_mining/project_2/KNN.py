from matplotlib.pyplot import (figure, hold, plot, title, xlabel, ylabel,
                               colorbar, imshow, xticks, yticks, show, suptitle)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import model_selection

from dtu_ml_data_mining.project_2.project_2 import *


# Maximum number of KNN
L = 30

# Multiply the number of neigbors
M = 2

# K-nearest neighbors
KNN = [e * M for e in range(1, L + 1)]

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist = list(range(1, 3))

# Two layer K-fold crossvalidation
K_1 = 10
K_2 = 10
OCV = model_selection.KFold(n_splits=K_1, shuffle=True)
ICV = model_selection.KFold(n_splits=K_2, shuffle=True)

# Error matrixs
error_val = np.empty((L, 2, K_2))
error_gen = np.empty((L, 2))
errors = np.zeros(K_1)


class ParamSet:
    """Container for test error and corresponding parameters"""
    def __init__(self, test_error, neighbors, norm):
        self.test_error = test_error
        self.neighbors = neighbors
        self.norm = norm


optimal_parameters = []

# Start two layer crossvalidation
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

        # Train and test data for every model
        for knn in KNN:
            for d in dist:
                knclassifier = KNeighborsClassifier(n_neighbors=knn, p=d)
                knclassifier.fit(X_train, y_train)
                y_est = knclassifier.predict(X_val)

                # Calculate validation error * X_val / X_par
                error_val[int(knn / M - 1), (d - 1), k_2] = np.sum(y_est != y_val, dtype=float) * (len(X_val) / len(X_par))

        k_2 += 1

    # Outer layer CV
    # Calculate Generalisation Error for every model
    for knn in KNN:
        for d in dist:
            error_gen[int(
                knn / M - 1)][d - 1] = np.sum(error_val[int(knn / M - 1)][d - 1], dtype=float)

    # find model with minimum error
    d_min_index = 0
    knn_min_index = 0
    min_error = 1000
    for knn in KNN:
        min_temp_index = np.argmin(error_gen[int(knn / M - 1)])

        if min_error > error_gen[int(knn / M - 1)][min_temp_index]:
            min_error = error_gen[int(knn / M - 1)][min_temp_index]
            d_min_index = min_temp_index
            knn_min_index = knn

    # Train on X_par
    knclassifier = KNeighborsClassifier(
        n_neighbors=knn_min_index, p=d_min_index + 1)
    knclassifier.fit(X_par, y_par)
    y_est = knclassifier.predict(X_test)

    # Compute true generalisation error for the model * X_test / X_k
    test_error = (np.sum(y_est != y_test, dtype=float) /
                  len(X_test)) * (len(X_test) / len(X_k))
    errors[k_1] = test_error

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_est)
    accuracy = 100 * cm.diagonal().sum() / cm.sum()
    error_rate = 100 - accuracy
    figure(2)
    imshow(cm, cmap='binary', interpolation='None')
    colorbar()
    xticks(range(2))
    yticks(range(2))
    xlabel('Predicted class')
    ylabel('Actual class')
    suptitle('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(
        accuracy, error_rate))
    title('KNN Model (Distance: {0}, KNN: {1})'.format(
        d_min_index + 1, knn_min_index))
    # show()

    # Save set of optimal parameters
    optimal_parameters.append(ParamSet(test_error, knn_min_index, d_min_index + 1))

    k_1 += 1


# Print the classification error rate
print('Error rate: {0}%'.format(100 * np.sum(errors)))

# Find optimal model by minimizing with respect to test_error
opt_param_set = min(optimal_parameters, key=lambda p: p.test_error)
print(('Optimal parameter set:\n'
       ' - Neighbors: {}\n'
       ' - Norm: {}').format(opt_param_set.neighbors, opt_param_set.norm))
