# exercise 6.1.1
import os

import graphviz
import numpy as np
from matplotlib.pylab import (
    figure, legend, plot, show, xlabel, ylabel, text,
    imshow, colorbar, xticks, yticks, suptitle, title, savefig, close,
)
from matplotlib.pylab import cm as colormap
from scipy.io import loadmat
from sklearn import model_selection, tree
from sklearn.metrics import confusion_matrix

from dtu_ml_data_mining.project_2.project_2 import *


## Cross validation ##

# Standardize and normalize the data #
# Subtract mean value from data
X_k = X_k - np.ones((N, 1)) * X_k.mean(0)

# Divide by standard deviation
X_k /= np.ones((N, 1)) * X_k.std(0)

# Join observations with response variable and shuffle everything together;
# then split and carry out analysis.
y = y.reshape(y.shape[0], 1)
X_k_with_grades = np.append(X_k, y, axis=1)

# Shuffle rows according to seed
np.random.seed(seed=20)
np.random.shuffle(X_k_with_grades)

X_k, y = X_k_with_grades[:, :-1], X_k_with_grades[:, -1]
y = y.A.ravel()

# Outer layer. Compute 5 different optimal models and their test errors
K_outer = 10
K_inner = 10
OCV = model_selection.KFold(n_splits=K_outer, shuffle=False)
ICV = model_selection.KFold(n_splits=K_inner, shuffle=False)

test_errors = np.zeros((K_outer,))
test_errors_weighted = np.zeros((K_outer,))
test_errors_weighted_ratio = np.zeros((K_outer,))

optimal_depths = {}

outer_iteration = 0
for fold_no_outer, (par_index, test_index) in enumerate(OCV.split(X_k), 1):
    # Reset generalization error estimate array
    gen_errors = np.zeros((19,))

    print(f'Outer {fold_no_outer}/{K_outer}')
    # extract training and test set for current OCV fold
    X_par = X_k[par_index, :]
    y_par = y[par_index]
    X_test = X_k[test_index, :]
    y_test = y[test_index]

    test_size_ratio = y_test.shape[0] / X_k.shape[0]

    for fold_no_inner, (train_index, val_index) in enumerate(ICV.split(X_par), 1):
        print(f'\tInner {fold_no_inner}/{K_inner}')

        X_train = X_par[train_index, :]
        y_train = y_par[train_index]
        X_val = X_par[val_index, :]
        y_val = y_par[val_index]

        val_size_ratio = y_val.shape[0] / y_par.shape[0]

        k = range(2, 21)  # tree complexity

        # Train each model
        for i, d in enumerate(k):
            # print(f'\t\tTraining model no. {i + 1} out of {len(k)} (depth: {d})')
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=d)
            dtc = dtc.fit(X_train, y_train)
            y_est_val = dtc.predict(X_val)

            # validation error
            val_error = sum(np.abs(y_est_val - y_val))

            # Add to cumulated generalization error for this particular model,
            # i.e. this particular choice for max depth.
            gen_errors[i] += val_error * val_size_ratio

    # Find optimal model and train on training data
    optimal_model_idx = np.argmin(gen_errors)
    optimal_depth = range(2, 21)[optimal_model_idx]
    dtc = tree.DecisionTreeClassifier(criterion='gini',
                                      max_depth=optimal_depth)
    dtc = dtc.fit(X_par, y_par)

    # Validate against test data to obtain the test error
    y_est_test = dtc.predict(X_test)

    test_error = sum(np.abs(y_est_test - y_test))
    test_errors[outer_iteration] = test_error
    test_errors_weighted[outer_iteration] = test_error * test_size_ratio

    test_error_ratio = test_error / len(y_test)
    test_errors_weighted_ratio[outer_iteration] = test_error_ratio * \
        test_size_ratio

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_est_test)
    accuracy = 100 * cm.diagonal().sum() / cm.sum()
    error_rate = 100 - accuracy
    figure(2)

    thresh = 30
    for i in (0, 1):
        for j in (0, 1):
            text(j, i, str(cm[i, j]),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    imshow(cm, cmap=colormap.Blues, vmin=0, vmax=80)
    bounds = list(range(0, 85, 5))
    colorbar(boundaries=bounds)
    xticks(range(2), ('failed', 'passed'))
    yticks(range(2), ('failed', 'passed'))
    xlabel('Predicted class')
    ylabel('Actual class')
    suptitle('Confusion matrix (Accuracy: {:.4f}%, Error Rate: {:.4f}%)'.format(
        accuracy, error_rate))
    title('DT Model (depth: {})'.format(optimal_depth))

    filepath = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(filepath,
                               f'confusion_matrices/dt/confmat_{outer_iteration + 1}.png')
    savefig(output_path, format='png', dpi=500, bbox_inches='tight')
    # show()
    close()

    optimal_depths[optimal_depth] = test_error_ratio

    outer_iteration += 1


gen_error = np.sum(test_errors_weighted)
gen_error_ratio = np.sum(test_errors_weighted_ratio)

# Print unweighted test errors
print('Unweighted test errors (Fold <fold_no>: <test_error>):')
for i, t in enumerate(test_errors, 1):
    print(f'Fold {i}: {t}')

# Use the newly calculated test errors to find the generalization error


print(f'Test errors (weighted with respect to size of test set): ',
      test_errors_weighted)
print(f'Generalization error: {gen_error}')

print(f'Test errors (ratios; weighted with respect to size of test set): ',
      test_errors_weighted_ratio)
print(f'Generalization error: {gen_error_ratio * 100}%')

print('The optimal model parameters (depth, test_error): ',
      min(optimal_depths.items(), key=lambda e: e[1]))


dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
dtc = dtc.fit(X_k, y)
out = tree.export_graphviz(dtc, out_file='tree_project_2.gvz',
                           feature_names=k_encoded_attr_names)

# src = graphviz.Source.from_file('tree_project_2.gvz')


#         y_est_val = dtc.predict(X_val)

#         misclass_rate_val = sum(np.abs(y_est_val - y_val)
#                                 ) / float(len(y_est_val))

#         # y_est_test = dtc.predict(X_test)
#         # y_est_train = dtc.predict(X_train)
#         # misclass_rate_test = sum(np.abs(y_est_test - y_test)) / float(len(y_est_test))
#         # misclass_rate_train = sum(np.abs(y_est_train - y_train)) / float(len(y_est_train))

#         Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train

#     # for i, t in enumerate(tc):
#     #     X_train = X_par[train_index, :]
#     #     y_train = y_par[train_index]
#     #     X_val = X_test[val_index, :]
#     #     y_val = y_test[val_index]


# # Tree complexity parameter - constraint on maximum depth
# tc = np.arange(2, 21, 1)

# # Simple holdout-set crossvalidation
# test_proportion = 0.5
# X_train, X_test, y_train, y_test = model_selection.train_test_split(
#     X_k, y, test_size=test_proportion)

# # Initialize variables
# Error_train = np.empty((len(tc), 1))
# Error_test = np.empty((len(tc), 1))

# for i, t in enumerate(tc):
#     # Fit decision tree classifier, Gini split criterion, different pruning levels
#     dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
#     dtc = dtc.fit(X_train, y_train)

#     # Evaluate classifier's misclassification rate over train/test data
#     y_est_test = dtc.predict(X_test)
#     y_est_train = dtc.predict(X_train)
#     misclass_rate_test = sum(np.abs(y_est_test - y_test)
#                              ) / float(len(y_est_test))
#     misclass_rate_train = sum(
#         np.abs(y_est_train - y_train)) / float(len(y_est_train))
#     Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train

# f = figure()
# plot(tc, Error_train)
# plot(tc, Error_test)
# xlabel('Model complexity (max tree depth)')
# ylabel('Error (misclassification rate)')
# legend(['Error_train', 'Error_test'])

# show()

# tc = np.arange(2, 21, 1)
# t = Error_test[0]
# k = 2
# for i, j in enumerate(tc):
#     v = Error_test[i]
#     if v < t:
#         t = v
#         k = j

# print(f'lowest error for depth {k}')
