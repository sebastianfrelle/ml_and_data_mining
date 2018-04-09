import os

import numpy as np
import sklearn.linear_model as lm
from matplotlib.pylab import cm as colormap
from matplotlib.pylab import (
    close, colorbar, figure, imshow, legend, plot, savefig, show, suptitle,
    text, title, xlabel, xticks, ylabel, yticks,
)
from sklearn import model_selection
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

# Define models for splitting outer and inner folds
no_models = 5
K_outer = 10
K_inner = 10
OCV = model_selection.KFold(n_splits=K_outer, shuffle=False)
ICV = model_selection.KFold(n_splits=K_inner, shuffle=False)


# Train model with different regularization strengths. sklearn method takes
# inverse regularization strength, so prepare by producing a list containing the
# inverse of our reg. strengths.
regularization_strengths = [10 ** e for e in range(-2, 3)]
inv_regs = [1 / e for e in regularization_strengths]

# Store generalization
test_errors = np.zeros((K_outer,))
test_errors_weighted = np.zeros((K_outer,))
test_errors_weighted_ratio = np.zeros((K_outer,))


optimal_params = {}

outer_iteration = 0
for fold_no_outer, (par_index, test_index) in enumerate(OCV.split(X_k), 1):
    print(f'Outer {fold_no_outer}/{K_outer}')
    gen_errors = np.zeros((len(inv_regs),))

    # Select data for outer fold
    X_par = X_k[par_index, :]
    y_par = y[par_index]
    X_test = X_k[test_index, :]
    y_test = y[test_index]

    # Ratio of test
    test_size_ratio = y_test.shape[0] / X_k.shape[0]

    for fold_no_inner, (train_index, val_index) in enumerate(ICV.split(X_par), 1):
        print(f'\tInner {fold_no_inner}/{K_inner}')

        # Select data for inner fold
        X_train = X_par[train_index, :]
        y_train = y_par[train_index]
        X_val = X_par[val_index, :]
        y_val = y_par[val_index]

        # Ratio of validation set length vs. par set length
        val_size_ratio = y_val.shape[0] / y_par.shape[0]

        # Train models using different regularization strengths
        for i, c in enumerate(inv_regs):
            model = lm.logistic.LogisticRegression(C=c)
            model = model.fit(X_train, y_train)

            # Classify
            y_est_val = model.predict(X_val)
            # y_est_fail_prob = model.predict_proba(X_val)[:, 0]

            val_error = sum(np.abs(y_est_val - y_val))
            gen_errors[i] += val_error * val_size_ratio

    # Find the optimal regularization strength. Initialize that model.
    optimal_model_idx = np.argmin(gen_errors)
    opt_inv_regularization_strength = inv_regs[optimal_model_idx]
    opt_regularization_strength = regularization_strengths[optimal_model_idx]
    opt_model = lm.logistic.LogisticRegression(
        C=opt_inv_regularization_strength)

    # Train that model on the par set, then validate against the test data to
    # obtain the test error.
    opt_model = opt_model.fit(X_par, y_par)
    y_est_test = opt_model.predict(X_test)

    test_error = sum(np.abs(y_est_test - y_test))
    test_errors[outer_iteration] = test_error
    test_errors_weighted[outer_iteration] = test_error * test_size_ratio

    test_error_weighted_ratio = test_error / len(y_test)
    test_errors_weighted_ratio[outer_iteration] = test_error_weighted_ratio * \
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
    suptitle('Confusion matrix (Accuracy: {:.4f}%, Error Rate: {:.4f}%)'
             .format(accuracy, error_rate))
    title('LogReg Model (Regularization strength: {})'
          .format(opt_regularization_strength))

    filepath = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(filepath,
                               f'confusion_matrices/logreg/confmat_{outer_iteration + 1}.png')
    savefig(output_path, format='png', dpi=500, bbox_inches='tight')
    # show()
    close()

    optimal_params[opt_inv_regularization_strength] = test_error_weighted_ratio

    outer_iteration += 1

gen_error = np.sum(test_errors_weighted)
gen_error_ratio = np.sum(test_errors_weighted_ratio)

# Print unweighted, absolute test errors
print('Unweighted test errors (Fold <fold_no>: <test_error>):')
for i, t in enumerate(test_errors, 1):
    print(f'Fold {i}: {t}')

print(('Test errors (weighted with respect to size of test set): '
       f'{test_errors_weighted}'))
print(f'Generalization error: {gen_error}')

print(('Test errors (weighted with respect to size of test set): '
       f'{test_errors_weighted_ratio}'))
print(f'Generalization error ratio: {gen_error_ratio * 100}%')


# Print the optimal inverse regularization strength based on minimizing the
# test errors
print('Optimal inverse regularization strength and its test error:')
print(min(optimal_params.items(), key=lambda e: e[1]))
