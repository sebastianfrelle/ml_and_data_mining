import numpy as np
from sklearn import model_selection
import sklearn.linear_model as lm

from dtu_ml_data_mining.project_2.project_2 import *

## Cross validation ##

# Define models for splitting outer and inner folds
no_models = 5
K_outer = 5
K_inner = 10
OCV = model_selection.KFold(n_splits=K_outer, shuffle=True)
ICV = model_selection.KFold(n_splits=K_inner, shuffle=True)


# Train model with different regularization strengths. sklearn method takes inverse regularization
# strength, so produce an inverse list.
regularization_strengths = [10 ** e for e in range(-2, 3)]
inv_regs = [1 / e for e in regularization_strengths]

# Store generalization
gen_errors = np.zeros((len(inv_regs),))
test_errors = np.zeros((K_outer,))

outer_iteration = 0
for fold_no_outer, (par_index, test_index) in enumerate(OCV.split(X_k), 1):
    print(f'Outer {fold_no_outer}/{K_outer}')

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

            misclass_rate_val = sum(np.abs(y_est_val - y_val)) / float(len(y_est_val))

            gen_errors[i] += misclass_rate_val * val_size_ratio

    # Find the optimal regularization strength. Build that model
    optimal_model_idx = np.argmin(gen_errors)
    opt_inv_regularization_strength = inv_regs[optimal_model_idx]
    opt_model = lm.logistic.LogisticRegression(C=opt_inv_regularization_strength)

    # Train on the par set, then validate against the test data to obtain the 
    # test error.
    opt_model = opt_model.fit(X_par, y_par)
    y_est_test = opt_model.predict(X_test)
    misclass_rate_test = sum(np.abs(y_est_test - y_test)) / float(len(y_est_test))

    test_errors[outer_iteration] = misclass_rate_test * test_size_ratio
    outer_iteration += 1

print(f'Test errors (weighted with respect to size of test set): {test_errors}')
gen_error = np.sum(test_errors)
print(f'Generalization error: {gen_error}')
