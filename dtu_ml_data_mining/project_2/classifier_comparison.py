# exercise 6.3.1

import numpy as np
from matplotlib.pyplot import boxplot, figure, show, xlabel, ylabel, savefig
from scipy import stats
from scipy.io import loadmat

import sklearn.linear_model as lm
from sklearn import model_selection, tree
from sklearn.neighbors import KNeighborsClassifier

from dtu_ml_data_mining.project_2.project_2 import *

## Cross-validation ##
# Define models for splitting outer and inner folds
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True)

# Initialize variables
# Models selected are logistic regression and KNN
Error_logreg = np.empty((K, 1))
Error_knn = np.empty((K, 1))

# Reference model errors
Error_ref = np.empty((K, 1))


def reference_prediction(dataset):
    """Predict all observations as class 1."""
    return np.ones((dataset.shape[0],))


## Model parameters ##
knn_params = {'n_neighbors': 14, 'p': 1}
logreg_params = {'C': 0.1}

k = 0
for fold_no, (train_index, test_index) in enumerate(CV.split(X_k, y), 1):
    print(f'CV-fold {fold_no} of {K}')

    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    # Fit and evaluate KNN classifier
    knn_model = KNeighborsClassifier(**knn_params)
    knn_model = knn_model.fit(X_train, y_train)
    y_knn = knn_model.predict(X_test)
    Error_knn[k] = 100 * np.sum(y_knn != y_test, dtype=float) / len(y_test)

    # Fit and evaluate Logistic Regression classifier
    model = lm.logistic.LogisticRegression(**logreg_params)
    model = model.fit(X_train, y_train)
    y_logreg = model.predict(X_test)
    Error_logreg[k] = 100 * \
        np.sum(y_logreg != y_test, dtype=float) / len(y_test)

    # Reference model. Predict all observations as class w/label 1
    y_ref = reference_prediction(X_test)
    Error_ref[k] = 100 * (y_ref != y_test).sum().astype(float) / len(y_test)

    k += 1

# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05.


def compare_classifier_performances(errors_1, errors_2, sig_level=0.05,
                                    names=('Classifier 1', 'Classifier 2')):
    """Compare the performances of two classifiers.

    Takes two vectors of test errors for cross-validation performed on two
    classifiers and attempts to determine whether the performances of the two
    are distinct to a significant level.
    """
    z = (errors_1 - errors_2)
    zb = z.mean()
    nu = K - 1
    sig = (z - zb).std() / np.sqrt(K - 1)

    zL = zb + sig * stats.t.ppf(sig_level / 2, nu)
    zH = zb + sig * stats.t.ppf(1 - sig_level / 2, nu)

    if zL <= 0 <= zH:
        print('Classifiers {} and {} are not significantly different'
              .format(*names))
    else:
        print(('Classifiers are significantly different, and model {} performs '
               'better than {}.').format(*(names if zb > 0 else names[::-1])))


compare_classifier_performances(Error_logreg, Error_knn,
                                names=('LogReg', 'KNN'))
compare_classifier_performances(Error_logreg, Error_ref,
                                names=('LogReg', 'Reference'))
compare_classifier_performances(Error_knn, Error_ref,
                                names=('KNN', 'Reference'))


# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_logreg, Error_knn, Error_ref), axis=1))
xlabel('Logistic Regression vs. KNN vs. most frequent class')
ylabel('Cross-validation error [%]')

savefig('./boxplot_classifier_comparison.eps', format='eps', dpi=1000)

show()

print('Ran Exercise 6.3.1')
