# exercise 11.1.5
from matplotlib.pyplot import figure, plot, legend, xlabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from sklearn import model_selection

from dtu_ml_data_mining.project_3.project_3 import *

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

# Range of K's to try
KRange = range(1, 11)
T = len(KRange)

covar_type = 'full'     # you can try out 'diag' as well
# number of fits with different initalizations, best result will be kept
reps = 3

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10, shuffle=True)

for t, K in enumerate(KRange):
    print('Fitting model for K={0}'.format(K))

    # Fit Gaussian mixture model
    gmm = GaussianMixture(n_components=K,
                          covariance_type=covar_type,
                          n_init=reps).fit(X)

    # For each crossvalidation fold
    for train_index, test_index in CV.split(X_k):

        # extract training and test set for current CV fold
        X_train = X[train_index]
        X_test = X[test_index]

        # Fit Gaussian mixture model to X_train
        gmm = GaussianMixture(n_components=K,
                              covariance_type=covar_type,
                              n_init=reps).fit(X_train)

        # compute negative log likelihood of X_test
        CVE[t] += -gmm.score_samples(X_test).sum()


# Plot results
figure(1)
plot(KRange, 2 * CVE, '-ok')
legend(['Crossvalidation'])
xlabel('K')
show()

print('Ran Exercise 11.1.5')
