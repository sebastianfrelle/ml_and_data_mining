# exercise 11.1.5
import numpy as np
from matplotlib.pyplot import figure, legend, plot, show, xlabel
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from toolbox.Tools.toolbox_02450 import clusterval
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

# def group_by_cluster(dataset, clusters):
#     if dataset.shape[0] != clusters.shape[0]:
#         print('Expected no. of observations to be identical')
#         return

#     n_clusters = np.unique(clusters).shape[0]
#     clustered = np.array((n_clusters, 1))  # to hold clustered values
#     for cluster_label, obs in zip(clusters, X_k[:]):
#         cluster_array_idx = cluster_label - 1
#         clustered[cluster_array_idx]


## Produce cluster labels ##
# GMM
gmm = GaussianMixture(n_components=3,
                      covariance_type=covar_type,
                      n_init=reps).fit(X_k)
clusters_gmm = gmm.predict(X_k)

# Hierarchical clustering
Method = 'complete'
Metric = 'euclidean'

Z = linkage(X_k, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 3
clusters_hierarchical = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
clusters_hierarchical = clusters_hierarchical - 1

# Compare clusters using adjusted_rand_score
rand_score, _, _ = clusterval(clusters_gmm, clusters_hierarchical)
print(rand_score)

# for t, K in enumerate(KRange):
#     print('Fitting model for K={0}'.format(K))

#     # Fit Gaussian mixture model
#     gmm = GaussianMixture(n_components=K,
#                           covariance_type=covar_type,
#                           n_init=reps).fit(X)

#     # For each crossvalidation fold
#     for train_index, test_index in CV.split(X_k):

#         # extract training and test set for current CV fold
#         X_train = X[train_index]
#         X_test = X[test_index]

#         # Fit Gaussian mixture model to X_train
#         gmm = GaussianMixture(n_components=K,
#                               covariance_type=covar_type,
#                               n_init=reps).fit(X_train)

#         # compute negative log likelihood of X_test
#         CVE[t] += -gmm.score_samples(X_test).sum()


# # # Plot results
# # figure(1)
# # plot(KRange, 2 * CVE, '-ok')
# # legend(['Crossvalidation'])
# # xlabel('K')
# # show()

# # print('Ran Exercise 11.1.5')
