# exercise 11.1.5
from matplotlib.pyplot import figure, plot, legend, xlabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from toolbox.Tools.toolbox_02450 import clusterplot
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

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'complete'
Metric = 'euclidean'

Z = linkage(X_k, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 3
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
clusterplot(X_k, cls.reshape(cls.shape[0], 1), y=y)

# Display dendrogram
max_display_levels = 3
figure(2, figsize=(10, 4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()
