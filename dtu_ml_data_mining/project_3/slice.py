from dtu_ml_data_mining.project_3.project_3 import *

# Join observations with response variable and shuffle everything together;
# then split and carry out analysis.
y = y.reshape(y.shape[0], 1)
X_k_with_grades = np.append(X_k, y, axis=1)

# Shuffle rows according to seed
np.random.seed(seed=20)
np.random.shuffle(X_k_with_grades)

X_k, y = X_k_with_grades[:, :-1], X_k_with_grades[:, -1]
y = y.A.ravel()

# Selections
cluster_1 = [95, 422, 343, 372, 3]
cluster_2 = [86, 103, 606, 412, 85]

# Shuffle rows according to seed
print('Cluster 1')
print(y[cluster_1])

print('Cluster 2')
print(y[cluster_2])


# # cluster 1

# 95, 422, 343, 372, 3

# # cluster 2

# 86, 103, 606, 412, 85