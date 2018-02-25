# exercise 2.1.1
import numpy as np
from project_1 import *

other_idxs = [2, 6, 7, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29]

# Finding binary Statistics
j = 0
X_b = np.mat(np.empty((13, 1)))
for i in binary_idxs:
    X_b[j, 0] = (np.count_nonzero(X[:, i]) / 650) * 100
    j = j + 1
# print(X_b)


# Finding Nominal Statistics
for i in nominal_idxs:
    unique_elements, counts_elements = np.unique(np.asarray(X[:, i]), return_counts=True)
    #print(np.asarray((unique_elements, counts_elements)))

# Finding Decrete Statistics
# for i in other_idxs:
#     print(i, np.mean(X[:, i]))

# for i in other_idxs:
#     print(i, np.var(X[:, i]))

for i in range(3):
    print(i, np.mean(grades[:, i]))


for i in range(3):
    print(i, np.var(grades[:, i]))
