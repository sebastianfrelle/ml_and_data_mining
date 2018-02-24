# exercise 2.1.1
import numpy as np
import xlrd

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show
from scipy.linalg import svd

from categoric2numeric import categoric2numeric

# Load xls sheet with data
doc = xlrd.open_workbook('project/student/student-por.xls').sheet_by_index(0)

# Extract attribute names (1st row, column 4 to 12)
attributeNames = doc.row_values(1, 0, 31)

# Extract class names to python list,
# then encode with integers (dict)

classLabels = doc.col_values(0, 1, 395)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

nominal_idxs = [0, 1, 3, 4, 5, 8, 9, 10, 11,
                15, 16, 17, 18, 19, 20, 21, 22]

transformed_attributes = {}

X = np.mat(np.empty((394, 32)))
for i in range(32):
    if i in nominal_idxs:
        classLabels = doc.col_values(i, 2, 396)
        classNames = sorted(set(classLabels))
        classDict = dict(zip(classNames, range(len(classNames))))
        transformed_attributes[attributeNames[i]] = classDict
        X[:, i] = np.mat([classDict[value] for value in classLabels]).T
    else:
        X[:, i] = np.mat(doc.col_values(i, 2, 396)).T

# one-out-of-k encoding
X_k = np.mat(np.empty(X.shape))
for i in range(32):
    if i in nominal_idxs:
        X_k[:, i] = categoric2numeric(X[:, i])[0]

# N = X.shape[0]
# C = len(classNames)

# ## PCA ##
# # Subtract mean value from data
# Y = X - np.ones((N, 1)) * X.mean(0)

# # PCA by computing SVD of Y
# U, S, V = svd(Y, full_matrices=False)
# V = V.T

# # Compute variance explained by principal components
# rho = (S * S) / (S * S).sum()
# Z = Y * V

# # Indices of the principal components to be plotted
# i = 0
# j = 1

# # Plot PCA of the data
# f = figure()
# title('NanoNose data: PCA')
# #Z = array(Z)
# for c in range(C):
#     # select indices belonging to class c:
#     class_mask = y.A.ravel() == c
#     plot(Z[class_mask, i], Z[class_mask, j], 'o')

# legend(classNames)
# xlabel('PC{0}'.format(i + 1))
# ylabel('PC{0}'.format(j + 1))

# # Plot variance explained
# figure()
# plot(range(1, len(rho) + 1), rho, 'o-')
# title('Variance explained by principal components')
# xlabel('Principal component')
# ylabel('Variance explained')
# show()

# print('Ran Exercise 2.1.3')
