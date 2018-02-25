# exercise 2.1.1
import numpy as np
import xlrd

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd

from categoric2numeric import categoric2numeric

# Load xls sheet with data
doc = xlrd.open_workbook('project/student/student-por.xls').sheet_by_index(0)

# Extract attribute names (1st row, column 4 to 12)
attributeNames = doc.row_values(1, 0, 33)

input_attribute_no = 32

# Extract class names to python list,
# then encode with integers (dict)
# classLabels = doc.col_values(0, 1, 649)
# classNames = sorted(set(classLabels))
# classDict = dict(zip(classNames, range(len(classNames))))

nominal_idxs = [0, 1, 3, 4, 5, 8, 9, 10, 11,
                15, 16, 17, 18, 19, 20, 21, 22]

binary_idxs = [0, 1, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 22]
nominal_idxs = [8, 9, 10, 11]

categorical_idxs = binary_idxs + nominal_idxs

# Read data into matrix-- not applying k coding yet
transformed_attributes = {}
X = np.mat(np.empty((649, input_attribute_no)))
for i in range(input_attribute_no):
    if i in categorical_idxs:
        classLabels = doc.col_values(i, 2, 651)
        classNames = sorted(set(classLabels))
        classDict = dict(zip(classNames, range(len(classNames))))
        transformed_attributes[attributeNames[i]] = classDict
        X[:, i] = np.mat([classDict[value] for value in classLabels]).T
    else:
        X[:, i] = np.mat(doc.col_values(i, 2, 651)).T

# one-out-of-k encoding
X_k = np.mat(np.empty((X.shape[0], 0)))
for i in range(input_attribute_no):
    if i in categorical_idxs:
        # Perform k coding
        # Convert to float data type (dtype=float) to enable division by float
        k_coded = np.mat(categoric2numeric(X[:, i])[0], dtype=np.float)
        k_coded /= np.sqrt(k_coded.shape[1])
        for j in range(k_coded.shape[1]):
            X_k = np.append(X_k, k_coded[:, j], axis=1)
    else:
        X_k = np.append(X_k, X[:, i], axis=1)

N = X_k.shape[0]  # no. of observations

## PCA ##
# Subtract mean value from data
Y = X_k - np.ones((N, 1)) * X_k.mean(0)

# Divide by standard deviation
Y /= np.ones((N, 1)) * Y.std(0)

# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

print(X_k.shape)


def main():
    # Plot variance explained
    figure()
    plot(range(1, len(rho) + 1), rho, 'o-')
    title('Variance explained by principal components: w/ grades')
    xlabel('Principal component')
    ylabel('Variance explained')
    show()

    projected_onto_first = Y * np.mat(V[:, 0]).T
    projected_onto_second = Y * np.mat(V[:, 1]).T

    # print(projected_onto_first.shape)
    # print(projected_onto_second.shape)

    figure()
    plot(projected_onto_first, projected_onto_second, 'o')
    show()


if __name__ == '__main__':
    main()
