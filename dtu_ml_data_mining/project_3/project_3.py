import numpy as np
import xlrd
from categoric2numeric import categoric2numeric

np.set_printoptions(precision=3, linewidth=200, suppress=True)

doc = xlrd.open_workbook('./dtu_ml_data_mining/project_3/student-por.xls').sheet_by_index(0)

attributeNames = doc.row_values(1, 0, 33)

nominal_idxs = [0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22]
binary_idxs = [0, 1, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 22]
nominal_idxs = [8, 9, 10, 11]
categorical_idxs = binary_idxs + nominal_idxs

input_attribute_no = 30
grades = np.mat(np.empty((649, 3)))

for i in range(3):
    grades[:, i] = np.mat(doc.col_values(input_attribute_no + i, 2, 651)).T

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

M = np.append(X, grades, axis=1)

no_categories = {}
# one-out-of-k encoding
X_k = np.mat(np.empty((X.shape[0], 0)))
for i in range(input_attribute_no):
    if i in categorical_idxs:
        # Perform k coding
        # Convert to float data type (dtype=float) to enable division by float
        k_coded = np.mat(categoric2numeric(X[:, i])[0], dtype=np.float)
        k_coded /= np.sqrt(k_coded.shape[1])

        c = k_coded.shape[1]
        no_categories[i] = c  # save no. of categories
        for j in range(c):
            X_k = np.append(X_k, k_coded[:, j], axis=1)
    else:
        X_k = np.append(X_k, X[:, i], axis=1)

k_encoded_attr_names = []
for i in range(input_attribute_no):
    try:
        for j in range(no_categories[i]):
            k_encoded_attr_names.append(f'{attributeNames[i]}_{j}')
    except KeyError:
        k_encoded_attr_names.append(attributeNames[i])


N = X_k.shape[0]  # no. of observations


# Encode grades into classes for pass/fail
y = grades[:, 2].copy()
for i in range(y.shape[0]):  # iterate using no. of columns in grades
    e = y[i, :]
    if e < 10:
        y[i, :] = 0
    else:
        y[i, :] = 1

y = y.A.ravel()