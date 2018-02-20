# exercise 2.1.1
import numpy as np
import xlrd

# Load xls sheet with data
doc = xlrd.open_workbook('project/student/student-por.xls').sheet_by_index(0)


# Extract attribute names (1st row, column 4 to 12)
attributeNames = doc.row_values(1, 0, 31)

# Extract class names to python list,
# then encode with integers (dict)

classLabels = doc.col_values(0, 1, 395)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

nominal_idxs = [0, 1, 3, 4, 5, 8, 9, 10, 11, 15,
                16, 17, 18, 19, 20, 21, 22]

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

print(transformed_attributes)
