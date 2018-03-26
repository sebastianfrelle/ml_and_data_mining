import numpy as np
import xlrd


doc = xlrd.open_workbook('student-por.xls').sheet_by_index(0)

attributeNames = doc.row_values(1, 0, 33)

print(attributeNames)