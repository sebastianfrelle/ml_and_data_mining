import numpy as np
from project_1 import *

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, subplot, hist, ylim, tight_layout, legend

other_idxs = [2, 6, 7, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29]

# Histogram
y = np.array([classDict[value] for value in classLabels])
N = len(y)
M = len(attributeNames)
C = len(classNames)

# Histogram of Nominal attributes
figure(figsize=(8, 6))
u = np.floor(np.sqrt(4))
v = np.ceil(float(4) / u)
j = 1
for i in nominal_idxs:
    subplot(u, v, j)
    if j == 1:
        title('Nominal Attributes')
    j = j + 1
    tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    hist(X[:, i])
    xlabel(attributeNames[i])
    ylabel('Observation')
    ylim(0, 500)


# Histogram of Binary attributes
figure(figsize=(10, 6))
u = np.floor(np.sqrt(13))
v = np.ceil(float(13) / u)
j = 1
for i in binary_idxs:
    subplot(u, v, j)
    if j == 1:
        title('Binary Attributes')
    j = j + 1
    tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    hist(X[:, i])
    xlabel(attributeNames[i])
    ylabel('Observation')
    ylim(0, N)


# Histogram of other attributes
figure(figsize=(10, 6))
u = np.floor(np.sqrt(13))
v = np.ceil(float(13) / u)
j = 1
for i in other_idxs:
    subplot(u, v, j)
    if j == 1:
        title('Remaining Attributes')
    j = j + 1
    tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    hist(X[:, i])
    xlabel(attributeNames[i])
    ylabel('Observation')
    ylim(0, N)


# Histogram of grades
gradeNames = ['G1', 'G2', 'G3']

figure(figsize=(8, 6))
title('Grades')
bins = np.linspace(0, 20, 10)
hist(grades, bins=bins, alpha=0.5, label=gradeNames)
xlabel('Grade')
ylabel('Observation')
legend(loc='upper right')

show()
