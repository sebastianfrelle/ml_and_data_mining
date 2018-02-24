import numpy as np
from matplotlib.pyplot import (
    figure, plot, title, xlabel, 
    ylabel, show, legend, subplot, savefig
)

from project_1 import *

np.set_printoptions(precision=3, linewidth=200, suppress=True)

M = np.append(X, grades, axis=1)

corrmat = np.corrcoef(M, rowvar=False)

print(corrmat[:, corrmat.shape[1] - 1])

figure(figsize=(20, 5))
for f, i in enumerate(range(M.shape[1] - 3, M.shape[1])):
    subplot(1, 3, f + 1)
    plot(M[:, i], M[:, M.shape[1] - 1], 'o')
    xlabel('G3')
    ylabel(attributeNames[i])

savefig('./grades_plot.eps', format='eps', dpi=1000)
show()
