import numpy as np
from matplotlib.pyplot import (
    figure, plot, title, xlabel,
    ylabel, show, legend, subplot, savefig, text,
)

from project_1 import *

M = np.append(X, grades, axis=1)

np.set_printoptions(precision=3, linewidth=200, suppress=True)

corrmat = np.corrcoef(M, rowvar=False)

print(corrmat)
np.savetxt('corrmat.txt', corrmat, delimiter=',', fmt='%.2f')

figure(figsize=(20, 5))
for f, i in enumerate(range(M.shape[1] - 3, M.shape[1])):
    subplot(1, 3, f + 1)
    plot(M[:, i], M[:, M.shape[1] - 1], 'o')
    text(3 + 20 // i, 15, 'hej')
    xlabel('G3')
    ylabel(attributeNames[i])

print(attributeNames)

savefig('./grades_plot.eps', format='eps', dpi=1000)
show()
