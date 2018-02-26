import numpy as np
from matplotlib.pyplot import (
    figure, plot, title, xlabel,
    ylabel, show, legend, subplot, savefig, text, scatter,
)

from project_1 import *

M = np.append(X, grades, axis=1)

np.set_printoptions(precision=3, linewidth=200, suppress=True)

corrmat = np.corrcoef(M, rowvar=False)

# print(corrmat)
# np.savetxt('corrmat.txt', corrmat, delimiter=',', fmt='%.2f')

# fedu->medu
# walc->dalc
# mjob->medu is much higher than fjob->fedu
# medu (0.24), fedu (0.21), failures (-0.39), higher (0.33)

plots = (
    (('medu', 6), ('fedu', 7)),  # medu->fedu
    (('dalc', 26), ('walc', 27)),  # walc->dalc
)

figure(figsize=(20, 5))
for i, attr in enumerate(plots):
    subplot(1, 2, i + 1)
    # text(3 + 20 // i, 15, f'corr={correlations[f]}')
    plot(M[:, attr[0][1]], M[:, attr[1][1]], 'o', markersize=3)
    xlabel(attr[0][0])
    ylabel(attr[1][0])

savefig('./attibutes_plot.eps', format='eps', dpi=1000)
show()
