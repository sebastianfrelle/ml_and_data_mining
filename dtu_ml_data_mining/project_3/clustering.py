import os

import numpy as np
from matplotlib.pylab import (
    figure, legend, plot, show, xlabel, ylabel, text,
    imshow, colorbar, xticks, yticks, suptitle, title, savefig, close,
)
from matplotlib.pylab import cm as colormap
from scipy.io import loadmat
from sklearn import model_selection, tree
from sklearn.metrics import confusion_matrix

from project_3 import *

print(y)