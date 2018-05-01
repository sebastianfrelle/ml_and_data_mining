import os
import re
import time
from subprocess import run
from sys import platform

import numpy as np
from matplotlib.pyplot import figure, legend, plot, show, xlabel
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.mixture import GaussianMixture

from dtu_ml_data_mining.project_3.project_3 import *
from toolbox.Tools.similarity import *
from toolbox.Tools.writeapriorifile import *

k_encoded_attr_names.append('pass=1/fail=0')

# Standardize and normalize the data #
# Subtract mean value from data
X_k = X_k - np.ones((N, 1)) * X_k.mean(0)

# Divide by standard deviation
X_k /= np.ones((N, 1)) * X_k.std(0)

# Join observations with response variable and shuffle everything together;
# then split and carry out analysis.
y = y.reshape(y.shape[0], 1)
X_k_with_grades = np.append(X_k, y, axis=1)

# Shuffle rows according to seed
np.random.seed(seed=20)
np.random.shuffle(X_k_with_grades)

X_k, y = X_k_with_grades[:, :-1], X_k_with_grades[:, -1]

X_k_bin, y_bin = binarize(X_k, y)

binarized_dataset = np.concatenate((X_k_bin, y_bin), axis=1)
WriteAprioriFile(binarized_dataset, filename="binarized_student_por.txt")

if platform.startswith('linux'): #== "linux" or platform == "linux2":
    ext = ''  # Linux
    dir_sep = '/'
elif platform.startswith('darwin'): #== "darwin":
    ext = 'MAC'  # OS X
    dir_sep = '/'
elif platform.startswith('win'): #== "win32":
    ext = '.exe'  # Windows
    dir_sep = '\\'
else:
    raise NotImplementedError()

filename = './binarized_student_por.txt'
minSup = 80
minConf = 80
maxRule = 4

print(k_encoded_attr_names)

# Run Apriori Algorithm
print('Mining for frequent itemsets by the Apriori algorithm')
status1 = run('.{0}toolbox{0}Tools{0}apriori{1} -f"," -s{2} -v"[Sup. %S]" {3} apriori_temp1.txt'
              .format(dir_sep, ext, minSup, filename ), shell=True)

if status1.returncode != 0:
    print('An error occurred while calling apriori, a likely cause is that minSup was set to high such that no '
          'frequent itemsets were generated or spaces are included in the path to the apriori files.')
    exit()
if minConf > 0:
    print('Mining for associations by the Apriori algorithm')
    status2 = run('.{0}toolbox{0}Tools{0}apriori{1} -tr -f"," -n{2} -c{3} -s{4} -v"[Conf. %C,Sup. %S]" {5} apriori_temp2.txt'
                  .format(dir_sep, ext, maxRule, minConf, minSup, filename ), shell=True)

    if status2.returncode != 0:
        print('An error occurred while calling apriori')
        exit()
print('Apriori analysis done, extracting results')

# Extract information from stored files apriori_temp1.txt and apriori_temp2.txt
f = open('apriori_temp1.txt', 'r')
lines = f.readlines()
f.close()
# Extract Frequent Itemsets
FrequentItemsets = [''] * len(lines)
sup = np.zeros((len(lines), 1))
for i, line in enumerate(lines):
    FrequentItemsets[i] = line[0:-1]
    sup[i] = re.findall(' [-+]?\d*\.\d+|\d+]', line)[0][1:-1]
os.remove('apriori_temp1.txt')

# Read the file
f = open('apriori_temp2.txt', 'r')
lines = f.readlines()
f.close()
# Extract Association rules
AssocRules = [''] * len(lines)
conf = np.zeros((len(lines), 1))
for i, line in enumerate(lines):
    AssocRules[i] = line[0:-1]
    conf[i] = re.findall(' [-+]?\d*\.\d+|\d+,', line)[0][1:-1]
os.remove('apriori_temp2.txt')

# sort (FrequentItemsets by support value, AssocRules by confidence value)
AssocRulesSorted = [AssocRules[item] for item in np.argsort(conf, axis=0).ravel()]
AssocRulesSorted.reverse()
FrequentItemsetsSorted = [FrequentItemsets[item] for item in np.argsort(sup, axis=0).ravel()]
FrequentItemsetsSorted.reverse()

# Print the results
time.sleep(.5)
print('\n')
print('RESULTS:\n')
print('Frequent itemsets:')
for i, item in enumerate(FrequentItemsetsSorted):
    try:
        indices = re.search(r"^[^\[]+", item).group(0).split(' ')
    except ValueError as ex:
        print(item)
        raise ex

    attr_names = [k_encoded_attr_names[int(i)] for i in indices]
    print('Item: {0}, column names: {1}'.format(item, attr_names))
print('\n')
print('Association rules:')
for i, item in enumerate(AssocRulesSorted):
    indices = re.search(r"^[^\[]+", item).group(0).split(' <- ')
    indices = [i.split(' ') for i in indices]
    indices = [i for sublist in indices for i in sublist]
    attr_names = [k_encoded_attr_names[int(i)] for i in indices if len(i) is not 0]
    print('Rule: {0}, column names: {1}'.format(item, attr_names))
