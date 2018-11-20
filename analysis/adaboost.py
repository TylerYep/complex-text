import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import util

import os, sys
sys.path.append('preprocess')
sys.path.append('algorithms')
import algs
from algs import Algorithm
from read_data import DataFeatures
"""
names = ["k_Nearest_Neighbors", "SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "Logistic_Regression", 'Dummy']
"""
N = 10
alg = util.load_pkl('algorithms/results/AdaBoost.pkl')
X = [eval(alg.results[:N]['clf_options'].values[i])['learning_rate'] for i in range(len(alg.results[:N]))]
Y = list(alg.results[:N]['test_acc'].values)
x = [eval(alg.results[N:2*N]['clf_options'].values[i])['learning_rate'] for i in range(len(alg.results[N:2*N]))]
y = list(alg.results[N:2*N]['test_acc'].values)
S = [eval(alg.results[2*N:3*N]['clf_options'].values[i])['learning_rate'] for i in range(len(alg.results[2*N:3*N]))]
T = list(alg.results[2*N:3*N]['test_acc'].values)
s = [eval(alg.results[3*N:]['clf_options'].values[i])['learning_rate'] for i in range(len(alg.results[3*N:]))]
t = list(alg.results[3*N:]['test_acc'].values)

plt.plot(s, t, label='1 estimator')
plt.plot(X, Y, label='50 estimators')
plt.plot(x, y, label='100 estimators')
plt.plot(S, T, label='150 estimators')
plt.xlabel('Learning Rate')
plt.ylabel('Test Accuracy')
plt.legend()
plt.show()
