import pandas as pd
import scipy.stats
import numpy as np
import util
import itertools
from tqdm import tqdm

import os, sys
sys.path.append('preprocess')
sys.path.append('algorithms')
import algs
from algs import Algorithm
from read_data import DataFeatures


#Harry
"""
LogisticRegression
DummyClassifier
SVM
Naive_Bayes
GaussianProccess
"""
#Tyler
"""
KNeighbours
DecisionTreeClassifier
RandomForestClassifier
MLP
AdaBoost
"""

wb_path = 'preprocess/weebit_features.pkl'


def get_results(alg: Algorithm, data, feature_lists, options_c, options_wc, options_tfidf):
    prod = itertools.product(feature_lists, options_c, options_wc, options_tfidf)
    for f, c, wc, t in tqdm(list(prod)):
        if not('tfidf' in f or t == {}): continue # Ignore the case where theres not tfidf in features but tfdif is not {}
        if not('word count' in f or wc == {}): continue
        alg.run(data, f, c, wc, t)

wc_opts = [{}, {'min_df':5}, {'max_df':0.8}, {'min_df':5, 'max_df':0.8}, {'min_df':5, 'max_df':0.8, 'binary':True}]
tfidf_opts = [{}, {'min_df':5}, {'max_df':0.8}, {'min_df':5, 'max_df':0.8}]
lr_opts = [{'penalty':'l2', 'C':10**i} for i in range(-3, 3)] + [{'penalty':'l1', 'C':10**i} for i in range(-3, 3)]
svm_opts = [ {'kernel':'rbf', 'C':1}, {'kernel':'rbf', 'C':0.8}, {'kernel':'rbf', 'C':0.6}, {'kernel':'linear', 'C':1}, {'kernel':'linear', 'C':0.8}, {'kernel':'linear', 'C':0.6} ]

features = [[x] for x in util.features] + [['word count', 'nl'], ['tfidf', 'nl']]#, ['word count', 'tfidf'], ['word count', 'tfidf', 'nl']]


#names = ["Nearest_Neighbors", "SVM", "Gaussian_Process",
#         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
#         "Naive_Bayes", "Logistic_Regression", 'Dummy']
if __name__ == "__main__":
    a = algs.load_alg('Logistic_Regression')
    data = util.load_pkl(wb_path)
    #param_dist = {'penalty':['l1', 'l2'], 'C':[10**i for i in range(-5, 5)]}
    #a.search(data, param_dist, ['word count', 'nl'], {'min_df':5, 'max_df':0.8}, {})
    #a.run(data, ['nl'])
    get_results(a, data,  [['word count','nl']],lr_opts, [{'min_df':5, 'max_df':0.8}], [{}])
    a.to_csv()
