import pandas as pd
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
lr_opts = [{'penalty':'l1', 'C':1}, {'penalty':'l1', 'C':0.8}, {'penalty':'l1', 'C':0.6}, {'penalty':'l2', 'C':1}, {'penalty':'l2', 'C':0.8}, {'penalty':'l2', 'C':0.6}]

features = [[x] for x in util.features] + [['word count', 'nl'], ['tfidf', 'nl'], ['word count', 'tfidf'], ['word count', 'tfidf', 'nl']]


#names = ["Nearest_Neighbors", "SVM", "Gaussian_Process",
#         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
#         "Naive_Bayes", "Logistic_Regression", 'Dummy']
if __name__ == "__main__":
    a = algs.load_alg('Logistic_Regression')
    data = util.load_pkl(wb_path)
    get_results(a, data, features, lr_opts, wc_opts, tfidf_opts)
    #get_results(a, data, [['nl']], [{}], [{}], [{}]) Dummy: features ..etc don't matter
    a.to_csv()
