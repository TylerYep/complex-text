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
"""
names = ["k_Nearest_Neighbors", "SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "Logistic_Regression", 'Dummy']
"""

# Harry
"""
Logistic_Regression - 0.71, 0.71
Dummy - 0.34, 0.32
SVM - 0.81, 0.71
Naive_Bayes - 0.53, 0.46
GaussianProccess - 1, 0.46
"""
# Tyler
"""
k_Nearest_Neighbors - 0.76, 0.62
Decision_Tree - 1.00, 0.67
Random_Forest - 0.99, 0.69
Neural_Net - 0.73, 0.68
AdaBoost - 0.74, 0.69
"""

wb_path = 'preprocess/weebit_features.pkl'


def get_results(alg: Algorithm, data, feature_lists, options_c, options_wc, options_tfidf):
    prod = itertools.product(feature_lists, options_c, options_wc, options_tfidf)
    for f, c, wc, t in tqdm(list(prod)):
        if not('tfidf' in f or t == {}): continue # Ignore the case where theres not tfidf in features but tfdif is not {}
        if not('word count' in f or wc == {}): continue
        alg.run(data, f, c, wc, t)

features = [[x] for x in util.features] + [['word count', 'nl'], ['tfidf', 'nl']]#, ['word count', 'tfidf'], ['word count', 'tfidf', 'nl']]
wc_opts = [{}, {'min_df':5}, {'max_df':0.8}, {'min_df':5, 'max_df':0.8}, {'min_df':5, 'max_df':0.8, 'binary':True}]
tfidf_opts = [{}, {'min_df':5}, {'max_df':0.8}, {'min_df':5, 'max_df':0.8}]
lr_opts = [{'penalty':'l2', 'C':10**i} for i in range(-3, 3)] + [{'penalty':'l1', 'C':10**i} for i in range(-3, 3)]
svm_opts = [ {'kernel':'rbf', 'C':1}, {'kernel':'rbf', 'C':0.8}, {'kernel':'rbf', 'C':0.6}, {'kernel':'linear', 'C':1}, {'kernel':'linear', 'C':0.8}, {'kernel':'linear', 'C':0.6} ]

def bit_twiddle_params(a, data, features):
    # best_options = {}
    # best_train, best_test = 0.0, 0.0
    for es in {50, 100, 150, 200}:
        for eta in tqdm(range(1, 11)):
            options = {'n_estimators': es, 'learning_rate': eta * 0.1}
            a.run(data, features, clf_options=options, wc_params={'min_df':5, 'max_df':0.8, 'binary':True})
            a.to_csv()
            #     acc_train = a.results.loc[len(a.results) - 1].train_acc
            #     acc_test = a.results.loc[len(a.results) - 1].test_acc
            #     if acc_train > best_train and acc_test > best_test:
            #         best_train, best_test = acc_train, acc_test
            #         best_options = options
            # print(best_options, best_train, best_test)


#names = ["Nearest_Neighbors", "SVM", "Gaussian_Process",
#         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
#         "Naive_Bayes", "Logistic_Regression", 'Dummy']
if __name__ == "__main__":
    data = util.load_pkl(wb_path)
    a= algs.load_alg('Logistic_Regression')
    a.run(data, ['word count', 'nl'],{'penalty': 'l1', 'C': 0.1},{'min_df': 5, 'max_df': 0.8})

    #bit_twiddle_params(a, data, ['word count', 'nl'])
    ##param_dist = {'penalty':['l1', 'l2'], 'C':[10**i for i in range(-5, 5)]}
    ##a.search(data, param_dist, ['word count', 'nl'], {'min_df':5, 'max_df':0.8}, {})
    ##a.run(data, ['nl'])
    #get_results(a, data,  [['word count','nl']],lr_opts, [{'min_df':5, 'max_df':0.8}], [{}])
    #a.to_csv()
