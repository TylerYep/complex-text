import pandas as pd
import numpy as np
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
    for f in feature_lists:
        for c in options_c:
            for wc in options_wc:
                for t in options_tfidf:
                    if 'tfidf' not in f: t = {}
                    if 'word count' not in f: wc = {}
                    alg.run(data, f, c, wc, t)

def bit_twiddle_params(a, data, features):
    best_options = {}
    best_train, best_test = 0.0, 0.0
    for solv in {'lbfgs','adam'}:
        for act in {'identity', 'logistic', 'tanh', 'relu'}:
            for eta in {0.00001, 0.0001, 0.001, 0.01, 0.1, 1}:
                options = {'solver' : solv, 'activation': act, 'learning_rate': 'constant', 'learning_rate_init': eta,
                'n_iter_no_change': 100, 'max_iter': 300}
                a.run(data, features, clf_options=options)
                a.to_csv()
                acc_train = a.results.loc[len(a.results) - 1].train_acc
                acc_test = a.results.loc[len(a.results) - 1].test_acc
                if acc_train > best_train and acc_test > best_test:
                    best_train, best_test = acc_train, acc_test
                    best_options = options
    print(best_options, best_train, best_test)


if __name__ == "__main__":
    a = algs.load_alg('Neural_Net')
    data = util.load_pkl(wb_path)
    bit_twiddle_params(a, data, ['nl'])

    #
    # a = algs.load_alg('Naive_Bayes')
    # features = util.features
    # data = util.load_pkl(wb_path)
    # get_results(a, data, features, nb_opts, wc_opts, tfidf_opts)
    # a.run(data, ['tfidf'],wc_params={'min_df':5}, tfidf_params={'min_df':5} )
    #a.run(data, ['word count'], wc_params={'min_df':4})
    #a.run(data, ['tfidf'], tfidf_params={'min_df':5})
    a.to_csv()
