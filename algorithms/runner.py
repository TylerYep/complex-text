import pandas as pd
import numpy as np
import util

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
    for f in feature_lists:
        for c in options_c:
            for wc in options_wc:
                for t in options_tfidf:
                    alg.run(data, f, c, wc, t)

if __name__ == "__main__":
    a = algs.load_alg('Naive_Bayes')
    data = util.load_pkl(wb_path)

    a.run(data, ['tfidf'],wc_params={'min_df':5}, tfidf_params={'min_df':5} )
    #a.run(data, ['word count'], wc_params={'min_df':4})
    #a.run(data, ['tfidf'], tfidf_params={'min_df':5})
    a.to_csv()
