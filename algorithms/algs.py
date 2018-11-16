import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn 
import pickle
import os, sys
import spacy
sys.path.append('../')
sys.path.append('../preprocess')
import util
from read_data import DataFeatures
from collections import Counter, defaultdict
wb_path = '../preprocess/weebit_features.pkl'

class Algorithm:
    def __init__(self, name, clf, clf_options={}):
        self.name = name
        self.clf = clf(clf_options**)
        self.results = {} # i.e. {'wc, min_df=5': results} 

    def get_fname(self):
        fname = name + str(clf_options)
        fname += '.pkl'

    def update_results(self, results, wc_params, tfidf_params):
        # TODO create a id string from clf_options, wc_params, and tfidf_params
        # set self.results
        pass

    def predict(self, x):
        return self.clf.predict(x)

    def train(self, data):
        # Returns train_error if possible?
        self.clf.fit(data)
        return #TODO

    def test(self, data):
        # Returns test_error, precision recall thing
        pass


def run(name, clf, data, features, clf_options={}, wc_params={}, tfidf_params={}):
    #TODO If name already exists, load it
    alg = Algorithm(name, clf, clf_options)

    #if something something in features: 
    data.get_wc(wc_params)
    data.get_tfidf(tfidf_params)

    features = [data.f_dict[f] for f in features]
    # TODO do a join here.

    train = features[data.train_indices]
    val = features[data.val_indices]

    train_err = alg.train(train)
    test_err = alg.test(val)

    alg.update_results(results, wc_params, tfidf_params)
    alg.save()


if __name__ == "__main__":
    wb = util.load_pkl(wb_path)
    wb.get_indices()


