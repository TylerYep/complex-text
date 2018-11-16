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
#TODO import all the algs

wb_path = '../preprocess/weebit_features.pkl'
features = ['word count', 'tfidf', 'nl']

class Algorithm:
    def __init__(self, name, model, clf_options={}):
        self.name = name
        self.model = model
        self.clf = self.model(**clf_options)
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
        #TODO implement
        # Returns test_error, precision recall thing
        pass


    def run(self, data, features, clf_options={}, wc_params={}, tfidf_params={}):
        # features \subset ['word count', 'tfidf', 'nl']

        #if something something in features: 
        self.clf = self.model(**clf_options)
        data.get_wc(wc_params)
        data.get_tfidf(tfidf_params)

        features = [data.f_dict[f] for f in features]
        # TODO do a join here.

        train = features[data.train_indices]
        val = features[data.val_indices]

        train_err = self.train(train)
        test_err = self.test(val)

        self.update_results(results, wc_params, tfidf_params)
        self.save()

def get_results(alg, data, features, options_c, options_wc, options_tfidf):
    # for c in options_c: for f in features ...
    #   alg.run(data, f, c, wc, tfdf)
    # TODO
    pass

def compare_models():
    # for name, clf in ...: get_results(alg)
    # TODO 
    pass

if __name__ == "__main__":
    pass

