import pandas as pd
import numpy as np
import sklearn
import pickle
#import spacy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV

import os, sys
import util
#from util import ASC

sys.path.append('preprocess')


def load_alg(name):
    path = 'algorithms/results/' + name +'.pkl'
    if os.path.isfile(path):
        return util.load_pkl(path)
    return Algorithm(name, util.model_dict[name])

class Algorithm:
    def __init__(self, name, model):
        """
        Args:
            name: name of model
            model: uninstantiated sklearn model
        """
        self.name = name
        self.model = model
        self.results = pd.DataFrame(columns=util.results_headers)

    def remove_dup(self):
        self.results = self.results.iloc[self.results.astype(str)\
        .drop_duplicates(subset=['model_type', 'features', 'clf_options', 'wc_params', 'tfidf_params']).index]

    def get_fname(self):
        # Returns the file path to save
        fname = self.name
        fname += '.pkl'
        return os.path.join('algorithms/results', fname)

    def save(self):
        util.save_pkl(self.get_fname(), self)

    def predict(self, x):
        return self.clf.predict(x)

    def train(self, x, y):
        self.clf.fit(x, y)
        preds = self.predict(x)

        return util.get_acc(y, preds)

    def eval(self, x, y):
        predictions = self.predict(x)

        test_error = util.get_acc(y, predictions)
        prfs = precision_recall_fscore_support(y, predictions)
        return test_error, prfs

    def run_preds(self, train_data, data, features, clf_options={}, wc_params={}, tfidf_params={}):
        """
        Arguments
            data: DataFeatures object
            features: list of features (\subset ['word count', 'tfidf', 'nl'])
            clf_options: dictionary of sklearn classifier options
            wc_params: dictionary of count vectorizer params
            tfidf_params: dictionary of tfidf vecorizer params
        """
        self.clf = self.model(**clf_options)

        train_x = train_data.get_joint_matrix(features, wc_params, tfidf_params)
        train_y = train_data.labels
        print(train_x.shape)

        val_x = data.get_joint_matrix(features, wc_params, tfidf_params)
        val_y = data.labels
        print(val_x.shape)

        train_acc = self.train(train_x, train_y)
        test_acc, prfs = self.eval(val_x, val_y)

        # Add a row to results
        row = (self.name, str(features), str(clf_options),
                str(wc_params), tfidf_params, train_acc,
                test_acc, prfs)
        self.results.loc[len(self.results)] = row
        self.save()


    def run(self, data, features, clf_options={}, wc_params={}, tfidf_params={}):
        """
        Arguments
            data: DataFeatures object
            features: list of features (\subset ['word count', 'tfidf', 'nl'])
            clf_options: dictionary of sklearn classifier options
            wc_params: dictionary of count vectorizer params
            tfidf_params: dictionary of tfidf vecorizer params
        """
        self.clf = self.model(**clf_options)
        X = data.get_joint_matrix(features, wc_params, tfidf_params)

        train_x = X[data.train_indices]
        train_y = data.labels[data.train_indices]
        val_x = X[data.val_indices]
        val_y = data.labels[data.val_indices]

        train_acc = self.train(train_x, train_y)
        test_acc, prfs = self.eval(val_x, val_y)

        # Add a row to results
        row = (self.name, str(features), str(clf_options),
                str(wc_params), tfidf_params, train_acc,
                test_acc, prfs)
        self.results.loc[len(self.results)] = row
        self.save()

    def search(self, data, param_dist, features, wc_params, tfidf_params):
        X = data.get_joint_matrix(features, wc_params, tfidf_params)
        y = data.labels
        r_search = RandomizedSearchCV(self.model(), param_dist, n_iter=20)
        r_search.fit(X, y)
        self.r = r_search

    def to_csv(self):
        self.results.to_csv(os.path.join('algorithms/results', self.name + '.csv'), index=False)

def combine_csv():
    # Loops through Algorithms and creates a combined csv of results
    files = [os.path.join('algorithms/results', f) for f in os.listdir('algorithms/results') if '.pkl' in f]
    algs = [util.load_pkl(f) for f in files]
    combined_results = pd.concat([a.results for a in algs])
    combined_results.to_csv(os.path.join('algorithms/results', 'combined_results.csv'), index=False)

if __name__ == "__main__":
    combine_csv()
    pass
