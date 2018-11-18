import pandas as pd
import numpy as np
import sklearn
import pickle
import spacy
from sklearn.metrics import precision_recall_fscore_support

import os, sys
sys.path.append('../')
import util
sys.path.append('../preprocess')
<<<<<<< HEAD
from read_data import DataFeatures

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support
from sklearn.dummy import DummyClassifier


names = ["Nearest_Neighbors", "SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "Logistic_Regression", 'Dummy']

models = [KNeighborsClassifier, SVC, GaussianProcessClassifier, DecisionTreeClassifier,
    RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB,
    LogisticRegression, DummyClassifier]

model_dict = dict(zip(names, models))
wb_path = '../preprocess/weebit_features.pkl'
features = ['word count', 'tfidf', 'nl']
results_headers = ['model_type', 'features', 'clf_options', 'wc_params', 'tfidf_params', 'train_acc', 'test_acc', 'prfs']

def get_acc(true, pred):
    return(np.mean(true == pred))
=======
>>>>>>> 07a62c734c2184021ca011e16f4dec530d909a6f

def load_alg(name):
    path = 'results/' + name +'.pkl'
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
        self.results = pd.DataFrame(columns=results_headers)

    def get_fname(self):
        # Returns the file path to save
        fname = self.name
        fname += '.pkl'
        return os.path.join('results', fname)

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
        if 'word count' in features:
            data.get_wc(wc_params)
        if 'tfidf' in features:
            data.get_tfidf(tfidf_params)

        f_dict = data.get_f_dict()
        X = [f_dict[f] for f in features]
        X = np.concatenate(tuple(X), axis=1)

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

    def to_csv(self):
        self.results.to_csv(os.path.join('results', self.name + '.csv'), index=False)

def combine_csv():
    # Loops through Algorithms and creates a combined csv of results

    files = [os.path.join('results', f) for f in os.listdir('results') if '.pkl' in f]
    algs = [util.load_pkl(f) for f in files]
    combined_results = pd.concat([a.results for a in algs])
    combined_results.to_csv(os.path.join('results', 'combined_results.csv'), index=False)

if __name__ == "__main__":
    # Test with runner
    pass




