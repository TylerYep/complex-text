import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pickle
#import spacy
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
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

    def conf_matrix(self, x, y):
        def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=0)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
        predictions = self.predict(x)
        cnf_matrix = confusion_matrix(y, predictions)
        np.set_printoptions(precision=2)
        plot_confusion_matrix(cnf_matrix, classes=['level '+ str(i) for i in [2, 3, 4]], normalize=True, title='Confusion Matrix for Logistic Regression')
        #plt.imshow(conf)
        plt.show()

        
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
        self.conf_matrix(val_x, val_y)
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
