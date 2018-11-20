import pickle
import numpy as np
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
from sklearn.dummy import DummyClassifier

class ASC:
    def __init__(self):
        pass

    def fit(self, X, y):
        avgs = []
        X = X[:, -1].flatten()
        for i in [2, 3, 4]:
            avgs.append(np.mean(X[y == i]))
        self.avgs = avgs

    def predict(self, X):
        X = X[:, -1].flatten()
        diffs = np.array([[np.abs(x-a) for a in self.avgs] for x in X])
        return np.argmin(diffs, axis=1) + 2

features = ['word count', 'tfidf', 'nl']

results_headers = ['model_type', 'features', 'clf_options', 'wc_params',
        'tfidf_params', 'train_acc', 'test_acc', 'prfs']

names = ["k_Nearest_Neighbors", "SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "Logistic_Regression", 'Dummy', 'ASC']

models = [KNeighborsClassifier, SVC, GaussianProcessClassifier, DecisionTreeClassifier,
    RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB,
    LogisticRegression, DummyClassifier, ASC]

model_dict = dict(zip(names, models))

def get_acc(true, pred):
    return(np.mean(true == pred))

def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


