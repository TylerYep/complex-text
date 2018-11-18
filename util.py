import pickle

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

features = ['word count', 'tfidf', 'nl']

results_headers = ['model_type', 'features', 'clf_options', 'wc_params',
        'tfidf_params', 'train_acc', 'test_acc', 'prfs']

names = ["Nearest_Neighbors", "SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "Logistic_Regression", 'Dummy']

models = [KNeighborsClassifier, SVC, GaussianProcessClassifier, DecisionTreeClassifier,
    RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB,
    LogisticRegression, DummyClassifier]

model_dict = dict(zip(names, models))

def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
