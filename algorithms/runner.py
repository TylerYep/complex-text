import algs

names = ["Nearest_Neighbors", "SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "Logistic_Regression", 'Dummy']

models = [KNeighborsClassifier, SVC, GaussianProcessClassifier, DecisionTreeClassifier,
    RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB,
    LogisticRegression, DummyClassifier]

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

x = util.load_pkl('weebit_features.pkl')
print(x.nl_matrix)

def get_results(alg: algs.Algorithm, data, feature_lists, options_c, options_wc, options_tfidf):
    for f in feature_lists:
        for c in options_c:
            for wc in options_wc:
                for t in options tfidf:
                    alg.run(data, f, c, wc, t)
