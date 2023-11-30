from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

class Models:

    def __init__(self, name: str, model):
        self.name = name
        self.model = model
        self.precision = 0
        self.recall = 0
        self.f1 = 0

def getModels():
    models = []
    random_forest_model = RandomForestClassifier(n_jobs=-1)
    models.append(Models('Random Forest', random_forest_model))

    logistic_regression_model = LogisticRegression(max_iter=2000)
    models.append(Models('Logistic Regression', logistic_regression_model))

    bernoulli_naive_bayes_model = BernoulliNB()
    models.append(Models('Bernoulli Naive Bayes', bernoulli_naive_bayes_model))

    gaussian_naive_bayes_model = GaussianNB()
    models.append(Models('Gaussian Naive Bayes', gaussian_naive_bayes_model))

    decision_tree_model = DecisionTreeClassifier()
    models.append(Models('Decision Tree',decision_tree_model))

    xgboost_model = XGBClassifier()
    models.append(Models('Gradient Boost',xgboost_model))

    mlp_class_model = MLPClassifier(hidden_layer_sizes=20, max_iter=2000, activation='relu', solver='lbfgs')
    models.append(Models('Multi-Layer Perceptron', mlp_class_model))

    return models

