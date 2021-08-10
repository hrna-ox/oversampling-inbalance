import numpy as np

# Import dimensionality reduction
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Import samplers
from imblearn.over_sampling import SMOTE, SMOTEN, SVMSMOTE, ADASYN

# Import models
from sklearn.svm import SVC as SVM
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.neural_network import MLPClassifier as MLP

# Import evaluation metrics
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score


class type_detector():
    """
    Type detector for processing
    """
    def __init__(self, binary_vars = [], cat_vars = [], cont_vars = []):
        self.binary_vars = binary_vars
        self.cat_vars    = cat_vars
        self.cont_vars   = cont_vars



class data_processor():
    """
    Data Processor class.
    """
    def __init__(self, mu = None, sigma = None):
        self.mu    = mu
        self.sigma = sigma

    def fit(self, data):
        mean_ = np.mean(data, axis = 0, keepdims = True)
        std_  = np.std(data, axis = 0, keepdims = True)

        self.mu = mean_
        self.sigma = std_

        return None

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


    def transform(self, data):
        return np.divide(data - self.mu, self.sigma)


    def fit_train_test(self, data_train, data_test):
        train_output = self.fit_transform(data_train)
        test_output  = self.transform(data_test)

        return train_output, test_output

class sampler():
    def __init__(self, method, **kwargs):
        if method == 'SMOTE':
            self.sampler = SMOTE(**kwargs)

        elif method == 'SMOTEN':
            self.sampler = SMOTEN(**kwargs)

        elif method == 'SVMSMOTE':
            self.sampler = SVMSMOTE(**kwargs)

        elif method == 'ADASYN':
            self.sampler = ADASYN(**kwargs)

        else:
            print("Sampling method provided not valid!!")

    def fit(self, X, y):
        return self.sampler.fit(X, y)

    def fit_resample(self, X, y):
        return self.sampler.fit_resample(X, y)

    def get_params(self, **kwargs):
        return self.sampler.get_params(**kwargs)



class model():
    def __init__(self, model, **kwargs):
        if model == "SVM":
            self.model = SVM(**kwargs)

        elif model == "KNN":
            self.model = KNN(**kwargs)

        elif model == "Tree":
            self.model = Tree(**kwargs)

        elif model == "Logistic":
            self.model = LogReg(**kwargs)

        elif model == "MLP":
            self.model = MLP(**kwargs)

        else:
            print("Model name not accepted. Valid input names are \n"
                  "SVM, KNN, Tree, Logistic, MLP")

    def do_something_else(self, **kwargs):
        print("Do something else! Feature to be added.")
        return None


    def fit_predict(self, X, y, X_test, **kwargs):
        # Fit data
        self.model.fit(X, y)

        return self.model.predict(X_test)      # predict output


class evaluator():
    """
    Evaluator class.
    Makes prediction
    """
    def __init__(self, **kwargs):
        self.bac = balanced_accuracy_score
        self.roc = roc_auc_score
        self.cm  = confusion_matrix
        self.f1  = f1_score
        self.rec = recall_score
        self.pre = precision_score
        self.results = []

    def evaluate(self, y_true, y_pred):
        output_dic = {
            "balanced_accuracy": self.bac(y_true, y_pred),
            "f1": self.f1(y_true, y_pred, average = "macro"),
            "recall": self.rec(y_true, y_pred, average = "macro"),
            "precision": self.pre(y_true, y_pred, average = "macro")
            # "roc_auc_score": self.roc(y_true, y_pred, average = "macro", multi_class = "ovr"),
            #"confusion matrix": self.cm(y_true, y_pred)
        }

        return output_dic



