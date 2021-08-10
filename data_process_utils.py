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



class data_visualiser():
    """
    Data Visualiser class
    """
    def __init__(self, num_components = 2, colormap_name = "tab10"):
        self.num_dimensions = num_components
        self.colors = get_cmap(colormap_name).colors


    def project(self, data, **kwargs):
        return TSNE(self.num_dimensions, **kwargs).fit_transform(data)

    def plot(self, data, ax, **kwargs):
        try:
            assert self._is_2D(data)

        except Exception as e:
            print("data not 2-dimensional")
            data = self.project(data)

        # Access data
        x, y = data[:, 0], data[:, 1]

        # Plot
        ax.scatter(x, y, **kwargs, label = "data")

        return ax

    def plot_per_class(self, X,y, ax,  **kwargs):
        """
        Scatter plot with different colors for each class
        """
        try:
            assert self._is_2D(X)
            assert self._samples_equal(X, y)

        except Exception as e:
            print("X input not 2-dimensional or X and y not of the same size.")
            X = self.project(X)

        # Compute num classes
        num_classes_ = self._num_classes(y)

        # plot
        for class_ in range(1, num_classes_ + 1):

            # Access data for the class
            X_class_ = X[self._get_class_ids(X, y, class_), :]
            x_scatter, y_scatter = X_class_[:, 0], X_class_[:, 1]

            ax.scatter(x_scatter, y_scatter, color = self.colors[class_], label = "Class {}".format(class_))

        return ax

    def plot_pre_post_sampling(self, data_og, data_samp, ax, **kwargs):
        """
        Scatter plot comparing data before and after sampling
        """
        # Obtain corresponding data
        X_og, y_og = data_og
        X_samp, y_samp = data_samp


        # Make plots
        ax[0] = self.plot_per_class(X_og, y_og, ax = ax[0])
        ax[1] = self.plot_per_class(X_samp, y_samp, ax = ax[1])

        return ax

    def _is_2D(self, data):
        return len(data.shape) == 2

    def _samples_equal(self, array_1, array_2):
        return array_1.shape[0] == array_2.shape[0]

    def _num_classes(self, y):
        return np.unique(y).size

    def _get_class_ids(self, X, y, class_id):
        return (y==class_id)





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



