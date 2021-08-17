import numpy as np
import pandas as pd

# Import samplers
from imblearn.over_sampling import SMOTE, SMOTEN, SVMSMOTE, ADASYN
from sklearn.model_selection import train_test_split

# Import models
from sklearn.svm import SVC as SVM
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.neural_network import MLPClassifier as MLP

# Import evaluation metrics
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score

import itertools

key_vars = ["data_key", "model", "method", "sampling_strategy"]

class type_detector():
    """
    Type detector for processing
    """
    def __init__(self, binary_vars = None, cat_vars = None, cont_vars = None):
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

        # Replace 0 variance by 1
        std_[std_ == 0] = 1.0

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
            valid_params = ["random_state", "k_neighbors", "n_jobs", "sampling_strategy"]
            kwargs = {key:value for key, value in kwargs.items() if key in valid_params}
            self.sampler = SMOTE(**kwargs)

        elif method == 'SMOTEN':
            valid_params = ["random_state", "k_neighbors", "n_jobs", "sampling_strategy"]
            kwargs = {key:value for key, value in kwargs.items() if key in valid_params}
            self.sampler = SMOTEN(**kwargs)

        elif method == 'SVMSMOTE':
            valid_params = ["random_state", "k_neighbors", "n_jobs", "sampling_strategy", "m_neighbors",
                          "svm_estimator", "out_step"]
            kwargs = {key:value for key, value in kwargs.items() if key in valid_params}
            self.sampler = SVMSMOTE(**kwargs)

        elif method == 'ADASYN':
            valid_params = ["random_state", "n_neighbors", "n_jobs", "sampling_strategy"]
            kwargs = {key:value for key, value in kwargs.items() if key in valid_params}
            self.sampler = ADASYN(**kwargs)

        else:
            self.sampler = None
            print("Sampling method provided not valid!!")
            raise ValueError

    def fit(self, X, y):
        return self.sampler.fit(X, y)

    def fit_resample(self, X, y):
        return self.sampler.fit_resample(X, y)

    def get_params(self, **kwargs):
        return self.sampler.get_params(**kwargs)



class model_class():
    def __init__(self, model, **kwargs):
        if model == "SVM":
            valid_params = ["C", "kernel", "degree", "gamma", "coef0", "shrinking", "probability", "tol",
                            "cache_size", "class_weight", "verbose", "max_iter", "random_state"]
            kwargs = {key:value for key, value in kwargs.items() if key in valid_params}
            self.model = SVM(**kwargs)

        elif model == "KNN":
            valid_params = ["n_neighbors", "weights", "algorithm", "leaf_size", "p", "metric", "metric_params", "n_jobs"]
            kwargs = {key:value for key, value in kwargs.items() if key in valid_params}
            self.model = KNN(**kwargs)

        elif model == "RandomForest":
            valid_params = ["n_estimators", "criterion", "max_depth", "min_samples_split", "min_samples_leaf",
                            "min_weight_fraction_leaf", "max_features", "max_leaf_nodes", "min_impurity_decrease",
                            "min_impurity_split", "bootstrap", "n_jobs", "random_state", "verbose"]
            kwargs = {key:value for key, value in kwargs.items() if key in valid_params}
            self.model = RandomForest(**kwargs)

        elif model == "Logistic":
            valid_params = ["penalty", "dual", "tol", "C", "fit_intercept", "intercept_scaling",
                            "random_state", "solver", "max_iter", "verbose", "n_jobs"]
            kwargs = {key:value for key, value in kwargs.items() if key in valid_params}
            self.model = LogReg(**kwargs)

        elif model == "MLP":
            valid_params = ["hidden_layer_sizes", "activation", "solver", "alpha", "batch_size",
                            "learning_rate", "learning_rate_init", "power_t", "max_iter", "shuffle",
                            "random_state", "tol", "verbose", "momentum"]
            kwargs = {key:value for key, value in kwargs.items() if key in valid_params}
            self.model = MLP(**kwargs)

        else:
            print("Model name not accepted. Valid input names are \n"
                  "SVM, KNN, Tree, Logistic, MLP")
            raise ValueError

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
        self.f1  = f1_score
        self.rec = recall_score
        self.pre = precision_score
        self.results = None

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

    def add_results(self, results):
        self.results = results

    def _sampled_vs_original_improvement(self):
        "Compute improvement through sampling vs original modelling"
        og_results_   = self.results[self.results["sampled"] == False]
        samp_results_ = self.results[self.results["sampled"] == True]

        key_vars = global key_vars
        og_results_.sort_values(key_vars, inplace = True)
        samp_results_.sort_values(key_vars, inplace = True)

        # Check first columns make sense
        assert np.sum(og_results_.iloc[:, :4] != samp_results_.iloc[:, :4]) == 0

        cols = og_results_.columns.tolist()
        metric_cols = cols[cols.index("sampled"):]



class Looper():
    def __init__(self, list_of_methods, list_of_strategies, list_of_models,
                 list_of_metrics = ["balanced_accuracy", "f1", "precision", "recall"], test_size = 0.5, seed = 4537):
        self.methods    = list_of_methods
        self.strategies = list_of_strategies
        self.models     = list_of_models
        self.metrics    = list_of_metrics
        self.test_size  = test_size
        self.seed       = seed

        # Build result output
        key_vars = global key_vars
        self.results    = pd.DataFrame(data = np.nan, index = [], columns =  key_vars + ["sampled"] + self.metrics)
        self.current_iteration = None

    def loop_through(self, dataset_dic):
        "Apply pipeline to dictionary of datasets, and compute metrics"
        data_names = list(dataset_dic.keys())

        for data_name, method, strategy, model in itertools.product(data_names, self.methods, self.strategies, self.models):
            # Update current attribute
            self.current_iteration = data_name, method, strategy, model

            # Get data and separate into train test splits
            X, y = dataset_dic[data_name]["X"], dataset_dic[data_name]["y"]
            self.run_pipeline(X, y, method, strategy, model)

        return None

    def run_pipeline(self, X, y, method, strategy, model, **kwargs):
        "Run pipeline to a particular instance of X, y data and method, strategy, model considerations."
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size,
                                                            random_state = self.seed, shuffle = True, stratify = y)

        sampler_ = self.get_sampler(method, **kwargs)

        try:
            X_samp, y_samp = sampler.fit_resample(X_train, y_train)

        except Exception as e:
            print("Exception for resampling: \n")
            print("Continuining")

        processor_original, processor_sampled = data_processor(), data_processor()
        x_train_og, x_test_og = processor_original.fit_train_test(X_train, X_test)
        x_train_samp, x_test_samp = processor_sampled.fit_train_test(X_samp, X_test)

        # Run model and predict
        model_      = self.get_model(model, **kwargs)
        y_pred_og   = model_.fit_predict(x_train_og, y_train, x_test_og)
        y_pred_samp = model_.fit_predict(x_train_samp, y_samp, x_test_samp)

        # Evaluate scores
        evaluator_  = evaluator()
        self._add_scores(list(evaluator_.evaluate(y_test, y_pred_og).values()), sampled = False)

    def _add_scores(self, scores, sampled):
        "add scores to result attribute"
        sampled_flag = sampled
        new_row = list(self.current_iteration) + [sampled_flag] + scores

        added_row = self.results.append(pd.Series(dict(zip(self.results.df.columns, new_row))),
                                           ignore_index = True)
        self.results = added_row

        return None

    def get_sampler(self, method, strategy, **kwargs):
        return sampler(method, strategy, **kwargs)

    def get_model(self, model, **kwargs):
        return model_class(model, **kwargs)