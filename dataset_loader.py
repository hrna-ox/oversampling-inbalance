"""
Python script for dataset loading
"""
from imblearn.datasets import fetch_datasets as fetch
from imblearn.datasets import make_imbalance
import sklearn.datasets as skl_datasets

import numpy as np
import pandas as pd

import time

def convert_classes_to_range(y):
    "Target might have other format types, so convert to range (0,..., num_classes - 1)"
    _values, num_classes = list(np.unique(y)), np.unique(y).size
    y_output = np.copy(y)

    for _value in _values:
        y_output[y == _value] = _values.index(_value)

    assert set(np.unique(y_output)) == set(range(num_classes))

    return y_output.astype(int)


def _make_sampling_strategy(y, ratio):
    """
    Given y and a max ratio between proportions, determine the relevant sample sizes
    """
    _values, _counts = np.unique(y, return_counts = True)

    # Ensure no class has more than ratio * min values
    max_value_ = int(np.floor(np.min(_counts) * ratio))
    output_counts_ = np.where(_counts > max_value_, max_value_, _counts)

    return dict(zip(_values.astype(int), output_counts_))

def _make_imbalance_dic(data_dic, low = 1.0, high = 8.0, seed=2323):
    # Data keys
    skl_keys = [key for key in list(data_dic.keys()) if "skl" in key]
    all_keys = list(data_dic.keys())

    # Make imbalance dataset for those in skl keys
    for _key in all_keys:

        # Load data
        X, y = data_dic[_key]["X"], data_dic[_key]["y"]

        # sample ratio
        ratio_ = np.random.uniform(low = low, high = high)
        strategy_ = _make_sampling_strategy(y, ratio_)

        # make imbalance dataset
        try:
            new_X, new_y = make_imbalance(X=X, y =y, sampling_strategy = strategy_ ,random_state = seed)
        except Exception as e:
            new_X, new_y = X, y
            print(e)
            print("Error making data {} imbalanced".format(_key))
            print("Sampling Error was {} strategy and {} ratio.".format(strategy_, ratio_))
            print("\n")
            pass

        # Updated
        data_dic[_key] = {"X": new_X, "y": new_y, "ratio": ratio_}

    return data_dic

def _convert_targets(data_dic):
    "Convert y to 0 - num classes format"
    keys_ = data_dic.keys()
    output_dic = data_dic.copy()

    for key_ in keys_:
        output_dic[key_]["y"] = convert_classes_to_range(output_dic[key_]["y"])

    return output_dic

def _make_transformations(dic, low, high):
    "Apply transformation to datasets"

    return _make_imbalance_dic(_convert_targets(dic), low, high)

def data_loader(seed = 2323, low = 1.0, high = 8.0):
    "Load and Generate dictionary of data."
    time_start = time.time()

    # Prepare output dic
    output_dic = {}

    # Load binary datasets from
    imblearn_datasets = fetch(random_state = seed)

    # Load them into output dic
    for data_name in list(imblearn_datasets.keys()):
        # access X, y
        X = imblearn_datasets[data_name]["data"]
        y = convert_classes_to_range(imblearn_datasets[data_name]["target"])

        assert X.shape[0] == y.shape[0]

        # Update
        output_dic[data_name] = {"X": X, "y": y}

    # Add other datasets
    glass_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv'
    ecoli_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ecoli.csv'
    thyroid_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/new-thyroid.csv'
    urls = [ecoli_url, thyroid_url, glass_url]

    for dataset_url in urls:
        # Load data
        data = pd.read_csv(dataset_url, header = None)
        X, y = data.values[:, :-1], data.values[:, -1]
        data_name = dataset_url.split("/")[-1].split(".")[0] + "-multiclass"
        output_dic[data_name] = {"X": X, "y": y}


    #1. Sklearn Iris
    iris_X, iris_y = skl_datasets.load_iris(return_X_y=True)
    iris_y = convert_classes_to_range(iris_y)
    output_dic["iris_skl"] = {"X": iris_X, "y": iris_y}

    #2. Digits dataset
    digits_X, digits_y = skl_datasets.load_digits(return_X_y=True)
    digits_y = convert_classes_to_range(digits_y)
    output_dic["digits_skl"] = {"X": digits_X, "y": digits_y}

    #3. Wine dataset
    wine_X, wine_y = skl_datasets.load_wine(return_X_y=True)
    wine_y = convert_classes_to_range(wine_y)
    output_dic["wine_skl"] = {"X": wine_X, "y": wine_y}

    #4. Breast cancer datasets
    breast_X, breast_y = skl_datasets.load_breast_cancer(return_X_y=True)
    breast_y = convert_classes_to_range(breast_y)
    output_dic["breast_skl"] = {"X": breast_X, "y": breast_y}

    #5. Covtype vectorized
    covtype_X, covtype_y = skl_datasets.fetch_covtype(random_state = seed, shuffle = True, return_X_y=True)
    covtype_y = convert_classes_to_range(covtype_y)
    output_dic["covtype_skl"] = {"X": covtype_X, "y": covtype_y}

    # Convert to right target format
    output_dic = _make_transformations(output_dic, low = low, high = high)

    print("Data loaded in {:.2f} minutes".format((time.time() - time_start) / 60))

    return output_dic