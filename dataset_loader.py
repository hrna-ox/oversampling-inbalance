"""
Python script for dataset loading
"""
from imblearn.datasets import fetch_datasets as fetch
from imblearn.datasets import make_imbalance
import sklearn.datasets as skl_datasets

import numpy as np


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

def _make_imbalance_dic(data_dic, low = 0.0, high = 0.0, seed=2323):
    # Data keys
    skl_keys = [key for key in list(data_dic.keys()) if "skl" in key]

    # Make imbalance dataset for those in skl keys
    for _key in skl_keys:

        # Load data
        X, y = data_dic[_key]["X"], data_dic[_key]["y"]

        # sample ratio
        ratio_ = np.random.uniform(low = low, high = high)
        strategy_ = _make_sampling_strategy(y, ratio_)

        # make imbalance dataset
        new_X, new_y = make_imbalance(X=X, y =y, sampling_strategy = strategy_ ,random_state = 2323)

        # Updated
        data_dic[_key] = {"X": new_X, "y": new_y}

    return data_dic

def data_loader(seed = 2323, low = 0.0, high = 1.0):
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

    #6. KDDCup99 dataset
    # kddcup_X, kddcup_y = skl_datasets.fetch_kddcup99(random_state = seed, shuffle = True, return_X_y=True)
    # kddcup_X = kddcup_X[:, [id_ for id_ in list(range(kddcup_X.shape[1])) if (not id_ in [1,2,3])]]
    # kddcup_y = convert_classes_to_range(kddcup_y)
    # output_dic["kddcup_skl"] = {"X": kddcup_X, "y": kddcup_y}

    # Load UCI data
    #7. GHZ indoor dataset
    output_dic = _make_imbalance_dic(output_dic, low = low, high = high)


    return output_dic