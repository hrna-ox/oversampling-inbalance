import os, sys
import utils

import pandas as pd
import numpy as np
import sklearn as skl

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

# Import Imbalanced data section
import imblearn as imbl
from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import SMOTE, SMOTEN, SVMSMOTE, ADASYN


#%% Define aux functions
def _get_rows_cols(N):
    if N==1:
        nrows, ncols = 1, 1
    else:
        nrows = int(np.ceil((N//2)))
        ncols = 2

    return nrows, ncols

def _scatterplot2D_per_class(X, y):
    """
    Given 2D entries X, and target label y, make a 2D plot with different colour per class
    """
    fig, ax = plt.subplots()

    # Iterate through classes
    classes = np.unique(y)
    colors = get_cmap("tab10").colors
    markers = list(Line2D.markers.keys())

    # Iterate through classes
    for class_ in classes:
        class_ids = (y == class_)
        class_x   = X[class_ids, :]

        # Make a scatter plot
        plt.scatter(class_x[:, 0], class_x[:, 1], color = colors[int(class_)], label = 'Class {}'.format(class_))

    plt.close()

    return ax




#%% List of models, datasets...
# binary_datasets = fetch_datasets()
glass_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv'
ecoli_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ecoli.csv'
thyroid_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/new-thyroid.csv'

urls = [ecoli_url, thyroid_url, glass_url]
overs_methods = [SMOTE, SMOTEN, SVMSMOTE, ADASYN]    #,  custom_SMOTE]

#%% Iterate through each url
for url in [thyroid_url]:
    # Load data
    data = pd.read_csv(url, header = None)
    X, y = data.values[:, :-1], data.values[:, -1]

    # Print basic data info
    data_info = utils._basic_data_info(X, y)
    print("Data Info: \n", data_info)

    # Process Data
    data_processor = utils.data_processor()
    data_processor.fit(X, y)
    x_train = data_processor.transform(X)[0]

    # Initialise plot
    nrows, ncols = utils._get_nrows_ncols(2* len(overs_methods))
    fig, ax = plt.subplots(nrows, ncols, sharex = True, sharey = True)
    axs = ax.reshape(-1)

    # Iterate through all methods
    for method_class in overs_methods:

        # Get method id and compute sampling
        id_ = overs_methods.index(method_class)
        method = method_class(random_state = 2323)
        X_sampled, y_sampled = method.fit_resample(X, y)

        # Comparison visualisation
        oversampled_data_info = utils._basic_data_info(X_sampled, y_sampled)
        og_data_info          = utils._basic_data_info(X, y)

        comparison_info  = utils._compare_pre_post_sampling(X, y, X_sampled, y_sampled)

        # Data visualisation
        x_rep = TSNE(n_components = 2).fit_transform(X)
        x_rep_samp = TSNE(n_components = 2).fit_transform(X_sampled)

        # Plot per class
        axs[2*id_] = _scatterplot2D_per_class(x_rep, y)
        axs[2*id_ + 1] = _scatterplot2D_per_class(x_rep_samp, y_sampled)

    fig.tight_layout()

    plt.show()












