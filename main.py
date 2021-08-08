import os, sys
import utils

import pandas as pd
import numpy as np
import sklearn as skl

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Import Imbalanced data section
import imblearn as imbl
from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import SMOTE, SMOTEN, SVMSMOTE, ADASYN

# Load Models
from sklearn.svm import SVC as SVM
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.neural_network import MLPClassifier as MLP

# Import from utils
import data_process_utils as utils

#%% List of models, datasets...
# binary_datasets = fetch_datasets()
# glass_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv'
# ecoli_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ecoli.csv'
# thyroid_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/new-thyroid.csv'
#
# urls = [ecoli_url, thyroid_url, glass_url]
# overs_methods = [SMOTE, SMOTEN, SVMSMOTE, ADASYN]    #,  custom_SMOTE]
# models = [SVM, KNN, Tree, LogReg, MLP]
#
#
#
# # Try with ecoli first
# data = pd.read_csv(glass_url, header = None)
# X, y = data.values[:, :-1], data.values[:, -1]
#
# # Print data info
# data_info = utils._basic_data_info(X, y)
# print("Data Info: \n", data_info)
#
#
#
# # Split data
# X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y, test_size = 0.4, random_state = 2323,
#                                                                         shuffle = True, stratify = y)
#
# # Process Data
# data_processor = utils.data_processor()
# data_processor.fit(X_train, y_train)
# x_train = data_processor.transform(X_train)[0]
# x_test  = data_processor.transform(X_test)[0]
#
#
# # Iterate through all methods
# list_of_methods = [SMOTE(random_state = 2323, k_neighbors = 3)]
#
# for method in list_of_methods:
#     # Apply method
#     x_train_samp, y_train_samp = method.fit_resample(x_train, y_train)
#
#     # Comparison visualisation
#     oversampled_data_info = utils._basic_data_info(x_train_new, y_train_new)
#     train_data_info       = utils._basic_data_info(x_train, y_train)
#
#     comparison_info  = utils._compare_pre_post_sampling(x_train, y_train, x_train_new, y_train_new)
#
#     # Data visualisation
#     fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex= True, sharey = True)
#     x_rep = TSNE(n_components = 2).fit_transform(x_train)
#     x_rep_samp = TSNE(n_components = 2).fit_transform(x_train_samp)
#
#     # Plot per class
#     function+to_plot
#
#
#
#
#     # Apply model
#     knn_model = KNN().fit(x_train, y_train)
#     y_pred = knn_model.predict(x_test)
#
#     knn_model_oversampled = KNN().fit(x_train_new, y_train_new)
#     y_over_pred = knn_model_oversampled.predict(x_test)
#
#
#
#     # Compare results
#     results_comp = utils._compare_results(y_pred, y_over_pred, y_test)
#     print(results_comp)

# def main():

# Load data
glass_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv'
data = pd.read_csv(glass_url, header=None)
X, y = data.values[:, :-1], data.values[:, -1]

# Process data
processor = utils.data_visualiser()

# Plot data
fig, ax = plt.subplots()
ax = processor.plot_per_class(X, y)

plt.show()




# if __name__ == "__main__":
#     main()






