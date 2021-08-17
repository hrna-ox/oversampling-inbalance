import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap



# %%
"""
Step 1: Plotting data information. The goal is to include: 
    - Data information (potentially include metadata visualisation)
    - Visualisation of the dataset (after projection perhaps?)
    - Visualisation of the dataset per target class
    - Including correlations between features.
"""
class data_plotter():
    def __init__(self, num_dimensions = 2, projection_method = 'tsne', seed = 2323):
        self.num_dimensions = num_dimensions
        self.colors = get_cmap("tab10").colors
        self.seed = seed
        self.mu   = None
        self.sigma= None

        # Decide on projection method
        self.projection_method = self._projection_method_from_string(projection_method)


    def plot_x_data(self, data, ax = None, norm = False, **kwargs):

        # guarantee right_format
        data = self._set_to_right_format(data, projected = True, norm = norm)

        # Check relevant axes grabbed
        if ax == None:
            ax = plt.gca()

        # Make plot
        x, y = data[:, 0], data[:, 1]

        # Plot scatter
        ax_title = "data" + "_norm" * (norm == True) + "_projected"
        ax.scatter(x, y, **kwargs)
        ax.set_title(ax_title)

        return ax


    def plot_x_data_per_class(self, X, y, ax,  **kwargs):
        "Scatter plot for each class"

        # guarantee right format
        X = self._set_to_right_format(X, projected = True, norm = False)

        # Check axes
        if ax == None:
            ax = plt.gca()

        # Compute num classes
        num_classes_ = self._compute_num_classes(y)

        # plot
        for class_ in range(num_classes_):

            # Access data for the class
            X_class_ = X[self._get_class_ids(X, y, class_), :]
            x_scatter, y_scatter = X_class_[:, 0], X_class_[:, 1]

            ax.scatter(x_scatter, y_scatter, color = self.colors[class_], label = "Class {}".format(class_), **kwargs)

        return ax

    def _get_class_ids(self, X, y, class_):
        "Class ids given target labels y, class label and X data."
        assert X.shape[0] == y.shape[0]

        return y==class_

    def plot_compare_2_datasets(self, data1, data2, ax , **kwargs):
        """
        Scatter plot comparing 2 sets of data (pre and post sampling)
        """
        # Obtain corresponding data
        X1, y1 = data1
        X2, y2 = data2

        # Make plots
        ax[0] = self.plot_per_class(X1, y1, ax = ax[0])
        ax[1] = self.plot_per_class(X2, y2, ax = ax[1])

        return ax

    def _compute_num_classes(self, y):
        return np.unique(y).size

    def _compute_max_min_ratio(self, y):
        _, _counts = np.unique(y, return_counts = True)

        return np.divide(np.max(_counts / np.sum(_counts)), np.min(_counts / np.sum(_counts)))

    def _projection_method_from_string(self, projection_method_string):
        assert isinstance(projection_method_string, str)

        if projection_method_string.lower() == 'tsne':
            method = TSNE(self.num_dimensions, random_state = self.seed)

        elif projection_method_string.lower() == 'pca':
            method = PCA(n_components = self.num_dimensions, random_state = self.seed)

        else:
            print("Not right string provided as input")
            raise ValueError

        return method

    def _compute_metadata(self, X, y):
        "Attributes to compute: Data size, dimensionality, number of classes, measure of class inbalance - currently max - min"
        data_size = X.shape[0]
        dim_size = X.shape[1]

        # Compute class information
        num_classes = self._compute_num_classes(y)
        max_min_prop = self._compute_max_min_ratio(y)

        # Return array with 4 characteristics
        return np.array([data_size, dim_size, num_classes, max_min_prop])

    def metadata_from_dic(self, data_dic):
        "Compute Pandas DataFrame with basic dataset metadata."
        metadata_df = pd.DataFrame(data = np.nan, index = list(data_dic.keys()),
                                   columns = ["size", "feats", "K", "max_min_ratio"])

        # Iterate through keys
        for key_ in data_dic.keys():
            X, y = data_dic[key_]["X"], data_dic[key_]["y"]
            metadata_df.loc[key_, :] = self._compute_metadata(X, y)

        return metadata_df

    def project(self, data, **kwargs):
        return self.projection_method.fit_transform(data)


    def _norm(self, data):
        self.mu = np.min(data, axis = 0, keepdims = True)
        self.sigma = np.max(data, axis = 0, keepdims = True) - self.mu

        return np.divide(data - self.mu, self.sigma)

    def _set_to_right_format(self, data, projected = True, norm = True):
        if norm == True:
            data = self._norm(data)

        if projected == True:
            data = self.project(data)

        return data