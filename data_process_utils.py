import numpy as np

# Import dimensionality reduction
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


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
    def __init__(self):
        return None



class data_visualiser():
    """
    Data Visualiser class
    """
    def __init__(self, num_components = 2, colormap_name = "tab10"):
        self.num_dimensions = num_components
        self.colors = get_cmap(colormap_name).colors


    def project(self, data, **kwargs):
        return TSNE(self.num_dimensions, **kwargs).fit_transform(data)

    def plot(self, data, **kwargs):
        try:
            assert self._is_2D(data)

        except Exception as e:
            print("data not 2-dimensional")
            data = self.project(data)

        # Access data
        x, y = data[:, 0], data[:, 1]

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(x, y, **kwargs, label = "data")

        return ax


    def plot_per_class(self, X,y, **kwargs):
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
        num_classes_ = self._num_classes(self, y)

        for class_ in range(num_classes_):

            # Access data for the class
            X_class_ = X[self._get_class_ids(X, y, class_), :]
            x_scatter, y_scatter = X_class_[:, 0], X_class_[:, 1]

            ax.scatter(x_scatter, y_scatter, c = self.colors[class_], label = "Class {}".format(class_))

        return ax

    def plot_pre_post_sampling(self, data_og, data_sampled, **kwargs):
        """
        Scatter plot comparing data before and after sampling
        """
        # Obtain corresponding data
        X_og, y_og, X_samp, y_samp = data_og, data_sampled

        # Make plots
        fig, ax = plt.subplots(nrows = 1, ncols = 2)
        ax[0] = self.plot_per_class(X_og, y_og)
        ax[1] = self.plot_per_class(X_samp, y_samp)

        return ax

    def _is_2D(self, data):
        return len(data.shape) == 2

    def _samples_equal(self, array_1, array_2):
        return array_1.shape[0] == array_2.shape[0]

    def _num_classes(self, y):
        return np.unique(y).size

    def _get_class_ids(self, X, y, class_id):
        return (y==class_id)

