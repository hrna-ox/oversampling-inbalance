import matplotlib.pyplot as plt
import utils_vis as utils
import dataset_loader as loader

import time


# %% Plot data
visualiser  = utils.data_plotter(num_dimensions = 2, projection_method = "pca", seed = 2323)

time_start = time.time()
data_loader = loader.data_loader()
print("Data loaded in {:.2f} minutes".format((time.time() - time_start) / 60))

# fig, ax = plt.subplots()
# visualiser.plot_x_data(data = X, ax = ax, norm = False)