import matplotlib.pyplot as plt
import utils_vis as utils
import dataset_loader as loader

import pandas as pd
import numpy as np
import time


# %% Plot data
visualiser  = utils.data_plotter(num_dimensions = 2, projection_method = "pca", seed = 2323)

time_start = time.time()
data_loader = loader.data_loader(low = 1.0, high = 50.0)
print("Data loaded in {:.2f} minutes".format((time.time() - time_start) / 60))

# Save Metadata into pandas df
metadata_df = pd.DataFrame(data = np.nan, index = [list(data_loader.keys())],
                           columns = ["size", "feats", "K", "max_ratio"])

for key_ in data_loader.keys():
    metadata_df.loc[key_, :] = visualiser._compute_metadata(data_loader[key_]["X"], data_loader[key_]["y"])

metadata_df.to_csv("Metadata.csv", index = True)

# for key in data_loader.keys():
#     fig, ax = plt.subplots()
#     visualiser.plot_x_data(data = data_loader[key]["X"], ax = ax, norm = False)
#     ax.set_title("Data ".format(key))
#     plt.show()
#
# for key in data_loader.keys():
#     fig, ax = plt.subplots()
#     visualiser.plot_x_data_per_class(X = data_loader[key]["X"], y = data_loader[key]["y"], ax = ax)
#     ax.set_title("Data {}".format(key))
#     plt.show()