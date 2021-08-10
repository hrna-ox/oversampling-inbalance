import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as TTS

# Import from utils
import data_process_utils as utils

#%% List of models, datasets...
def main():
        return None

for i in range(1):

        # binary_datasets = fetch_datasets()
        glass_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv'
        ecoli_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ecoli.csv'
        thyroid_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/new-thyroid.csv'

        urls = [ecoli_url, thyroid_url, glass_url]
        # urls = [glass_url]
        overs_methods = ["SMOTE", "SMOTEN", "SVMSMOTE", "ADASYN"]    #,  custom_SMOTE]
        strategies    = ["minority", "not minority", "not majority", "all", "auto"]
        models = ["SVM", "KNN", "Tree", "Logistic"]

        # save results
        results_df = pd.DataFrame(data = np.nan, index = [], columns = ["dataset", "method", "strategy", "model", "og_or_samp",
                                                            "balanced_accuracy", "f1", "precision", "recall"])

        # Iterate through dataset, oversampling method and models.
        for dataset_url in tqdm(urls):
                # Load data
                data = pd.read_csv(dataset_url, header=None)
                X, y = data.values[:, :-1], data.values[:, -1]

                # Plot data
                vis_processor = utils.data_visualiser()
                fig, ax = plt.subplots()
                ax = vis_processor.plot_per_class(ax = ax, X = X, y = y)
                plt.show()

                # Process data
                X_train_og, X_test_og, y_train_og, y_test_og = TTS(
                        X, y, test_size = 0.3, random_state = 2323, shuffle = True, stratify = y)

                # Sample data
                for method_ in overs_methods:
                        for oversampling_strategy_ in strategies:
                                data_sampler = utils.sampler(method = method_,
                                                        sampling_strategy = oversampling_strategy_)
                                try:
                                        X_train_samp, y_train_samp = data_sampler.fit_resample(X_train_og, y_train_og)
                                except Exception as e:
                                        print("Exception in resampling: ", e)
                                        continue

                                # Visualise data
                                data_og, data_samp = (X_train_og, y_train_og), (X_train_samp, y_train_samp)
                                # fig, axs = plt.subplots(nrows = 1, ncols = 2)
                                # axs = vis_processor.plot_pre_post_sampling(ax = axs, data_og = data_og, data_samp = data_samp)
                                # axs[0].set_title("Original Data projection")
                                # axs[1].set_title("Sampled Data projection")
                                # plt.show()

                                # normalise for input
                                processor_og, processor_samp = utils.data_processor(), utils.data_processor()

                                x_norm_train_og, x_norm_test_og      = processor_og.fit_train_test(X_train_og, X_test_og)
                                x_norm_train_samp, x_norm_test_samp  = processor_samp.fit_train_test(X_train_samp, X_test_og)

                                # Run model
                                for model_ in models:
                                        model_og, model_samp = utils.model(model_), utils.model(model_)
                                        y_pred_og  = model_og.fit_predict(x_norm_train_og, y_train_og, x_norm_test_og)
                                        y_pred_samp= model_og.fit_predict(x_norm_train_samp, y_train_samp, x_norm_test_samp)

                                        # Evaluate according to metrics
                                        evaluator = utils.evaluator()

                                        # Compute metrics for original
                                        scores = list(evaluator.evaluate(y_test_og, y_pred_og).values())
                                        new_row = [dataset_url, method_,oversampling_strategy_, model_, "og"] + scores
                                        results_df = results_df.append(
                                                pd.Series(dict(zip(results_df.columns, new_row))),
                                                ignore_index = True
                                        )

                                        # Compute metrics for sampled
                                        scores_samp = list(evaluator.evaluate(y_test_og, y_pred_samp).values())
                                        new_row = [dataset_url, method_, oversampling_strategy_, model_, "samp"] + scores_samp
                                        results_df = results_df.append(
                                                pd.Series(dict(zip(results_df.columns, new_row))),
                                                ignore_index = True
                                        )


        results_df.to_csv("Oversampling_results.csv")

        # Print results difference
        results_og = results_df.query('og_or_samp == "og"')
        results_samp = results_df.query('og_or_samp == "samp"')
        assert np.sum(results_og.iloc[:, :4].values != results_samp.iloc[:, :4].values)  == 0

        # print average distribution
        test = results_og
        test.iloc[:, -4:] = test.iloc[:, -4:] - results_samp.iloc[:, -4:].values

        # Print results by method
        output_desc_ = test.groupby("method").apply(lambda x: x.iloc[:, -4:].describe())
        print(output_desc_)



if __name__ == "__main__":
        main()






