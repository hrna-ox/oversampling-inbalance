import pandas as pd
import numpy as np



from sklearn.metrics import accuracy_score, precision_score, recall_score\
    , f1_score


def _get_nrows_ncols(N):
    if N==1:
        nrows, ncols = 1, 1

    else:
        nrows = int(np.ceil(N//2))
        ncols = 2

    return nrows, ncols


def _class_distribution(y):
    """
    Given a vector of labels y, obtain the class distribution

    returns: 2 arrays with the number of classes and corresponding number of appearances in the data.
    If with_perc is True, then add 3rd array with percentage information.
    """
    unique, counts = np.unique(y, return_counts = True)

    percentages = counts / np.sum(counts)

    return unique, counts, percentages



def _basic_data_info(X, y):
    """
    summary description of dataset
    """
    num_samples, num_feats = X.shape    # start with X properties

    # Compute distribution
    classes, counts, percs = _class_distribution(y)
    num_classes = classes.size

    # Return data info dictionary
    output_dic = {
        "Num_samples": num_samples,
        "Num_feats": num_feats,
        "Num_classes": num_classes,
        "classes": classes,
        "counts": counts,
        "percs": percs
    }

    return output_dic



class data_processor():
    def __init__(self, mu = None, sigma = None):

        # Other transformation params
        self.mu = 0
        self.sigma = 1

    def fit(self, X, y):

        # Identify mu and sigma from data
        mu = np.min(X, axis = 0, keepdims = True)
        sigma = np.max(X, axis = 0, keepdims = True) - np.min(X, axis = 0, keepdims = True)

        # Update attributes
        self.mu = mu
        self.sigma = sigma


    def fit_transform(self, X, y):

        return self.transform(self.fit(X, y))


    def transform(self, *args):

        #
        output_tuple = []

        # Iterate through each arg
        for arg in args:
            transformed_data = np.divide(arg - self.mu, self.sigma)
            output_tuple     = output_tuple + [transformed_data]

        return tuple(output_tuple)


def _compare_pre_post_sampling(X_train, y_train, X_new, y_new):
    """
    Comparison information for old and new data
    """
    train_data_info = _basic_data_info(X_train, y_train)
    new_data_info   = _basic_data_info(X_new, y_new)

    print("\nNum samples increased from {} to {} samples\n".format(train_data_info["Num_samples"], new_data_info["Num_samples"]))

    # Create pandas Dataframe
    df = pd.DataFrame(np.nan, index = train_data_info['classes'], columns = ['og_dist', 'og_prop', 'new_dist', 'new_prop'])
    df.iloc[:, 0] = train_data_info["counts"]
    df.iloc[:, 1] = train_data_info["percs"]
    df.iloc[:, 2] = new_data_info["counts"]
    df.iloc[:, 3] = new_data_info["percs"]

    df.index.name = "classes"

    # Difference in distributions
    print("Count comparison is as follows: \n", df)





def _compare_results(y_pred, y_pred_sampled, y_true):

    """
    Compare results between y_pred, and y_pred from oversampling
    """
    scores_og = _compute_scores(y_pred, y_true)
    scores_samp = _compute_scores(y_pred_sampled, y_true)

    # Aggreggate both results
    result_comp = pd.concat({"Og": scores_og, "samp": scores_samp}, axis = 1)

    return result_comp

def _compute_scores(y_pred, y_true):
    """
    Compute all scores at once
    """
    auc = accuracy_score(y_true = y_true, y_pred = y_pred)
    pre = precision_score(y_true, y_pred, average = "macro")
    rec = recall_score(y_true, y_pred, average = "macro")
    f1  = f1_score(y_true, y_pred, average = "macro")

    return pd.Series(data = [auc, pre, rec, f1], index = ['acc', 'pre', 'rec', 'f1'])


