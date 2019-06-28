import numpy as np


def normalize(data, min=None, max=None):
    """
        Min-max normalization (scaling of the features between -1 and 1) of the data.
        :param data: input data of shape (n_splits, n_days, None, 5)
        :param min: if None, min and max are computed, fitted to the data, else transformation is applied
        :param max: see min
        :return: normalized data of shape (n_splits, n_days, None, 5), list of min (one vector per split),
                list of max (one vector per split)
    """
    # if min and max are not given, we need to fit the min max scaling
    if min is None and max is None:
        min, max = [], []
        for split in data:
            min_max = np.array([[df.min().values, df.max().values] for df in split])
            min.append(np.min(min_max[:, 0], axis=0))
            max.append(np.max(min_max[:, 1], axis=0))

    # transform the data
    data_norm = [
        [2 * (day - min_split) / (max_split - min_split) - 1 for day in split]
        for split, min_split, max_split in zip(data, min, max)
    ]

    return data_norm, min, max
