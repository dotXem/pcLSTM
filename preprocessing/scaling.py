def standardize(data, mean=None, std=None):
    """
        Zero-mean and unit-variance standardization of the data.
        :param data: input data of shape (n_splits, None, 110)
        :param mean: if None, mean and std are computed, fitted to the data
        :param std: see min
        :return: standardized data of shape (n_splits, None, 110), list of means (one vector per split),
                list of stds (one vector per split)
    """
    # if min and max are not given, we need to fit the min max scaling
    if mean is None and std is None:
        mean, std = [], []
        for split in data:
            mean.append(split.mean().values)
            std.append(split.std().values)

    # transform the data
    data_norm = [(split - mean_split) / std_split for split, mean_split, std_split in zip(data, mean, std)]

    return data_norm, mean, std
