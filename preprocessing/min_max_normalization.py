import numpy as np
from tools.timeit import timeit


# @timeit
def normalize(data, params, fit=False):
    # first we compute the min and max
    if fit:
        min, max = [], []
        for split in data:
            min_max = np.array([[df.min().values, df.max().values] for df in split])
            min.append(np.min(min_max[:, 0], axis=0))
            max.append(np.max(min_max[:, 1], axis=0))
        params["min"], params["max"] = min, max

    # transform the data
    min, max = params["min"], params["max"]

    data = [
        [2 * (day - min_split) / (max_split - min_split) - 1 for day in split]
        for split, min_split, max_split in zip(data, min, max)
    ]

    return data, params
