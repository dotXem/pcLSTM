import pandas as pd
from preprocessing.reshape import reshape_day, reshape_samples
from preprocessing.filtering import filter
from preprocessing.min_max_normalization import normalize
from preprocessing.train_valid_test_splitting import split
from tools.timeit import timeit

"""
Preprocessing steps:
    - reshape.py the data into days (extended days for T1DMS)
    - split the data into train, valid and test sets
    - (low pass filter the data)
    - min max normalization
"""

@timeit
def preprocessing(file, params):
    # load the data
    data = pd.read_csv(file)

    # reshape.py the dataframe into days
    data = reshape_day(data, params=params["reshape_day"])

    # split the days into training, validation and testing sets
    train, valid, test = split(data, params["split"])

    # low pass filter all the training sets and the inputs of the validation and testing sets
    train = filter(train, params=params["filter"])
    valid = filter(valid, params=params["filter"], only_inputs=True)
    test = filter(test, params=params["filter"], only_inputs=True)

    # normalization of the data
    train, params["normalize"] = normalize(train, params=params["normalize"], fit=True)
    valid, params["normalize"] = normalize(valid, params=params["normalize"])
    test, params["normalize"] = normalize(test, params=params["normalize"])

    # generate the samples (with past values as input)
    train = reshape_samples(train, params=params["reshape_samples"])
    valid = reshape_samples(valid, params=params["reshape_samples"])
    test = reshape_samples(test, params=params["reshape_samples"])

    return train, valid, test, params
