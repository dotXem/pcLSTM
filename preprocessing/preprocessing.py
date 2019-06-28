import pandas as pd
from preprocessing.reshape import reshape_day, reshape_samples_with_history
from preprocessing.scaling import normalize
from preprocessing.cross_validation import split


def preprocessing(file, hist, ph, freq, cv):
    """
        Preprocessing pipeline:
            1. reshape the data into days of data
            2. compute the training, validation, testing sets
            3. normalize the data
            4. format the samples with the available history
        :param file: path to the file where the data is
        :param hist: history in minutes (e.g., 180)
        :param ph: prediction horizon in minutes (e.g., 30)
        :param freq: sampling frequency (e.g.,  5)
        :param cv: cross-validation factor (e.g., 4)
        :return: training splits, validations splits, testings splits, min per splits, max per splits (for back-scaling)
    """
    # load the data
    data = pd.read_csv(file)

    # reshape.py the dataframe into days
    data = reshape_day(data, ph, freq)

    # split the days into training, validation and testing sets
    train, valid, test = split(data, cv)

    # normalization of the data
    train, min, max = normalize(train)
    valid, _, _ = normalize(valid, min, max)
    test, _, _ = normalize(test, min, max)

    # generate the samples (with past values as input)
    train = reshape_samples_with_history(train, hist, freq)
    valid = reshape_samples_with_history(valid, hist, freq)
    test = reshape_samples_with_history(test, hist, freq)

    return train, valid, test, min, max
