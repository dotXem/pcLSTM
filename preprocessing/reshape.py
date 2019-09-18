from misc import day_len
import numpy as np
import pandas as pd


def reshape_day(data, ph, freq):
    """
        Reshape the data from the files to an array of dataframes representing the days.
        :param data: DataFrame of shape (None, 4), with columns (datetime, glucose, insulin, CHO)
        :param ph: prediction horizon in minutes (e.g., 30)
        :param freq: sampling frequency of the time-series (e.g., 5)
        :return: array of n days, each being a DataFrame of shape (None, 5),
        with columns (glucose, insulin, CHO, glucose_at_ph-1, glucose_at_ph)
    """

    samples_per_day = day_len // freq
    ph_freq = ph // freq
    n_days = len(data.index) // samples_per_day

    # remove timestamps and split into days
    days = [
        data.ix[d * samples_per_day:(d + 1) * samples_per_day - 1, ["glucose", "insulin", "CHO"]].reset_index(
            drop=True)
        for d in range(n_days)
    ]

    # for every day, compute the objective prediction, based on the prediction horizon
    def compute_y(day, ph):
        y1 = day.ix[ph - 1:len(day.index) - 2, "glucose"].copy().reset_index(drop=True)
        y2 = day.ix[ph:len(day.index) - 1, "glucose"].copy().reset_index(drop=True)
        day_y = day.ix[:len(day.index) - ph - 1].copy().reset_index(drop=True)
        day_y["y_ph-1"], day_y["y_ph"] = y1, y2
        return day_y

    days = [compute_y(day, ph_freq) for day in days]

    return days


def reshape_samples_with_history(data, hist, freq):
    """
        Final reshape of the input data to compute the samples accounting for the amount of history.
        :param data: input data of shape (n_splits, n_days_per_split, None, 5)
        :param hist: history in minutes (e.g., 180 - 3 hours)
        :param freq: sampling frequency in minutes (e.g., 5)
        :return: data of the shape (n_splits, None, 3 * hist / freq + 2) - days are collasped inside the split
    """
    day_len_freq = len(data[0][0])
    hist_freq = hist // freq

    columns = np.array([
        ["glucose_" + str(_) for _ in np.arange(hist_freq)],
        ["insulin_" + str(_) for _ in np.arange(hist_freq)],
        ["CHO_" + str(_) for _ in np.arange(hist_freq)],
    ]).transpose().ravel()
    columns = np.append(columns, ["y_ph-1", "y_ph"])

    splits = []
    for split in data:
        days = []
        for day in split:
            # create the samples by adding the past values accounting for the amount of history
            X = np.array([
                day.ix[j: j + day_len_freq - hist_freq, ["glucose", "insulin", "CHO"]].values for j in
                range(hist_freq)
            ])
            # the for loop is done the other way around to speed up, but we need to transpose after
            X = np.rollaxis(X, 1, 0)

            X = np.reshape(X, (X.shape[0], -1))
            y1 = day.ix[hist_freq-1:, ["y_ph-1"]].values.reshape(-1, 1)
            y2 = day.ix[hist_freq-1:, ["y_ph"]].values.reshape(-1, 1)

            days.append(pd.DataFrame(data=np.c_[X, y1, y2], columns=columns))
        splits.append(pd.concat(days).reset_index(drop=True))

    return splits
