from misc import day_len
import numpy as np
import pandas as pd
from tools.timeit import timeit


# @timeit
def reshape_day(data, params):
    ph = params["ph"]
    freq = params["freq"]
    hist = params["hist"]
    samples_per_day = day_len // freq
    ph_freq = ph // freq
    hist_freq = hist // freq
    n_days = len(data.index) // samples_per_day

    if params["dataset"] == "T1DMS":
        # data reshaping to comply to the GLYFE benchmark, which is :
        # we take data from the previous day to account for the length of the history
        days = [
            data.ix[(i + 1) * samples_per_day - ph_freq - hist_freq:(i + 2) * samples_per_day,
            ["glucose", "insulin", "CHO"]].reset_index(drop=True) for i in range(len(data.index) // samples_per_day - 1)
        ]
    else:
        # remove timestamps and split into days
        days = [
            data.ix[d * samples_per_day:(d + 1) * samples_per_day - 1, ["glucose", "insulin", "CHO"]].reset_index(
                drop=True)
            for d in range(n_days)
        ]

    # for every day, compute the objective prediction, based on the prediction horizon
    def compute_y(day, ph):
        y1 = day.ix[ph - 1:len(day.index) - 1, "glucose"].copy().reset_index(drop=True)
        y2 = day.ix[ph:, "glucose"].copy().reset_index(drop=True)
        day_y = day.ix[:len(day.index) - ph - 1].copy().reset_index(drop=True)
        day_y["y_ph-1"], day_y["y_ph"] = y1, y2
        return day_y

    days = [compute_y(day, ph_freq) for day in days]

    return days


# @timeit
def reshape_samples(data, params):
    hist = params["hist"]
    freq = params["freq"]
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
            # X = np.array([
            #     day.ix[j: j + hist_freq - 1, ["glucose", "insulin", "CHO"]].values for j in
            #     range(day_len_freq - hist_freq + 1)
            # ])

            X = np.array([
                day.ix[j: j + day_len_freq - hist_freq + 1 - 1, ["glucose", "insulin", "CHO"]].values for j in
                range(hist_freq)
            ])
            X = np.rollaxis(X, 1, 0)

            X = np.reshape(X, (X.shape[0], -1))
            y1 = day.ix[hist_freq - 1:, ["y_ph-1"]].values.reshape(-1, 1)
            y2 = day.ix[hist_freq - 1:, ["y_ph"]].values.reshape(-1, 1)

            days.append(pd.DataFrame(data=np.c_[X, y1, y2], columns=columns))
        splits.append(pd.concat(days).reset_index(drop=True))

    return splits
