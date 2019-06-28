from misc import day_len
import numpy as np


def reshape_day(results, hist, ph, freq):
    """
        Partition the results into days of results
        :param results: results of shape (None, 2)
        :param hist: history in minutes (e.g., 180)
        :param ph: prediction horizon in minutes (e.g., 30)
        :param freq: sampling frequency in minutes (e.g., 5)
        :return: reshape results of shape (n_days, None, 2)
    """
    preds_per_day = day_len // freq - ph // freq - hist // freq

    return np.reshape(results, (-1, preds_per_day, 2))
