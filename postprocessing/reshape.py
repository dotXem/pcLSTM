from misc import day_len
import numpy as np


def reshape_day(results, hist, ph, freq, dataset):
    if dataset == "T1DMS":
        preds_per_day = day_len // freq
    else:
        preds_per_day = day_len // freq - ph // freq - hist // freq + 1

    return np.reshape(results, (-1, preds_per_day, 2))
