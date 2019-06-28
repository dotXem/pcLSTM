from scipy.signal import correlate
import numpy as np

def time_lag(y_true, y_pred):
    """
        Compute the time-lag (TL) metric, as the shifting number maximizing the correlation
        between the prediction and the ground truth.
        :param y_true: ground truth of shape (n_days, None)
        :param y_pred: predictions of shape (n_days, None)
        :return: mean of daily time-lags
    """

    lags = []
    for y_true_d, y_pred_d in zip(y_true, y_pred):
        y_true_d_norm = (y_true_d - np.mean(y_true_d)) / np.std(y_true_d)
        y_pred_d_norm = (y_pred_d - np.mean(y_pred_d)) / np.std(y_pred_d)
        lags.append(len(y_true_d_norm) - np.argmax(correlate(y_true_d_norm, y_pred_d_norm)) - 1)
    return np.mean(lags)
