import numpy as np


def dRMSE(y_true, y_pred):
    dy_true = np.diff(y_true)
    dy_pred = np.diff(y_pred)
    return np.sqrt(np.mean((dy_true - dy_pred) ** 2))
