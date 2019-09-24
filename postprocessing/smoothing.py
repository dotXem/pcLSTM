import sys
from tools.printd import printd
import numpy as np
import os
from evaluation.results import Results
import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing
import misc


def moving_average(x, N):
    """
        Apply a moving average smoothing to the input data
        :param x: input time-series
        :param N: wideness of the moving avergae window (from 1 to 5)
        :return: smoothed time-series
    """
    ma = pd.Series(x).rolling(window=N).mean()
    ma[pd.isna(ma)] = x[pd.isna(ma)]
    return ma.values


def exponential_smoothing(x, alpha):
    """
        Apply an exponential smoothing to the input data
        :param x: input time-series
        :param alpha: smoothness (between 0 and 1) coefficient
        :return: smoothed time-series
    """
    return SimpleExpSmoothing(x).fit(alpha).fittedvalues


def smooth_results(model_name, ph, dataset, subject, smoothing_technique, save=False, print=True, *kwargs):
    """

        :param model_name: name of the model (e.g., "ELM")
        :param ph: prediction horizon in minutes (e.g., 30)
        :param dataset: name of the dataset
        :param subject: name of the subject
        :param smoothing_technique: either "moving_average"/"ma" or "exponential_smoothing"/"es"
        :param save: boolean activating (or not) the saving of the results
        :param print: boolean activating (or not) the printing of the results
        :param kwargs: parameters to be passed to the smoothing technique
        :return: smoothed results
    """
    if smoothing_technique in ["moving_average", "ma"]:
        technique = moving_average
        acr = "ma" + str(kwargs[0])
    elif smoothing_technique in ["exponential_smoothing", "es"]:
        technique = exponential_smoothing
        acr = "es" + str(kwargs[0])
    else:
        sys.exit(-1)

    file = os.path.join("results", "ph" + str(ph), model_name,
                        "ph" + str(ph) + "_" + dataset + subject + "_" + model_name + "_.res")
    res = Results(model_name, ph, dataset, subject, misc.freq, file=file)
    splits_ma = []
    for split in res.results:
        y_trues = split[0, :, :]
        y_preds = split[1, :, :]
        y_preds_ma = []
        for day in y_preds:
            y_preds_ma.append(technique(day, *kwargs))
        splits_ma.append([y_trues, y_preds_ma])

    splits_ma = [np.rollaxis(np.array(split_ma), 0, 3) for split_ma in splits_ma]
    res_smoothed = Results(model_name + acr, ph, dataset, subject, misc.freq, results=splits_ma)
    if print:
        printd(res.get_results())
        printd(res_smoothed)
    if save:
        res_smoothed.save()
        return res_smoothed
    else:
        return res_smoothed


def grid_search(model_name, ph, dataset, subject, technique, grid, save=False):
    """
        Performs a grid search of the smoothing technique parametrs, prints the results for each run.
        :param model_name: name of the model (e.g., "ELM")
        :param ph: prediciton horizon in minutes (e.g., 30)
        :param dataset: name of the dataset (can be "all")
        :param subject: name of the subject (can be "all")
        :param technique: smoothing technique, either "moving_average"/"ma" or "exponential_smoothing"/"es"
        :param grid: list of parameters to be passed to the smoothing technique
        :param save: if the results shall be saved
        :return: /
    """
    if dataset == "all":
        datasets_subjects = np.array(
            np.concatenate([[[k, v_] for v_ in v] for k, v in misc.datasets_subjects_dict.items()],
                           axis=0)).reshape(-1, 2)
    else:
        if subject == "all":
            datasets_subjects = np.array(
                np.concatenate([[dataset, v_] for v_ in misc.datasets_subjects_dict[dataset]],
                               axis=0)).reshape(-1, 2)
        else:
            datasets_subjects = np.array([[dataset, subject]]).reshape(-1, 2)

    for val in grid:
        res = []
        for dataset, subject in datasets_subjects:
            res.append(smooth_results(model_name, ph, dataset, subject, technique, save, False, val).get_results())

        printd(val, dict(pd.DataFrame(data=res,columns=list(res[0].keys())).mean()))
