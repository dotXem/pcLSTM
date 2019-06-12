import matplotlib.pyplot as plt
from evaluation.rmse import RMSE
from evaluation.drmse import dRMSE
from evaluation.cg_ega.cg_ega import CG_EGA
import numpy as np

"""
Define the Results object that compute all the results given y_true, and y_pred
"""


class Results():
    def __init__(self, results, freq):
        self.results = np.rollaxis(np.array(results), 3, 1)
        self.freq = freq

    def get_results(self):
        # RMSE
        rmse = np.mean([RMSE(*res) for res in self.results])

        # dRMSE
        drmse = np.mean([dRMSE(*res) for res in self.results])

        # CG-EGA
        # cg_ega = CG_EGA(*results, freq).simplified_CG_EGA()
        cg_ega = np.mean([CG_EGA(*res, self.freq).simplified_CG_EGA() for res in self.results], axis=0)

        return np.concatenate([[rmse], [drmse], cg_ega])

    def plot(self, split=0, day=0, hold=True):
        y_true = self.results[split, 0, day, :]
        y_pred = self.results[split, 1, day, :]
        plt.figure()
        plt.plot(y_true,label="y_true")
        plt.plot(y_pred,label="y_pred")
        plt.title("Prediction VS ground truth for day " + str(day) + " of split " + str(split))
        plt.xlabel("time (" + str(self.freq) + "min)")
        plt.ylabel("glucose (mg/dL)")
        plt.show()
