import numpy as np
from evaluation.cg_ega.tools import derivatives, _all, _any
from evaluation.cg_ega.p_ega import P_EGA
from evaluation.cg_ega.r_ega import R_EGA


class CG_EGA():
    def __init__(self, y_true, y_pred, freq):
        self.freq = freq
        self.p_ega = P_EGA(y_true, y_pred, freq).full_P_EGA()
        self.r_ega = R_EGA(y_true, y_pred, freq).full_R_EGA()
        self.dy_true, self.y_true = derivatives(y_true, self.freq)
        self.dy_pred, self.y_pred = derivatives(y_pred, self.freq)

    # full version of the CG-EGA with the cartesian product of the P-EGA and R-EGA with glycemia zones distinction
    def full_CG_EGA(self):
        """
        Compute the Continuous Glucose-Error Grid Analysis from "Evaluating the accuracy of continuous glucose-monitoring
        sensors: continuous glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et
        al., 2004. It is computed by combining the R-EGA and P-EGA.
        :param y_true: true glucose values;
        :param y_pred: predicted glucose values;
        :param hist: amount of history use to make the predictions;
        :param ph: prediction horizon;
        :return: numpy array of CG-EGA classification, divided by glycemia regions (hypo, eu, hyper);
        """

        y_true, dy_true, y_pred, dy_pred = self.y_true, self.dy_true, self.y_pred, self.dy_pred
        p_ega, r_ega = self.p_ega, self.r_ega

        # compute the glycemia regions
        hypoglycemia = np.less_equal(y_true, 70).reshape(-1, 1)
        euglycemia = _all([
            np.less_equal(y_true, 180),
            np.greater(y_true, 70)
        ]).reshape(-1, 1)
        hyperglycemia = np.greater(y_true, 180).reshape(-1, 1)

        # apply region filter and convert to 0s and 1s
        P_hypo = np.reshape(np.concatenate([np.reshape(p_ega[:, 0], (-1, 1)),
                                            np.reshape(p_ega[:, 3], (-1, 1)),
                                            np.reshape(p_ega[:, 4], (-1, 1))], axis=1).astype("int32") *
                            hypoglycemia.astype("int32"),
                            (-1, 3))
        P_eu = np.reshape(np.concatenate([np.reshape(p_ega[:, 0], (-1, 1)),
                                          np.reshape(p_ega[:, 1], (-1, 1)),
                                          np.reshape(p_ega[:, 2], (-1, 1))], axis=1).astype("int32") *
                          euglycemia.astype("int32"),
                          (-1, 3))
        P_hyper = np.reshape(p_ega.astype("int32") * hyperglycemia.astype("int32"), (-1, 5))

        R_hypo = np.reshape(r_ega.astype("int32") * hypoglycemia.astype("int32"), (-1, 8))
        R_eu = np.reshape(r_ega.astype("int32") * euglycemia.astype("int32"), (-1, 8))
        R_hyper = np.reshape(r_ega.astype("int32") * hyperglycemia.astype("int32"), (-1, 8))

        CG_EGA_hypo = np.dot(np.transpose(R_hypo), P_hypo)
        CG_EGA_eu = np.dot(np.transpose(R_eu), P_eu)
        CG_EGA_hyper = np.dot(np.transpose(R_hyper), P_hyper)

        return CG_EGA_hypo, CG_EGA_eu, CG_EGA_hyper

    # simplified version of the CG-EGA (AP, BE, EP rates)
    def simplified_CG_EGA(self):
        AP_hypo, BE_hypo, EP_hypo, AP_eu, BE_eu, EP_eu, AP_hyper, BE_hyper, EP_hyper = self.detailed_CG_EGA(count=True)
        sum = (AP_hypo + BE_hypo + EP_hypo + AP_eu + BE_eu + EP_eu + AP_hyper + BE_hyper + EP_hyper)
        return (AP_hypo + AP_eu + AP_hyper) / sum, (BE_hypo + BE_eu + BE_hyper) / sum, (
                EP_hypo + EP_eu + EP_hyper) / sum

    # simplified version of the CG-EGA with score of the different glycemia zones
    def detailed_CG_EGA(self, count=False):
        """
        Simplify the CG-EGA results into Accurate Predictions, Benign Errors and Erroneous Predictions, following
        Compute the Rate-Error Grid Analysis from "Evaluating the accuracy of continuous glucose-monitoring sensors:
        continuous glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et al., 2004.
        :param y_true: true glucose values;
        :param y_pred: predicted glucose values;
        :param hist: amount of history use to make the predictions;
        :param ph: prediction horizon;
        :return: numpy array count of predictions per AB/BE/EP in every glycemia region (hypo, eu, hyper);
        """

        CG_EGA_hypo, CG_EGA_eu, CG_EGA_hyper = self.full_CG_EGA()

        filter_AP_hypo = [
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        filter_BE_hypo = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ]

        filter_EP_hypo = [
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]

        AP_hypo = np.sum(CG_EGA_hypo * filter_AP_hypo)
        BE_hypo = np.sum(CG_EGA_hypo * filter_BE_hypo)
        EP_hypo = np.sum(CG_EGA_hypo * filter_EP_hypo)

        filter_AP_eu = [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        filter_BE_eu = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        filter_EP_eu = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]

        AP_eu = np.sum(CG_EGA_eu * filter_AP_eu)
        BE_eu = np.sum(CG_EGA_eu * filter_BE_eu)
        EP_eu = np.sum(CG_EGA_eu * filter_EP_eu)

        filter_AP_hyper = [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        filter_BE_hyper = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        filter_EP_hyper = [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]

        AP_hyper = np.sum(CG_EGA_hyper * filter_AP_hyper)
        BE_hyper = np.sum(CG_EGA_hyper * filter_BE_hyper)
        EP_hyper = np.sum(CG_EGA_hyper * filter_EP_hyper)

        if not count:
            sum_hypo = (AP_hypo + BE_hypo + EP_hypo)
            sum_eu = (AP_eu + BE_eu + EP_eu)
            sum_hyper = (AP_hyper + BE_hyper + EP_hyper)

            AP_hypo, BE_hypo, EP_hypo = AP_hypo / sum_hypo, BE_hypo / sum_hypo, EP_hypo / sum_hypo
            AP_eu, BE_eu, EP_eu = AP_eu / sum_eu, BE_eu / sum_eu, EP_eu / sum_eu
            AP_hyper, BE_hyper, EP_hyper = AP_hyper / sum_hyper, BE_hyper / sum_hyper, EP_hyper / sum_hyper
            
        return AP_hypo, BE_hypo, EP_hypo, AP_eu, BE_eu, EP_eu, AP_hyper, BE_hyper, EP_hyper
