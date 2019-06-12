import numpy as np
from evaluation.cg_ega.tools import derivatives, _all, _any


class R_EGA():
    def __init__(self, y_true, y_pred, freq):
        self.freq = freq
        self.dy_true, self.y_true = derivatives(y_true, self.freq)
        self.dy_pred, self.y_pred = derivatives(y_pred, self.freq)

    def full_R_EGA(self):
        """
            Compute the Rate-Error Grid Analysis from "Evaluating the accuracy of continuous glucose-monitoring sensors:
            continuous glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et al., 2004.
            :param y_true: true glucose values;
            :param y_pred: predicted glucose values;
            :param hist: amount of history use to make the predictions;
            :param ph: prediction horizon;
            :return: numpy array of R-EGA prediction classification;
            """
        y_true, dy_true, y_pred, dy_pred = self.y_true, self.dy_true, self.y_pred, self.dy_pred

        A = _any([
            _all([  # upper and lower
                np.greater_equal(dy_pred, dy_true - 1),
                np.less_equal(dy_pred, dy_true + 1)
            ]),
            _all([  # left
                np.less_equal(dy_pred, dy_true / 2),
                np.greater_equal(dy_pred, dy_true * 2)
            ]),
            _all([  # right
                np.less_equal(dy_pred, dy_true * 2),
                np.greater_equal(dy_pred, dy_true / 2)
            ])
        ])

        B = _all([
            np.equal(A, False),  # not in A but satisfies the cond below
            _any([
                _all([
                    np.less_equal(dy_pred, -1),
                    np.less_equal(dy_true, -1)
                ]),
                _all([
                    np.less_equal(dy_pred, dy_true + 2),
                    np.greater_equal(dy_pred, dy_true - 2)
                ]),
                _all([
                    np.greater_equal(dy_pred, 1),
                    np.greater_equal(dy_true, 1)
                ])
            ])
        ])

        uC = _all([
            np.less(dy_true, 1),
            np.greater_equal(dy_true, -1),
            np.greater(dy_pred, dy_true + 2)
        ])

        lC = _all([
            np.less_equal(dy_true, 1),
            np.greater(dy_true, -1),
            np.less(dy_pred, dy_true - 2)
        ])

        uD = _all([
            np.less_equal(dy_pred, 1),
            np.greater_equal(dy_pred, -1),
            np.greater(dy_pred, dy_true + 2)
        ])

        lD = _all([
            np.less_equal(dy_pred, 1),
            np.greater_equal(dy_pred, -1),
            np.less(dy_pred, dy_true - 2)
        ])

        uE = _all([
            np.greater(dy_pred, 1),
            np.less(dy_true, -1)
        ])

        lE = _all([
            np.less(dy_pred, -1),
            np.greater(dy_true, 1)
        ])

        return np.concatenate([A, B, uC, lC, uD, lD, uE, lE], axis=1)