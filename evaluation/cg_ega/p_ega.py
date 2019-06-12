import numpy as np
from evaluation.cg_ega.tools import derivatives, _all, _any

class P_EGA():
    def __init__(self, y_true, y_pred, freq):
        self.freq = freq
        self.dy_true, self.y_true = derivatives(y_true, self.freq)
        self.dy_pred, self.y_pred = derivatives(y_pred, self.freq)

    def full_P_EGA(self):
        """
        Compute the Prediction-Error Grid Analysis from "Evaluating the accuracy of continuous glucose-monitoring sensors:
        continuous glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et al., 2004.
        :param y_true: true glucose values;
        :param y_pred: predicted glucose values;
        :return: numpy array of P-EGA prediction classification;
        """
        y_true, dy_true, y_pred, dy_pred = self.y_true, self.dy_true, self.y_pred, self.dy_pred

        # if the true rate are big, we accept bigger mistake (region borders are modified)
        mod = np.zeros_like(y_true)
        mod[_any([
            _all([
                np.greater(dy_true, -2),
                np.less_equal(dy_true, -1)]),
            _all([
                np.less(dy_true, 2),
                np.greater_equal(dy_true, 1),
            ])
        ])] = 10

        mod[_any([
            _all([
                np.less_equal(dy_true, -2)
            ]),
            _all([
                np.greater_equal(dy_true, 2)
            ])
        ])] = 20

        A = _any([
            _all([
                np.less_equal(y_pred, 70 + mod),
                np.less_equal(y_true, 70)
            ]),
            _all([
                np.less_equal(y_pred, y_true * 6 / 5 + mod),
                np.greater_equal(y_pred, y_true * 4 / 5 - mod)
            ])
        ])

        E = _any([
            _all([
                np.greater(y_true, 180),
                np.less(y_pred, 70 - mod)
            ]),
            _all([
                np.greater(y_pred, 180 + mod),
                np.less_equal(y_true, 70)
            ])
        ])

        D = _any([
            _all([
                np.greater(y_pred, 70 + mod),
                np.greater(y_pred, y_true * 6 / 5 + mod),
                np.less_equal(y_true, 70),
                np.less_equal(y_pred, 180 + mod)
            ]),
            _all([
                np.greater(y_true, 240),
                np.less(y_pred, 180 - mod),
                np.greater_equal(y_pred, 70 - mod)
            ])
        ])

        C = _any([
            _all([
                np.greater(y_true, 70),
                np.greater(y_pred, y_true * 22 / 17 + (180 - 70 * 22 / 17) + mod)
            ]),
            _all([
                np.less_equal(y_true, 180),
                np.less(y_pred, y_true * 7 / 5 - 182 - mod)
            ])
        ])

        # B being the weirdest zone in the P-EGA, we compute it last by saying
        # it's all the points that have not been classified yet.
        B = _all([
            np.equal(A, False),
            np.equal(C, False),
            np.equal(D, False),
            np.equal(E, False),
        ])

        return np.concatenate([A, B, C, D, E], axis=1)