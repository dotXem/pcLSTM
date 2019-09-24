from sklearn.svm import SVR as skSVR
from tools.timeit import timeit

params = {
    "C": 5e1,
    "epsilon":1e-1,
    "gamma": 0.5e-3,
    "shrinking": True,
}


class SVR():
    """
    The SVR model is based on Support Vector Regression.
    Parameters:
        - C: loss
        - epsilon: wideness of the no-penalty tube
        - gamma: kernel coefficient
        - shrinking: wether or not tp ise the shrinkin heuristic
    """

    CACHE_SIZE = 8000  # in Mb, allow more cache to the fitting of the model

    def __init__(self, params):
        self.C = params["C"]
        self.epsilon = params["epsilon"]
        self.kernel = "rbf"
        self.gamma = params["gamma"]
        self.shrinking = params["shrinking"]

    @timeit
    def fit(self, x, y):
        y = y["y_ph"]

        # define the model
        self.model = skSVR(C=self.C,
                           epsilon=self.epsilon,
                           kernel=self.kernel,
                           gamma=self.gamma,
                           shrinking=self.shrinking,
                           cache_size=self.CACHE_SIZE)

        # fit the model
        self.model.fit(x, y)

    @timeit
    def predict(self, x, y):
        y_true = y["y_ph"].values

        # predict
        y_pred = self.model.predict(x)

        return y_true, y_pred
