import numpy as np
from scipy.special import expit
from misc import *
from sklearn.linear_model import Ridge
from tools.timeit import timeit

params = {
    "neurons": 1e5,
    "l2": 5e2,
}


class ELM():
    """
    The ELM model is based on Extreme Learning Machine by Huang et al., 2004 .
    Parameters:
        - neurons: number of neurons in the single hidden layer
        - l2: L2 regularization factor
    """

    def __init__(self, params):
        self.neurons = params["neurons"]
        self.l2 = params["l2"]

    @timeit
    def fit(self, x, y):
        x = x.values
        y = y["y_ph"].values

        n_inputs = int(np.shape(x)[1])

        # initialize the random hidden layer with weights and bias
        np.random.seed(seed)
        self.W = np.random.normal(size=(n_inputs, int(self.neurons)))
        self.b = np.random.normal(size=(1, int(self.neurons)))

        # compute the outputs of the hidden layer
        H = expit(x @ self.W + self.b)

        # fit the outputs of the hidden layer to the objective values
        self.model = Ridge(self.l2)
        self.model.fit(H, y)

    @timeit
    def predict(self, x, y):
        x = x.values
        y_true = y["y_ph"].values

        # compute the output of the hidden layer
        H = expit(x @ self.W + self.b)

        # predict
        y_pred = self.model.predict(H)

        return y_true, y_pred
