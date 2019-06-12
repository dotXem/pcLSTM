import numpy as np
from scipy.special import expit
from misc import *
from sklearn.linear_model import Ridge
from tools.timeit import timeit

params = {
    "neurons": 100000,
    "l2": 100,
}

class ELM():
    def __init__(self, neurons, l2):
        self.neurons = neurons
        self.l2 = l2

    @timeit
    def fit(self, x, y):
        y = y["y_ph"]

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
        y_true = y["y_ph"].values

        # compute the output of the hidden layer
        H = expit(x @ self.W + self.b)

        # predict
        y_pred = self.model.predict(H)

        return y_true, y_pred


