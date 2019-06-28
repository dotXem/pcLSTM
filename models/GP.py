from sklearn.gaussian_process.kernels import DotProduct, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from tools.timeit import timeit
import misc

params = {
    "alpha": 1e-2,
    "sigma_0": 1e-8,
}


class GP():
    """
    The GP model is based on Gaussian Processes and DotProduct kernel.
    Parameters:
        - alpha: noise added to the observations to ease the fitting of the model
        - sigma_0: inhomogeneity coefficient of the kernel
    """

    def __init__(self, params):
        self.alpha = params["alpha"]
        self.sigma_0 = params["sigma_0"]

    @timeit
    def fit(self, x, y):
        y = y["y_ph"]

        # define the GP kernel
        kernel = DotProduct(sigma_0=self.sigma_0)
        # kernel = RBF(length_scale=1e2)

        # define the model
        self.model = GaussianProcessRegressor(kernel=kernel,
                                              alpha=self.alpha,
                                              random_state=misc.seed)

        # fit the model
        self.model.fit(x, y)

    @timeit
    def predict(self, x, y):
        y_true = y["y_ph"].values

        # predict
        y_pred = self.model.predict(x)

        return y_true, y_pred
