from torch.utils.data import TensorDataset
import os
import torch.nn as nn
from misc import *
import torch
from tools.pytorch.training import fit, predict
import misc
from tools.timeit import timeit
from tools.pytorch.loss_functions import pcMSE
from tools.printd import printd

params = {
    "n_in": 3,
    "n_out": 2,
    "n_hidden": 1,
    "n_neurons": 128,

    # training
    "loss": "pcMSE",
    "dropout": 0.0,
    "l2": 1e-4,
    "epochs": 1000,
    "batch_size": 10,
    "lr": 5e-3,
    "patience": 12,

    "c": 2e0,  # coherence factor

}


class pcLSTM():
    """
    The pcLSTM model is based on the Prediction-Coherent LSTM-based Recurrent Neural Network.
    Parameters:
        - n_in: number of features per time-step
        - n_out: number of outputs (shall be 2)
        - n_hidden: number of hidden layers
        - n_neurons: number of neurons per hidden layer
        - loss: loss function used (shall be "pcMSE")
        - dropout: dropout coefficient
        - l2: L2 penalty applied to the weights
        - epochs: maximal number of training epochs
        - batch_size: size of the mini-batch
        - lr: initial learning rate
        - patience: number of non-improving epochs after which we early stop the training
        - c: coherence factor, weights the importance of the predicted variation accuracy in the cMSE loss
        function
    """

    def __init__(self, params):
        self.n_in = params["n_in"]
        self.n_out = params["n_out"]
        self.n_hidden = params["n_hidden"]
        self.n_neurons = params["n_neurons"]
        self.dropout = params["dropout"]
        self.loss = params["loss"]
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]
        self.lr = params["lr"]
        self.l2 = params["l2"]
        self.patience = params["patience"]
        self.c = params["c"]

        if params["loss"] == "pcMSE":
            self.loss = pcMSE(self.c)
        elif params["loss"] == "MSE":
            self.loss = nn.MSELoss()

        self.model = torchpcLSTM(self.n_in, self.n_out, self.n_hidden, self.n_neurons, self.dropout)
        self.model.cuda()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)

        printd("is CUDA available ?", torch.cuda.is_available())

    @timeit
    def fit(self, x_train, y_train, x_valid, y_valid):
        # convert x and y into datasets
        train_ds = self.to_dataset(x_train, y_train)
        valid_ds = self.to_dataset(x_valid, y_valid)

        # create the checkpoint file
        self.checkpoint_file = os.path.join(misc.path, "tmp", "checkpoints", "pcLSTM_ph-" + str(ph) + "_checkpoint.pt")

        # train the model, resulting model saved in self.checkpoint_file
        fit(self.epochs, self.batch_size, self.model, self.loss, self.opt, train_ds, valid_ds, self.patience,
            None, self.checkpoint_file)

    @timeit
    def predict(self, x, y):
        # convert x and y to datasets
        ds = self.to_dataset(x, y)

        y_true, y_pred = predict(self.model, self.checkpoint_file, ds)

        return y_true, y_pred

    def to_dataset(self, x, y):
        x = x.values.reshape(-1, len(x.columns) // self.n_in, self.n_in)
        y = y.values.reshape(-1, 2)

        x = torch.Tensor(x).cuda()
        y = torch.Tensor(y).cuda()

        return TensorDataset(x, y)


class torchpcLSTM(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, n_neurons, dropout):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.dropout = dropout

        # batch_first=True => input/ouput w/ shape (batch,seq,feature)
        self.lstm = nn.LSTM(self.n_in, self.n_neurons, self.n_hidden, dropout=self.dropout, batch_first=True)

        self.linear = nn.Linear(self.n_neurons, 1)

    def forward(self, xb):
        xb, _ = self.lstm(xb)
        xb = self.linear(xb[:, -self.n_out:, :])
        return xb
