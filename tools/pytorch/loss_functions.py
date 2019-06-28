import torch.nn as nn
import torch


class pcMSE(nn.Module):
    def __init__(self, c):
        super(pcMSE, self).__init__()

        self.c = c

    def forward(self, x, y):
        # needed with batch norm
        x = x.reshape(-1, 2)

        mse = torch.mean((x[:, 1] - y[:, 1]) ** 2)
        dx = x[:, 1] - x[:, 0]
        dy = y[:, 1] - y[:, 0]
        dmse = torch.mean(((dx - dy)) ** 2)

        return mse + self.c * dmse
