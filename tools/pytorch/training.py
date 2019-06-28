import torch
import numpy as np
from torch.utils.data import DataLoader
from tools.pytorch.early_stopping import EarlyStopping
from tools.printd import printd


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, batch_size, model, loss_func, opt, train_ds, valid_ds, patience, scheduler_params, checkpoint_file):
    # create the dataloaders (using the batch_size) from the datasets
    train_dl, valid_dl = ds_to_dl(train_ds, valid_ds, batch_size)

    # instantiate early stopping
    early_stopping = EarlyStopping(patience=patience,
                                   path=checkpoint_file)  # default patience= 7

    if scheduler_params is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=scheduler_params["factor"],
                                                               verbose=True, patience=scheduler_params["patience"])
    else:
        scheduler = None

    for epoch in range(epochs):
        # activate training and train
        model.train()
        zip(*[loss_batch(model, loss_func, xb, yb, opt) for xb, yb in train_dl])

        # activate evaluation and evaluate
        model.eval()
        early_stopping, res = evaluate(epoch, early_stopping, model, loss_func, [train_dl, valid_dl])

        if scheduler is not None: scheduler.step(res[-1])

        # if there is no improvement, we earlystop
        if early_stopping.early_stop:
            printd("Early Stopped.")
            break


def evaluate(epoch, early_stopping, model, loss_func, dls):
    dls_names = ["[train]", "[valid]"]
    # do not apply gradient because we do not train
    with torch.no_grad():
        loss = []
        # for both training and validation dataloaders
        for dl, name in zip(dls, dls_names):
            # compute the loss
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in dl])
            loss_dl = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            loss.append(loss_dl)

            # if the dataloader is the validation set: check if we earlystop
            if name == "[valid]":
                early_stopping(loss_dl, model)

    # concat the results and print
    res = np.r_[epoch, np.c_[dls_names, loss].ravel()]
    printd(*res)

    return early_stopping, res


def predict(model, checkpoint, test_ds):
    model_checkpoint = torch.load(checkpoint)
    model.load_state_dict(model_checkpoint)
    dl = DataLoader(test_ds, batch_size=len(test_ds))

    preds = model(dl.dataset.tensors[0]).cpu().detach().numpy()[:, -1].reshape(-1, 1)
    trues = dl.dataset.tensors[1].cpu().numpy()[:, -1].reshape(-1, 1)

    return trues, preds


def ds_to_dl(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
