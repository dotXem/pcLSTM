import sklearn.model_selection as sk_model_selection
import numpy as np
import misc
from tools.timeit import timeit


# @timeit
def split(data, params):
    cv = params["cv"]

    train, valid, test = [], [], []

    kf_1 = sk_model_selection.KFold(n_splits=cv, shuffle=True, random_state=misc.seed)
    for train_valid_index, test_index in kf_1.split(np.arange(len(data))):
        train_valid_fold = [data[i] for i in train_valid_index]
        test_fold = [data[i].copy() for i in test_index]
        kf_2 = sk_model_selection.KFold(n_splits=cv - 1, shuffle=False)  # we already shuffled once
        for train_index, valid_index in kf_2.split(train_valid_fold):
            train_fold = [train_valid_fold[i].copy() for i in train_index]
            valid_fold = [train_valid_fold[i].copy() for i in valid_index]
            train.append(train_fold)
            valid.append(valid_fold)
            test.append(test_fold)

    return train, valid, test