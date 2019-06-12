import misc
import os
from preprocessing.preprocessing import preprocessing
from postprocessing.postprocessing import postprocessing
from models.ELM import ELM, params as ELM_params
import numpy as np
from evaluation.results import Results

dataset = "IDIAB"
subject = "1"
file = os.path.join("data", "dynavolt", dataset, dataset + "_subject" + subject + ".csv")

freq = misc.freqs[dataset]

params_preprocessing = {
    "reshape_day": {
        "ph": misc.ph,
        "hist": misc.hist,
        "dataset": dataset,
        "freq": freq,
    },
    "split": {
        "cv": misc.cv,
    },
    "filter": {
        "cutoff": 6e-4 / freq,
        "order": 4,
        "fs": 1 / (60 * freq),
    },
    "normalize": {
        "min": -1,
        "max": -1,
    },
    "reshape_samples": {
        "hist": misc.hist,
        "freq": freq,
    }
}


def main():
    """
    #TODO write the description
    :return:
    """

    """ PREPROCESSING """
    train_sets, valid_sets, test_sets, params = preprocessing(file, params_preprocessing)

    # TODO REMOVE - one split testing
    split_number = 0
    train_sets, valid_sets, test_sets = [train_sets[split_number]], [valid_sets[split_number]], [
        test_sets[split_number]]
    params["normalize"]["min"] = [params["normalize"]["min"][split_number]]
    params["normalize"]["max"] = [params["normalize"]["max"][split_number]]

    """ CROSS-VALIDATION """
    results = []
    for i, [train, valid, test] in enumerate(zip(train_sets, valid_sets, test_sets)):
        train_x, train_y = train.iloc[:, :-2], train.iloc[:, -2:]
        valid_x, valid_y = valid.iloc[:, :-2], valid.iloc[:, -2:]
        test_x, test_y = test.iloc[:, :-2], test.iloc[:, -2:]

        model = ELM(neurons=ELM_params["neurons"], l2=ELM_params["l2"])
        model.fit(x=train_x, y=train_y)

        y_true, y_pred = model.predict(x=valid_x, y=valid_y)
        results.append(np.c_[y_true, y_pred])

    """ POST-PROCESSING """
    results = postprocessing(results.copy(),
                             hist=misc.hist,
                             ph=misc.ph,
                             freq=freq,
                             min=params["normalize"]["min"],
                             max=params["normalize"]["max"],
                             dataset=dataset)

    """ EVALUATION """

    res = Results(results, freq)
    metrics = res.get_results()
    print(metrics)
    res.plot()


    pass


if __name__ == "__main__":
    """
    ArgParser with subject, model, ph
    """
    main()
