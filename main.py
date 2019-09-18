import sys
import argparse
import misc
import os
from preprocessing.preprocessing import preprocessing
from postprocessing.postprocessing import postprocessing
from tools.printd import printd
import numpy as np
from evaluation.results import Results
from pydoc import locate
from tools.compute_subjects_list import compute_subjects_list


def main(dataset, subject, Model, params, ph, eval="valid", print=True, plot=False, save=True, excel_file=None):
    printd(dataset, subject, Model.__name__)

    file = os.path.join("data", "dynavolt", dataset, dataset + "_subject" + subject + ".csv")

    """ PREPROCESSING """
    train_sets, valid_sets, test_sets, norm_min, norm_max = preprocessing(file, misc.hist, ph, misc.freq, misc.cv)

    """ CROSS-VALIDATION """
    results = []
    for i, [train, valid, test] in enumerate(zip(train_sets, valid_sets, test_sets)):
        train_x, train_y = train.iloc[:, :-2], train.iloc[:, -2:]
        valid_x, valid_y = valid.iloc[:, :-2], valid.iloc[:, -2:]
        test_x, test_y = test.iloc[:, :-2], test.iloc[:, -2:]

        model = Model(params)
        if Model.__name__ in misc.nn_models:
            model.fit(x_train=train_x, y_train=train_y, x_valid=valid_x, y_valid=valid_y)
        else:
            model.fit(x=train_x, y=train_y)

        if eval == "valid":
            y_true, y_pred = model.predict(x=valid_x, y=valid_y)
        elif eval == "test":
            y_true, y_pred = model.predict(x=test_x, y=test_y)
        results.append(np.c_[y_true, y_pred])

    """ POST-PROCESSING """
    results = postprocessing(results.copy(),
                             hist=misc.hist,
                             ph=misc.ph,
                             freq=misc.freq,
                             min=norm_min,
                             max=norm_max)

    """ EVALUATION """
    res = Results(Model.__name__, misc.ph, dataset, subject, misc.freq, results=np.array(results))
    metrics = res.get_results()
    if print: printd(metrics)
    if save: res.save()
    if plot: res.plot()
    if excel_file is not None: res.to_excel(params, len(res.results), file_name=excel_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--ph", type=int)
    parser.add_argument("--eval", type=str)
    parser.add_argument("--excel", type=str)
    parser.add_argument("--print", type=bool)
    parser.add_argument("--plot", type=bool)
    parser.add_argument("--save", type=bool)
    args = parser.parse_args()

    model_name = args.model if args.model is not None else sys.exit(-1)
    ph = args.ph if args.ph is not None else 30
    eval = args.eval if args.eval is not None else "valid"
    excel = args.excel if args.excel is not None else None
    print = args.print if args.print is not None else True
    plot = args.plot if args.plot is not None else False
    save = args.save if args.save is not None else False

    Model = locate("models." + model_name + "." + model_name)
    params = locate("models." + model_name + ".params")

    datasets_subjects = compute_subjects_list(args.dataset, args.subject)
    for dataset, subject in datasets_subjects:
        main(dataset=dataset, subject=subject, Model=Model, params=params, ph=ph, eval=eval)
