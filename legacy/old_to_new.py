import misc
import numpy as np
from evaluation.results import Results
import os


def old_to_new(model_name, ph):
    """
    Convert the all results (numpy array) into current results (numpy array) that can be read by the Results object
    :param model_name: name of the model (e.g., "ELM")
    :param ph: prediction horizon (e.g., 30)
    :return: /
    """
    n_preds_per_day = misc.day_len // misc.freq - misc.hist // misc.freq - ph // misc.freq
    datasets_subjects = np.array(np.concatenate([[[k, v_] for v_ in v] for k, v in misc.datasets_subjects_dict.items()],
                                                axis=0)).reshape(-1, 2)
    for dataset, subject in datasets_subjects:
        file_name = model_name + "_ph-" + str(ph) + "_" + dataset + "_subject" + subject + "_results.npy"
        res = np.load(os.path.join("results", "ph" + str(ph), model_name, file_name), allow_pickle=True)
        n_days = [res_.shape[0] // n_preds_per_day for res_ in res]
        res = [res_.reshape(n_days_, -1, 2) for res_, n_days_ in zip(res, n_days)]
        res_obj = Results(model_name, ph, dataset, subject, misc.freq, results=res)
        res_obj.save()
