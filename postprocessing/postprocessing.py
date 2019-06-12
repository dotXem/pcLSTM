from postprocessing.rescaling import rescaling
from postprocessing.reshape import reshape_day


def postprocessing(results, hist, ph, freq, min, max, dataset):
    new_results = []
    for results_split, min_split, max_split in zip(results, min, max):
        # undo min-max normalization
        new_results_split = rescaling(results_split, min_split, max_split)

        # reshape the predictions into days
        new_results_split = reshape_day(new_results_split, hist, ph, freq, dataset)

        new_results.append(new_results_split)

    return new_results
