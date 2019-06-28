from postprocessing.rescaling import rescaling
from postprocessing.reshape import reshape_day


def postprocessing(results, hist, ph, freq, min, max):
    """
        Post-processing pipeline:
            For every split
            1. scale the data back with min and max
            2. reshape the predictions
        :param results:
        :param hist:
        :param ph:
        :param freq:
        :param min: list of min (one vector per split)
        :param max: list of max (one vector per split)
        :return:
    """
    new_results = []
    for results_split, min_split, max_split in zip(results, min, max):
        # undo min-max normalization
        new_results_split = rescaling(results_split, min_split, max_split)

        # reshape the predictions into days
        new_results_split = reshape_day(new_results_split, hist, ph, freq)

        new_results.append(new_results_split)

    return new_results
