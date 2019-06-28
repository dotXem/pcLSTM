def rescaling(results, min, max):
    """
        Scale back the results that have previously been mix-max-normalized.
        :param results: results of shape (None, 2)
        :param min: vector of max values (one per initial feature)
        :param max: vector of max values (one per initial feature)
        :return: rescaled results
    """
    min_y, max_y = min[-1], max[-1]
    return (results + 1) / 2 * (max_y - min_y) + min_y
