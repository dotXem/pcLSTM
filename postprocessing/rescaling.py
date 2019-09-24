def rescaling(results, mean, std):
    """
        Scale back the results that have previously been standardized.
        :param results: results of shape (None, 2)
        :param mean: vector of mean values (one per initial feature)
        :param std: vector of std values (one per initial feature)
        :return: rescaled results
    """
    min_y, max_y = mean[-1], std[-1]

    return results * max_y + min_y