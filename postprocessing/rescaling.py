def rescaling(results, min, max):
    min_y, max_y = min[-1], max[-1]
    return (results + 1) / 2 * (max_y - min_y) + min_y
