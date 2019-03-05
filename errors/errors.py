def get_mean_squared_error(y1, y2):
    for x, y in zip(y1, y2):
        sum += (x - y) * (x - y)
    return sum / len(y1)
