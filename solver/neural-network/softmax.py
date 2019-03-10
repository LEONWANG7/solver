import numpy as np


def softmax(L):
    """
    用于多类别分类。
    :param L:
    :return:
    """
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i * 1.0 / sumExpL)
    return result
