import numpy as np


def powell(x):
    return sum([abs(x[i]) ** (i + 1) for i in range(len(x))])


def xinshe(x):
    ss = np.sum(np.square(np.sin(x))) - np.exp(-np.sum(np.square(x)))
    return ss * np.exp(-np.sum(np.square(np.sin(np.sqrt(np.abs(x))))))
