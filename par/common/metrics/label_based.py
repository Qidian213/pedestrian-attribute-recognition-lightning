import numpy as np


def compute_mean_accuracy(predictions, labels):
    p = np.sum((labels == 1), axis=0)
    n = np.sum((labels == 0), axis=0)
    tp = np.sum((labels == 1) * (predictions == 1), axis=0)
    tn = np.sum((labels == 0) * (predictions == 0), axis=0)

    acc_pos = tp / (p + 1e-20)
    acc_neg = tn / (n + 1e-20)
    mA = (acc_pos + acc_neg) / 2

    return mA
