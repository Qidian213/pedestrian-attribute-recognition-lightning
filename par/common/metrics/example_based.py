import numpy as np


def example_based_metrics(predictions, labels):
    p = np.sum((labels == 1), axis=1)
    predicted_p = np.sum((predictions == 1), axis=1)

    intersection = np.sum((labels * predictions > 0), axis=1)
    union = np.sum((labels + predictions > 0), axis=1)

    accuracy = np.mean(intersection / (union + 1e-20))
    precision = np.mean(intersection / (predicted_p + 1e-20))
    recall = np.mean(intersection / (p + 1e-20))
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1
