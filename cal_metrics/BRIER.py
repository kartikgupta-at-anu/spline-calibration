import numpy as np
from sklearn.metrics import mean_squared_error


def brier_score(y_pred, y):
    loss = mean_squared_error(y, y_pred)
    return loss


def top1_brier_score(score, acc):
    loss = np.mean((score - acc)**2)
    return loss