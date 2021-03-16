import numpy as np
from sklearn.metrics import log_loss


def nll(y_pred, y):
    loss = log_loss(y, y_pred)
    return loss
