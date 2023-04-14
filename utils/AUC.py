import numpy as np
import torch

def confusion_matrix(y_true, y_pred):
    y_true = y_true.detach().numpy().astype(int).reshape(-1)
    y_pred = torch.sum(y_pred, axis = -1)
    y_pred = y_pred.detach().numpy().astype(int).reshape(-1)
    
    TP = y_true & y_pred
    TN = 1 - (y_true | y_pred)
    FP = (y_true | y_pred) - y_true
    FN = (y_true | y_pred) - y_pred
    
    return np.sum(TP, axis = -1), np.sum(TN, axis = -1), np.sum(FP, axis = -1), np.sum(FN, axis = -1)

