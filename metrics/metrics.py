import torch

import numpy as np

def error(y_true, y_pred):
    temp1 = torch.argmax(y_pred, axis = -1)
    #temp2 = torch.argmax(y_true, axis = -1)
    temp2 = y_true
    return torch.mean((temp1 == temp2) / 1.)


def jaccard(y_pred, y_true, smooth = 100):
    y_pred = torch.argmax(y_pred, axis = 1).reshape(-1)
    y_true = y_true.reshape(-1)

    intersection = torch.sum(y_true * y_pred)
    
    #total_pix = y_pred.shape[0] + y_true.shape[0]
    
    total_pix = y_pred.sum() + y_true.sum()
    
    res =  (intersection) / (total_pix - intersection + 1e-6)
    
    return res
    

# def jaccard(y_pred, y_true, smooth = 100):
#     intersection = (y_true * y_pred).abs().sum(dim = -1)
#     sum_ = torch.sum(y_true.abs() + y_pred.abs(), dim = -1)
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     return (1 - jac) * smooth

def avg_surf_distance(y_pred, y_true):
    pass

def hausdorff_distance(y_pred, y_true):
    pass