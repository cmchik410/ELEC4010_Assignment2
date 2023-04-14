import torch
import numpy as np


def Q2_encoder(targets):
    N, C, H, W = targets.shape
    
    return targets.reshape(N, H, W)

def Q1_encoder(targets, n_classes):
    N = targets.shape[0]
    return torch.zeros(N, n_classes).scatter_(1, targets, 1)

# def Q2_encoder(targets, n_classes):
#     N, C, H, W = targets.shape

#     new_shape = (N, ) + (n_classes, ) + (H, ) + (W, )

#     targets = targets.reshape(-1)

#     temp = np.eye(n_classes)[targets]

#     temp = temp.reshape(new_shape)

#     return torch.Tensor(temp)