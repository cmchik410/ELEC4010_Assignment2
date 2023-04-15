import torch

import numpy as np

def error(y_true, y_pred):
    temp1 = torch.argmax(y_pred, axis = -1)
    #temp2 = torch.argmax(y_true, axis = -1)
    temp2 = y_true
    return torch.mean((temp1 == temp2) / 1.)

