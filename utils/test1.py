from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score
from medpy.metric.binary import sensitivity, specificity
from torchmetrics.classification import AUROC

def testing(model, data_loader, loss_fcn, metrics, device):
    
    model.train(False)
    
    progBar = tqdm(data_loader, nrows = 4)
    
    test_loss = 0
    test_acc = 0
    AUC = 0

    auroc = AUROC(task = "multiclass", num_classes = 2)
    
    for i, data in enumerate(progBar, start = 1):
        X_batch, y_true = data["image"].to(device), data["label"].reshape(-1).to(device)
        
        y_pred = model(X_batch)
        
        loss = loss_fcn(y_pred, y_true)
        
        acc = metrics(y_true, y_pred)

        y_pred = y_pred.detach().cpu()
        y_true = y_true.detach().cpu()

        running_auc = auroc(y_pred, y_true)
        
        test_loss = (test_loss * (i - 1) + loss.item()) / i
        test_acc = (test_acc * (i - 1) + acc.item()) / i
        AUC = (AUC * (i - 1) + running_auc.item()) / i
        
        progBar.set_description("Testing: ")
        progBar.set_postfix(test_loss = test_loss, test_acc = test_acc, auc = AUC)

    return test_loss, test_acc, AUC
