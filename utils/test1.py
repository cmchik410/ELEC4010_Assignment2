from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_curve, auc
from medpy.metric.binary import sensitivity, specificity

def testing(model, data_loader, loss_fcn, metrics):
    
    model.train(False)
    
    progBar = tqdm(data_loader, nrows = 4)
    
    test_loss = 0
    test_acc = 0
    sens = 0
    spec = 0
    
    all_pred = []
    all_true = []
    
    for i, data in enumerate(progBar, start = 1):
        X_batch, y_true = data["image"], data["label"].reshape(-1)
        
        y_pred = model(X_batch)
        
        loss = loss_fcn(y_pred, y_true)
        
        acc = metrics(y_true, y_pred)
        
        test_loss = (test_loss * (i - 1) + loss.item()) / i
        test_acc = (test_acc * (i - 1) + acc.item()) / i
        
        progBar.set_description("Testing: ")
        progBar.set_postfix(test_loss = test_loss, test_acc = test_acc)
        
        y_pred = y_pred.detach().numpy()
        y_pred = np.argmax(y_pred, axis = -1)
        y_true = y_true.detach().numpy()
        
        for i in y_pred:
            all_pred.append(i)
        
        for j in y_true:
            all_true.append(j)
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
        
    TP = np.sum((all_pred & all_true)) / 1.0
    TN = np.sum((1 - all_pred) & (1 - all_true)) / 1.0
    FP = np.sum((all_pred ^ all_true) * all_pred) / 1.0
    FN = np.sum((all_pred ^ all_true) * all_true) / 1.0

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    
    return test_loss, test_acc, TPR, FPR
