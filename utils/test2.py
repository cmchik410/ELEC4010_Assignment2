import numpy as np
from tqdm import tqdm


def testing(model, data_loader, loss_fcn, metrics, encoder, device):
    model.train(False)
    
    progBar = tqdm(data_loader)
    
    test_acc_dict = {"dice" : 0, "jaccard" : 0, "asd" : 0, "hd95" : 0}
    
    test_loss = 0  
    
    for i, data in enumerate(progBar, start = 1):
        X_batch, y_true = data["image"].to(device), data["label"].to(device)
        
        y_pred = model(X_batch)

        y_true = encoder(y_true)
        
        loss = loss_fcn(y_pred, y_true)
        
        y_pred = y_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis = 1)
        y_true = y_true.detach().cpu().numpy()

        y_pred = 1 - y_pred
        y_true = 1 - y_true
        
        acc0 = metrics[0](y_pred, y_true)
        acc1 = metrics[1](y_pred, y_true)
        acc2 = metrics[2](y_pred, y_true)
        acc3 = metrics[3](y_pred, y_true)
        
        test_loss = (test_loss * (i - 1) + loss.item()) / i
        test_acc_dict["dice"] = (test_acc_dict["dice"] * (i - 1) + acc0) / i
        test_acc_dict["jaccard"] = (test_acc_dict["jaccard"] * (i - 1) + acc1) / i
        test_acc_dict["asd"] = (test_acc_dict["asd"] * (i - 1) + acc2) / i
        test_acc_dict["hd95"] = (test_acc_dict["hd"] * (i - 1) + acc3) / i
        
        progBar.set_description("Testing: ")
        progBar.set_postfix(train_loss = test_loss, dice = test_acc_dict["dice"],
                            jaccard = test_acc_dict["jaccard"], asd = test_acc_dict["asd"], hd95 = test_acc_dict["hd95"])
        

    return test_loss, test_acc_dict