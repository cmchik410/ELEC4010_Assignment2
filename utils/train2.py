import numpy as np
from tqdm import tqdm

from utils.test2 import testing

def training(model, train_loader, test_loader, loss_fcn, optimizer, epochs, metrics):
    
    print("\n Training \n" + "=" * 100)
    
    train_loss_dict = {}
    train_acc_dict = {}
    test_loss_dict = {}
    test_acc_dict = {}
        
    history = {}
    
    for epoch in range(epochs):
        model.train(True)
        
        progBar = tqdm(train_loader)
        
        train_loss = 0
        
        acc_dict = {"dice" : 0, "jaccard" : 0, "asd" : 0, "hd95" : 0}   

        for i, data in enumerate(progBar, start = 1):
            X_batch, y_true = data["image"], data["label"]

            optimizer.zero_grad()

            y_pred = model(X_batch)
            
            loss = loss_fcn(y_pred, y_true)

            loss.backward()

            optimizer.step()
            
            train_loss = (train_loss * (i - 1) + loss.item()) / i
            
            y_pred = y_pred.detach().numpy()
            y_true = y_true.detach().numpy()
            
            acc0 = metrics[0](y_pred, y_true)
            acc1 = metrics[1](y_pred, y_true)
            acc2 = metrics[2](y_pred, y_true)
            acc3 = metrics[3](y_pred, y_true)
        
            acc_dict["dice"] = (acc_dict["dice"] * (i - 1) + acc0) / i
            acc_dict["jaccard"] = (acc_dict["jaccard"] * (i - 1) + acc1) / i
            acc_dict["asd"] = (acc_dict["asd"] * (i - 1) + acc2) / i
            acc_dict["hd95"] = (acc_dict["hd95"] * (i - 1) + acc3) / i
                
            progBar.set_description("Epoch [%d / %d]" % (epoch + 1, epochs))
        
            progBar.set_postfix(train_loss = train_loss, dice = acc_dict["dice"], jaccard = acc_dict["jaccard"],
                                ASD = acc_dict["asd"], HD95 = acc_dict["hd95"])
            
        test_loss, test_acc = testing(model, test_loader, loss_fcn, metrics)
        
        train_loss_dict[epoch] = train_loss
        train_acc_dict[epoch] = acc_dict
        test_loss_dict[epoch] = test_loss
        test_acc_dict[epoch] = test_acc
        
        print()   
        
    history["train_loss"] = train_loss_dict
    history["train_acc"] = train_acc_dict
    history["test_loss"] = test_loss_dict
    history["test_acc"] = test_acc_dict 

    model.train(False)
    
    return model, history