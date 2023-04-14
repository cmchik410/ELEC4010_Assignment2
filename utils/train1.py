from tqdm import tqdm
import copy
import torch
from utils.test1 import testing

def training(model, train_loader, test_loader, loss_fcn, optimizer, scheduler, epochs, metrics, device):
    
    print("\n Training \n" + "=" * 100)
    
    train_loss_dict = {}
    train_acc_dict = {}
    test_loss_dict = {}
    test_acc_dict = {}
    AUC_dict = {}
        
    history = {}
    
    best_acc = 0

    best_model = None
    
    for epoch in range(epochs):
        model.train(True)
        
        progBar = tqdm(train_loader, nrows = 4)
        
        train_loss = 0
        train_acc = 0     

        for i, data in enumerate(progBar, start = 1):
            X_batch, y_true = data["image"].to(device), data["label"].to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)
               
            loss = loss_fcn(y_pred, y_true)

            loss.backward()

            optimizer.step()
            
            acc = metrics(y_true, y_pred)
            
            train_loss = (train_loss * (i - 1) + loss.item()) / i
            train_acc = (train_acc * (i - 1) + acc.item()) / i
                
            progBar.set_description("Epoch [%d / %d]" % (epoch + 1, epochs))
        
            progBar.set_postfix(train_loss = train_loss, train_acc = train_acc)
        
        train_loss_dict[epoch] = train_loss
        train_acc_dict[epoch] = train_acc
            
        test_loss, test_acc, AUC = testing(model, test_loader, loss_fcn, metrics, device)
        
        test_loss_dict[epoch] = test_loss
        test_acc_dict[epoch] = test_acc
        AUC_dict[epoch] = AUC
        
        scheduler.step(test_loss)
        
        if test_acc > best_acc:
            best_acc = train_acc
            best_model = model
        else:
            model = best_model

        print()

    history["train_loss"] = train_loss_dict
    history["train_acc"] = train_acc_dict
    history["test_loss"] = test_loss_dict
    history["test_acc"] = test_acc_dict 
    history["AUC"] = AUC_dict
    
    return best_model, history
