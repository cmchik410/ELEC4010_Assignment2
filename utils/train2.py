import numpy as np
from tqdm import tqdm

from utils.test2 import testing

def training(model, train_loader, test_loader, batch_size, n_classes, 
             loss_fcn, optimizer, epochs, metrics, encoder):
    
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
        
        temp_acc_dict = {"dice" : 0, "jaccard" : 0, "asd" : 0, "hd" : 0}   

        for i, data in enumerate(progBar, start = 1):
            X_batch, y_true = data["image"], data["label"]

            optimizer.zero_grad()

            y_pred = model(X_batch)
            
            y_true = encoder(y_true)
            
            loss = loss_fcn(y_pred, y_true)

            loss.backward()

            optimizer.step()
            
            train_loss = (train_loss * (i - 1) + loss.item()) / i
            
            y_pred = y_pred.detach().numpy()
            y_pred = np.argmax(y_pred, axis = 1)
            y_true = y_true.detach().numpy()
            
            acc1 = metrics[0](y_true, y_pred)
            acc2 = metrics[1](y_true, y_pred)
            acc3 = metrics[2](y_true, y_pred)
            acc4 = metrics[3](y_true, y_pred)
        
            temp_acc_dict["dice"] = (temp_acc_dict["dice"] * (i - 1) + acc1) / i
            temp_acc_dict["jaccard"] = (temp_acc_dict["jaccard"] * (i - 1) + acc2) / i
            temp_acc_dict["asd"] = (temp_acc_dict["asd"] * (i - 1) + acc3) / i
            temp_acc_dict["hd"] = (temp_acc_dict["hd"] * (i - 1) + acc4) / i
                
            progBar.set_description("Epoch [%d / %d]" % (epoch + 1, epochs))
        
            progBar.set_postfix(train_loss = train_loss, dice_loss = temp_acc_dict["dice"], 
                                jaccard = temp_acc_dict["jaccard"], asd = temp_acc_dict["asd"], hd = temp_acc_dict["hd"])
            
        test_loss, test_acc = testing(model, test_loader, n_classes, loss_fcn, metrics, encoder)
        
        train_loss_dict[epoch] = train_loss
        train_acc_dict[epoch] = temp_acc_dict
        test_loss_dict[epoch] = test_loss
        test_acc_dict[epoch] = test_acc
        
        print()   
        
    history["train_loss"] = train_loss_dict
    history["train_acc"] = train_acc_dict
    history["test_loss"] = test_loss_dict
    history["test_acc"] = test_acc_dict 

    model.train(False)
    
    return model, history