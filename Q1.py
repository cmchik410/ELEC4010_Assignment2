from sys import float_repr_style
import yaml
import json
import argparse
import torch

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary

from utils.train1 import training
from utils.test1 import testing
from net.resnet import ResNet50
from utils.data_utils import ISBI2016
from metrics.metrics import error


def main(**kwarg):
    root_dir = kwargs["root_dir"]
    train_dir = kwargs["train_dir"]
    train_label_file = kwargs["train_label_file"]
    test_dir = kwargs["test_dir"]
    test_label_file = kwargs["test_label_file"]
    dimensions = kwargs["dimensions"]
    channels = kwargs["channels"]
    n_classes = kwargs["n_classes"]
    batch_size = kwargs["batch_size"]
    epochs = kwargs["epochs"]
    lr = kwargs["learning_rate"]
    momentum = kwargs["momentum"]
    save_model_path = kwargs["save_model_path"]

    img_shape = (channels, ) + tuple(dimensions)

    label_dict = {"benign" : 0, "malignant" : 1}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = 0.5, std = 0.5)
    ])

    ISBI_train = ISBI2016(root_dir = root_dir,
                          split = train_dir,
                          csv_file = train_label_file,
                          balanced = False,
                          label_dict = label_dict,
                          transform = transform
                          )
    
    ISBI_test = ISBI2016(root_dir = root_dir,
                          split = test_dir,
                          csv_file = test_label_file,
                          balanced = False,
                          label_dict = None,
                          transform = transform
                          )

    train_loader = DataLoader(dataset = ISBI_train,
                              batch_size = batch_size,
                              shuffle = True)
    
    test_loader = DataLoader(dataset = ISBI_test,
                             batch_size = batch_size,
                             shuffle = True)
    
    model = ResNet50(channels, n_classes)
    #summary(model, input_data = img_shape)
    weights = torch.tensor([0.3, 0.7])
    loss_fcn = nn.CrossEntropyLoss(weight = weights)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.9, patience = 2, verbose = True)
    metrics = error

    model, history = training(model, train_loader, test_loader, loss_fcn, optimizer, scheduler, epochs, metrics)

    torch.save(model.state_dict(), save_model_path)

    with open("Q1_history.txt", 'w') as fp:
        json.dump(history, fp)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "ELEC4010 Assigment 2 Q1")

    parser.add_argument("--cfg", default = "configs/Q1.yaml", help = "path to Q1 config file", type = str)

    args = parser.parse_args()

    with open(args.cfg, "r") as fp:
        kwargs = yaml.load(fp, Loader = yaml.FullLoader)

    main(**kwargs)