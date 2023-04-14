import yaml
import json
import argparse
import torch

from torch import nn
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import DataLoader

from utils.data_utils import MRI
from utils.encoder import Q2_encoder
from utils.train2 import training
from net.unet import uNet

from medpy.metric.binary import dc
from medpy.metric.binary import jc, asd, hd

def main(**kwargs):
    root_dir = kwargs["root_dir"]
    train_csv_file = kwargs["train_csv_file"]
    test_csv_file = kwargs["test_csv_file"]
    dimensions = kwargs["dimensions"]
    channels = kwargs["channels"]
    n_classes = kwargs["n_classes"]
    loss_name = kwargs["loss_fcn"]
    epochs = kwargs["epochs"]
    batch_size = kwargs["batch_size"]
    lr = kwargs["learning_rate"]
    momentum = kwargs["momentum"]
    save_model_path = kwargs["save_model_path"]

    img_shape = (channels, ) + tuple(dimensions)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(dimensions, antialias = True),
        transforms.Normalize(mean = 0.5, std = 0.5)
    ])
    
    MRI_train = MRI(root_dir = root_dir, csv_file = train_csv_file, transform = transform)
    
    MRI_test = MRI(root_dir = root_dir, csv_file = test_csv_file, transform = transform)

    train_loader = DataLoader(dataset = MRI_train, batch_size = batch_size, shuffle = True)

    test_loader = DataLoader(dataset = MRI_test, batch_size = batch_size, shuffle = True)

    model = uNet(channels, n_classes)
    #summary(model, input_data = img_shape)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    metrics = [dc, jc, asd, hd] 
    encoder = Q2_encoder

    model, history = training(model, train_loader, test_loader, batch_size, n_classes, 
                                          loss_fcn, optimizer, epochs, metrics, encoder) 
    
    torch.save(model.state_dict(), save_model_path)

    with open("Q2_history.txt", 'w') as fp:
        json.dump(history, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "ELEC4010 Assigment 2 Q2")

    parser.add_argument("--cfg", default = "configs/Q2_cross.yaml", help = "path to Q2 config file", type = str)

    args = parser.parse_args()

    with open(args.cfg, "r") as fp:
        kwargs = yaml.load(fp, Loader = yaml.FullLoader)

    main(**kwargs)
