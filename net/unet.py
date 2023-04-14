import torch
from torch import nn

class downBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, "same")
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, "same")
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x
    

class upBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upBlock, self).__init__()

        self.convTran1 = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, "same")
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, "same")
   
        self.relu = nn.ReLU()

    def forward(self, prev, x):
        x = self.convTran1(x)
        x = self.relu(self.conv2(x))
        x = torch.cat([prev, x], 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x


class uNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        super(uNet, self).__init__()

        self.down1 = downBlock(in_ch, 64)
        self.down2 = downBlock(64, 128)
        self.down3 = downBlock(128, 256)
        self.down4 = downBlock(256, 512)
        
        self.down5 = downBlock(512, 1024)

        self.up1 = upBlock(1024, 512)
        self.up2 = upBlock(512, 256)
        self.up3 = upBlock(256, 128)
        self.up4 = upBlock(128, 64)

        self.conv1 = nn.Conv2d(64, 2, 3, 1, "same")
        self.conv2 = nn.Conv2d(2, n_classes, 3, 1, "same")

        self.maxPool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        c1 = self.down1(x)
        x = self.maxPool(c1)
        c2 = self.down2(x)
        x = self.maxPool(c2)
        c3 = self.down3(x)
        x = self.maxPool(c3)
        x = self.down4(x)
        c4 = self.drop(x)
        x = self.maxPool(c4)

        x = self.down5(x)
        x = self.drop(x)

        x = self.up1(c4, x)
        x = self.up2(c3, x)
        x = self.up3(c2, x)
        x = self.up4(c1, x)
        
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))

        return x
    