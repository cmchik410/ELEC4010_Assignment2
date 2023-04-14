from torch import nn

class convBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expansion):
        super(convBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, stride, "valid")
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 1, 1, "valid")
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.conv3 = nn.Conv2d(out_ch, out_ch * expansion, 1, 1, "valid")
        self.bn3 = nn.BatchNorm2d(out_ch * expansion)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        return x
    

class resBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, repeat, expansion):
        super(resBlock, self).__init__()
        
        self.repeat = repeat
        
        self.up = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch * expansion, 1, stride, "valid"),
                        nn.BatchNorm2d(out_ch * expansion)
        )
        
        self.layer1 = convBlock(in_ch, out_ch, stride, expansion)

        self.layer2 = convBlock(out_ch * expansion, out_ch, 1, expansion)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        flag = False
        for i in range(self.repeat):
            identity = x.clone()
            if not flag:
                x = self.layer1(x)
                identity = self.up(identity)
                flag = True
            else:
                x = self.layer2(x)
            
            x += identity

        return self.relu(x)
    

class ResNet50(nn.Module):
    def __init__(self, in_ch, n_classes):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(3, 2, 1)

        self.resBlock1 = resBlock(64, 64, 1, 3, 4)
        self.resBlock2 = resBlock(256, 128, 2, 4, 4)
        self.resBlock3 = resBlock(512, 256, 2, 6, 4)
        self.resBlock4 = resBlock(1024, 512, 2, 3, 4)

        self.avgPool1 = nn.AvgPool2d(7)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, n_classes)
        
        self.act = nn.Softmax()

    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxPool1(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
        x = self.avgPool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        
        return self.act(x)
