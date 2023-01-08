import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv_kernals = [6,64,64,64,64,128,128,128,128]

    def block(self, in_channels, out_channels):
        itr_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        return itr_block

    def fc(self):
        fc_block = nn.Sequential(
            nn.Flatten(),
            nn.AdaptiveAvgPool1d(512),
            nn.Dropout(0.5),
            nn.Linear(512,1024),
            nn.Linear(1024,8),
        )
        return fc_block
    
    def conv_block(self):       
        lst = []
        counter = 0
        for i in range(len(self.conv_kernals)-1):
            itr_block = self.block(self.conv_kernals[i],self.conv_kernals[i+1])
            lst.append(itr_block)
            counter+=1
            if counter%2==0:
                lst.append(nn.MaxPool2d(2,stride=2))
        conv_layers = nn.Sequential(*lst)
        return conv_layers

    def model(self):
        conv_layers = self.conv_block().to(device)
        fc_layers = self.fc().to(device)
        m = nn.Sequential(*[conv_layers,fc_layers])
        return m