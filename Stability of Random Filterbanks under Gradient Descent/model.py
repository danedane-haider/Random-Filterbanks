import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class TDFilterbank(torch.nn.Module):
    def __init__(self, spec, w):
        super().__init__()
        
        self.psi = torch.nn.Conv1d(
            in_channels=1,
            out_channels=spec["J"],
            kernel_size=spec["T"],
            stride=spec["stride"],
            padding=0,
            bias=False)
        
        self.spec = spec

        self.psi.weight.data = w[:, :spec["T"]].unsqueeze(1).float()        
    
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[-1])
        x = F.pad(x, (0, self.spec["T"]-1), mode='circular',)
        Wx = torch.abs(self.psi(x))
        # hann = torch.hann_window(spec["N"]//spec["stride"]).unsqueeze(0).unsqueeze(0)
        # phi = torch.ones(spec["J"], spec["J"], spec["N"]//spec["stride"])*hann
        # Ux = F.conv1d(Wx, phi, bias=None, stride=1, padding=0)

        Ux = F.avg_pool1d(Wx, kernel_size=self.spec["N"]//self.spec["stride"], stride=1)
        return Ux
    
class Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(32, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x
    
class ClassifierConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []
        dense_layers = []

        # First Convolution Block with Sigmoid and BatchNorm
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(10, 5), stride=2, padding=0)
        self.sig1 = nn.Sigmoid()
        self.bn1 = nn.BatchNorm2d(8)
        conv_layers += [self.conv1, self.sig1, self.bn1]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=2, padding=0)
        self.sig2 = nn.Sigmoid()
        self.bn3 = nn.BatchNorm2d(8)
        conv_layers += [self.conv3, self.sig2, self.bn3]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d((16,64))
        self.lin = nn.Linear(in_features=8*16*64, out_features=1024)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.lin2 = nn.Linear(in_features=1024, out_features=14)
        # softmax
        self.softmax = nn.Softmax(dim=1)
        dense_layers += [self.lin, self.relu, self.drop, self.lin2, self.softmax]

        # Wrap the Blocks
        self.conv = nn.Sequential(*conv_layers)
        self.dense = nn.Sequential(*dense_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)
        # Adaptive pool and flatten for input to Dense layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        # Run the Dense layer
        x = self.dense(x)
        return x