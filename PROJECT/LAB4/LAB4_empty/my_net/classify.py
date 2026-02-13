#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: Zhenghao Li
# Date: 2024-11-08

import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F

class emotionNet(nn.Module):
    def __init__(self, printtoggle):
        super().__init__()
        self.print = printtoggle
      
        ### write your codes here ###
        #############################
        # step1:
        # Define the functions you need: convolution, pooling, activation, and fully connected functions.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0) # 根据图示
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0) # 根据图示
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.LeakyReLU()

        self.fc1 = nn.Linear(in_features=4096, out_features=7) 
        self.relu4 = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.4)


    def forward(self, x):
        #Step 2
        # Using the functions your defined for forward propagate
        # First block
        # convolution -> maxpool -> relu
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # Second block
        # convolution -> maxpool -> relu
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # Third block
        # convolution -> maxpool -> relu
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        # Flatten for linear layers
        x = torch.flatten(x, start_dim=1)

        print(f"Shape after flatten: {x.shape}") # 这一行会打印出 (Batch_size, 4096)
        # fully connect layer
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        
        return x
'''
def makeEmotionNet(printtoggle=False):
    model = emotionNet(False)

    #L_{CE} loss function
    lossfun = nn.CrossEntropyLoss()
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = .0005, weight_decay=1e-5)

    return model, lossfun, optimizer
'''

def makeEmotionNet(printtoggle=False):
    #model = emotionNet()
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 7)
    #loss function
    lossfun = nn.CrossEntropyLoss()
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = .0005, weight_decay=1e-5)
    return model, lossfun, optimizer