#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: Zhenghao Li
# Date: 2024-11-08

import torch
import my_net

batchsize    = 48
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#If you have a Mac
#device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
print("Training Device: ", device)

# Load dataset
train_loader, classes= my_net.utility.loadTrain("./images", batchsize)

# Set model, lossfunc and optimizer
model,lossfun,optimizer = my_net.classify.makeEmotionNet(False)

# Start training process
epochs = 75
losses, accuracy, _ = my_net.utility.function2trainModel(model, device, train_loader, lossfun, optimizer, epochs)

print("--------------------------")
print("Loss and accuracy in every iteration")
for i, (loss, acc) in enumerate(zip(losses, accuracy)):
    print(f"Iteration {i}, loss：{loss:.2f}, accuracy: {acc:.2f}")

# Plot and save training curves
my_net.utility.plot_training_curves(losses, accuracy, save_path='./training_curves.png')

PATH = './face_expression_excel.pth'
torch.save(model.state_dict(), PATH)