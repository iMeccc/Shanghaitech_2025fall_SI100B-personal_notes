#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: Zhenghao Li
# Date: 2024-11-08

import torch
import my_net

batchsize    = 48
#If you have a PC
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#If you have a Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Training Device: ", device)

# Load dataset
train_loader, classes= my_net.utility.loadTrain("./images_bonus", batchsize)

# Set model, lossfunc and optimizer
model,lossfun,optimizer = my_net.classify.makeEmotionNet(False)

# Start training process
epochs = 100
losses, accuracy, _ = my_net.utility.function2trainModel(model, device, train_loader, lossfun, optimizer, epochs)

print("--------------------------")
print("Loss and accuracy in every iteration")
for i, (loss, acc) in enumerate(zip(losses, accuracy)):
    print(f"Iteration {i}, loss：{loss:.2f}, accuracy: {acc:.2f}")

# Plot and save training curves
save_path='./training_curves_7class_lr05.png'
my_net.utility.plot_training_curves(losses, accuracy, save_path)

PATH = './face_expression_7class_lr05.pth'
torch.save(model.state_dict(), PATH)