#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: Zhenghao Li
# Email: lizhenghao@shanghaitech.edu.cn
# Institute: SIST
# Date: 2024-12-10

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
# from classify import makeEmotionNet
import os

transform = {
        "training":
    transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),  # 随机水平翻转
     transforms.RandomRotation(10),  # 随机旋转 ±10度
     transforms.RandomCrop(44),  # 随机裁剪到44x44大小
     transforms.Resize((48, 48)),  # 调整回48x48大小
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "evaluate":
    transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}

#transform = transforms.Compose(
#    [
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def imshow_with_labels(images, labels, classes):
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15,15))
    axs = axs.flatten()

    for i in range(grid_size**2):
        if i< num_images:
            # img = to_pil_image(images[i])
            # extract that image (need to transpose it back to 32x32x3)
            img = images.data[i].numpy().transpose((1,2,0))
            img = img/2 + .5 # undo normalization
            label = classes[labels[i]]
            axs[i].imshow(img, cmap='gray')
            axs[i].set_title(f'Label: {label}')
            axs[i].axis('off')
        else:
            axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def load_data(path, batch_size, DataType):
    Dataset = ImageFolder(path, transform=transform[DataType])
    if batch_size == 0:
        batch_size = len(Dataset)
    Dataloader = DataLoader(Dataset ,batch_size=batch_size,shuffle=True, drop_last=True)
    return Dataset, Dataloader

def loadTrain(path, batch_size):
    train_path = os.path.join(path, "train")
    trainset, train_loader = load_data(train_path, batch_size, "training")
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # try again
    print('Data shapes (train/test):')
    print( images.data.shape )
    
    # and the range of pixel intensity values
    print('\nData value range:')
    print( (torch.min(images.data),torch.max(images.data)) )
    
    # Show images
    imshow_with_labels(images, labels, trainset.classes)

    return train_loader, trainset.classes

def loadTest(path, batch_size=0):
    test_path = os.path.join(path, "validation")
    testset, test_loader = load_data(test_path, batch_size, "evaluate")

    return test_loader, testset.classes

def function2trainModel(model, device, train_loader, lossFun, optimizer, epochs):
    #model,lossfun,optimizer = makeFaceNet()

    model.to(device)

    # initialize losses
    trainLoss = np.zeros(epochs)
    #devLoss   = torch.zeros(epochs)
    trainAcc  = np.zeros(epochs)
    #devAcc    = torch.zeros(epochs)

    for epochi in range(epochs):
        # loop over training data batches
        model.train() # switch to train mode
        batchLoss = []
        batchAcc = []
        for batch_idx, (X,y) in enumerate(train_loader):
            # push data to GPU
            X = X.to(device)
            y = y.to(device)
            # forward pass and loss
            yHat = model(X)
            loss = lossFun(yHat, y)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss and accuracy for this batch
            batchLoss.append(loss.item())
            batchAcc.append(torch.mean((torch.argmax(yHat, dim=1)==y).float()).item())
            print(f"Epoch: {epochi+1}/{epochs}, Batch: {batch_idx}, {batch_idx+1}/{len(train_loader)}")

        # end of batch loop
        # get average losses and accuracies across the batches
        trainLoss[epochi] = np.mean(batchLoss)
        trainAcc[epochi] = 100*np.mean(batchAcc)
    return trainLoss, trainAcc, model

def plot_training_curves(losses, accuracy, save_path):
    """
    绘制并保存 Loss 和 Accuracy 曲线
    """
    epochs = range(len(losses))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training process')

    # 绘制 Loss
    ax1.plot(epochs, losses, label='loss')
    ax1.set_title('loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # 绘制 Accuracy
    ax2.plot(epochs, accuracy, label='accuracy', color='red')
    ax2.set_title('accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap="Blues", save_path='./confusion_matrix.png'):
    """
    绘制并保存混淆矩阵
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    # 设置坐标轴标签
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    # 在每个格子里填上数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
        
    plt.show()

#trainLoss,trainAcc,model = function2trainModel()
