#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: Zhenghao Li
# Date: 2024-11-08

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__)) # -> '.../PROJECT/LAB5'
project_dir = os.path.dirname(current_dir) # -> '.../PROJECT'
sys.path.insert(0, project_dir)
import torch
from LAB4.LAB4_empty import my_net
import numpy as np

batch_size=48
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model, lossfun, optimizer = my_net.classify.makeEmotionNet(False)
model_path = os.path.join(project_dir, 'LAB6', 'face_expression_excel.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
images_path = os.path.join(project_dir, 'LAB4', 'images')
test_loader, classes = my_net.utility.loadTest(images_path, batch_size)
X,y = next(iter(test_loader))

model.eval()
## Test in one batch
with torch.no_grad():
    X = X.to(device)
    yHat = model(X)

    ##Step 1 Obtain predicted labels
    new_labels = torch.argmax(yHat, dim=1)
    y_true = y.cpu().numpy()
    y_pred = new_labels.cpu().numpy()
    ##Show first 48 predicted labels
    my_net.utility.imshow_with_labels(X[:batch_size], new_labels[:batch_size], classes)

    #Step 2
    ##Calculate the accuracy for each category prediction, as well as the overall accuracy
    #Print them to the screen.
    ## "happy:xx.xx%, neutral:xx.xx%, sad:xx.xx%, total:xx.xx%"
    total_accuracy = np.mean(y_true == y_pred) * 100
    class_correct = {i: 0 for i in range(len(classes))}
    class_total = {i: 0 for i in range(len(classes))}
    for i in range(len(y_true)):
        label = y_true[i]
        pred = y_pred[i]
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1

    acc_str_parts = []
    for i, class_name in enumerate(classes):
        # 使用三元运算符避免除以零的错误
        accuracy = (class_correct[i] / class_total[i]) * 100 if class_total[i] > 0 else 0
        acc_str_parts.append(f"{class_name}:{accuracy:.2f}%")

    print(f"{', '.join(acc_str_parts)}, total:{total_accuracy:.2f}%")
    #Step 3
    ##Calculate the recall for each category prediction, as well as the overall recall
    #Print them to the screen.
    ## "happy:xx.xx%, neutral:xx.xx%, sad:xx.xx%, total:xx.xx%"
    recall_str_parts = []
    for i, class_name in enumerate(classes):
        # Recall = TP / (TP + FN)
        tp = class_correct[i]
        tp_plus_fn = class_total[i]
        recall = (tp / tp_plus_fn) * 100 if tp_plus_fn > 0 else 0
        recall_str_parts.append(f"{class_name}:{recall:.2f}%")
    total_recall = np.mean([ (class_correct[i] / class_total[i]) if class_total[i] > 0 else 0 for i in range(len(classes)) ]) * 100
    print(f"{', '.join(recall_str_parts)}, total:{total_recall:.2f}%")

## Get the accuracy and recall in full dataset
##Step 4
print("\n--- Evaluating on Full Dataset ---")

total_samples = 0
total_correct = 0
# 创建一个混淆矩阵来累加所有结果
num_classes = len(classes)
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

# 确保模型在评估模式
model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        y_true_batch = y.cpu().numpy()
        
        yHat = model(X)
        y_pred_batch = torch.argmax(yHat, dim=1).cpu().numpy()

        total_samples += len(y_true_batch)
        total_correct += np.sum(y_true_batch == y_pred_batch)

        for i in range(len(y_true_batch)):
            confusion_matrix[y_true_batch[i], y_pred_batch[i]] += 1

# Plot and save confusion matrix
print("Confusion Matrix:")
print(confusion_matrix)
my_net.utility.plot_confusion_matrix(confusion_matrix, classes, save_path='./confusion_matrix.png')

overall_accuracy = (total_correct / total_samples) * 100
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

true_positives = np.diag(confusion_matrix)
false_negatives = np.sum(confusion_matrix, axis=1) - true_positives
denominators = true_positives + false_negatives
recall_per_class = np.divide(true_positives, denominators, out=np.zeros_like(true_positives, dtype=float), where=denominators!=0)
overall_recall_macro = np.mean(recall_per_class) * 100
print(f"Overall Recall: {overall_recall_macro:.2f}%")


print("\n--- Per-class Metrics ---")
true_positives = np.diag(confusion_matrix)
false_positives = np.sum(confusion_matrix, axis=0) - true_positives
false_negatives = np.sum(confusion_matrix, axis=1) - true_positives

for i, class_name in enumerate(classes):
    tp = true_positives[i]
    fp = false_positives[i]
    fn = false_negatives[i]
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
  
    print(f"Class: {class_name}")
    print(f"  Accuracy: {(tp + np.sum(confusion_matrix) - (fp + fn + tp)) / total_samples * 100:.2f}% ")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall: {recall * 100:.2f}%")

