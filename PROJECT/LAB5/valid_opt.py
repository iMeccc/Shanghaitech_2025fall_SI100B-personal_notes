import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 环境与路径设置 ---
# 使用 __file__ 确保路径的健壮性
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    sys.path.insert(0, project_dir)
    from LAB4.LAB4_empty.my_net import classify, utility
except (NameError, ImportError) as e:
    print("路径或导入错误，请确保文件结构正确。")
    print(f"Error details: {e}")
    sys.exit(1)


# --- 2. 辅助函数定义 (代码复用) ---
def calculate_and_print_metrics(confusion_matrix, classes, title="--- Metrics ---"):
    """
    根据混淆矩阵计算并打印详细的评估指标。
    """
    print(f"\n{title}")
    
    # 从混淆矩阵中提取 TP, FP, FN
    true_positives = np.diag(confusion_matrix)
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives
    total_samples = np.sum(confusion_matrix)
    
    if total_samples == 0:
        print("No samples to evaluate.")
        return

    # 计算总准确率 (微平均)
    overall_accuracy = np.sum(true_positives) / total_samples
    
    # 计算宏平均召回率
    denominators = true_positives + false_negatives
    recall_per_class = np.divide(true_positives, denominators, out=np.zeros_like(true_positives, dtype=float), where=denominators!=0)
    macro_recall = np.mean(recall_per_class)
    
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Overall Recall (Macro Avg): {macro_recall * 100:.2f}%")
    
    print("\n--- Per-class Metrics ---")
    for i, class_name in enumerate(classes):
        tp = true_positives[i]
        fp = false_positives[i]
        fn = false_negatives[i]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = recall_per_class[i] # 直接使用上面计算好的值
        
        print(f"Class: {class_name}")
        print(f"  Precision: {precision * 100:.2f}%")
        print(f"  Recall: {recall * 100:.2f}%")


# --- 3. 主程序入口 ---
if __name__ == '__main__':
    # --- 配置 ---
    batch_size = 48
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 加载模型和数据 ---
    try:
        # 模型加载
        model_name = 'face_expression_resnet.pth'
        model_folder = 'LAB7'
        model, _, _ = classify.makeEmotionNet(False)
        model_path = os.path.join(project_dir, model_folder, model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # 一次性设置

        # 数据加载
        image_name = 'images'
        images_path = os.path.join(project_dir, 'LAB4', image_name)
        test_loader, classes = utility.loadTest(images_path, batch_size)
    except Exception as e:
        print(f"加载模型或数据时出错: {e}")
        sys.exit(1)

    # --- 4. 单批次评估 (预览) ---
    print("\n--- Evaluating on One Batch (Preview) ---")
    try:
        X_batch, y_batch = next(iter(test_loader))
        X_batch = X_batch.to(device)
        
        with torch.no_grad():
            yHat_batch = model(X_batch)
            y_pred_batch = torch.argmax(yHat_batch, dim=1).cpu().numpy()
            y_true_batch = y_batch.cpu().numpy()

        # 为单批次构建临时混淆矩阵
        num_classes = len(classes)
        cm_batch = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(len(y_true_batch)):
            cm_batch[y_true_batch[i], y_pred_batch[i]] += 1

        # 调用复用函数打印指标
        calculate_and_print_metrics(cm_batch, classes, title="--- Metrics on One Batch ---")
        
        # 显示图片预览
        utility.imshow_with_labels(X_batch.cpu(), torch.from_numpy(y_pred_batch), classes)

    except StopIteration:
        print("测试数据加载器为空，无法进行单批次预览。")


    # --- 5. 完整数据集评估 ---
    print("\n--- Evaluating on Full Dataset ---")
    total_samples = 0
    num_classes = len(classes)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y_true_batch = y.cpu().numpy()
            
            yHat = model(X)
            y_pred_batch = torch.argmax(yHat, dim=1).cpu().numpy()
            
            total_samples += len(y_true_batch)
            for i in range(len(y_true_batch)):
                confusion_matrix[y_true_batch[i], y_pred_batch[i]] += 1
    
    # 再次调用复用函数打印指标
    calculate_and_print_metrics(confusion_matrix, classes, title="--- Final Metrics on Full Dataset ---")

    # 绘制并保存混淆矩阵热力图
    save_path='./confusion_matrix_resnet.png'
    utility.plot_confusion_matrix(confusion_matrix, classes, save_path)