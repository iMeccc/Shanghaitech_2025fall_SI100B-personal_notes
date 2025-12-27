import cv2
import torch
import torch.nn as nn
import numpy as np
import os, sys
from torchvision import transforms

# script_dir -> .../PROJECT/LAB6
script_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir -> .../PROJECT （确保 PROJECT 下有 LAB4 文件夹）
project_dir = os.path.dirname(script_dir)

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from LAB4.LAB4_empty.my_net.classify import emotionNet

class Detector:
    def __init__(self, CascadePath, ModelPath, device='cpu'):
        self.classes = ['happy', 'neutral', 'sad'] # 确保类别顺序和数量与你的模型一致
        self.device = torch.device(device)

        # --- Step 1: 加载模型 (已修正) ---
        self.face_cascade = cv2.CascadeClassifier(CascadePath)
        if self.face_cascade.empty():
            raise IOError(f"无法加载 Haar Cascade 模型于: {CascadePath}")

        # 标准加载方式：先创建结构，再加载权重
        self.model = emotionNet(printtoggle=False)
        self.model.load_state_dict(torch.load(ModelPath, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # 现在 self.model 是一个模型对象，可以调用 .eval()
        
        # 定义图像预处理流程
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((48, 48), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def process(self, img):
        display_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        # --- 优化：将所有检测到的人脸打包成一个批次 ---
        face_rois = [] # 存储裁剪的 BGR 人脸图像
        face_coords = [] # 存储对应的坐标
        
        for (x, y, w, h) in faces:
            face_rois.append(img[y:y+h, x:x+w])
            face_coords.append((x, y, w, h))
        
        # 如果没有检测到人脸，直接返回原图
        if not face_rois:
            return display_img
            
        # --- 统一进行预处理 ---
        batch_tensors = []
        for roi in face_rois:
            # BGR -> RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = self.transform(roi_rgb)
            batch_tensors.append(tensor)
            
        # 使用 torch.stack 将 tensor 列表堆叠成一个批处理张量
        input_batch = torch.stack(batch_tensors).to(self.device)

        # --- 一次性进行模型推理 ---
        with torch.no_grad():
            outputs = self.model(input_batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_indices = torch.max(probabilities, 1)

        # --- 在图上绘制所有结果 ---
        for i, (x, y, w, h) in enumerate(face_coords):
            label = self.classes[int(predicted_indices[i].item())]
            conf = confidence[i].item() * 100
            
            label_text = f"{label}: {conf:.1f}%"
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        return display_img

# --- Step 3: 主程序部分 ---
if __name__ == '__main__':
    # 确保文件都在脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    cascade_file = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
    model_file = os.path.join(script_dir, 'face_expression_excel.pth')
    demo_image_file = os.path.join(script_dir, 'demo.png')

    # 读取图片
    img = cv2.imread(demo_image_file)
    if img is None:
        raise FileNotFoundError(f"{demo_image_file} not found or could not be read.")

    # 初始化检测器
    compute_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {compute_device}")
    try:
        detector = Detector(CascadePath=cascade_file, ModelPath=model_file, device=compute_device)
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit()

    # 处理并显示结果
    ret = detector.process(img)
    
    cv2.imwrite('result.jpg', ret)
    cv2.imshow('Result', ret)
    cv2.waitKey(0)
    cv2.destroyAllWindows()