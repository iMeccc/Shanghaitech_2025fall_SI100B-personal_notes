import torch
import torch.nn as nn
import cv2
import numpy as np

# --- 1. 定义与训练时完全相同的模型架构 ---
# 这一步至关重要，因为 torch.load 只加载权重，不加载结构
class emotionNet(nn.Module):
    def __init__(self, printtoggle=False):
        super().__init__()
        self.print = printtoggle

        # 第一个处理模块
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 第二个处理模块
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 第三个处理模块
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # 全连接层
        # 输入特征数 D = 256 * 4 * 4 = 4096
        self.fc1 = nn.Linear(in_features=4096, out_features=3) 
        self.relu4 = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        # 你的 forward 函数逻辑
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = torch.flatten(x, start_dim=1)
        
        x = self.relu4(self.fc1(x))
        return x

# --- 2. 加载模型和工具 ---

# 定义类别标签
classes = ['happy', 'neutral', 'sad']

# 实例化模型
model = emotionNet()

# 加载训练好的权重
try:
    model.load_state_dict(torch.load('face_expression.pth', map_location=torch.device('cpu')))
except FileNotFoundError:
    print("错误：找不到模型文件 'face_expression.pth'。请确保它和脚本在同一个目录下。")
    exit()

# 设置为评估模式
model.eval()

# 加载 OpenCV 的人脸检测器
try:
    face_cascade = cv2.CascadeClassifier(r'C:\CODE\ShanghaiTech\2025fall_SI100B\PROJECT\LAB2\haar-cascade-files\haarcascade_frontalface_default.xml')
except cv2.error:
    print("错误：找不到人脸检测器文件 'haarcascade_frontalface_default.xml'。请下载并放在脚本目录下。")
    exit()

# --- 3. 初始化摄像头 ---
cap = cv2.VideoCapture(0) # 0 代表默认的摄像头
if not cap.isOpened():
    print("错误：无法打开摄像头。")
    exit()

# --- 4. 实时处理循环 ---
while True:
    # 逐帧读取视频
    ret, frame = cap.read()
    if not ret:
        print("无法接收视频帧。退出...")
        break

    # 将帧转换为灰度图（用于人脸检测）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # 遍历检测到的每一张人脸
    for (x, y, w, h) in faces:
        # a. 提取人脸区域 (ROI - Region of Interest)
        roi_color = frame[y:y+h, x:x+w]

        # b. 预处理人脸图像，使其符合模型输入要求
        #    - 缩放到 48x48
        #    - 归一化到 [-1, 1]
        #    - 维度变换 HWC -> CHW -> BCHW
        #    - 转换为 PyTorch Tensor
        face = cv2.resize(roi_color, (48, 48))
        face = face.astype(np.float32) / 127.5 - 1.0  # 归一化
        face = np.transpose(face, (2, 0, 1))           # HWC to CHW
        face = np.expand_dims(face, axis=0)            # Add batch dimension
        input_tensor = torch.from_numpy(face).float()

        # c. 使用模型进行推理
        with torch.no_grad(): # 推理时不需要计算梯度
            output_scores = model(input_tensor)

        # d. 解读输出结果
        probabilities = torch.softmax(output_scores, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)
        predicted_idx = int(predicted_index.item())
        confidence_val = float(confidence.item())
        predicted_label = f"{classes[predicted_idx]}: {confidence_val*100:.1f}%"

        # e. 在原始视频帧上绘制结果
        # 绘制人脸边界框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # 绘制情绪标签
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示最终的视频帧
    cv2.imshow('Real-time Emotion Detection', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. 释放资源 ---
cap.release()
cv2.destroyAllWindows()