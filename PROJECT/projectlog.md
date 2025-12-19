### LAB1
setting up environments and install packages, then get familiar with the foundamental image processions.
- Obstacles: environments set up and workspace choosing.

### LAB2
learn basic functions of opencv including read, show image, draw rectangle/circle, write words, cropping image, write to local.
then haar-cascade classifer, able to detect edges of objects and return the range of it. NMS algorithm is applied to reduce the overlapping parts.   

**OpenCV 基础操作学习**

#### 📸 1. 图像读取与显示
- **读取**：
  - `cv2.imread(path)`：支持绝对路径或相对路径。
  - 返回值是 `numpy.ndarray`（n维数组），形状为 `(H, W, C)`（高、宽、通道（即RGB图和灰度图））。
  - 用 `img.shape` 查看图像尺寸，用 `img.dtype` 查看数据类型（通常是 uint8（无符号8位整数），取值范围 [0, 255]）。
```python
# 获取图像高度和宽度
h, w = img.shape[:2]

# 将图像转为灰度（如果原来是彩色）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 裁剪图像
cropped = img[100:300, 200:400]

# 修改某个像素的颜色
img[50, 100] = [0, 255, 0]  # 设为绿色
```
```python
import cv2

# 读取彩色图像
img = cv2.imread('cat.jpg')  # shape: (480, 640, 3)

# 分离通道
b, g, r = cv2.split(img)

# b.shape: (480, 640) —— 蓝色通道
# g.shape: (480, 640) —— 绿色通道
# r.shape: (480, 640) —— 红色通道

# 合并通道
merged = cv2.merge([b, g, r])
```
- **显示**：
  - `cv2.imshow('title', img)`：创建一个窗口显示图像。
  - `cv2.waitKey(0)`：等待任意键按下，否则窗口会立即关闭。
- ✅ **关键**：确保路径正确，图像文件存在。

#### 🖍️ 2. 图像基本操作
- **绘制矩形**：
  - `cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)`
  - 用于标注目标区域。
- **添加文字**：
  - `cv2.putText(img, text, org, font, scale, color, thickness)`
  - 用于在图像上添加说明。
- **裁剪与缩放**：
  - **裁剪**：使用 NumPy 的切片 `img[y1:y2, x1:x2]`。
  - **缩放**：`cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)`
  - `INTER_LINEAR` 是默认的插值方法，适合一般用途。

#### 💾 3. 图像保存
- **保存**：
  - `cv2.imwrite('filename.jpg', img)`：将处理后的图像保存到本地。
  - 文件名后缀决定保存格式（如 `.jpg`, `.png`）。

#### 🧩 4. 核心概念：图像即数组
- **本质**：OpenCV 中的图像是一个 NumPy 数组。
  - 可以直接用 NumPy 操作进行像素级处理（如 `img[100, 200] = [0, 0, 255]`）。
  - 裁剪、缩放、颜色空间转换等操作都基于数组切片和函数调用。

#### 🎯 5. 综合应用：按步骤处理图像
1. 读取图像。
2. 在图像上绘制矩形。
3. 裁剪指定区域。
4. 缩放图像。
5. 添加文字。
6. 保存结果。

> ✅ **核心洞见**：  
> **OpenCV 的强大在于其“图像即数组”的设计哲学。**  
> 你可以像操作普通数据一样操作图像，结合 NumPy 和 OpenCV API，实现从简单到复杂的各种视觉任务。

### LAB3
卷积神经网络（CNN）
#### 🔍 1. **卷积 = 滑动模板匹配**
- **操作**：用一个小窗口（卷积核）在图像上滑动。
- **计算**：窗口内像素与卷积核逐元素相乘后求和。
- **输出**：一个**响应图**，值越大表示局部区域越“像”该卷积核。
- ✅ **每个卷积核 = 一个特征探测器**（如边缘、纹理、模糊等）。

#### 🧩 2. **特征提取是分层的**
- **浅层**：提取低级特征（边缘、角点）。
- **深层**：组合低级特征，形成高级语义（眼睛、轮子、文字）。
- ✅ **多个卷积核并行工作 → 同时提取多种特征**。

#### ⚙️ 3. **仿射变换 = 特征组合器**
- 全连接层（`y = Wx + b`）对扁平化后的特征向量进行**加权组合**。
- **每行权重 `W[i, :]` = 一种“判断逻辑”**：
  - 强调某些特征（如“有胡须 + 圆眼” → 猫），
  - 抑制其他特征。
- ✅ **变换的目的不是“连接”，而是“学习如何组合特征”**。

#### 📦 4. **整体流程 = 编码 + 决策**
1. **编码阶段**（卷积 + 激活 + 池化 + 扁平化）  
   → 将原始像素压缩为**高维语义特征向量**（“原始数据库”）。
2. **决策阶段**（全连接 + 激活）  
   → 用不同“变换”对特征向量加权，做出最终判断。

#### 💡 核心洞见
> **卷积核决定“看什么”（What to look for）。卷积核的选取有很多种，每种可以对应提取原始图片的某个特征。  
> 全连接权重决定“怎么判”（How to decide）。存在很多种不同的变换（连接权重），每种变换可以通过某种运算强调原始数据的某一个特点。**  

---

### LAB4
使用 PyTorch 构建并训练神经网络

> ✅ **核心洞见**：
> **PyTorch = 神经网络的“乐高积木盒”。**
> 我们不再需要手动实现复杂的数学运算（如卷积、反向传播），而是通过**声明式**地“搭建”网络层，像拼乐高一样构建模型。PyTorch 负责所有底层的计算、优化和 GPU 加速。

#### 🧠 1. **训练的“灵魂”：损失函数与反向传播**
- **目标**：模型训练的唯一目标是**最小化损失 (Loss)**。
- **损失函数 `Loss = f(预测值, 真实值)`**：
  - 一个数学函数，用来衡量模型“猜得有多差”。
  - 分类问题常用**交叉熵损失 (Cross-Entropy Loss)**。
    - **原理**：如果模型对正确答案的预测概率很高，损失就小；反之，损失就大。
- **反向传播 (Backpropagation)**：
  - 计算出损失后，模型会“复盘”，从后往前计算出**每一个权重**对这次“犯错”应负的“责任”（即**梯度**）。
- **优化器 (Optimizer)**：
  - 得到所有权重的“责任”后，优化器（如 Adam）负责去**微调**每一个权重，让模型下次能做得更好。

#### 📦 2. **PyTorch 核心“积木” (`torch.nn` 模块)**
从现在开始，我们用 PyTorch 提供的现成“积木块”（类）来代替之前手写的函数。

##### **`nn.Conv2d` - 卷积层**
- **作用**：提取图像特征，是 CNN 的核心。
- **接口**：`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`
- **参数详解**：
  - `in_channels` (整数): **输入通道数**。必须与上一层的输出通道数匹配。对于第一层，它通常是 3（彩色图）。
  - `out_channels` (整数): **输出通道数**。由你**自己定义**，代表你希望这一层提取出多少种不同的特征。
  - `kernel_size` (整数或元组): 卷积核的大小，如 `3` (代表 3x3) 或 `(3, 5)`。
  - `stride` (整数): 步长。
  - `padding` (整数): 填充。
- ✅ **关键**：你只需定义“积木”的规格，PyTorch 会**自动创建并管理**内部的权重和偏置。

##### **`nn.MaxPool2d` - 最大池化层**
- **作用**：缩小特征图尺寸（下采样），减少计算量，提取最显著的特征。
- **接口**：`nn.MaxPool2d(kernel_size, stride=None)`
- **参数详解**：
  - `kernel_size`: 池化窗口的大小，如 `2` (代表 2x2)。
  - `stride`: 步长。如果省略，默认等于 `kernel_size`。

##### **`nn.LeakyReLU` - 激活函数**
- **作用**：为网络引入**非线性**，让模型能学习更复杂的模式。
- **接口**：`nn.LeakyReLU(negative_slope=0.01)`
- ✅ **用法**：通常紧跟在 `nn.Conv2d` 或 `nn.Linear` 之后。

##### **`torch.flatten` - 展平层**
- **作用**：将多维的特征图“压扁”成一个一维向量，为送入全连接层做准备。
- **接口**：`torch.flatten(x, start_dim=1)`
- **参数详解**：
  - `start_dim=1`: **极其重要！** 它的意思是“保持第 0 维（批次维度 Batch_size）不变，将从第 1 维开始的所有后续维度（C, H, W）全部展平”。
  - ✅ **这完美地处理了批处理数据，是你之前 `reshape(batch_size, -1)` 的等效实现。**

##### **`nn.Linear` - 全连接层**
- **作用**：对展平后的特征进行加权组合，做出最终的分类决策。
- **接口**：`nn.Linear(in_features, out_features)`
- **参数详解**：
  - `in_features` (整数): **输入特征数**。必须与 `flatten` 之后得到的向量长度**完全匹配**。你需要手动计算这个值。
  - `out_features` (整数): **输出特征数**。对于分类问题，它等于你的**类别总数**（比如 3 或 7）。

#### 🏗️ 3. **搭建模型的“两步走”**
所有 PyTorch 模型都通过定义一个继承自 `nn.Module` 的类来构建。

1.  **`__init__(self)` 方法 (声明积木)**
    - **作用**：这是模型的“蓝图”。在这里，你需要把你将要用到的**所有带参数的网络层**（如 `nn.Conv2d`, `nn.Linear`）都实例化，并作为类的属性（如 `self.conv1 = nn.Conv2d(...)`）。
    - ✅ **只声明，不连接。**

2.  **`forward(self, x)` 方法 (连接流水线)**
    - **作用**：这是模型的“执行流程”。它定义了当数据 `x` 输入时，应该按照怎样的**顺序**流过你在 `__init__` 中声明的那些层。
    - **返回值**：返回网络的最终输出（通常是未经 Softmax 的原始分数 Logits）。

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 在这里声明所有需要的“积木”
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.fc1 = nn.Linear(4096, 3)

    def forward(self, x):
        # 在这里定义数据如何流过这些“积木”
        x = self.conv1(x)
        # ... (经过其他层) ...
        x = self.fc1(x)
        return x
```

#### 🚀 4. **训练与推理**
- **训练 (Training)**：
  - 将模型设置为 `model.train()` 模式。
  - 循环遍历数据集，执行**前向传播、计算损失、反向传播、权重更新**。
  - 训练结束后，使用 `torch.save(model.state_dict(), 'model.pth')` 保存学到的**权重**。
- **推理 (Inference)**：
  - 创建一个和训练时**一模一样**的模型结构实例。
  - 使用 `model.load_state_dict(torch.load('model.pth'))` 加载训练好的权重。
  - **关键**：将模型设置为 `model.eval()` 模式，这会关闭 Dropout 等只在训练时使用的功能，确保预测结果的确定性。