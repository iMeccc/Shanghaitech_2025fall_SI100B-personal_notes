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