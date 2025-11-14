
# NumPy 核心知识点速查手册 (面向机器学习)

## 1. 为什么 NumPy 是基石？

在机器学习中，所有数据（图片、文本、表格）最终都会被表示为**数字数组**。NumPy 的核心数据结构 `ndarray` 是一个高性能的多维数组，它比 Python 自带的 `list` 在处理大规模数值运算时快几个数量级。它是几乎所有数据科学库（包括 PyTorch）的底层依赖。

**核心优势**:
- **速度 (Speed)**: 底层由 C 语言实现，运算极快。
- **效率 (Efficiency)**: 支持“向量化”操作，代码简洁，可读性强。
- **生态 (Ecosystem)**: 与 PyTorch、Pandas 等库无缝集成。

---
## 2. 导入 NumPy

这是标准的导入方式，`np` 是社区公认的别名。

```python
import numpy as np
```

---
## 3. NumPy 的核心: `ndarray` (N-维数组)

一个 `ndarray` 是一个由**相同数据类型**元素组成的网格。

### 关键属性

- `ndarray.ndim`: 维度数量。向量是 1 维，矩阵是 2 维。
- `ndarray.shape`: 数组的形状，一个元组，表示每个维度的大小 (e.g., `(3, 4)` 表示 3 行 4 列)。
- `ndarray.size`: 数组中元素的总数。
- `ndarray.dtype`: 元素的数据类型 (e.g., `float64`, `int32`)。

```python
# 创建一个 2x3 的矩阵
a = np.array([,])

print(f"维度 (ndim): {a.ndim}")
print(f"形状 (shape): {a.shape}")
print(f"总大小 (size): {a.size}")
print(f"数据类型 (dtype): {a.dtype}")
```

---
## 4. 创建数组的常用方法

### 4.1 从 Python 列表创建

这是最基础的方式。

```python
# 一维数组
arr1 = np.array()
print(f"一维数组: {arr1}")

# 二维数组
arr2 = np.array([[1.0, 2.0], [3.0, 4.0]])
print(f"二维数组:\n{arr2}")
```

### 4.2 创建特定内容的数组 (非常常用)

在初始化模型参数或数据时极其有用。

```python
# 创建一个 2x3 的全零数组
zeros_arr = np.zeros((2, 3))

# 创建一个 3x2 的全一数组
ones_arr = np.ones((3, 2))

# 创建一个内容未初始化的数组 (内容是内存中的随机值，但速度最快)
empty_arr = np.empty((2, 2))

print(f"全零数组:\n{zeros_arr}")
print(f"全一数组:\n{ones_arr}")
```

### 4.3 创建序列数组

```python
# 类似 Python 的 range(start, stop, step)
# 创建从 10 到 24，步长为 5 的数组
arr_arange = np.arange(10, 25, 5)
print(f"arange 示例: {arr_arange}")

# 在区间内生成指定数量的等间距点 (包含端点)
# 创建从 0 到 2 之间，均匀取 9 个点
arr_linspace = np.linspace(0, 2, 9)
print(f"linspace 示例: {arr_linspace}")
```

### 4.4 创建随机数组 (极其常用)

```python
# 创建一个 2x3 的，[0, 1) 之间均匀分布的随机数组
rand_uniform = np.random.rand(2, 3)

# 创建一个 2x3 的，服从标准正态分布 (均值为0，方差为1) 的随机数组
rand_normal = np.random.randn(2, 3)

# 创建一个 3x4 的，[0, 10) 之间的随机整数数组
rand_int = np.random.randint(0, 10, size=(3, 4))

print(f"均匀分布:\n{rand_uniform}")
print(f"正态分布:\n{rand_normal}")
print(f"随机整数:\n{rand_int}")
```

---
## 5. 数组的索引与切片

这是 NumPy 最强大的功能之一，是数据处理的基础。

### 5.1 一维数组

与 Python 列表类似。

```python
a = np.arange(10)**2  # [ 0  1  4  9 16 25 36 49 64 81]

# 索引
print(f"索引 2 的元素: {a}")

# 切片 [start:end:step]
print(f"索引 2 到 5 的元素: {a[2:5]}")
```

### 5.2 多维数组

使用 `[row, column]` 格式进行索引。`:` 代表该维度的所有元素。

```python
matrix = np.array([,
                  ,
                  ])

# 获取单个元素 (第 2 行, 第 3 列)
print(f"元素 (1, 2): {matrix}") # 输出 12

# 获取一整行 (第 1 行)
print(f"第 1 行: {matrix[1, :]}") # 或者简写为 matrix

# 获取一整列 (第 2 列)
print(f"第 2 列: {matrix[:, 2]}")

# 获取子矩阵 (前 2 行, 从第 1 列到第 2 列)
sub_matrix = matrix[:2, 1:3]
print(f"子矩阵:\n{sub_matrix}")
```

### 5.3 布尔索引 (极其强大)

使用布尔条件数组来筛选数据。

```python
data = np.arange(12).reshape(3, 4)
print(f"原始数据:\n{data}")

# 创建布尔条件
condition = data > 5
print(f"条件矩阵 (是否 > 5):\n{condition}")

# 使用条件来筛选
print(f"大于 5 的所有元素: {data[condition]}")
```
---
## 6. 基本运算

### 6.1 逐元素运算

NumPy 会自动对数组中的每个元素执行操作，无需 `for` 循环。

```python
a = np.array()
b = np.array()

print(f"a - b = {a - b}")
print(f"b ** 2 = {b ** 2}")
print(f"a * 10 = {a * 10}")
print(f"np.sin(a) = {np.sin(a)}") # NumPy 提供了大量通用函数 (ufunc)
```

### 6.2 矩阵运算

- `*` 是逐元素乘法。
- `@` 是真正的矩阵乘法。

```python
A = np.array([,])
B = np.array([,])

print(f"逐元素乘法 A * B:\n{A * B}")
print(f"矩阵乘法 A @ B:\n{A @ B}")
```

### 6.3 广播 (Broadcasting)

当对不同形状的数组进行运算时，NumPy 会自动扩展（广播）较小的数组以匹配较大的数组。

```python
matrix = np.ones((3, 4)) # 3x4 的全一矩阵
vector = np.arange(4)     #

# 广播会自动将 vector "向下复制" 3 次，使其与 matrix 形状匹配
result = matrix + vector
print(f"广播加法的结果:\n{result}")
```

---
## 7. 常用函数与操作

### 7.1 聚合函数

```python
data = np.random.rand(2, 3) * 10 # 生成 2x3 的 [0, 10) 随机数
print(f"数据:\n{data}")

print(f"总和: {data.sum()}")
print(f"最小值: {data.min()}")
print(f"最大值: {data.max()}")

# 沿着坐标轴计算 (axis=0 是列, axis=1 是行)
print(f"每列的总和: {data.sum(axis=0)}")
print(f"每行的最大值: {data.max(axis=1)}")
```

### 7.2 形状操作

在数据送入机器学习模型前，调整形状是必备技能。

```python
data = np.floor(np.random.rand(3, 4) * 10) # 3x4 随机整数矩阵
print(f"原始形状: {data.shape}")

# 展平数组 (返回一个新的一维数组)
print(f"展平后: {data.flatten()}")

# 重塑数组 (不改变原始数据)
reshaped = data.reshape(2, 6)
print(f"重塑为 2x6:\n{reshaped}")

# 转置矩阵
print(f"转置后:\n{data.T}")
```

---
## 8. 与 PyTorch 的衔接

NumPy 数组和 PyTorch 张量可以非常高效地相互转换，并且共享底层内存（只要它们在 CPU 上），这意味着转换过程几乎是零成本的。

```python
# 导入 PyTorch
import torch

# 1. 从 NumPy 数组创建 PyTorch 张量
numpy_array = np.ones((2, 3))
pytorch_tensor = torch.from_numpy(numpy_array)

print("--- NumPy to PyTorch ---")
print(f"NumPy 数组:\n{numpy_array}")
print(f"PyTorch 张量:\n{pytorch_tensor}")

# 修改 NumPy 数组会影响 PyTorch 张量
numpy_array = 99
print(f"\n修改 NumPy 后，PyTorch 张量变为:\n{pytorch_tensor}")


# 2. 从 PyTorch 张量转换回 NumPy 数组
tensor_to_convert = torch.tensor([,])
numpy_from_tensor = tensor_to_convert.numpy()

print("\n--- PyTorch to NumPy ---")
print(f"PyTorch 张量:\n{tensor_to_convert}")
print(f"转换后的 NumPy 数组:\n{numpy_from_tensor}")
```
