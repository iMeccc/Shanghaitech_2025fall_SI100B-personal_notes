import torch
import torch.nn as nn

# --- 关键步骤 1: 导入你的模型定义 ---
# Python 需要知道 emotionNet 这个类是什么样的。
# 你可以把 emotionNet 类的定义复制到这个文件里，
# 或者如果你的类在一个叫 model.py 的文件里，你可以这样导入：
# from model import emotionNet

# 假设你直接把类定义复制过来了
class emotionNet(nn.Module):
    def __init__(self, printtoggle):
        super().__init__()
        self.print = printtoggle
      
        ### write your codes here ###
        #############################
        # step1:
        # Define the functions you need: convolution, pooling, activation, and fully connected functions.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0) # 根据图示
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0) # 根据图示
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.LeakyReLU()

        self.fc1 = nn.Linear(in_features=4096, out_features=7) 
        self.relu4 = nn.LeakyReLU()


    def forward(self, x):
        #Step 2
        # Using the functions your defined for forward propagate
        # First block
        # convolution -> maxpool -> relu
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # Second block
        # convolution -> maxpool -> relu
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # Third block
        # convolution -> maxpool -> relu
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        # Flatten for linear layers
        x = torch.flatten(x, start_dim=1)

        print(f"Shape after flatten: {x.shape}") # 这一行会打印出 (Batch_size, 4096)
        # fully connect layer
        x = self.fc1(x)
        x = self.relu4(x)

        return x

# --- 关键步骤 2: 创建模型实例并加载权重 ---
print("正在创建模型实例...")
# 实例化一个和你训练时一模一样的“空壳”
model = emotionNet(False)

# 找到你训练好的 .pth 文件
pth_file_path = 'face_expression_bonus.pth'
print(f"正在从 '{pth_file_path}' 加载权重...")

# 将训练好的“灵魂”（权重）注入到“空壳”中
model.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))

# 将模型设置为评估模式，这是导出的标准做法
model.eval()
print("权重加载成功，模型已设置为评估模式。")


# --- 关键步骤 3: 创建虚拟输入并导出 ---
# 创建一个符合模型输入的“虚拟”输入张量 (Batch=1, Channels=3, Height=48, Width=48)
dummy_input = torch.randn(1, 3, 48, 48)

# 定义要导出的 ONNX 文件名
onnx_file_path = "emotion_model_bonus.onnx"
print(f"准备将模型导出到 '{onnx_file_path}'...")

# 执行导出
torch.onnx.export(
    model,                      # 要导出的模型
    (dummy_input,),             # 一个符合输入尺寸的虚拟张量（作为 tuple 传入）
    onnx_file_path,             # 导出的文件路径
    export_params=True,         # 确保权重也被导出
    opset_version=18,           # 一个常用的 ONNX 版本号
    do_constant_folding=True,   # (可选) 进行一些优化
    input_names=['input'],      # (可选) 给输入节点起个名字
    output_names=['output'],    # (可选) 给输出节点起个名字
    dynamic_axes={'input' : {0 : 'batch_size'},    # (可选, 但推荐)
                  'output' : {0 : 'batch_size'}}   # 标记 batch 维度是动态的
)

print("="*30)
print(f"模型已成功导出！请用 Netron 打开 '{onnx_file_path}' 文件。")
