import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitCNN(nn.Module):
    """
    卷积神经网络模型，用于MNIST手写数字识别
    """
    def __init__(self):
        super(DigitCNN, self).__init__()
        # 第一个卷积层, 1个输入通道，32个输出通道，3x3卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二个卷积层, 32个输入通道，64个输出通道，3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Dropout层，防止过拟合
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x的初始形状: [batch_size, 1, 28, 28]
        x = self.conv1(x)  # -> [batch_size, 32, 28, 28]
        x = F.relu(x)
        x = self.conv2(x)  # -> [batch_size, 64, 28, 28]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # -> [batch_size, 64, 14, 14]
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)  # -> [batch_size, 64, 7, 7]
        x = torch.flatten(x, 1)  # -> [batch_size, 64*7*7]
        x = self.fc1(x)  # -> [batch_size, 128]
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # -> [batch_size, 10]
        output = F.log_softmax(x, dim=1)  # 应用log_softmax进行分类
        return output

# 自定义的模型，添加注意力机制
class AttentionDigitCNN(nn.Module):
    """
    带有注意力机制的卷积神经网络模型
    """
    def __init__(self):
        super(AttentionDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        # 应用注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights  # 应用注意力权重
        
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) 