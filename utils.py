import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torchvision import transforms
from PIL import Image

# 图像预处理转换
def get_transform():
    """
    返回用于预处理手写数字图像的转换
    """
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

# 加载预训练模型
def load_model(model, model_path, device):
    """
    加载预训练模型
    
    Args:
        model: 模型对象
        model_path: 模型权重文件路径
        device: 运行设备 (cpu/cuda)
    
    Returns:
        加载了权重的模型
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 图像预处理
def preprocess_image(image_path):
    """
    预处理单个图像文件
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        预处理后的图像张量
    """
    transform = get_transform()
    image = Image.open(image_path).convert('L')  # 转换为灰度
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    return image_tensor

# 从OpenCV图像预处理
def preprocess_cv_image(cv_image):
    """
    预处理OpenCV格式的图像
    
    Args:
        cv_image: OpenCV格式的图像(numpy数组)
        
    Returns:
        预处理后的图像张量
    """
    # 转换为灰度
    if len(cv_image.shape) == 3:
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = cv_image
    
    # 调整尺寸为28x28
    resized_image = cv2.resize(gray_image, (28, 28))
    
    # 图像增强和规范化
    _, binary_image = cv2.threshold(resized_image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # 转换为PIL图像
    pil_image = Image.fromarray(binary_image)
    
    # 应用变换
    transform = get_transform()
    image_tensor = transform(pil_image).unsqueeze(0)
    
    return image_tensor

# 从模型获取预测
def get_prediction(model, image_tensor, device):
    """
    使用模型进行预测
    
    Args:
        model: 预训练模型
        image_tensor: 预处理后的图像张量
        device: 运行设备
    
    Returns:
        预测的数字和概率
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.exp(outputs)
        _, predicted = torch.max(outputs, 1)
        probability = probabilities[0][predicted.item()].item()
        
    return predicted.item(), probability

# 绘制热力图可视化
def draw_attention_heatmap(model, image_tensor, device, output_path=None):
    """
    为注意力模型绘制热力图
    
    Args:
        model: 带注意力机制的模型
        image_tensor: 输入图像张量
        device: 运行设备
        output_path: 输出图像路径
    
    Returns:
        热力图图像
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # 注册钩子
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 尝试获取注意力权重
    if hasattr(model, 'attention'):
        hook = model.attention.register_forward_hook(get_activation('attention'))
        
        # 前向传播
        with torch.no_grad():
            output = model(image_tensor)
            
        hook.remove()
        
        if 'attention' in activation:
            # 获取注意力权重
            attention_map = activation['attention'][0, 0].cpu().numpy()
            
            # 调整大小以匹配原图
            attention_map = cv2.resize(attention_map, (28, 28))
            
            # 归一化为0-1
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
            
            # 创建热力图
            plt.figure(figsize=(10, 5))
            
            # 原始图像
            plt.subplot(1, 2, 1)
            plt.imshow(image_tensor[0, 0].cpu().numpy(), cmap='gray')
            plt.title("Original Image")
            plt.axis('off')
            
            # 热力图
            plt.subplot(1, 2, 2)
            plt.imshow(attention_map, cmap='jet')
            plt.title("Attention Heatmap")
            plt.axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
            
            return attention_map
    
    return None

# 准备模型评估结果目录
def prepare_results_dir():
    """准备结果目录"""
    if not os.path.exists('results'):
        os.makedirs('results')
    return os.path.abspath('results')

# 获取所有预测和概率
def get_all_predictions(model, dataloader, device):
    """
    获取数据集中所有图像的预测结果
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        预测、真实标签和预测概率
    """
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.exp(output)
            
            _, preds = torch.max(output, 1)
            batch_probs = [probs[i][preds[i]].item() for i in range(len(preds))]
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            probabilities.extend(batch_probs)
    
    return np.array(predictions), np.array(true_labels), np.array(probabilities) 