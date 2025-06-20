import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from model import DigitCNN, AttentionDigitCNN
import argparse
import logging

# 设置日志
def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join('logs', f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

# 加载MNIST数据集
def load_data(batch_size=64):
    # 数据预处理和增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转±10度
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 下载并加载数据
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('data', train=False, transform=test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

# 训练函数
def train(model, device, train_loader, optimizer, epoch, logger):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    train_loss /= len(train_loader)
    accuracy = 100. * correct / total
    logger.info(f'Train set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    return train_loss, accuracy

# 测试函数
def test(model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    logger.info(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return test_loss, accuracy

# 可视化一些图像及其预测结果
def visualize_predictions(model, device, test_loader, num_samples=10):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    with torch.no_grad():
        images, labels = images[:num_samples], labels[:num_samples]
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 显示图像和预测结果
    plt.figure(figsize=(12, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i][0].cpu().numpy(), cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        plt.title(f'Pred: {predicted[i]}\nTrue: {labels[i]}', color=color)
        plt.axis('off')
    
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(os.path.join('results', 'predictions.png'))
    plt.close()

# 可视化特征图
def visualize_feature_maps(model, device, test_loader):
    model.eval()
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images[0:1].to(device)  # 只取第一张图像
    
    # 提取第一个卷积层的特征图
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册钩子
    if isinstance(model, DigitCNN) or isinstance(model, AttentionDigitCNN):
        model.conv1.register_forward_hook(get_activation('conv1'))
    
    # 前向传播
    with torch.no_grad():
        output = model(images)
    
    # 可视化特征图
    if 'conv1' in activation:
        feature_maps = activation['conv1'][0].cpu()
        plt.figure(figsize=(12, 12))
        
        # 显示原图
        plt.subplot(1, 2, 1)
        plt.imshow(images[0][0].cpu(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # 显示特征图 (16个通道中的前9个)
        plt.subplot(1, 2, 2)
        plt.imshow(torch.mean(feature_maps, dim=0), cmap='viridis')  # 通道平均
        plt.title('Average Feature Map')
        plt.axis('off')
        
        if not os.path.exists('results'):
            os.makedirs('results')
        plt.savefig(os.path.join('results', 'feature_maps.png'))
        plt.close()

# 可视化训练过程
def plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(os.path.join('results', 'training_history.png'))
    plt.close()

# 可视化混淆矩阵
def plot_confusion_matrix(model, device, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算混淆矩阵
    conf_matrix = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_targets, all_preds):
        conf_matrix[t, p] += 1
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 8))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    
    # 设置标签
    classes = range(10)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # 添加文本注释
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(os.path.join('results', 'confusion_matrix.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='MNIST Handwritten Digit Classification')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'attention'],
                        help='Model architecture (cnn or attention)')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--save-model', action='store_true', default=True, help='Save the trained model')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 检查是否使用GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 设置日志
    logger = setup_logger()
    logger.info(f"Using device: {device}")
    logger.info(f"Model: {args.model}")
    
    # 加载数据
    train_loader, test_loader = load_data(args.batch_size)
    
    # 创建模型
    if args.model == 'cnn':
        model = DigitCNN().to(device)
        logger.info("Using standard CNN model")
    else:
        model = AttentionDigitCNN().to(device)
        logger.info("Using attention-based CNN model")
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    
    # 训练和测试
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    best_accuracy = 0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n--- Epoch {epoch} ---")
        
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, logger)
        test_loss, test_acc = test(model, device, test_loader, logger)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            if args.save_model:
                if not os.path.exists('models'):
                    os.makedirs('models')
                torch.save(model.state_dict(), os.path.join('models', f'mnist_{args.model}.pt'))
                logger.info(f"Model saved with accuracy: {test_acc:.2f}%")
    
    # 可视化训练历史
    plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies)
    
    # 可视化一些预测结果
    visualize_predictions(model, device, test_loader)
    
    # 可视化特征图
    visualize_feature_maps(model, device, test_loader)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(model, device, test_loader)
    
    logger.info("Training completed!")
    logger.info(f"Best test accuracy: {best_accuracy:.2f}%")
    logger.info(f"Results and visualizations saved in 'results' directory")

if __name__ == '__main__':
    main() 