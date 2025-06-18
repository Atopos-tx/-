import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image

# 数据增强与预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小为224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(30),  # 随机旋转
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # RGB图像归一化
])

# 加载数据集
data_dir = 'C:/Users/JNU/Desktop/flowers/flowers'  # 花卉数据集的路径
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 划分训练集、验证集、测试集（70%训练，20%验证，10%测试）
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 类别名称
class_names = dataset.classes  # 自动从文件夹名称中提取类别

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        # 第一层卷积 + ReLU + 最大池化
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输入3通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入32通道，输出64通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输入64通道，输出128通道
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)  # 2x2池化
        # 全连接层1
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # 展平后输入尺寸
        # 全连接层2
        self.fc2 = nn.Linear(512, num_classes)  # 输出5个类别

    def forward(self, x):
        # 卷积 -> ReLU -> 池化（连续的三个卷积层）
        x = self.pool(torch.relu(self.conv1(x)))  # 卷积1 + ReLU + 池化
        x = self.pool(torch.relu(self.conv2(x)))  # 卷积2 + ReLU + 池化
        x = self.pool(torch.relu(self.conv3(x)))  # 卷积3 + ReLU + 池化
        # 展平图像（将多维特征图展平为一维）
        x = x.view(x.size(0), -1)  # 展平特征图
        # 全连接层1 -> ReLU
        x = torch.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)  # 返回类别分数
        return x

# 创建模型实例
model = SimpleCNN(num_classes=5)

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam优化器

# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10):
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    best_accuracy = 0.0  # 用于保存最好的模型
    best_model_wts = None
    best_epoch = 0  # 记录最佳epoch

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, "
              f"Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_time:.2f}s")

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # 每个epoch结束后评估模型
        val_accuracy = evaluate_model(model, val_loader)
        val_accuracies.append(val_accuracy)
        # 如果当前验证精度是最佳的，则保存模型权重
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1  # 记录最佳epoch
            best_model_wts = model.state_dict()  # 保存最佳权重
        # 更新学习率
        scheduler.step()
    # 加载最好的模型权重
    model.load_state_dict(best_model_wts)
    plot_confusion_matrix(model, val_loader)
    return train_losses, train_accuracies, val_accuracies, best_accuracy, best_epoch

# 评估函数
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

# 绘制训练过程的函数
def plot_training_progress(train_accuracies, train_losses, val_accuracies):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 混淆矩阵绘制函数
def plot_confusion_matrix(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# 展示预测结果的函数
def display_predictions(model, test_loader, num_images=15):
    model.eval()
    images_shown = 0
    # 修改为 3 行 num_images 列
    rows = 3  # 行数
    cols = num_images // 3  # 每行展示的图片数（取 num_images 的整数除以3）
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))

    axes = axes.flatten()

    with torch.no_grad():
        for inputs, labels in test_loader:
            for i in range(min(num_images - images_shown, inputs.size(0))):
                image = inputs[i].permute(1, 2, 0)
                image = image.numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                image = image.clip(0, 1)

                label = labels[i].item()
                output = model(inputs[i].unsqueeze(0).to(device))
                _, predicted = torch.max(output, 1)

                ax = axes[images_shown]
                ax.imshow(image)
                ax.axis('off')

                true_label = class_names[label]
                pred_label = class_names[predicted.item()]

                if label == predicted.item():
                    ax.set_title(f"True: {true_label}\nPred: {pred_label}", color='green')
                else:
                    ax.set_title(f"True: {true_label}\nPred: {pred_label}", color='red')
                    ax.add_patch(plt.Rectangle((0, 0), image.shape[1], image.shape[0], linewidth=3, edgecolor='red', facecolor='none'))

                images_shown += 1
                if images_shown >= num_images:
                    break
            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.show()

# 训练模型
train_losses, train_accuracies, val_accuracies, best_accuracy, best_epoch = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10)

# 输出最佳epoch的信息
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print(f"Best Epoch: {best_epoch}")

# 绘制训练过程
plot_training_progress(train_accuracies, train_losses, val_accuracies)

# 绘制混淆矩阵
plot_confusion_matrix(model, val_loader)

# 在测试集上展示分类结果
display_predictions(model, test_loader, num_images=15)

# 清理缓存
torch.cuda.empty_cache()
