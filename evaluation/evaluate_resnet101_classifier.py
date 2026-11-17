import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn
from sklearn.metrics import confusion_matrix
import numpy as np

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集（假设测试集的路径是data_dir/test）
data_dir = '/data/MM_AIGC/split_20_test/originals/DM'  # 请替换成你的数据文件夹路径
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

# 使用DataLoader加载测试数据
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载训练好的模型
model = models.resnet101(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)

# 使用GPU训练（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 加载保存的最佳模型
model.load_state_dict(torch.load('best_modelDM.pth'))
model.eval()

# 测试模型并计算准确率
test_correct = 0
test_total = 0
class_correct = [0] * 4  # 用于存储每个类别的正确预测数
class_total = [0] * 4  # 用于存储每个类别的总样本数
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # 获取预测结果
        _, predicted = torch.max(outputs, 1)

        # 统计每个类别的正确和总数
        for i in range(labels.size(0)):
            label = labels[i]
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        # 收集所有标签和预测
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# 计算测试集准确率
test_accuracy = 100 * test_correct / test_total

# 打印测试集准确率
print(f'Test Accuracy: {test_accuracy:.2f}%')

# 打印每个类别的平均准确率
for i, class_name in enumerate(test_dataset.classes):
    if class_total[i] > 0:  # 确保该类有样本
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print(f'Accuracy for class {class_name}: {class_accuracy:.2f}%')
    else:
        print(f'No samples for class {class_name}')

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_predictions)

# 打印混淆矩阵
print("Confusion Matrix:")
print(cm)
