import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 配置参数
class Config:
    batch_size = 64
    lr = 1e-3
    epochs = 500
    save_interval = 100  # 每隔 10 轮保存权重
    num_classes = 9  # 已知类别数量（BigGAN, StyleGAN, StyleGAN2, ProGAN, SD1.5, Wukong, Glide, ADM, REAL）


# 分类器模型
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),  # 输入维度为 2（StyleGAN_score, ADM_score）
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, Config.num_classes)  # 输出维度为 9（对应 9 个类别）
        )

    def forward(self, x):
        return self.fc(x)


# 数据集类
class ClassificationDataset(Dataset):
    def __init__(self, csv_file):
        """
        :param csv_file: 包含差异性得分和标签的 CSV 文件路径
                         第四列：StyleGAN_score
                         第五列：ADM_score
                         第六列：标签（0-8，对应已知类别）
        """
        # 使用 pandas 加载 CSV 文件
        data = pd.read_csv(csv_file)

        # 提取第四列和第五列（差异性分数）
        self.scores = data.iloc[:, [1, 2]].values.astype(np.float32)

        # 提取第六列（标签）
        self.labels = data.iloc[:, 0].values.astype(np.int64)

        # 确保标签范围正确
        assert all(0 <= label < Config.num_classes for label in self.labels), "Labels must be within known classes"

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return torch.tensor(self.scores[idx]), torch.tensor(self.labels[idx])


# 训练函数
def train_classifier(csv_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化数据集和数据加载器
    dataset = ClassificationDataset(csv_file)
    dataloader = DataLoader(dataset,
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)

    # 初始化模型、损失函数和优化器
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)

    # 训练循环
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for scores, labels in dataloader:
            scores = scores.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(scores)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(dataloader)
        accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{Config.epochs}] '
              f'Train Loss: {avg_train_loss:.4f} '
              f'Accuracy: {accuracy:.4f}')

        # 每隔 10 轮保存权重
        if (epoch + 1) % Config.save_interval == 0:
            weight_path = f'classifier_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), weight_path)
            print(f"Model weights saved to {weight_path}")


# 推理函数
def infer_classifier(csv_file, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = Classifier().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 加载测试数据
    data = pd.read_csv(csv_file)
    scores = data.iloc[:, [3, 4]].values.astype(np.float32)  # 提取第四列和第五列
    scores_tensor = torch.tensor(scores, dtype=torch.float32).to(device)

    # 推理
    with torch.no_grad():
        outputs = model(scores_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

    # 映射类别索引到具体名称
    categories = [
        "BigGAN", "StyleGAN", "StyleGAN2",
        "ProGAN", "SD1.5", "Wukong",
        "Glide", "ADM", "REAL"
    ]
    predicted_categories = [categories[idx] for idx in predicted.cpu().numpy()]

    # 打印结果
    for i, category in enumerate(predicted_categories):
        print(f"Sample {i + 1}: Predicted Category = {category}")


if __name__ == "__main__":
    # 训练分类器
    csv_file = "lpips_scores_per_category.csv"  # 包含差异性得分和标签的 CSV 文件
    train_classifier(csv_file)

    # 推理分类器
    weights_path = "./classifier_epoch_50.pth"  # 训练好的权重文件
    infer_classifier(csv_file, weights_path)