import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image

# 配置参数
class Config:
    img_size = 256
    batch_size = 64
    lr = 1e-4
    epochs = 100
    save_interval = 10
    dm_models = ['ADM', 'VQDM', 'Wukong', 'SD1.5']  # 目标分类的 DM 模型
    num_classes = len(dm_models)
    # EfficientNet 预训练权重的归一化参数
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

# 特征提取器（EfficientNet-B4）
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的 EfficientNet-B4
        backbone = models.efficientnet_b4(pretrained=True)
        # 去掉最后的分类层，保留特征提取层
        self.feature_extractor = nn.Sequential(*backbone.features)
        # 添加全局平均池化层
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # 输出 1792 维特征

# 分类器模型（结合原始特征和差异分数）
class CombinedClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 原始图像特征分支
        self.feature_extractor = FeatureExtractor()
        # 差异分数输入分支
        self.score_fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU()
        )
        # 融合层（EfficientNet-B4 特征维度为 1792）
        self.combined_fc = nn.Sequential(
            nn.Linear(1792 + 64, 512),  # 修改输入维度
            nn.ReLU(),
            nn.Linear(512, Config.num_classes)
        )
    
    def forward(self, image, stylegan_score, adm_score):
        # 提取图像特征
        image_features = self.feature_extractor(image)
        # 处理差异分数
        score_features = self.score_fc(torch.stack([stylegan_score, adm_score], dim=1))
        # 融合特征
        combined = torch.cat([image_features, score_features], dim=1)
        return self.combined_fc(combined)

# 数据集类（结合图像和差异分数）
class CombinedDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.normalize_mean, std=Config.normalize_std)  # 使用 EfficientNet 归一化参数
        ])
        self.dm_model_map = {model: idx for idx, model in enumerate(Config.dm_models)}
        self.samples = []
        
        for _, row in self.data.iterrows():
            orig_model = row['Category']
            if orig_model in Config.dm_models:
                image_name = row['Image Name']
                stylegan_score = row['StyleGAN_Score']
                adm_score = row['ADM_Score']
                self.samples.append({
                    'image_path': os.path.join(root_dir, 'originals', orig_model, image_name),
                    'stylegan_score': stylegan_score,
                    'adm_score': adm_score,
                    'label': self.dm_model_map[orig_model]
                })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 加载原始图像
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        return (
            image,
            torch.tensor([sample['stylegan_score']], dtype=torch.float32),
            torch.tensor([sample['adm_score']], dtype=torch.float32),
            torch.tensor(sample['label'], dtype=torch.long)
        )

# 划分数据集（80%训练，10%验证，10%测试）
def split_dataset(dataset):
    # 划分训练集和临时集（80%训练，20%临时）
    train_indices, temp_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        stratify=[s['label'] for s in dataset.samples],
        random_state=42
    )
    # 划分验证集和测试集（各占10%）
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        stratify=[dataset.samples[i]['label'] for i in temp_indices],
        random_state=42
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

# 训练函数
def train(csv_file, root_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CombinedDataset(csv_file, root_dir)
    train_set, val_set, test_set = split_dataset(dataset)
    
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False)
    
    model = CombinedClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    
    best_val_acc = 0.0
    for epoch in range(Config.epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, stylegan_scores, adm_scores, labels in train_loader:
            images = images.to(device)
            stylegan_scores = stylegan_scores.to(device)
            adm_scores = adm_scores.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, stylegan_scores.squeeze(), adm_scores.squeeze())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, stylegan_scores, adm_scores, labels in val_loader:
                images = images.to(device)
                stylegan_scores = stylegan_scores.to(device)
                adm_scores = adm_scores.to(device)
                labels = labels.to(device)
                
                outputs = model(images, stylegan_scores.squeeze(), adm_scores.squeeze())
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        print(f'Epoch [{epoch + 1}/{Config.epochs}] '
              f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'dm_classifier_efficientnet_best.pth', pickle_protocol=4)
            print("Best model saved with Val Acc:", best_val_acc)
        
        # 定期保存模型
        if (epoch + 1) % Config.save_interval == 0:
            torch.save(model.state_dict(), f'dm_classifier_efficientnet_epoch_{epoch + 1}.pth', pickle_protocol=4)
    
    # 测试阶段
    model.load_state_dict(torch.load('dm_classifier_efficientnet_best.pth', map_location=device, weights_only=True))
    model.eval()
    results = []
    with torch.no_grad():
        for images, stylegan_scores, adm_scores, labels in test_loader:
            images = images.to(device)
            stylegan_scores = stylegan_scores.to(device)
            adm_scores = adm_scores.to(device)
            labels = labels.to(device)
            
            outputs = model(images, stylegan_scores.squeeze(), adm_scores.squeeze())
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)
            
            # 记录结果
            for i in range(len(images)):
                results.append({
                    "Image Name": dataset.samples[test_set.indices[i]]['image_path'].split('/')[-1],
                    "StyleGAN_Score": stylegan_scores[i].item(),
                    "ADM_Score": adm_scores[i].item(),
                    "True Class": Config.dm_models[labels[i].item()],
                    "Predicted Class": Config.dm_models[predicted[i].item()],
                    "Confidence": max_probs[i].item()
                })
    
    pd.DataFrame(results).to_csv("dm_classifier_efficientnet_test_results.csv", index=False)
    print("Test results saved to dm_classifier_efficientnet_test_results.csv")

if __name__ == "__main__":
    csv_file = "data123.csv"  # 包含差异分数和图像名称的 CSV 文件
    root_dir = "./data_root"  # 数据集根目录
    train(csv_file, root_dir)