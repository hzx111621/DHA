import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

# 配置参数
class Config:
    img_size = 256
    batch_size = 64
    lr = 1e-4
    epochs = 100
    save_interval = 10
    dm_models = ['ADM', 'SD1.5', 'VQDM', 'Wukong']  # 目标分类的 DM 模型
    num_classes = len(dm_models)
    data_root = "./data_root"
    # EfficientNet-B4 标准归一化参数
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

# 数据集类（仅原始图像）
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_map = {model: idx for idx, model in enumerate(Config.dm_models)}
        
        # 遍历所有类别目录
        for model in Config.dm_models:
            model_dir = os.path.join(root_dir, 'originals', model)
            if not os.path.exists(model_dir):
                continue
            # 收集图像文件
            for img_name in os.listdir(model_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(model_dir, img_name), self.class_map[model]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 分类模型（EfficientNet-B4）
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的 EfficientNet-B4
        backbone = models.efficientnet_b4(pretrained=True)
        # 修改分类层（原分类层是 Linear(in_features=1792, out_features=1000)）
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),  # 保留原 dropout 层
            nn.Linear(num_ftrs, Config.num_classes)
        )
        self.model = backbone
    
    def forward(self, x):
        return self.model(x)

# 数据划分
def split_dataset(dataset):
    train_val_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=0.1,
        stratify=[label for _, label in dataset.samples],
        random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.1 / 0.9,  # 10% of original data
        stratify=[dataset.samples[i][1] for i in train_val_idx],
        random_state=42
    )
    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx)
    )

# 训练函数
def train():
    # 数据预处理（适配 EfficientNet）
    transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.normalize_mean, std=Config.normalize_std)
    ])
    # 加载数据集
    full_dataset = ImageDataset(root_dir=Config.data_root, transform=transform)
    train_set, val_set, test_set = split_dataset(full_dataset)
    
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    
    # 训练循环
    best_val_acc = 0.0
    for epoch in range(Config.epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        print(f'Epoch [{epoch+1}/{Config.epochs}] '
              f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'image_classifier_efficientnet_best.pth')
            print("Best model saved with Val Acc:", best_val_acc)
        
        # 定期保存模型
        if (epoch + 1) % Config.save_interval == 0:
            torch.save(model.state_dict(), f'image_classifier_efficientnet_epoch_{epoch+1}.pth')
    
    # 测试阶段
    model.load_state_dict(torch.load('image_classifier_efficientnet_best.pth', map_location=device))
    model.eval()
    results = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)
            
            for i in range(len(images)):
                results.append({
                    "Image": test_set.dataset.samples[test_set.indices[i]][0],
                    "True Class": Config.dm_models[labels[i].item()],
                    "Predicted Class": Config.dm_models[predicted[i].item()],
                    "Confidence": max_probs[i].item()
                })
    
    pd.DataFrame(results).to_csv("image_classification_efficientnet_test_results.csv", index=False)
    print("Test results saved to image_classification_efficientnet_test_results.csv")

if __name__ == "__main__":
    train()