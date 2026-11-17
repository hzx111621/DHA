import os
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


# 配置参数
class Config:
    img_size = 256
    batch_size = 64
    lr = 3e-4  # 调整学习率
    epochs = 100
    # 模型类别定义
    GAN_MODELS = ['StyleGAN', 'ProGAN','CylcleGAN',  'REAL']
    DM_MODELS = ['ADM', 'VQDM','SD1.5']
    ALL_MODELS = GAN_MODELS + DM_MODELS


# 纯差异提取器
class PureDiffExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享特征提取器
        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        # 差异回归网络
        self.diff_net = nn.Sequential(
            nn.Linear(2048 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),  # 调整Dropout比例
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        feat1 = self.feature_extractor(x1).flatten(1)
        feat2 = self.feature_extractor(x2).flatten(1)
        combined = torch.cat([feat1, feat2], dim=1)
        return self.diff_net(combined).squeeze()


# 自定义分段损失函数
class SegmentAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        # 动态权重分配
        # weights = torch.ones_like(targets)
        # weights[torch.isclose(targets, torch.tensor(0.0))] = 1.0
        # weights[torch.isclose(targets, torch.tensor(0.3))] = 2.0
        # weights[torch.isclose(targets, torch.tensor(0.7))] = 1.5

        # 计算加权MSE
        loss = self.mse(outputs, targets)
        return torch.mean(loss)
        # weighted_loss = loss * weights

        # 增加分段约束项
        # segment_penalty = torch.mean(torch.abs(outputs - targets) * (1 - torch.exp(-5 * torch.abs(outputs - targets))))
        # return torch.mean(weighted_loss) + 0.2 * segment_penalty




# 数据集类（保持原实现不变）
class PureDiffDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.pairs = []
        self.transform = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        # 构建图像对路径
        orig_base = os.path.join(root_dir, 'originals')
        recon_base = os.path.join(root_dir, 'reconstructed')
        if not os.path.exists(orig_base) or not os.path.exists(recon_base):
            raise ValueError(f"Originals or Reconstructed directory not found at {root_dir}")
        # 遍历所有原始模型
        for model_type in os.listdir(orig_base):
            model_path = os.path.join(orig_base, model_type)
            if not os.path.isdir(model_path):
                continue
            # 动态获取类别
            categories = [cat for cat in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, cat))]
            for category in categories:
                category_path = os.path.join(model_path, category)
                if not os.path.exists(category_path):
                    continue
                # 加载类别目录下的图像文件
                orig_files = [
                    f for f in os.listdir(category_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                if not orig_files:
                    continue
                # 动态检查重建目录
                for recon_method in ['StyleGAN', 'ADM']:
                    recon_dir = os.path.join(recon_base, recon_method)
                    if os.path.exists(recon_dir):
                        self._find_and_add_pairs(category_path, recon_dir, category, recon_method)
        # 划分训练集和验证集
        np.random.seed(42)  # 固定随机种子以确保划分一致
        indices = np.arange(len(self.pairs))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * 0.8)  # 80% 训练集，20% 验证集
        if split == 'train':
            self.pairs = [self.pairs[i] for i in indices[:split_idx]]
        elif split == 'val':
            self.pairs = [self.pairs[i] for i in indices[split_idx:]]
    def _find_and_add_pairs(self, orig_dir, recon_dir, orig_model, recon_method):
        orig_files = set(
            f for f in os.listdir(orig_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        for root, _, files in os.walk(recon_dir):
            recon_files = set(
                f for f in files
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            )
            common_files = orig_files & recon_files
            for img_name in common_files:
                orig_path = os.path.join(orig_dir, img_name)
                recon_path = os.path.join(root, img_name)
                self.pairs.append((orig_path, recon_path, orig_model, recon_method))
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        orig_path, recon_path, orig_model, recon_method = self.pairs[idx]
        seed = np.random.randint(2147483647)
        orig_img = PILImage.open(orig_path).convert('RGB')
        recon_img = PILImage.open(recon_path).convert('RGB')
        torch.manual_seed(seed)
        orig_tensor = self.transform(orig_img)
        torch.manual_seed(seed)
        recon_tensor = self.transform(recon_img)
        target = self._calculate_target(orig_model, recon_method)
        return orig_tensor, recon_tensor, torch.tensor(target, dtype=torch.float32)
    def _calculate_target(self, orig_model, recon_method):
        if orig_model == recon_method:
            return 0.0
        elif orig_model == "REAL":
            if recon_method in ["StyleGAN", "ADM"]:
                return 0.7
        elif (orig_model in Config.GAN_MODELS and recon_method == "StyleGAN") or \
             (orig_model in Config.DM_MODELS and recon_method == "ADM"):
            return 0.3
        else:
            return 0.7

# 训练函数
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化数据集和数据加载器
    train_dataset = PureDiffDataset('./data_root', split='train')
    val_dataset = PureDiffDataset('./data_root', split='val')
    train_loader = DataLoader(train_dataset,
                              batch_size=Config.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=Config.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # 初始化模型和优化器
    model = PureDiffExtractor().to(device)
    criterion = SegmentAwareLoss()  # 使用新的损失函数
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

    # 训练循环
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0
        for orig, recon, targets in train_loader:
            orig = orig.to(device)
            recon = recon.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(orig, recon)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # 验证集评估
        model.eval()
        val_loss = 0.0
        category_scores = {model_type: {"StyleGAN": [], "ADM": []} for model_type in Config.ALL_MODELS}

        with torch.no_grad():
            for orig, recon, targets in val_loader:
                orig = orig.to(device)
                recon = recon.to(device)
                targets = targets.to(device)

                outputs = model(orig, recon)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # 收集每个类别的得分
                for output, orig_model, recon_method in zip(outputs.cpu().numpy(),
                                                            [Config.ALL_MODELS[int(t)] for t in targets.cpu().numpy()],
                                                            [pair[3] for pair in val_dataset.pairs]):
                    category_scores[orig_model][recon_method].append(output)

        avg_val_loss = val_loss / len(val_loader)

        # 打印损失
        print(f'Epoch [{epoch + 1}/{Config.epochs}] '
              f'Train Loss: {avg_train_loss:.4f} '
              f'Val Loss: {avg_val_loss:.4f}')

        # 保存模型
        if (epoch + 1) % 10 == 0:
            save_path = f'pure_diff_extractor_mse_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")

    # 最终保存模型
    torch.save(model.state_dict(), 'pure_diff_extractor_80_final.pth')


if __name__ == '__main__':
    train()