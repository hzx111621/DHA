import os
import numpy as np
from collections import defaultdict
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# 配置参数
class Config:
    img_size = 256
    batch_size = 64
    lr = 3e-4
    epochs = 100
    GAN_MODELS = ['StyleGAN', 'CylcleGAN', 'StyleGAN2', 'BigGAN']
    DM_MODELS = ['ADM', 'SD1.5', 'Wukong', 'VQDM']
    ALL_MODELS = GAN_MODELS + DM_MODELS + ['REAL']

# 纯差异提取器（使用 ResNet50）
class PureDiffExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # 去掉最后的全连接层
        # 差异回归网络
        self.diff_net = nn.Sequential(
            nn.Linear(2048 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        feat1 = self.feature_extractor(x1).flatten(1)  # 展平特征图
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
        weights = torch.ones_like(targets)
        weights[torch.isclose(targets, torch.tensor(0.0))] = 1.0
        weights[torch.isclose(targets, torch.tensor(0.3))] = 2.0
        weights[torch.isclose(targets, torch.tensor(0.7))] = 1.5

        # 计算加权MSE
        loss = self.mse(outputs, targets)
        weighted_loss = loss * weights

        # 分段约束项
        segment_penalty = torch.mean(torch.abs(outputs - targets) * (1 - torch.exp(-5 * torch.abs(outputs - targets))))
        return torch.mean(weighted_loss) + 0.2 * segment_penalty

# 数据集类
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
        orig_base = os.path.join(root_dir, 'originals')
        recon_base = os.path.join(root_dir, 'reconstructed')
        
        if not os.path.exists(orig_base) or not os.path.exists(recon_base):
            raise ValueError(f"Directory not found at {root_dir}")

        # 遍历所有原始模型
        # 遍历所有原始模型
        for model_type in os.listdir(orig_base):
            model_path = os.path.join(orig_base, model_type)
            if not os.path.isdir(model_path):
                continue
        
            # 特殊处理 REAL 类别（无子目录）
            if model_type == "REAL":
                orig_files = [
                    f for f in os.listdir(model_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                if not orig_files:
                    print(f"No images found for REAL in {model_path}")
                    continue
                # 动态检查重建目录
                for recon_method in ['StyleGAN', 'ADM']:
                    recon_dir = os.path.join(recon_base, recon_method, "REAL")
                    if not os.path.exists(recon_dir):
                        print(f"Reconstructed directory not found: {recon_dir}")
                        continue
                    self._find_and_add_pairs(model_path, recon_dir, model_type, recon_method)
            else:
                # 其他类别按原逻辑处理子目录
                categories = [cat for cat in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, cat))]
                for category in categories:
                    category_path = os.path.join(model_path, category)
                    if not os.path.exists(category_path):
                        continue
                    orig_files = [
                        f for f in os.listdir(category_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ]
                    if not orig_files:
                        continue
                    # 动态检查重建目录
                    for recon_method in ['StyleGAN', 'ADM']:
                        recon_dir = os.path.join(recon_base, recon_method, model_type, category)
                        if os.path.exists(recon_dir):
                            self._find_and_add_pairs(category_path, recon_dir, model_type, recon_method)

        # 划分训练集和验证集
        np.random.seed(42)
        indices = np.arange(len(self.pairs))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * 0.8)
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
                print(f"Added pair: {orig_path} -> {recon_path}")  # 调试信息

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            orig_path, recon_path, orig_model, recon_method = self.pairs[idx]
        except IndexError:
            print(f"Error: Index {idx} is out of range for dataset of length {len(self.pairs)}")
            raise
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
                return 0.5  # REAL 类别的目标值
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

    # 打印样本分布
    train_counts = defaultdict(int)
    for _, _, _, orig_model in train_dataset.pairs:
        train_counts[orig_model] += 1
    print("Train sample distribution:", dict(train_counts))

    val_counts = defaultdict(int)
    for _, _, _, orig_model in val_dataset.pairs:
        val_counts[orig_model] += 1
    print("Validation sample distribution:", dict(val_counts))

    # 初始化模型和优化器
    model = PureDiffExtractor().to(device)
    criterion = SegmentAwareLoss()
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
        category_scores = defaultdict(lambda: defaultdict(list))

        with torch.no_grad():
            for orig, recon, targets in val_loader:
                orig = orig.to(device)
                recon = recon.to(device)
                targets = targets.to(device)

                outputs = model(orig, recon)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # 收集每个类别的得分
                for idx in range(len(outputs)):
                    orig_model = val_dataset.pairs[idx][2]
                    recon_method = val_dataset.pairs[idx][3]
                    output = outputs[idx].item()
                    category_scores[orig_model][recon_method].append(output)

        avg_val_loss = val_loss / len(val_loader)

        # 打印损失和 REAL 类别得分
        print(f'Epoch [{epoch + 1}/{Config.epochs}] '
              f'Train Loss: {avg_train_loss:.4f} '
              f'Val Loss: {avg_val_loss:.4f}')

        # 打印 REAL 类别的平均得分
        real_stylegan_scores = category_scores["REAL"]["StyleGAN"]
        real_adm_scores = category_scores["REAL"]["ADM"]
        real_stylegan_avg = np.mean(real_stylegan_scores) if real_stylegan_scores else 0
        real_adm_avg = np.mean(real_adm_scores) if real_adm_scores else 0
        print(f"  REAL Scores: StyleGAN={real_stylegan_avg:.4f}, ADM={real_adm_avg:.4f}")

        # 保存模型
        if (epoch + 1) % 10 == 0:
            save_path = f'pure_diff_extractor_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")

    torch.save(model.state_dict(), 'pure_diff_extractor_final.pth')

if __name__ == '__main__':
    train()