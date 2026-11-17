import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# 配置参数
class Config:
    img_size = 256
    batch_size = 64
    lr = 1e-4
    epochs = 200
    save_interval = 10  # 每隔 10 轮保存权重并打印分数
    perturbation_range = 0.1  # 扰动范围
    num_classes = 9  # 类别数量（REAL, StyleGAN, StyleGAN2, BigGAN, ProGAN, ADM, Glide, Wukong, SD1.5）


# 差异提取器和分类器
class DifferenceExtractorWithClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享特征提取器
        backbone = models.resnet50(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # 差异性得分预测分支
        self.score_predictor = nn.Sequential(
            nn.Linear(2048 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)  # 输出单个分数
        )

        # 分类器分支
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, Config.num_classes)  # 输出类别预测
        )

        # 分数扰动模块
        self.perturbation_layer = nn.Sequential(
            nn.Linear(1, 1),  # 对分数进行轻微扰动
            nn.Tanh()
        )

    def forward(self, x1, x2):
        # 提取特征
        feat1 = self.feature_extractor(x1).flatten(1)
        feat2 = self.feature_extractor(x2).flatten(1)
        combined = torch.cat([feat1, feat2], dim=1)

        # 预测差异性得分
        scores = self.score_predictor(combined)
        perturbed_scores = scores + 0.1 * self.perturbation_layer(scores)  # 轻微扰动

        # 预测类别
        class_logits = self.classifier(feat1)  # 使用原始图像的特征预测类别

        return perturbed_scores.squeeze(), class_logits


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
        orig_img = Image.open(orig_path).convert('RGB')
        recon_img = Image.open(recon_path).convert('RGB')
        torch.manual_seed(seed)
        orig_tensor = self.transform(orig_img)
        torch.manual_seed(seed)
        recon_tensor = self.transform(recon_img)
        target_score = self._calculate_target(orig_model, recon_method)
        target_class = self._get_class_label(orig_model)
        return orig_tensor, recon_tensor, torch.tensor(target_score, dtype=torch.float32), torch.tensor(target_class,
                                                                                                        dtype=torch.long)

    def _calculate_target(self, orig_model, recon_method):
        # 固定分数规则
        if orig_model == "StyleGAN":
            return 0.0 if recon_method == "StyleGAN" else 0.7
        elif orig_model == "ADM":
            return 0.7 if recon_method == "StyleGAN" else 0.0
        elif orig_model == "REAL":
            return 0.7

        # 需要扰动的分数规则
        base_score = None
        if orig_model in ['StyleGAN2', 'BigGAN', 'ProGAN']:
            base_score = 0.3 if recon_method == "StyleGAN" else 0.7
        elif orig_model in ['Glide', 'Wukong', 'SD1.5']:
            base_score = 0.7 if recon_method == "StyleGAN" else 0.3

        # 添加随机扰动
        if base_score is not None:
            perturbation = np.random.uniform(-Config.perturbation_range, Config.perturbation_range)
            return base_score + perturbation

        # 默认返回值
        return 0.7

    def _get_class_label(self, orig_model):
        # 定义类别标签映射
        class_map = {
            "REAL": 0,
            "StyleGAN": 1,
            "StyleGAN2": 2,
            "BigGAN": 3,
            "ProGAN": 4,
            "ADM": 5,
            "Glide": 6,
            "Wukong": 7,
            "SD1.5": 8
        }
        return class_map.get(orig_model, 0)


# 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, pred_scores, true_scores, pred_classes, true_classes):
        # MSE 损失（差异性得分）
        score_loss = self.mse_loss(pred_scores, true_scores)

        # 分类损失
        class_loss = self.cross_entropy_loss(pred_classes, true_classes)

        # 总损失
        total_loss = score_loss + class_loss
        return total_loss


# 计算每个类别的平均差异得分
def calculate_average_scores(model, dataloader, device):
    model.eval()
    category_scores = {
        "REAL": {"StyleGAN": [], "ADM": []},
        "StyleGAN": {"StyleGAN": [], "ADM": []},
        "StyleGAN2": {"StyleGAN": [], "ADM": []},
        "BigGAN": {"StyleGAN": [], "ADM": []},
        "ProGAN": {"StyleGAN": [], "ADM": []},
        "ADM": {"StyleGAN": [], "ADM": []},
        "Glide": {"StyleGAN": [], "ADM": []},
        "Wukong": {"StyleGAN": [], "ADM": []},
        "SD1.5": {"StyleGAN": [], "ADM": []}
    }

    global_index = 0  # 全局索引
    with torch.no_grad():
        for orig, recon, targets_score, targets_class in dataloader:
            orig = orig.to(device)
            recon = recon.to(device)
            outputs_score, _ = model(orig, recon)

            # 获取类别和重建方法信息
            for i in range(len(outputs_score)):
                orig_path, recon_path, orig_model, recon_method = dataloader.dataset.pairs[global_index + i]
                if orig_model in category_scores and recon_method in ["StyleGAN", "ADM"]:
                    category_scores[orig_model][recon_method].append(outputs_score[i].item())

            global_index += len(outputs_score)  # 更新全局索引

    # 计算平均值
    average_scores = {}
    for category, scores_dict in category_scores.items():
        avg_stylegan = np.mean(scores_dict["StyleGAN"]) if scores_dict["StyleGAN"] else 0.0
        avg_adm = np.mean(scores_dict["ADM"]) if scores_dict["ADM"] else 0.0
        average_scores[category] = {"StyleGAN": avg_stylegan, "ADM": avg_adm}

    return average_scores


# 计算分类准确率
def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for orig, recon, _, targets_class in dataloader:
            orig = orig.to(device)
            recon = recon.to(device)
            targets_class = targets_class.to(device)
            _, outputs_class = model(orig, recon)
            _, predicted = torch.max(outputs_class, 1)
            total += targets_class.size(0)
            correct += (predicted == targets_class).sum().item()
    accuracy = correct / total
    return accuracy


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
    model = DifferenceExtractorWithClassifier().to(device)
    criterion = CustomLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    # 训练循环
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0
        for orig, recon, targets_score, targets_class in train_loader:
            orig = orig.to(device)
            recon = recon.to(device)
            targets_score = targets_score.to(device)
            targets_class = targets_class.to(device)
            optimizer.zero_grad()
            outputs_score, outputs_class = model(orig, recon)
            loss = criterion(outputs_score, targets_score, outputs_class, targets_class)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # 验证集评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for orig, recon, targets_score, targets_class in val_loader:
                orig = orig.to(device)
                recon = recon.to(device)
                targets_score = targets_score.to(device)
                targets_class = targets_class.to(device)
                outputs_score, outputs_class = model(orig, recon)
                loss = criterion(outputs_score, targets_score, outputs_class, targets_class)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        # 计算分类准确率
        train_accuracy = calculate_accuracy(model, train_loader, device)
        val_accuracy = calculate_accuracy(model, val_loader, device)

        print(f'Epoch [{epoch + 1}/{Config.epochs}] '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} '
              f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

        # 每隔 10 轮保存权重并计算平均分数
        if (epoch + 1) % Config.save_interval == 0:
            # 保存权重
            weight_path = f'difference_extractor_with_classifier_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), weight_path)
            print(f"Model weights saved to {weight_path}")

            # 计算并打印平均分数
            average_scores = calculate_average_scores(model, val_loader, device)
            print("Average Scores by Category:")
            for category, scores in average_scores.items():
                print(f"{category}: StyleGAN={scores['StyleGAN']:.4f}, ADM={scores['ADM']:.4f}")


if __name__ == "__main__":
    train()