import os
import sys

print("--- DEBUG INFO ---")
print(f"Current Working Directory: {os.getcwd()}")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Script Directory: {SCRIPT_DIR}")
DINOV2_CODE_PATH = os.path.join(SCRIPT_DIR, 'dinov2')
print(f"Attempting to add this path for DINOv2 code: {DINOV2_CODE_PATH}")
print(f"Does this path exist? {os.path.exists(DINOV2_CODE_PATH)}")
print("--- END DEBUG INFO ---")

# ... 你的其他代码从这里开始 ...



import os
import numpy as np
from collections import defaultdict
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import timm
import matplotlib.pyplot as plt

# 配置参数
class Config:
    img_size = 252
    batch_size = 32
    lr = 1e-4
    epochs = 200
    # 真正的生成模型列表
    GAN_MODELS = ['StyleGAN', 'CylcleGAN', 'StyleGAN2', 'BigGAN'] 
    DM_MODELS = ['ADM', 'SD1.5', 'Wukong', 'VQDM']
    
    # 包含 REAL 用于数据遍历
    ALL_MODELS = GAN_MODELS + DM_MODELS + ['REAL'] 
    
    magnitude_loss_weight = 0.5
    semantic_loss_weight = 1

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib.util # 导入我们需要的库

# ------------------- 终极离线加载方案 (正确动态加载版) -------------------

# 步骤 1: 定义 dinov2 源代码和权重的路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DINOV2_REPO_PATH = os.path.join(SCRIPT_DIR, 'dinov2')
HUB_CONF_PATH = os.path.join(DINOV2_REPO_PATH, 'hubconf.py')
DINOV2_LOCAL_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "weights", "dinov2_vits14_pretrain.pth")

# 步骤 2: 将 dinov2 的根目录添加到 sys.path
# 这是为了让 hubconf.py 内部的 "from dinov2 import models" 能够正确工作
if DINOV2_REPO_PATH not in sys.path:
    sys.path.insert(0, DINOV2_REPO_PATH)

# 步骤 3: 使用 importlib 动态加载 hubconf.py 文件为一个模块
try:
    print(f"Dynamically loading module from: {HUB_CONF_PATH}")
    spec = importlib.util.spec_from_file_location("hubconf", HUB_CONF_PATH)
    hubconf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hubconf)
    print("Module 'hubconf' loaded successfully.")
except Exception as e:
    print(f"\nFATAL ERROR: Failed to dynamically load hubconf.py.")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------------


class HASNet(nn.Module):
    def __init__(self, embed_dim=384, n_heads=6):
        super().__init__()
        
        print("Creating DINOv2 model structure from local files...")
        # 现在我们可以安全地使用刚刚动态加载的 hubconf 模块了
        self.backbone = hubconf.dinov2_vits14(pretrained=False)
        
        print(f"Loading weights from local file: {DINOV2_LOCAL_WEIGHTS_PATH}")
        if not os.path.exists(DINOV2_LOCAL_WEIGHTS_PATH):
             print(f"错误: 权重文件未找到! 路径: {DINOV2_LOCAL_WEIGHTS_PATH}")
             sys.exit(1)
        state_dict = torch.load(DINOV2_LOCAL_WEIGHTS_PATH, map_location='cpu')
        self.backbone.load_state_dict(state_dict)
        print("DINOV2 weights loaded successfully in offline mode.")

        # --- 模型的其余部分保持不变 ---
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.magnitude_head = nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1), nn.Sigmoid())
        self.semantic_head = nn.Sequential(nn.Linear(embed_dim, 512), nn.GELU(), nn.Linear(512, 128))

# 请将 HASNet 类中的整个 forward 方法替换为以下内容
# training/train_hasnet_dinov2.py -> HASNet -> forward()

    # training/train_hasnet_dinov2.py -> HASNet -> forward()

# --- 请用这个完整的、最终修正版的 forward 方法替换您文件中的旧版本 ---

# --- training/train_hasnet_dinov2.py -> HASNet -> forward() ---
# --- 这是最终的、经过修正的、可以直接运行的版本 ---

    def forward(self, x1, x2):
        self.backbone.eval()
        with torch.no_grad():
            feat1 = self.backbone.forward_features(x1)['x_norm_patchtokens'] # Shape: (B, N, D)
            feat2 = self.backbone.forward_features(x2)['x_norm_patchtokens'] # Shape: (B, N, D)

        # --- 核心修正：适配 batch_first=False 的行为 ---
        # F.multi_head_attention_forward 默认期望 (N, B, D)
        # 所以我们需要交换 batch 和 patch 维度
        feat1 = feat1.transpose(0, 1) # Shape: (N, B, D)
        feat2 = feat2.transpose(0, 1) # Shape: (N, B, D)

        attended_feat, attn_weights = F.multi_head_attention_forward(
            query=feat1,
            key=feat2,
            value=feat2,
            embed_dim_to_check=self.cross_attn.embed_dim,
            num_heads=self.cross_attn.num_heads,
            in_proj_weight=self.cross_attn.in_proj_weight,
            in_proj_bias=self.cross_attn.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.cross_attn.dropout,
            out_proj_weight=self.cross_attn.out_proj.weight,
            out_proj_bias=self.cross_attn.out_proj.bias,
            training=self.training,
            need_weights=True,
            average_attn_weights=False,
        )
        
        # --- 核心修正：将输出维度交换回来 ---
        # attended_feat 的输出维度是 (N, B, D)，我们需要 (B, N, D)
        attended_feat = attended_feat.transpose(0, 1) # Shape: (B, N, D)

        # --- 后续计算现在是完全正确的 ---
        attended_feat = self.attn_norm(attended_feat)
        raw_signature_vector = torch.mean(attended_feat, dim=1)
        magnitude = self.magnitude_head(raw_signature_vector).squeeze(-1)
        semantic_embedding = F.normalize(self.semantic_head(raw_signature_vector), p=2, dim=1)
        
        # --- saliency_map 的计算现在也是正确的 ---
        # attn_weights 的形状现在是 (B, H, N_query, N_key) -> (32, 6, 324, 324)
        saliency_scores_per_patch = attn_weights.mean(dim=(1, 3)) # Shape: (B, N_query) -> (32, 324)
        
        num_patches = attended_feat.shape[1] # 从正确的 (B, N, D) 形状中获取 N
        h = w = int(np.sqrt(num_patches))

        if h * w != num_patches:
            raise ValueError(f"The number of patches ({num_patches}) is not a perfect square. Check your image transforms.")

        batch_size = saliency_scores_per_patch.shape[0] # 现在 batch_size 会是 32
        saliency_map = saliency_scores_per_patch.view(batch_size, h, w)
        
        saliency_map = F.interpolate(
            saliency_map.unsqueeze(1),
            size=(Config.img_size, Config.img_size),
            mode='bilinear',
            align_corners=False
        )
        
        saliency_map = saliency_map.squeeze(1)

        return {
            "magnitude": magnitude,
            "semantic_embedding": semantic_embedding,
            "saliency_map": saliency_map,
            "attention_weights": attn_weights
        }
# ----------------------------------------------------------------------------
# 2. 损失函数升级: 复合损失
# ----------------------------------------------------------------------------
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss

class HASNetLoss(nn.Module):
    def __init__(self, magnitude_weight=1.0, semantic_weight=0.5):
        super().__init__()
        self.magnitude_weight = magnitude_weight
        self.semantic_weight = semantic_weight
        self.magnitude_criterion = nn.MSELoss() # 使用简单的 MSE，你也可以换回你的 SegmentAwareLoss
        self.semantic_criterion = SupervisedContrastiveLoss()

    def forward(self, outputs, targets):
        loss_mag = self.magnitude_criterion(outputs['magnitude'], targets['magnitude'])
        # 只有当一个 batch 中有正样本对时才计算对比损失
        unique_labels = torch.unique(targets['label'])
        if len(unique_labels) < len(targets['label']):
            loss_sem = self.semantic_criterion(outputs['semantic_embedding'], targets['label'])
        else:
            loss_sem = torch.tensor(0.0, device=outputs['magnitude'].device)

        total_loss = self.magnitude_weight * loss_mag + self.semantic_weight * loss_sem
        return total_loss, loss_mag, loss_sem

# ----------------------------------------------------------------------------
# 3. 数据集升级: 增加类别标签
# ----------------------------------------------------------------------------
class HASNetDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.pairs = []
        self.class_to_idx = {}
        
        print("--- Starting Dataset Construction ---")
        self._build_pairs()
        
        print(f"--- Dataset Construction Finished ---")
        print(f"Total pairs found: {len(self.pairs)}")
        print(f"Total classes for contrastive loss: {len(self.class_to_idx)}")
        
        # 打印每种目标分数的样本数量，用于调试
        self._print_target_distribution()

        # 划分训练集和验证集
        np.random.seed(42)
        indices = np.arange(len(self.pairs))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * 0.8)
        
        if split == 'train':
            self.pairs = [self.pairs[i] for i in indices[:split_idx]]
            print(f"Train split size: {len(self.pairs)}")
        elif split == 'val':
            self.pairs = [self.pairs[i] for i in indices[split_idx:]]
            print(f"Validation split size: {len(self.pairs)}")

# -------------------- 为您的两层目录结构量身定制的 _build_pairs 函数 --------------------
# -------------------- 最终优化版：适配“REAL in GAN”结构 --------------------
    def _build_pairs(self):
        """
        健壮地遍历您的两层目录结构 (GAN/xxx, DM/xxx)。
        这个版本将 REAL 视为 GAN 或 DM 目录下的一个普通子目录。
        """
        orig_base = os.path.join(self.root_dir, 'originals')
        recon_base = os.path.join(self.root_dir, 'reconstructed')

        if not os.path.exists(orig_base):
            print(f"FATAL ERROR: Originals directory not found at: {orig_base}")
            return

        print(f"--- Found top-level directories: {os.listdir(orig_base)} ---")

        # 步骤1：遍历顶层目录，即 ['DM', 'GAN']
        for top_level_dir in os.listdir(orig_base):
            top_level_path = os.path.join(orig_base, top_level_dir)
            if not os.path.isdir(top_level_path):
                continue

            # 步骤2：遍历子模型目录，例如 'StyleGAN', 'BigGAN', 'REAL'
            for orig_model in os.listdir(top_level_path):
                orig_model_path = os.path.join(top_level_path, orig_model)
                if not os.path.isdir(orig_model_path):
                    continue

                # 步骤3：遍历两种重建方法
                for recon_method in ['StyleGAN', 'ADM']:
                    # 构建重建路径
                    # 假设重建目录也遵循 GAN/DM 的二级结构, 例如: reconstructed/StyleGAN/GAN/REAL
                    recon_model_path = os.path.join(recon_base, recon_method, top_level_dir, orig_model)
                    
                    if not os.path.exists(recon_model_path):
                        # 如果上面的路径找不到，再尝试一个更扁平的结构，以增加兼容性
                        # 例如: reconstructed/StyleGAN/REAL
                        flat_recon_path = os.path.join(recon_base, recon_method, orig_model)
                        if os.path.exists(flat_recon_path):
                             recon_model_path = flat_recon_path
                        else:
                             # 如果两种路径都找不到，说明这个组合没有对应的重建数据，跳过
                             continue
                    
                    print(f"Processing pair: Original='{orig_model}' (from {top_level_dir}) -> Reconstructed by='{recon_method}'")
                    # orig_model 的值会是 'REAL', 'StyleGAN' 等，这正是我们需要的
                    self._find_and_add_pairs(orig_model_path, recon_model_path, orig_model, recon_method)

    def _find_and_add_pairs(self, orig_dir, recon_dir, orig_model, recon_method):
        """
        在一个给定的 (原始目录, 重建目录) 对中，查找同名文件并添加到数据集中。
        """
        # 使用 os.walk 遍历所有子目录，以处理像 StyleGAN/cat, StyleGAN/dog 这样的结构
        for dirpath, _, filenames in os.walk(orig_dir):
            if not filenames:
                continue

            # 计算相对路径，以便在重建目录中找到对应的位置
            relative_path = os.path.relpath(dirpath, orig_dir)
            current_recon_dir = os.path.join(recon_dir, relative_path)

            if not os.path.exists(current_recon_dir):
                continue

            for img_name in filenames:
                # 检查文件扩展名
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                orig_path = os.path.join(dirpath, img_name)
                recon_path = os.path.join(current_recon_dir, img_name)
                
                # 最终确认重建文件是否存在
                if os.path.exists(recon_path):
                    self.pairs.append((orig_path, recon_path, orig_model, recon_method))
                    
                    # 更新用于对比学习的类别标签
                    artifact_class = f"{orig_model}_vs_{recon_method}"
                    if artifact_class not in self.class_to_idx:
                        self.class_to_idx[artifact_class] = len(self.class_to_idx)
    
    def _print_target_distribution(self):
        """打印数据集中每种目标分数的样本数量。"""
        counts = defaultdict(int)
        for _, _, orig_model, recon_method in self.pairs:
            target = self._calculate_magnitude_target(orig_model, recon_method)
            counts[target] += 1
        print("--- Target Score Distribution ---")
        for target, count in sorted(counts.items()):
            print(f"Target {target:.1f}: {count} samples")
        print("---------------------------------")


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig_path, recon_path, orig_model, recon_method = self.pairs[idx]
        
        orig_img = PILImage.open(orig_path).convert('RGB')
        recon_img = PILImage.open(recon_path).convert('RGB')
        
        orig_tensor = self.transform(orig_img)
        recon_tensor = self.transform(recon_img)
        
        magnitude_target = self._calculate_magnitude_target(orig_model, recon_method)
        artifact_class = f"{orig_model}_vs_{recon_method}"
        label = self.class_to_idx[artifact_class]
        
        targets = {
            "magnitude": torch.tensor(magnitude_target, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long)
        }
        
        return orig_tensor, recon_tensor, targets

    def _calculate_magnitude_target(self, orig_model, recon_method):
        """
        更新后的逻辑：
        - 0.0: 完全匹配 (如 StyleGAN -> StyleGAN)
        - 0.7: 真实图像 (REAL) 被任何方法重建
        - 0.3: 同族生成模型重建 (如 BigGAN -> StyleGAN)
        - 0.7: 跨族重建 (如 SD1.5 -> StyleGAN)
        """
        
        # 优先级 1: 完全匹配
        if orig_model == recon_method:
            return 0.0

        # 优先级 2: 只要原图是真实图像 (REAL)，无论被谁重建，目标都是 0.7
        if orig_model == 'REAL':
            return 0.7

        # 优先级 3: 处理生成模型内部的“大类匹配”
        # 此时 orig_model 只可能是 GAN 或 DM 类
        is_gan_family_match = (orig_model in Config.GAN_MODELS and recon_method == 'StyleGAN')
        is_dm_family_match = (orig_model in Config.DM_MODELS and recon_method == 'ADM')
        
        if is_gan_family_match or is_dm_family_match:
            return 0.3

        # 优先级 4: 跨架构重建 (如 DM -> StyleGAN)
        return 0.7
# ----------------------------------------------------------------------------
# 4. 训练流程升级: 增加可视化
# ----------------------------------------------------------------------------
def visualize_saliency_maps(epoch, images, saliency_maps, save_dir='visualizations'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    images = images.cpu()
    saliency_maps = saliency_maps.cpu().detach().numpy()
    
    # 反标准化
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    num_images = images.shape[0]
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))
    
    for i in range(num_images):
        img = inv_normalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        saliency_map = saliency_maps[i]
        
        # 原始图像
        axes[0, i].imshow(img)
        axes[0, i].set_title("Original Image")
        axes[0, i].axis('off')
        
        # 显著图叠加
        axes[1, i].imshow(img)
        axes[1, i].imshow(saliency_map, cmap='jet', alpha=0.5)
        axes[1, i].set_title("Artifact Saliency")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1}_saliency_map.png'))
    plt.close()
    print(f"Saved saliency map visualization to {save_dir}/epoch_{epoch+1}_saliency_map.png")


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = HASNetDataset('./data_root', split='train')
    val_dataset = HASNetDataset('./data_root', split='val')
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = HASNet().to(device)
    criterion = HASNetLoss(Config.magnitude_loss_weight, Config.semantic_loss_weight)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

    for epoch in range(Config.epochs):
        model.train()
        total_loss, total_mag_loss, total_sem_loss = 0, 0, 0
        
        for orig, recon, targets in train_loader:
            orig, recon = orig.to(device), recon.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            outputs = model(orig, recon)
            loss, loss_mag, loss_sem = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mag_loss += loss_mag.item()
            total_sem_loss += loss_sem.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_mag_loss = total_mag_loss / len(train_loader)
        avg_sem_loss = total_sem_loss / len(train_loader)
        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (orig, recon, targets) in enumerate(val_loader):
                orig, recon = orig.to(device), recon.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                outputs = model(orig, recon)
                loss, _, _ = criterion(outputs, targets)
                val_loss += loss.item()

                # 每隔五轮，保存第一批验证集图像用于可视化
                if (epoch + 1) % 5 == 0 and i == 0:
                    visualize_saliency_maps(epoch, orig[:5], outputs['saliency_map'][:5])

        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch + 1}/{Config.epochs}] | Train Loss: {avg_train_loss:.4f} '
              f'(Mag: {avg_mag_loss:.4f}, Sem: {avg_sem_loss:.4f}) | Val Loss: {avg_val_loss:.4f}')

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f'hasnet0.5_1_epoch_{epoch + 1}.pth')

    torch.save(model.state_dict(), 'hasnet0.5_1_final.pth')

if __name__ == '__main__':
    # 确保你的数据根目录是 './data_root'
    # 目录结构示例:
    # ./data_root/
    # ├── originals/
    # │   ├── REAL/
    # │   ├── StyleGAN/
    # │   │   └── category1/ ...
    # │   └── ADM/
    # │       └── category1/ ...
    # └── reconstructed/
    #     ├── StyleGAN/
    #     │   ├── REAL/
    #     │   └── StyleGAN/
    #     │       └── category1/ ...
    #     └── ADM/
    #         ├── REAL/
    #         └── ...
    train()