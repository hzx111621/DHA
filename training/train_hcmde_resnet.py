import os
import numpy as np
from collections import defaultdict
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

# ### 功能新增 ###: 引入 Matplotlib 用于生成热力图
import matplotlib.pyplot as plt

# =================================================================================
# 1. 配置参数 (与之前相同)
# =================================================================================
class Config:
    img_size = 256
    batch_size = 32
    lr = 3e-4
    epochs = 100
    GAN_MODELS = ['StyleGAN', 'CylcleGAN', 'StyleGAN2', 'BigGAN']
    DM_MODELS = ['ADM', 'SD1.5', 'Wukong', 'VQDM']
    ALL_MODELS = GAN_MODELS + DM_MODELS + ['REAL']
    model_to_idx = {name: i for i, name in enumerate(ALL_MODELS)}

# =================================================================================
# 2. H-CMDE 模型架构 (与之前相同)
# =================================================================================
# (CrossAttention, DecoderBlock, ResNetEncoder, HCMDE 类的代码在此处，与上一版完全相同，为简洁省略)
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        B, C, H, W = query.shape
        query = query.view(B, C, -1).permute(0, 2, 1)
        key = key.view(B, C, -1).permute(0, 2, 1)
        value = value.view(B, C, -1).permute(0, 2, 1)
        attn_output, _ = self.attn(query, key, value)
        return attn_output.permute(0, 2, 1).view(B, C, H, W)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        f0 = self.initial(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

class HCMDE(nn.Module):
    def __init__(self, d_model=512, n_heads=8, pretrained_encoder=True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained_encoder)
        self.cross_attention = CrossAttention(embed_dim=2048, num_heads=n_heads)
        self.bottleneck_conv = nn.Conv2d(2048 * 3, 2048, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.v_diff_mlp = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, d_model))
        self.decoder_block1 = DecoderBlock(2048, 1024, 1024)
        self.decoder_block2 = DecoderBlock(1024, 512, 512)
        self.decoder_block3 = DecoderBlock(512, 256, 256)
        self.final_upconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.map_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.score_head_mlp = nn.Linear(d_model + 1, 1)

    def forward(self, I_orig, I_recon):
        features_orig = self.encoder(I_orig)
        features_recon = self.encoder(I_recon)
        F_o1, F_o2, F_o3, F_o4 = features_orig
        F_r1, F_r2, F_r3, F_r4 = features_recon

        F_o4_ctx = self.cross_attention(F_o4, F_r4, F_r4)
        F_r4_ctx = self.cross_attention(F_r4, F_o4, F_o4)

        diff_abs = torch.abs(F_o4_ctx - F_r4_ctx)
        D_semantic = torch.cat([diff_abs, F_o4_ctx, F_r4_ctx], dim=1)
        x = self.bottleneck_conv(D_semantic)
        pooled_vec = self.pool(x).flatten(1)
        V_diff = self.v_diff_mlp(pooled_vec)

        d1 = self.decoder_block1(x, torch.abs(F_o3 - F_r3))
        d2 = self.decoder_block2(d1, torch.abs(F_o2 - F_r2))
        d3 = self.decoder_block3(d2, torch.abs(F_o1 - F_r1))
        final_features = self.final_upconv(d3)

        final_features = F.interpolate(final_features, size=I_orig.shape[2:], mode='bilinear', align_corners=False)
        M_diff_logits = self.map_conv(final_features)
        M_diff = torch.sigmoid(M_diff_logits)

        s_map_agg = self.pool(M_diff_logits).flatten(1)
        combined_score_features = torch.cat([V_diff, s_map_agg], dim=1)
        S_diff_logits = self.score_head_mlp(combined_score_features)
        S_diff = torch.sigmoid(S_diff_logits)

        return M_diff, V_diff, S_diff.squeeze()

# =================================================================================
# 3. 损失函数 (修改VGGPerceptualLoss为ResNetPerceptualLoss)
# =================================================================================
# SegmentAwareLoss (与之前相同)
class SegmentAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        weights = torch.ones_like(targets)
        weights[torch.isclose(targets, torch.tensor(0.0))] = 1.0
        weights[torch.isclose(targets, torch.tensor(0.3))] = 2.0
        weights[torch.isclose(targets, torch.tensor(0.5))] = 1.5
        weights[torch.isclose(targets, torch.tensor(0.7))] = 1.5
        
        loss = self.mse(outputs, targets)
        weighted_loss = loss * weights
        segment_penalty = torch.mean(torch.abs(outputs - targets) * (1 - torch.exp(-5 * torch.abs(outputs - targets))))
        
        return torch.mean(weighted_loss) + 0.2 * segment_penalty

# ### RESNET替换 ###: VGGPerceptualLoss替换为ResNetPerceptualLoss
class ResNetPerceptualLoss(nn.Module):
    """ L_map: 基于预训练ResNet的感知特征差异损失 """
    def __init__(self):
        super().__init__()
        # 使用预训练的ResNet50作为特征提取器
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 冻结ResNet网络，只用作特征提取器
        for p in resnet.parameters():
            p.requires_grad = False
            
        # 提取各个阶段作为感知特征的切片
        # stage 1: conv1 -> layer1 (输出 256 通道)
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1) 
        # stage 2: layer2 (输入 256 通道, 输出 512 通道)
        self.stage2 = resnet.layer2 
        # stage 3: layer3 (输入 512 通道, 输出 1024 通道)
        self.stage3 = resnet.layer3
        
        # 将所有阶段放入列表，便于迭代计算损失
        self.feature_stages = [self.stage1, self.stage2, self.stage3]

    def forward(self, M_diff_pred, I_orig, I_recon):
        loss = 0.0
        
        # 图像预处理 (与原代码相同)
        I_orig_resized = F.interpolate(I_orig, size=(224, 224), mode='bilinear', align_corners=False, recompute_scale_factor=False)
        I_recon_resized = F.interpolate(I_recon, size=(224, 224), mode='bilinear', align_corners=False, recompute_scale_factor=False)

        # 累积特征
        orig_x = I_orig_resized
        recon_x = I_recon_resized
        
        for stage_net in self.feature_stages:
            orig_x = stage_net(orig_x)
            recon_x = stage_net(recon_x)
            
            # 使用当前阶段的输出来计算感知差异图 D_gt
            D_gt = torch.abs(orig_x - recon_x).mean(dim=1, keepdim=True) # 在通道维度取平均
            
            # ... 后续的损失计算保持不变 ...
            M_diff_resized = F.interpolate(M_diff_pred, size=D_gt.shape[2:], mode='bilinear', align_corners=False, recompute_scale_factor=False)
            loss += F.mse_loss(M_diff_resized, D_gt)
            
        return loss / len(self.feature_stages)

# info_nce_loss (与之前相同)
def info_nce_loss(features, labels, temperature=0.1):
    if len(torch.unique(labels)) <= 1:
        return torch.tensor(0.0, device=features.device)
        
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T)
    mask.fill_diagonal_(False)
    
    positives = similarity_matrix[mask].view(features.size(0), -1)
    negatives = similarity_matrix[~mask].view(features.size(0), -1)
    
    if positives.size(1) == 0:
        return torch.tensor(0.0, device=features.device)
        
    logits = torch.cat([positives, negatives], dim=1)
    nce_labels = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
    logits = logits / temperature
    
    return F.cross_entropy(logits, nce_labels)

# =================================================================================
# 4. 数据集类 (与之前相同)
# =================================================================================
# (EnhancedDiffDataset 类的代码在此处，与上一版完全相同，为简洁省略)
class EnhancedDiffDataset(Dataset):
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
            
        for model_type in os.listdir(orig_base):
            model_path = os.path.join(orig_base, model_type)
            if not os.path.isdir(model_path):
                continue
                
            if model_type == "REAL":
                for recon_method in ['StyleGAN', 'ADM']:
                    recon_dir = os.path.join(recon_base, recon_method, "REAL")
                    if os.path.exists(recon_dir):
                        self._find_and_add_pairs(model_path, recon_dir, model_type, recon_method)
            else:
                categories = [cat for cat in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, cat))]
                for category in categories:
                    category_path = os.path.join(model_path, category)
                    for recon_method in ['StyleGAN', 'ADM']:
                        recon_dir = os.path.join(recon_base, recon_method, model_type, category)
                        if os.path.exists(recon_dir):
                            self._find_and_add_pairs(category_path, recon_dir, model_type, recon_method)
                            
        np.random.seed(42)
        indices = np.arange(len(self.pairs))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * 0.8)
        
        self.all_pairs = [self.pairs[i] for i in indices]
        
        if split == 'train':
            self.pairs = self.all_pairs[:split_idx]
        elif split == 'val':
            self.pairs = self.all_pairs[split_idx:]

    def _find_and_add_pairs(self, orig_dir, recon_dir, orig_model, recon_method):
        orig_files = set(f for f in os.listdir(orig_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        recon_files = set(f for f in os.listdir(recon_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        
        common_files = orig_files & recon_files
        
        for img_name in common_files:
            self.pairs.append((
                os.path.join(orig_dir, img_name),
                os.path.join(recon_dir, img_name),
                orig_model,
                recon_method
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig_path, recon_path, orig_model, recon_method = self.pairs[idx]
        
        seed = np.random.randint(2147483647)
        orig_img = PILImage.open(orig_path).convert('RGB')
        recon_img = PILImage.open(recon_path).convert('RGB')
        
        # 确保应用相同的随机变换
        torch.manual_seed(seed)
        orig_tensor = self.transform(orig_img)
        torch.manual_seed(seed)
        recon_tensor = self.transform(recon_img)
        
        target_score = self._calculate_target(orig_model, recon_method)
        source_label = Config.model_to_idx.get(orig_model, -1)
        
        return orig_tensor, recon_tensor, torch.tensor(target_score, dtype=torch.float32), torch.tensor(source_label, dtype=torch.long)

    def _calculate_target(self, orig_model, recon_method):
        if orig_model == recon_method:
            return 0.0
        elif orig_model == "REAL":
            return 0.5
        elif (orig_model in Config.GAN_MODELS and recon_method == "StyleGAN") or \
             (orig_model in Config.DM_MODELS and recon_method == "ADM"):
            return 0.3
        else:
            return 0.7

# =================================================================================
# 5. 可视化函数 (与之前相同)
# =================================================================================
def save_visualization_images(orig_imgs, recon_imgs, heatmaps, epoch, output_dir, num_images=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    num_to_save = min(num_images, orig_imgs.size(0))
    
    for i in range(num_to_save):
        # 准备图像和热力图
        orig, recon, heatmap = orig_imgs[i].detach().cpu(), recon_imgs[i].detach().cpu(), heatmaps[i].detach().cpu().squeeze()
        
        # 反归一化
        orig = orig * 0.5 + 0.5
        recon = recon * 0.5 + 0.5
        
        orig_np = orig.permute(1, 2, 0).numpy()
        recon_np = recon.permute(1, 2, 0).numpy()
        heatmap_np = heatmap.numpy()

        # 生成热力图 PIL 图像
        fig, ax = plt.subplots()
        ax.imshow(heatmap_np, cmap='hot', interpolation='nearest')
        ax.axis('off')
        fig.canvas.draw()
        heatmap_pil = PILImage.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)

        # 转换 NumPy 图像为 PIL 图像
        orig_pil = PILImage.fromarray((orig_np * 255).astype(np.uint8))
        recon_pil = PILImage.fromarray((recon_np * 255).astype(np.uint8))
        
        # 调整热力图大小以匹配其他图像
        if heatmap_pil.size != orig_pil.size:
            heatmap_pil = heatmap_pil.resize(orig_pil.size)
            
        # 拼接图像
        total_width = orig_pil.width * 3
        height = orig_pil.height
        composite_image = PILImage.new('RGB', (total_width, height))
        
        composite_image.paste(orig_pil, (0, 0))
        composite_image.paste(recon_pil, (orig_pil.width, 0))
        composite_image.paste(heatmap_pil, (orig_pil.width * 2, 0))
        
        # 保存图像
        save_path = os.path.join(output_dir, f'epoch_{epoch+1}_sample_{i}.png')
        composite_image.save(save_path)


# =================================================================================
# 6. 训练函数 (修改loss_map_fn为ResNet版本)
# =================================================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualization_dir = 'visualizations'

    # 1. 数据加载
    train_dataset = EnhancedDiffDataset('./data_root', split='train')
    val_dataset = EnhancedDiffDataset('./data_root', split='val')
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 模型、损失函数和优化器
    model = HCMDE().to(device)
    loss_task_fn = SegmentAwareLoss()
    # ### RESNET替换 ###: 使用新的 ResNetPerceptualLoss
    loss_map_fn = ResNetPerceptualLoss().to(device)  
    loss_vec_fn = info_nce_loss
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
    
    lambda_task, lambda_map, lambda_vec = 1.0, 0.5, 0.25

    # 3. 训练循环
    for epoch in range(Config.epochs):
        model.train()
        total_loss, task_loss_val, map_loss_val, vec_loss_val = 0.0, 0.0, 0.0, 0.0
        
        for orig, recon, targets, sources in train_loader:
            orig, recon, targets, sources = orig.to(device), recon.to(device), targets.to(device), sources.to(device)
            
            optimizer.zero_grad()
            
            M_diff, V_diff, S_diff = model(orig, recon)
            
            loss_task = loss_task_fn(S_diff, targets)
            loss_map = loss_map_fn(M_diff, orig, recon)
            loss_vec = loss_vec_fn(V_diff, sources)
            
            loss = (lambda_task * loss_task + lambda_map * loss_map + lambda_vec * loss_vec)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            task_loss_val += loss_task.item()
            map_loss_val += loss_map.item()
            vec_loss_val += loss_vec.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_task_loss = task_loss_val / len(train_loader)
        avg_map_loss = map_loss_val / len(train_loader)
        avg_vec_loss = vec_loss_val / len(train_loader)
        
        scheduler.step()
        
        print(f'Epoch [{epoch + 1}/{Config.epochs}] 训练总损失: {avg_train_loss:.4f} | '
              f'任务损失: {avg_task_loss:.4f}, 地图损失: {avg_map_loss:.4f}, 向量损失: {avg_vec_loss:.4f}')

        # 4. 验证循环
        model.eval()
        val_task_loss = 0.0
        with torch.no_grad():
            visualized_this_epoch = False
            for orig_val, recon_val, targets_val, _ in val_loader:
                orig_val, recon_val, targets_val = orig_val.to(device), recon_val.to(device), targets_val.to(device)
                
                M_diff_val, _, S_diff_val = model(orig_val, recon_val)
                val_task_loss += loss_task_fn(S_diff_val, targets_val).item()

                # 可视化保存
                if (epoch + 1) % 5 == 0 and not visualized_this_epoch:
                    save_visualization_images(
                        orig_imgs=orig_val,
                        recon_imgs=recon_val,
                        heatmaps=M_diff_val,
                        epoch=epoch,
                        output_dir=visualization_dir
                    )
                    visualized_this_epoch = True

            avg_val_loss = val_task_loss / len(val_loader)
            print(f'  验证集任务损失: {avg_val_loss:.4f}')
            if (epoch + 1) % 5 == 0:
                print(f"  可视化结果已保存至 '{visualization_dir}' 文件夹")

        # 5. 模型保存
        if (epoch + 1) % 10 == 0:
            save_path = f'hcmde_diff_extractor_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"模型权重已保存至 {save_path}")

    torch.save(model.state_dict(), 'hcmde_diff_extractor_final.pth')


if __name__ == '__main__':
    train()