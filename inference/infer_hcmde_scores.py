import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models # <-- 修正了 NameError: name 'models' is not defined
from PIL import Image as PILImage
import numpy as np
import csv
from typing import List, Tuple

# =================================================================================
# 1. 配置参数
# =================================================================================
class Config:
    img_size = 256
    batch_size = 32 # 推理时可调小，以适应GPU显存
    lr = 3e-4
    epochs = 100
    GAN_MODELS = ['StyleGAN', 'CylcleGAN', 'StyleGAN2', 'BigGAN']
    DM_MODELS = ['ADM', 'SD1.5', 'Wukong', 'VQDM']
    ALL_MODELS = GAN_MODELS + DM_MODELS + ['REAL']
    model_to_idx = {name: i for i, name in enumerate(ALL_MODELS)}

# =================================================================================
# 2. H-CMDE 模型架构
# =================================================================================

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
        # NameError 发生在这里，现在已通过顶部导入修正
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
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU(inplace=True)
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
# 3. 数据集类 (为推理添加了路径输出)
# =================================================================================
class EnhancedDiffDataset(Dataset):
    def __init__(self, root_dir, split='all'):
        self.pairs = []
        self.transform = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        
        orig_base = os.path.join(root_dir, 'originals')
        recon_base = os.path.join(root_dir, 'reconstructed')
        
        if not os.path.exists(orig_base) or not os.path.exists(recon_base): 
            raise ValueError(f"Directory not found at {root_dir}")
            
        for model_type in os.listdir(orig_base):
            model_path = os.path.join(orig_base, model_type)
            if not os.path.isdir(model_path): continue
            
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
        
        # 确保推理时使用所有数据，并保留原始顺序
        if split != 'all':
            np.random.seed(42)
            indices = np.arange(len(self.pairs))
            np.random.shuffle(indices)
            split_idx = int(len(indices) * 0.8)
            self.all_pairs = [self.pairs[i] for i in indices]
            if split == 'train': self.pairs = self.all_pairs[:split_idx]
            elif split == 'val': self.pairs = self.all_pairs[split_idx:]
        
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
        
        orig_img = PILImage.open(orig_path).convert('RGB')
        recon_img = PILImage.open(recon_path).convert('RGB')
        
        orig_tensor = self.transform(orig_img)
        recon_tensor = self.transform(recon_img)
        
        target_score = self._calculate_target(orig_model, recon_method)
        source_label = Config.model_to_idx.get(orig_model, -1)
        
        # 返回路径信息，以便追踪结果
        return orig_tensor, recon_tensor, orig_path, recon_path, torch.tensor(target_score, dtype=torch.float32), torch.tensor(source_label, dtype=torch.long)
        
    def _calculate_target(self, orig_model, recon_method):
        if orig_model == recon_method: return 0.0
        elif orig_model == "REAL": return 0.5
        elif (orig_model in Config.GAN_MODELS and recon_method == "StyleGAN") or (orig_model in Config.DM_MODELS and recon_method == "ADM"): return 0.3
        else: return 0.7

# =================================================================================
# 4. 差异分数提取主函数
# =================================================================================

def extract_diff_scores(model_path: str, data_root_dir: str, output_csv: str):
    """
    加载模型，在整个数据集上运行推理，并保存差异分数和向量。
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # 1. 初始化模型并加载权重
    # pretrained_encoder=False 在推理时是合理的，因为它只加载模型权重，而不是imagenet预训练权重。
    model = HCMDE(d_model=512, n_heads=8, pretrained_encoder=False).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 模型权重从 {model_path} 加载成功。")
    except Exception as e:
        print(f"❌ 加载模型权重失败: {e}")
        return

    model.eval()

    # 2. 准备数据集和 DataLoader
    inference_dataset = EnhancedDiffDataset(data_root_dir, split='all')
    inference_loader = DataLoader(
        inference_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    print(f"数据集大小: {len(inference_dataset.pairs)} 对图像")

    # 3. 推理循环
    results = []
    
    with torch.no_grad():
        for i, data in enumerate(inference_loader):
            # 解包数据：注意顺序与 __getitem__ 的输出顺序一致
            orig_t, recon_t, orig_paths, recon_paths, target_scores, source_labels = data
            
            orig_t = orig_t.to(device)
            recon_t = recon_t.to(device)

            # 获取模型的输出
            M_diff, V_diff, S_diff = model(orig_t, recon_t)
            
            # 将输出从 CUDA 转移到 CPU 并转换为 NumPy 数组
            S_diff_np = S_diff.cpu().numpy()
            V_diff_np = V_diff.cpu().numpy()
            
            for j in range(len(S_diff_np)):
                idx = i * Config.batch_size + j # 计算当前样本在总数据集中的索引
                
                result = {
                    'index': idx,
                    'orig_path': orig_paths[j],
                    'recon_path': recon_paths[j],
                    'orig_model': inference_dataset.pairs[idx][2],
                    'recon_method': inference_dataset.pairs[idx][3],
                    'target_score': target_scores[j].item(),
                    'predicted_score_S_diff': S_diff_np[j].item(),
                    'diff_vector_V_diff': V_diff_np[j].tolist() # 差异向量作为列表保存
                }
                results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1} / {len(inference_loader)} 批次...")

    print("推理完成。开始保存结果...")

    # 4. 保存结果到 CSV 文件
    
    if not results:
        print("没有结果需要保存。")
        return

    keys = list(results[0].keys())
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ 所有 {len(results)} 对图像的结果已保存到: {output_csv}")


if __name__ == '__main__':
    # =========================================================================
    # 请在这里修改您的模型和数据路径
    # =========================================================================
    MODEL_WEIGHTS_PATH = 'hcmde_diff_extractor_epoch_10.pth'  # 替换为您的模型路径
    DATA_ROOT_DIR = './data_root'                          # 替换为您的数据根目录
    OUTPUT_CSV_PATH = 'inference_results.csv'              # 结果文件名
    # =========================================================================

    extract_diff_scores(MODEL_WEIGHTS_PATH, DATA_ROOT_DIR, OUTPUT_CSV_PATH)