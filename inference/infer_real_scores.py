import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd


# 配置参数
class Config:
    img_size = 256
    weights_path = "./difference_extractor_epoch_200.pth"  # 差异提取器模型权重路径
    output_csv = "./reconstruction_scores_real.csv"  # 输出 CSV 文件路径

    # 指定文件夹路径
    originals_dir = "/data/MM_AIGC/data_root/originals/REAL"  # 原始图像文件夹路径
    stylegan_recon_dir = "/data/MM_AIGC/data_root/reconstructed/StyleGAN/REAL"  # StyleGAN 重建后图像文件夹路径
    adm_recon_dir = "/data/MM_AIGC/data_root/reconstructed/ADM/REAL"  # ADM 重建后图像文件夹路径


# 差异提取器
class DifferenceExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 共享特征提取器
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])

        # 分数预测分支
        self.score_predictor = torch.nn.Sequential(
            torch.nn.Linear(2048 * 2, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1)  # 输出单个分数
        )

        # 分数扰动模块
        self.perturbation_layer = torch.nn.Sequential(
            torch.nn.Linear(1, 1),  # 对分数进行轻微扰动
            torch.nn.Tanh()
        )

    def forward(self, x1, x2):
        # 提取特征
        feat1 = self.feature_extractor(x1).flatten(1)
        feat2 = self.feature_extractor(x2).flatten(1)
        combined = torch.cat([feat1, feat2], dim=1)

        # 预测分数
        scores = self.score_predictor(combined)

        # 引入扰动
        perturbed_scores = scores + 0.1 * self.perturbation_layer(scores)  # 轻微扰动
        return perturbed_scores.squeeze()


# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加 batch 维度


# 计算差异性分数并保存到 CSV
def calculate_and_save_scores():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = DifferenceExtractor().to(device)
    model.load_state_dict(torch.load(Config.weights_path, map_location=device))
    model.eval()

    # 初始化结果列表
    results = []

    # 获取所有原始图像文件
    original_images = [f for f in os.listdir(Config.originals_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for orig_file in original_images:
        orig_path = os.path.join(Config.originals_dir, orig_file)
        stylegan_recon_path = os.path.join(Config.stylegan_recon_dir, orig_file)
        adm_recon_path = os.path.join(Config.adm_recon_dir, orig_file)

        # 检查重建图像是否存在
        if not os.path.exists(stylegan_recon_path) or not os.path.exists(adm_recon_path):
            print(f"Missing reconstruction for {orig_file}, skipping...")
            continue

        # 预处理图像
        original_image = preprocess_image(orig_path).to(device)
        stylegan_recon = preprocess_image(stylegan_recon_path).to(device)
        adm_recon = preprocess_image(adm_recon_path).to(device)

        # 计算分数
        with torch.no_grad():
            stylegan_score = model(original_image, stylegan_recon).item()
            adm_score = model(original_image, adm_recon).item()

        # 保存结果
        results.append({
            "Original_Image": orig_path,
            "StyleGAN_Reconstructed_Image": stylegan_recon_path,
            "ADM_Reconstructed_Image": adm_recon_path,
            "StyleGAN_Score": stylegan_score,
            "ADM_Score": adm_score
        })

    # 将结果保存为 CSV 文件
    df = pd.DataFrame(results)
    df.to_csv(Config.output_csv, index=False)
    print(f"Difference scores saved to {Config.output_csv}")


if __name__ == "__main__":
    # 检查路径是否有效
    if not os.path.exists(Config.originals_dir):
        raise FileNotFoundError(f"Original images directory not found at {Config.originals_dir}")
    if not os.path.exists(Config.stylegan_recon_dir):
        raise FileNotFoundError(f"StyleGAN reconstructed images directory not found at {Config.stylegan_recon_dir}")
    if not os.path.exists(Config.adm_recon_dir):
        raise FileNotFoundError(f"ADM reconstructed images directory not found at {Config.adm_recon_dir}")

    # 计算分数并保存到 CSV
    calculate_and_save_scores()