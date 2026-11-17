import os
import torch
import pandas as pd
from PIL import Image as PILImage
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 导入你定义的模型类 (确保 HASNet 类在当前文件或已导入)
from training.train_hasnet_dinov2 import HASNet, Config

## --- 为了脚本独立性，请确保以下 transform 与训练时完全一致 ---
test_transform = transforms.Compose([
    transforms.Resize((252, 252)),  # 必须与 Config.img_size 一致
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@torch.no_grad()
def get_score(model, device, orig_path, recon_path):
    """计算单对图像的差异性分数 (Magnitude)"""
    if not os.path.exists(orig_path) or not os.path.exists(recon_path):
        return None

    orig_img = test_transform(PILImage.open(orig_path).convert('RGB')).unsqueeze(0).to(device)
    recon_img = test_transform(PILImage.open(recon_path).convert('RGB')).unsqueeze(0).to(device)

    output = model(orig_img, recon_img)
    # 提取 magnitude 分数 (0-1 之间)
    score = output['magnitude'].item()
    return score


def run_inference(model_path, data_root, output_csv='results_scores1.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 初始化模型并加载权重
    print(f"Loading model from {model_path}...")
    model = HASNet().to(device)
    # 注意：如果模型保存时包含 module. 前缀（DataParallel），需处理
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    results = []

    # 2. 设定路径 (根据你的描述)
    orig_base = os.path.join(data_root, 'originals')
    gan_recon_base = os.path.join(data_root, 'reconstructed', 'StyleGAN')
    dm_recon_base = os.path.join(data_root, 'reconstructed', 'ADM')

    # 3. 遍历原始图像目录
    # 假设结构是 originals/类别/图片名.jpg
    print("Starting inference...")
    for root, dirs, files in os.walk(orig_base):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # 获取相对路径，用于在重建目录中定位
            rel_path = os.path.relpath(os.path.join(root, filename), orig_base)

            orig_path = os.path.join(orig_base, rel_path)
            gan_path = os.path.join(gan_recon_base, rel_path)
            dm_path = os.path.join(dm_recon_base, rel_path)

            # 计算分数
            gan_score = get_score(model, device, orig_path, gan_path)
            dm_score = get_score(model, device, orig_path, dm_path)

            # 只有当两个分数都能计算时才记录（或者根据需求保留一个）
            if gan_score is not None or dm_score is not None:
                results.append({
                    'Image_Name': rel_path,
                    'GAN_Recon_Score': gan_score if gan_score is not None else "N/A",
                    'DM_Recon_Score': dm_score if dm_score is not None else "N/A"
                })

    # 4. 导出 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Inference completed. Results saved to {output_csv}")


if __name__ == '__main__':
    # 修改为你的模型文件路径和数据根目录
    MODEL_WEIGHTS = 'hasnet1­_0.5_final.pth'
    DATA_ROOT = './data_root'

    run_inference(MODEL_WEIGHTS, DATA_ROOT)