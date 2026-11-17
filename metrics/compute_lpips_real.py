import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from lpips import LPIPS  # 需要安装 lpips 包：pip install lpips
import csv

# 配置参数
class Config:
    img_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像预处理
def preprocess_image(image_path, transform):
    img = Image.open(image_path).convert('RGB')
    return transform(img)

# 主函数
def compute_lpips_and_save_to_csv(originals_dir, reconstructed_stylegan_dir, reconstructed_adm_dir, output_csv):
    # 初始化设备和模型
    device = Config.device
    lpips_model = LPIPS(net='vgg').to(device).eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    
    # 获取所有图像文件名
    original_files = [f for f in os.listdir(originals_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    stylegan_files = set(os.listdir(reconstructed_stylegan_dir))
    adm_files = set(os.listdir(reconstructed_adm_dir))
    
    # 打开 CSV 文件并写入表头
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image Name", "StyleGAN_LPIPS", "ADM_LPIPS"])
        
        # 遍历原始图像文件
        for img_name in original_files:
            if img_name not in stylegan_files or img_name not in adm_files:
                print(f"Skipping {img_name} due to missing reconstruction.")
                continue
            
            # 加载图像
            orig_path = os.path.join(originals_dir, img_name)
            stylegan_path = os.path.join(reconstructed_stylegan_dir, img_name)
            adm_path = os.path.join(reconstructed_adm_dir, img_name)
            
            orig_img = preprocess_image(orig_path, transform).unsqueeze(0).to(device)
            stylegan_img = preprocess_image(stylegan_path, transform).unsqueeze(0).to(device)
            adm_img = preprocess_image(adm_path, transform).unsqueeze(0).to(device)
            
            # 计算 StyleGAN 和 ADM 的 LPIPS 分数
            stylegan_score = lpips_model(orig_img, stylegan_img).squeeze().item()
            adm_score = lpips_model(orig_img, adm_img).squeeze().item()
            
            # 写入 CSV 文件
            writer.writerow([img_name, stylegan_score, adm_score])

if __name__ == "__main__":
    originals_dir = "/data/MM_AIGC/data_root/originals/REAL/"  # 原始图像文件夹路径
    reconstructed_stylegan_dir = "/data/MM_AIGC/data_root/reconstructed/StyleGAN/REAL/"  # StyleGAN 重建图像文件夹路径
    reconstructed_adm_dir = "/data/MM_AIGC/data_root/reconstructed/ADM/REAL/"  # ADM 重建图像文件夹路径
    output_csv = './lpips_scores_folder_images.csv'  # 输出 CSV 文件路径
    compute_lpips_and_save_to_csv(originals_dir, reconstructed_stylegan_dir, reconstructed_adm_dir, output_csv)
