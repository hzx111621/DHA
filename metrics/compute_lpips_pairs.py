import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from lpips import LPIPS  # 需要安装 lpips 包：pip install lpips
import csv

# 配置参数
class Config:
    img_size = 256
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集类
class PureDiffDataset(Dataset):
    def __init__(self, root_dir):
        self.pairs = []
        self.transform = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
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
    
    def _find_and_add_pairs(self, orig_dir, recon_dir, category, recon_method):
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
                self.pairs.append((orig_path, recon_path, category, recon_method))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        orig_path, recon_path, category, recon_method = self.pairs[idx]
        orig_img = Image.open(orig_path).convert('RGB')
        recon_img = Image.open(recon_path).convert('RGB')
        orig_tensor = self.transform(orig_img)
        recon_tensor = self.transform(recon_img)
        return orig_tensor, recon_tensor, category, recon_method

# 主函数
def compute_lpips_and_save_to_csv(dataset_root, output_csv):
    # 初始化设备和模型
    device = Config.device
    lpips_model = LPIPS(net='vgg').to(device).eval()
    
    # 初始化数据集和数据加载器
    dataset = PureDiffDataset(dataset_root)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 打开 CSV 文件并写入表头
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category", "Image Name", "StyleGAN_LPIPS", "ADM_LPIPS"])
        
        # 遍历数据集并计算 LPIPS 分数
        for orig, recon, category, recon_method in dataloader:
            orig = orig.to(device)
            recon = recon.to(device)
            
            # 计算 LPIPS 分数
            lpips_scores = lpips_model(orig, recon).squeeze().detach().cpu().numpy()
            
            # 获取图像名称
            image_names = [os.path.basename(pair[0]) for pair in dataloader.dataset.pairs]
            
            # 写入每张图像的分数
            for i in range(len(lpips_scores)):
                img_name = image_names[i]
                score = lpips_scores[i]
                if recon_method[i] == "StyleGAN":
                    writer.writerow([category[i], img_name, score, ""])
                elif recon_method[i] == "ADM":
                    writer.writerow([category[i], img_name, "", score])

if __name__ == "__main__":
    dataset_root = './data_root'  # 数据集根目录
    output_csv = './lpips_scores_per_image.csv'  # 输出 CSV 文件路径
    compute_lpips_and_save_to_csv(dataset_root, output_csv)