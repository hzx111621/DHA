import os
import csv
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet

# 配置参数
class Config:
    img_size = 256
    batch_size = 32
    #ALL_MODELS = ['StyleGAN', 'CycleGAN', 'BigGAN', 'ProGAN', 'ADM', 'Glide', 'Wukong', 'SD1.5', 'REAL']

# 纯差异提取器
class PureDiffExtractor(nn.Module):
    def __init__(self):
        super().__init__()
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


# 图像预处理
def preprocess_image(image_path, transform):
    img = PILImage.open(image_path).convert('RGB')
    return transform(img)

# 加载模型权重
def load_model(weights_path, device):
    model = PureDiffExtractor().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

# 测试函数
def test(original_dir, stylegan_dir, adm_dir, weights_path, output_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    model = load_model(weights_path, device)
    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    # 收集所有图像文件名
    original_files = [f for f in os.listdir(original_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    stylegan_files = set(os.listdir(stylegan_dir))
    adm_files = set(os.listdir(adm_dir))
    # 打开CSV文件准备写入
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # 写入表头
        writer.writerow(['Image', 'StyleGAN_Score', 'ADM_Score'])
        # 遍历每个原始图像
        for orig_file in original_files:
            if orig_file not in stylegan_files or orig_file not in adm_files:
                print(f"Skipping {orig_file} due to missing reconstruction.")
                continue
            # 加载图像
            orig_path = os.path.join(original_dir, orig_file)
            stylegan_path = os.path.join(stylegan_dir, orig_file)
            adm_path = os.path.join(adm_dir, orig_file)
            orig_tensor = preprocess_image(orig_path, transform).unsqueeze(0).to(device)
            stylegan_tensor = preprocess_image(stylegan_path, transform).unsqueeze(0).to(device)
            adm_tensor = preprocess_image(adm_path, transform).unsqueeze(0).to(device)
            # 计算差异性得分
            with torch.no_grad():
                stylegan_score = model(orig_tensor, stylegan_tensor).item()
                adm_score = model(orig_tensor, adm_tensor).item()
            # 写入CSV文件
            writer.writerow([orig_file, stylegan_score, adm_score])
            print(f"Processed {orig_file}: StyleGAN={stylegan_score:.4f}, ADM={adm_score:.4f}")
    print(f"Results saved to {output_csv}")

if __name__ == '__main__':
    # 输入路径
    # cycle
    # original_dir = 'split_20_test/originals/DM/ADM'  # 原始图像文件夹
    # stylegan_dir = 'data_root/reconstructed/StyleGAN/DM/ADM'  # StyleGAN 重建图像文件夹
    # adm_dir = 'data_root/reconstructed/ADM/DM/ADM'  # ADM 重建图像文件夹
    # weights_path = 'pure_diff_extractor_3_epoch_100.pth'  # 模型权重路径
    # output_csv = './resultsADM.csv'  # 输出CSV文件路径

    # original_dir = 'split_20_test/originals/DM/SD1.5'  # 原始图像文件夹
    # stylegan_dir = 'data_root/reconstructed/StyleGAN/DM/SD1.5'  # StyleGAN 重建图像文件夹
    # adm_dir = 'data_root/reconstructed/ADM/DM/SD1.5'  # ADM 重建图像文件夹
    # weights_path = 'pure_diff_extractor_3_epoch_100.pth'  # 模型权重路径
    # output_csv = './resultsSD1.csv'  # 输出CSV文件路径
    #
    # original_dir = 'split_20_test/originals/DM/Wukong'  # 原始图像文件夹
    # stylegan_dir = 'data_root/reconstructed/StyleGAN/DM/Wukong'  # StyleGAN 重建图像文件夹
    # adm_dir = 'data_root/reconstructed/ADM/DM/Wukong'  # ADM 重建图像文件夹
    # weights_path = 'pure_diff_extractor_3_epoch_100.pth'  # 模型权重路径
    # output_csv = './resultswu.csv'  # 输出CSV文件路径

    # original_dir = 'split_20_test/originals/DM/VQDM'  # 原始图像文件夹
    # stylegan_dir = 'data_root/reconstructed/StyleGAN/DM/VQDM'  # StyleGAN 重建图像文件夹
    # adm_dir = 'data_root/reconstructed/ADM/DM/VQDM'  # ADM 重建图像文件夹
    # weights_path = 'pure_diff_extractor_3_epoch_100.pth'  # 模型权重路径
    # output_csv = './resultsVQ.csv'  # 输出CSV文件路径

    # original_dir = 'split_20_test/originals/DM/DM-Glide'  # 原始图像文件夹
    # stylegan_dir = 'data_root/reconstructed/StyleGAN/DM/Glide'  # StyleGAN 重建图像文件夹
    # adm_dir = 'data_root/reconstructed/ADM/DM/Glide'  # ADM 重建图像文件夹
    # weights_path = 'pure_diff_extractor_3_epoch_100.pth'  # 模型权重路径
    # output_csv = './resultsGLIDE.csv'  # 输出CSV文件路径

    # original_dir = 'split_20_test/originals/GAN/StyleGAN'  # 原始图像文件夹
    # stylegan_dir = 'data_root/reconstructed/StyleGAN/GAN/StyleGAN'  # StyleGAN 重建图像文件夹
    # adm_dir = 'data_root/reconstructed/ADM/GAN/StyleGAN'  # ADM 重建图像文件夹
    # weights_path = 'pure_diff_extractor_3_epoch_100.pth'  # 模型权重路径
    # output_csv = './resultsSTYLE1.csv'  # 输出CSV文件路径

    # original_dir = 'split_20_test/originals/GAN/REAL'  # 原始图像文件夹
    # stylegan_dir = 'data_root/reconstructed/StyleGAN/GAN/REAL'  # StyleGAN 重建图像文件夹
    # adm_dir = 'data_root/reconstructed/ADM/GAN/REAL'  # ADM 重建图像文件夹
    # weights_path = 'pure_diff_extractor_3_epoch_100.pth'  # 模型权重路径
    # output_csv = './resultsREAL.csv'  # 输出CSV文件路径

    # original_dir = 'split_20_test/originals/GAN/ProGAN'  # 原始图像文件夹
    # stylegan_dir = 'data_root/reconstructed/StyleGAN/GAN/ProGAN'  # StyleGAN 重建图像文件夹
    # adm_dir = 'data_root/reconstructed/ADM/GAN/ProGAN'  # ADM 重建图像文件夹
    # weights_path = 'pure_diff_extractor_3_epoch_100.pth'  # 模型权重路径
    # output_csv = './resultspro.csv'  # 输出CSV文件路径

    # original_dir = 'split_20_test/originals/GAN/BigGAN'  # 原始图像文件夹
    # stylegan_dir = 'data_root/reconstructed/StyleGAN/GAN/BigGAN'  # StyleGAN 重建图像文件夹
    # adm_dir = 'data_root/reconstructed/ADM/DM/ADM'  # ADM 重建图像文件夹1
    # weights_path = 'pure_diff_extractor_3_epoch_100.pth'  # 模型权重路径
    # output_csv = './resultsbig.csv'  # 输出CSV文件路径

    # original_dir = 'split_20_test/originals/GAN/CylcleGAN'  # 原始图像文件夹
    # stylegan_dir = 'data_root/reconstructed/StyleGAN/GAN/CylcleGAN'  # StyleGAN 重建图像文件夹
    # adm_dir = 'data_root/reconstructed/ADM/GAN/CycleGAN'  # ADM 重建图像文件夹1
    # weights_path = 'pure_diff_extractor_3_epoch_100.pth'  # 模型权重路径
    # output_csv = './resultscy.csv'  # 输出CSV文件路径

    original_dir = 'D:\pythonProject5\GAN——new/result\horse'  # 原始图像文件夹
    stylegan_dir = 'D:\pythonProject5\GAN——new/result\horse_recon'  # StyleGAN 重建图像文件夹
    adm_dir = 'D:/pythonProject5/new_data_recon_DM'  # ADM 重建图像文件夹1
    weights_path = 'pure_diff_extractor_final_0.7.pth'  # 模型权重路径
    output_csv = './results1111.csv'  # 输出CSV文件路径

    # 运行测试
    test(original_dir, stylegan_dir, adm_dir, weights_path, output_csv)