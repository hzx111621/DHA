import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import warnings
from torchvision import transforms
import argparse # 引入参数解析库

# --- 关键：从训练脚本中导入必要的类 -------------------
try:
    from training.train_hasnet_dinov2 import HASNet, HASNetDataset
except ImportError as e:
    print("错误: 无法从 'training/train_hasnet_dinov2.py' 导入必要的类。")
    print("请确保 analysis/analyze_hasnet_comparisons.py 和你的训练脚本在同一个目录下。")
    print(f"原始错误: {e}")
    exit()
# -------------------------------------------------------------------

warnings.filterwarnings("ignore", category=FutureWarning)


# -------------------- 请用这个最终版本替换 analysis/analyze_hasnet_comparisons.py 中的 analyze_dataset 函数 --------------------

# -------------------- 请用这个最终版本替换 analysis/analyze_hasnet_comparisons.py 中的 analyze_dataset 函数 --------------------

def analyze_dataset(model_path, data_root_path, split='val', base_output_dir='analysis_results_all'):
    """
    对指定的数据集划分进行全面分析。
    现在会为每个样本保存一张组合图，包含：
    1. 原始图像 (Original Image)
    2. 重建图像 (Reconstructed Image)
    3. 显著图叠加在重建图像上 (Saliency on Reconstructed) 并附带颜色条。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===== Analyzing '{split}' split on device: {device} =====")

    # --- 1. 创建输出目录 ---
    # !! 核心修改：创建一个专门存放组合图的目录 !!
    output_dir = os.path.join(base_output_dir, split)
    comparison_dir = os.path.join(output_dir, 'comparison_visualizations') # 新的目录
    os.makedirs(comparison_dir, exist_ok=True)
    print(f"结果将保存到: {output_dir}")
    print(f"可视化组合图将保存到: {comparison_dir}")

    # --- 2. 加载模型 (无变化) ---
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到 at {model_path}")
        return
    model = HASNet().to(device)
    if torch.__version__ >= "2.1.0":
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    else:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("模型加载成功.")

    # --- 3. 准备数据集 (无变化) ---
    print(f"正在准备 '{split}' 数据集...")
    dataset = HASNetDataset(data_root_path, split=split)
    if len(dataset) == 0:
        print(f"错误: '{split}' 数据集为空，无法进行分析。")
        return
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"'{split}' 数据集样本数量: {len(dataset)}")
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    # --- 4. 遍历数据集，提取所有信息 ---
    results = []
    all_semantic_vectors = []
    all_labels = []
    category_scores = defaultdict(list)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    with torch.no_grad():
        for i, (orig_tensor, recon_tensor, targets) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Extracting from {split}"):
            # --- a. 获取模型输出和元数据 (无变化) ---
            orig_tensor, recon_tensor = orig_tensor.to(device), recon_tensor.to(device)
            outputs = model(orig_tensor, recon_tensor)
            magnitude_score = outputs['magnitude'].item()
            orig_path, _, orig_model, recon_method = dataset.pairs[i]
            img_filename = os.path.basename(orig_path)
            category_key = f"{orig_model}_vs_{recon_method}"
            results.append({'filename': img_filename, 'category': category_key, 'predicted_score': magnitude_score})
            category_scores[category_key].append(magnitude_score)
            semantic_vector = outputs['semantic_embedding'].squeeze().cpu().numpy()
            label_idx = targets['label'].item()
            all_semantic_vectors.append(semantic_vector)
            all_labels.append(label_idx)

            # --- b. !! 核心修改：创建并保存三合一的组合图 !! ---
            if i < 200: # 仍然只保存前200张
                # (1) 准备三张需要可视化的图像
                orig_img_vis = inv_normalize(orig_tensor.squeeze()).permute(1, 2, 0).cpu().numpy()
                orig_img_vis = np.clip(orig_img_vis, 0, 1)

                recon_img_vis = inv_normalize(recon_tensor.squeeze()).permute(1, 2, 0).cpu().numpy()
                recon_img_vis = np.clip(recon_img_vis, 0, 1)

                saliency_map_tensor = outputs['saliency_map'].squeeze().cpu()
                saliency_map_vis = saliency_map_tensor.numpy()
                saliency_map_vis = (saliency_map_vis - saliency_map_vis.min()) / (saliency_map_vis.max() - saliency_map_vis.min() + 1e-8)

                # (2) 使用 subplots 将三张图画在一起
                fig, axes = plt.subplots(1, 3, figsize=(24, 8)) # 1行3列的图

                # 子图1: 原始图像
                axes[0].imshow(orig_img_vis)
                axes[0].set_title('Original Image', fontsize=16)
                axes[0].axis('off')

                # 子图2: 重建图像
                axes[1].imshow(recon_img_vis)
                axes[1].set_title('Reconstructed Image', fontsize=16)
                axes[1].axis('off')

                # 子图3: 显著图叠加在重建图像上
                axes[2].imshow(recon_img_vis)
                # 将 imshow 的返回值保存下来，用于创建颜色条
                im = axes[2].imshow(saliency_map_vis, cmap='jet', alpha=0.5)
                axes[2].set_title('Saliency on Reconstructed', fontsize=16)
                axes[2].axis('off')

                # (3) 为第三张子图添加颜色条 (图注)
                fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

                # (4) 添加总标题并保存图像
                fig.suptitle(f"Sample {i:04d}: {category_key}\nPredicted Score: {magnitude_score:.4f}", fontsize=20)
                
                output_filename = f"comparison_{i:04d}_{category_key}_{img_filename}.png"
                output_path = os.path.join(comparison_dir, output_filename)
                
                # 使用 tight_layout 调整布局，防止标题重叠
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig) # 关闭图形，释放内存

    # --- 后续的步骤5和6 (保存CSV, t-SNE) 完全保持不变 ---
    # ... (代码省略) ...

    # --- 后续的步骤5和6 (保存CSV, t-SNE) 完全保持不变 ---
    # ... (代码省略) ...
    # --- 5. 保存CSV并打印统计结果 ---
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f'{split}_scores.csv'), index=False)
    print(f"\n--- '{split}' split 平均分数组合统计 ---")
    sorted_avg_scores = sorted({k: np.mean(v) for k, v in category_scores.items()}.items(), key=lambda item: item[1], reverse=True)
    for category, avg_score in sorted_avg_scores:
        print(f"{category:<30}: {avg_score:.4f}")

    # --- 6. 执行t-SNE降维与可视化 ---
    print(f"\n--- 正在为 '{split}' split 执行t-SNE降维... ---")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(np.array(all_semantic_vectors))
    print("t-SNE降维完成.")
    
    print("--- 正在绘制t-SNE可视化图 ---")
    plt.figure(figsize=(20, 16))
    unique_labels = sorted(list(set(all_labels)))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    markers = ['o', 's', 'P', 'X', '*', 'D', 'v', '^', '<', '>', 'p', 'h']
    for i, label_idx in enumerate(unique_labels):
        indices = [j for j, l in enumerate(all_labels) if l == label_idx]
        x, y = tsne_results[indices, 0], tsne_results[indices, 1]
        class_name = idx_to_class[label_idx]
        color, marker = colors(i), markers[i % len(markers)]
        plt.scatter(x, y, color=color, marker=marker, label=class_name, alpha=0.7, s=50)
    plt.title(f't-SNE Visualization of Semantic Embeddings ({split.capitalize()} Set)', fontsize=24)
    plt.xlabel('t-SNE Dimension 1', fontsize=16)
    plt.ylabel('t-SNE Dimension 2', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    tsne_path = os.path.join(output_dir, f'tsne_semantic_space_{split}.png')
    plt.savefig(tsne_path, dpi=300)
    plt.close()
    print(f"t-SNE可视化图已保存到: {tsne_path}")
    print(f"\n===== '{split}' split 分析完成! =====")

if __name__ == '__main__':
    # --- 配置参数 ---
    # 使用 argparse 来从命令行接收参数，更加灵活
    parser = argparse.ArgumentParser(description="Analyze HAS-Net model on train and/or validation sets.")
    parser.add_argument('--model', type=str, default='hasnet_final.pth', help='Path to the trained model weights.')
    parser.add_argument('--data_root', type=str, default='./data_root', help='Path to the root of the dataset.')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], help="List of splits to analyze (e.g., 'train', 'val').")
    parser.add_argument('--output_dir', type=str, default='analysis_results_all3', help='Directory to save analysis results.')
    
    args = parser.parse_args()

    # 循环分析指定的每个数据集划分
    for split in args.splits:
        if split not in ['train', 'val']:
            print(f"警告: 无效的数据集划分 '{split}'，已跳过。请选择 'train' 或 'val'。")
            continue
        analyze_dataset(
            model_path=args.model,
            data_root_path=args.data_root,
            split=split,
            base_output_dir=args.output_dir
        )