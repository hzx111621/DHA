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
import argparse

# --- 关键：从训练脚本中导入必要的类 -------------------
try:
    from training.train_hasnet_dinov2 import HASNet, HASNetDataset, Config
except ImportError as e:
    print("错误: 无法从 'training/train_hasnet_dinov2.py' 导入必要的类。")
    print("请确保 analysis/analyze_hasnet_attention.py 和你的训练脚本在同一个目录下。")
    print(f"原始错误: {e}")
    exit()
# -------------------------------------------------------------------

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def analyze_dataset(model_path, data_root_path, split='val', base_output_dir='analysis_results_full'):
    """
    对指定的数据集划分进行全面分析，包括：
    1. 保存全新的“四合一”组合分析图 (动态选择高/低差异Patch)
    2. 生成分数CSV报告和t-SNE可视化。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===== Analyzing '{split}' split on device: {device} =====")

    # --- 1. 创建所有需要的输出目录 ---
    output_dir = os.path.join(base_output_dir, split)
    combined_visuals_dir = os.path.join(output_dir, 'combined_visualizations')
    os.makedirs(combined_visuals_dir, exist_ok=True)
    
    print(f"所有分析结果将保存到: {output_dir}")
    print(f"组合可视化图将保存到: {combined_visuals_dir}")

    # --- 2. 加载模型 ---
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到 at {model_path}")
        return
    model = HASNet().to(device)
    try:
        if torch.__version__ >= "2.1.0":
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        else:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("模型加载成功.")
    except Exception as e:
        print(f"加载模型权重时发生错误: {e}")
        return

    # --- 3. 准备数据集 ---
    print(f"正在准备 '{split}' 数据集...")
    dataset = HASNetDataset(data_root_path, split=split)
    if len(dataset) == 0:
        print(f"错误: '{split}' 数据集为空，无法进行分析。")
        return
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"'{split}' 数据集样本数量: {len(dataset)}")
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    # --- 4. 遍历数据集，提取并保存所有信息 ---
    results = []
    all_semantic_vectors = []
    all_labels = []
    category_scores = defaultdict(list)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    SAVE_VISUALS_LIMIT = 500 

    with torch.no_grad():
        for i, (orig_tensor, recon_tensor, targets) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Analyzing {split} set"):
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

            # --- b. 【新的四合一可视化方案】 ---
            if i < SAVE_VISUALS_LIMIT:
                
                # --- (1) 准备基础数据 ---
                attention_weights = outputs['attention_weights'].squeeze(0).cpu() # Shape: (Heads, N_q, N_k)
                patch_grid_size = int(np.sqrt(attention_weights.shape[1])) # e.g., 18
                N_q = attention_weights.shape[1]
                
                # 准备原始图和重建图 (Panel 1, 2, 3, 4的背景)
                orig_img_vis = inv_normalize(orig_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
                orig_img_vis = np.clip(orig_img_vis, 0, 1)
                recon_img_vis = inv_normalize(recon_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
                recon_img_vis = np.clip(recon_img_vis, 0, 1)

                # --- (2) 随机选择 3 个 Query Patch ---
                # 为了可重现性，我们使用一个固定的种子，或者从N_q中随机选择3个不重复的索引
                # 这里我们简单随机选择，如果需要，可以替换为固定的索引 (例如 10, 50, 100)
                random_indices = np.random.choice(N_q, size=3, replace=False)
                
                patches_to_visualize = {
                    'patch_1': random_indices[0],
                    'patch_2': random_indices[1],
                    'patch_3': random_indices[2]
                }

                # --- (3) 开始绘制 4 面板图像 ---
                fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=120)
                fig.suptitle(f'Analysis: {category_key} | {os.path.splitext(img_filename)[0]}\nScore: {magnitude_score:.4f}', fontsize=16)

                # --- Panel 1: 原始图像 (无红框) ---
                axes[0].imshow(orig_img_vis)
                axes[0].set_title('(a) Original Image (Query)')
                axes[0].axis('off')
                
                # --- Panel 2, 3, 4: Patch Attention 叠加在重建图上 ---
                panel_index = 1
                for query_patch_name, patch_idx in patches_to_visualize.items():
                    
                    # 计算 Patch-to-Patch 注意力图
                    # (Heads, N_q, N_k) -> [:, patch_idx, :] -> (Heads, N_k) -> mean over Heads(dim 0) -> (N_k,)
                    attention_for_patch = attention_weights[:, patch_idx, :].mean(dim=0)
                    patch_attention_heatmap = attention_for_patch.reshape(patch_grid_size, patch_grid_size).numpy()
                    
                    # 绘制重建图作为背景
                    axes[panel_index].imshow(recon_img_vis)
                    
                    # 叠加 Patch Attention 热力图 ("马赛克"风格)
                    im_patch = axes[panel_index].imshow(patch_attention_heatmap, 
                                                        cmap='coolwarm', 
                                                        alpha=0.6, 
                                                        interpolation='nearest', # 保持马赛克风格
                                                        extent=(0, Config.img_size, Config.img_size, 0))
                    
                    # 计算 Query Patch 坐标 (用于标题，可选)
                    y, x = divmod(patch_idx, patch_grid_size)
                    
                    axes[panel_index].set_title(f'({chr(97 + panel_index)}) Patch Attention (Loc: {x},{y})')
                    axes[panel_index].axis('off')
                    fig.colorbar(im_patch, ax=axes[panel_index], shrink=0.8)
                    
                    panel_index += 1

                # --- (4) 保存图像 ---
                plt.tight_layout(rect=[0, 0.03, 1, 0.93])
                base_filename = f"{i:04d}_{category_key}_{os.path.splitext(img_filename)[0]}"
                # 注意：这次只保存一个文件，因为它是一个组合图
                combined_visual_path = os.path.join(combined_visuals_dir, f"{base_filename}_rand3_attention_v2.png")
                plt.savefig(combined_visual_path)
                plt.close(fig)

    # --- 5. 保存CSV并打印统计结果 ---
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f'{split}_scores_report.csv'), index=False)
    print(f"\n--- '{split}' split 平均分数组合统计 ---")
    sorted_avg_scores = sorted({k: np.mean(v) for k, v in category_scores.items()}.items(), key=lambda item: item[1])
    for category, avg_score in sorted_avg_scores:
        print(f"{category:<35}: {avg_score:.4f}")

    # --- 6. 执行t-SNE降维与可视化 ---
    print(f"\n--- 正在为 '{split}' split 执行t-SNE降维... ---")
    tsne = TSNE(n_components=2, verbose=0, perplexity=min(30, len(all_semantic_vectors)-1), n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(np.array(all_semantic_vectors))
    print("t-SNE降维完成.")
    
    print("--- 正在绘制t-SNE可视化图 ---")
    plt.figure(figsize=(20, 16), dpi=150)
    unique_labels = sorted(list(set(all_labels)))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    markers = ['o', 's', 'P', 'X', '*', 'D', 'v', '^', '<', '>', 'p', 'h']
    
    for i, label_idx in enumerate(unique_labels):
        indices = [j for j, l in enumerate(all_labels) if l == label_idx]
        x, y = tsne_results[indices, 0], tsne_results[indices, 1]
        class_name = idx_to_class[label_idx]
        plt.scatter(x, y, color=colors(i), marker=markers[i % len(markers)], label=class_name, alpha=0.8, s=60, edgecolors='w', linewidths=0.5)
        
    plt.title(f't-SNE Visualization of Semantic Embeddings ({split.capitalize()} Set)', fontsize=24, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=16)
    plt.ylabel('t-SNE Dimension 2', fontsize=16)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0., fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    tsne_path = os.path.join(output_dir, f'tsne_semantic_space_{split}.png')
    plt.savefig(tsne_path)
    plt.close()
    print(f"t-SNE可视化图已保存到: {tsne_path}")
    print(f"\n===== '{split}' split 分析圆满完成! =====")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="对HAS-Net模型进行全面的分析、评估与可视化。")
    parser.add_argument('--model', type=str, default='hasnet_final.pth', help='已训练模型的权重文件路径 (.pth)。')
    parser.add_argument('--data_root', type=str, default='./data_root', help='数据集的根目录路径。')
    parser.add_argument('--splits', nargs='+', default=['train'], help="需要分析的数据集划分，例如: 'train' 'val'。")
    parser.add_argument('--output_dir', type=str, default='analysis_results_full14', help='保存所有分析结果的根目录。')
    
    args = parser.parse_args()

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