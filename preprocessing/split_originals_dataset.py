import os
import shutil
import random
from sklearn.model_selection import train_test_split


def split_originals_data(original_dir, split_20_dir, split_80_dir, test_size=0.2, random_state=42):
    """
    将originals目录中的数据按比例分割为两个新目录，保持目录结构不变

    参数:
    original_dir (str): 原始originals目录路径
    split_20_dir (str): 保存20%数据的目录路径
    split_80_dir (str): 保存80%数据的目录路径
    test_size (float): 分割比例（默认0.2）
    random_state (int): 随机种子（默认42）
    """
    random.seed(random_state)

    # 创建目标目录
    os.makedirs(split_20_dir, exist_ok=True)
    os.makedirs(split_80_dir, exist_ok=True)

    # 遍历所有模型类型目录
    for model_type in os.listdir(original_dir):
        model_path = os.path.join(original_dir, model_type)
        if not os.path.isdir(model_path):
            continue

        # 创建对应的目标模型目录
        model_20 = os.path.join(split_20_dir, model_type)
        model_80 = os.path.join(split_80_dir, model_type)
        os.makedirs(model_20, exist_ok=True)
        os.makedirs(model_80, exist_ok=True)

        # 处理每个类别目录
        for category in os.listdir(model_path):
            category_path = os.path.join(model_path, category)
            if not os.path.isdir(category_path):
                continue

            # 收集图像文件
            images = [f for f in os.listdir(category_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                continue

            # 分割数据
            train_imgs, val_imgs = train_test_split(
                images,
                test_size=test_size,
                random_state=random_state
            )

            # 创建类别目录
            category_20 = os.path.join(model_20, category)
            category_80 = os.path.join(model_80, category)
            os.makedirs(category_20, exist_ok=True)
            os.makedirs(category_80, exist_ok=True)

            # 复制文件
            for img in val_imgs:
                src = os.path.join(category_path, img)
                dst = os.path.join(category_20, img)
                shutil.copy2(src, dst)

            for img in train_imgs:
                src = os.path.join(category_path, img)
                dst = os.path.join(category_80, img)
                shutil.copy2(src, dst)

    print(f"数据分割完成：{split_20_dir} (20%), {split_80_dir} (80%)")


if __name__ == "__main__":
    # 配置路径
    ORIGINAL_DIR = "/data/MM_AIGC/originals"
    SPLIT_20_DIR = "./split_20"
    SPLIT_80_DIR = "./split_80"

    # 执行分割
    split_originals_data(
        original_dir=ORIGINAL_DIR,
        split_20_dir=SPLIT_20_DIR,
        split_80_dir=SPLIT_80_DIR,
        test_size=0.2,
        random_state=42
    )