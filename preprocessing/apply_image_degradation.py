#import os
#from PIL import Image
#
#
#def compress_and_save_images(source_dir, target_dir, quality=95):
#    """
#    对指定文件夹下的所有图像进行JPEG压缩，并按原文件夹结构保存到新地址。
#
#    :param source_dir: 原始文件夹路径
#    :param target_dir: 目标文件夹路径
#    :param quality: JPEG压缩质量因子，默认为95
#    """
#    # 确保目标文件夹存在
#    os.makedirs(target_dir, exist_ok=True)
#
#    # 遍历源文件夹中的所有文件和子文件夹
#    for root, dirs, files in os.walk(source_dir):
#        # 计算当前文件夹在目标文件夹中的对应路径
#        relative_path = os.path.relpath(root, source_dir)
#        target_path = os.path.join(target_dir, relative_path)
#
#        # 创建目标文件夹（如果不存在）
#        os.makedirs(target_path, exist_ok=True)
#
#        # 遍历当前文件夹中的所有文件
#        for file in files:
#            source_file_path = os.path.join(root, file)
#            target_file_path = os.path.join(target_path, file)
#
#            try:
#                # 打开图像文件
#                with Image.open(source_file_path) as img:
#                    # 如果图像是非JPEG格式，转换为RGB模式以支持JPEG保存
#                    if img.mode != "RGB":
#                        img = img.convert("RGB")
#
#                    # 保存压缩后的图像
#                    img.save(target_file_path, format="JPEG", quality=quality)
#                    print(f"Compressed and saved: {target_file_path}")
#            except Exception as e:
#                # 捕获异常并跳过无法处理的文件
#                print(f"Failed to process {source_file_path}: {e}")

#
## 示例用法
##if __name__ == "__main__":
#    # 原始文件夹路径
#    source_directory = "D:\MM_AIGC\split_20_test"
#    # 目标文件夹路径
#    target_directory = "data_test_JPEG_95"
#
#    # 调用函数进行JPEG压缩并保存
#    compress_and_save_images(source_directory, target_directory)

import os
from PIL import Image, ImageFilter


def apply_gaussian_blur_and_save(source_dir, target_dir, blur_radius=1):
   """
   对指定文件夹下的所有图像应用高斯模糊，并按原文件夹结构保存到新地址。

   :param source_dir: 原始文件夹路径
   :param target_dir: 目标文件夹路径
   :param blur_radius: 高斯模糊的半径，默认为1
   """
   # 确保目标文件夹存在
   os.makedirs(target_dir, exist_ok=True)

   # 遍历源文件夹中的所有文件和子文件夹
   for root, dirs, files in os.walk(source_dir):
       # 计算当前文件夹在目标文件夹中的对应路径
       relative_path = os.path.relpath(root, source_dir)
       target_path = os.path.join(target_dir, relative_path)

       # 创建目标文件夹（如果不存在）
       os.makedirs(target_path, exist_ok=True)

       # 遍历当前文件夹中的所有文件
       for file in files:
           source_file_path = os.path.join(root, file)
           target_file_path = os.path.join(target_path, file)

           try:
               # 打开图像文件
               with Image.open(source_file_path) as img:
                   # 应用高斯模糊
                   blurred_img = img.filter(ImageFilter.GaussianBlur(blur_radius))

                   # 如果图像是非JPEG格式，转换为RGB模式以支持JPEG保存
                   if img.mode != "RGB":
                       blurred_img = blurred_img.convert("RGB")

                   # 保存处理后的图像
                   blurred_img.save(target_file_path, format="JPEG")
                   print(f"Applied Gaussian blur and saved: {target_file_path}")
           except Exception as e:
               # 捕获异常并跳过无法处理的文件
               print(f"Failed to process {source_file_path}: {e}")


# 示例用法
if __name__ == "__main__":
   # 原始文件夹路径
   source_directory = "/data/MM_AIGC/split_20"
   # 目标文件夹路径
   target_directory = "data_test_blur"

   # 调用函数进行高斯模糊并保存
   apply_gaussian_blur_and_save(source_directory, target_directory)