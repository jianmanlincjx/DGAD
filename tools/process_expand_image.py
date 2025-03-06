import os
import cv2
import numpy as np
from tqdm import tqdm

def resize_and_pad_image(image_path, target_size=(512, 512)):
    """用 cv2 将图像调整为目标大小，缺失部分填充为黑色"""
    # 读取图片
    img = cv2.imread(image_path)
    
    # 获取原始图片的大小
    height, width = img.shape[:2]
    
    # 计算填充的区域
    left = (target_size[0] - width) // 2
    top = (target_size[1] - height) // 2
    right = left + width
    bottom = top + height

    # 使用 cv2 resize 调整图片大小
    resized_img = cv2.resize(img, (target_size[0], target_size[1]))
    
    return resized_img

def process_images_in_pair(source_dir, target_dir, mask_dir, object_dir, output_source_dir, output_target_dir, output_mask_dir, output_object_dir):
    """处理四个目录中的配对图像，保证一致性"""
    # 遍历 source 目录中的所有 PNG 文件
    for filename in tqdm(os.listdir(source_dir)):
        if filename.endswith('.png'):
            source_file_path = os.path.join(source_dir, filename)
            target_file_path = os.path.join(target_dir, filename)
            mask_file_path = os.path.join(mask_dir, filename)
            object_file_path = os.path.join(object_dir, filename)

            # 确保对应的文件在四个目录中都存在
            if os.path.exists(target_file_path) and os.path.exists(mask_file_path) and os.path.exists(object_file_path):
                # 处理 source 图像
                processed_source = resize_and_pad_image(source_file_path)
                processed_target = resize_and_pad_image(target_file_path)
                processed_mask = resize_and_pad_image(mask_file_path)
                processed_object = resize_and_pad_image(object_file_path)

                # 保存处理后的图像
                cv2.imwrite(os.path.join(output_source_dir, filename), processed_source)
                cv2.imwrite(os.path.join(output_target_dir, filename), processed_target)
                cv2.imwrite(os.path.join(output_mask_dir, filename), processed_mask)
                cv2.imwrite(os.path.join(output_object_dir, filename), processed_object)

            else:
                print(f"Warning: Missing corresponding files for {filename}.")

# 输入和输出目录
base_dir = '/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new'
source_dir = f'{base_dir}/source'
target_dir = f'{base_dir}/target'
mask_dir = f'{base_dir}/mask'
object_dir = f'{base_dir}/object'

output_source_dir = f'{base_dir}/source_processed'
output_target_dir = f'{base_dir}/target_processed'
output_mask_dir = f'{base_dir}/mask_processed'
output_object_dir = f'{base_dir}/object_processed'

# 确保输出目录存在，如果不存在则创建
os.makedirs(output_source_dir, exist_ok=True)
os.makedirs(output_target_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_object_dir, exist_ok=True)

# 处理配对的图像
process_images_in_pair(source_dir, target_dir, mask_dir, object_dir, output_source_dir, output_target_dir, output_mask_dir, output_object_dir)
