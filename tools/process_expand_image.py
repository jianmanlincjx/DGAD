# import os
# from PIL import Image
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # 定义文件夹路径
# mask_dir = '/data/JM/code/BrushNet-main/dataset_big/完整物体插入配对数据压缩包/LVIS/mask'
# target_dir = '/data/JM/code/BrushNet-main/dataset_big/完整物体插入配对数据压缩包/LVIS/target'
# object_dir = '/data/JM/code/BrushNet-main/dataset_big/完整物体插入配对数据压缩包/LVIS/object'

# # 创建 object 文件夹（如果不存在）
# os.makedirs(object_dir, exist_ok=True)

# # 获取 mask 和 target 文件夹中的所有文件，并按顺序排序
# mask_files = sorted(os.listdir(mask_dir))
# target_files = sorted(os.listdir(target_dir))

# # 确保两个文件夹中的文件数量一致
# assert len(mask_files) == len(target_files), "mask 文件夹和 target 文件夹中的文件数量不一致！"

# def process_image(mask_file, target_file):
#     """合成单个图像，并保存到 object 文件夹"""
#     # 检查文件扩展名
#     if not mask_file.endswith('.png') or not target_file.endswith('.png'):
#         return None
    
#     # 加载 mask 和 target 图像
#     mask_path = os.path.join(mask_dir, mask_file)
#     target_path = os.path.join(target_dir, target_file)
    
#     mask = Image.open(mask_path).convert("L")  # 转为灰度图像
#     target = Image.open(target_path).convert("RGBA")  # 转为RGBA图像以便合成
    
#     # 确保目标图像和掩码图像大小一致
#     if mask.size != target.size:
#         mask = mask.resize(target.size)
    
#     # 使用 mask 图像作为遮罩，合成 object 图像
#     object_image = Image.composite(target, Image.new("RGBA", target.size), mask)

#     # 生成新的文件名
#     object_file_name = mask_file  # 使用 mask 的文件名来保持配对关系

#     # 保存合成后的图像
#     object_image_path = os.path.join(object_dir, object_file_name)
#     object_image.save(object_image_path)

#     return f"合成并保存: {object_file_name}"

# def process_images_in_parallel(mask_files, target_files):
#     """使用线程池并行处理图像"""
#     with ThreadPoolExecutor() as executor:
#         futures = []
#         # 提交每一对 mask 和 target 图像进行处理
#         for mask_file, target_file in zip(mask_files, target_files):
#             futures.append(executor.submit(process_image, mask_file, target_file))
        
#         # 等待任务完成并输出结果
#         for future in as_completed(futures):
#             result = future.result()
#             if result:
#                 print(result)

# # 调用多线程处理函数
# process_images_in_parallel(mask_files, target_files)


import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def process_single_image(filename, source_dir, target_dir, mask_dir, object_dir, output_source_dir, output_target_dir, output_mask_dir, output_object_dir):
    """处理单对图像"""
    source_file_path = os.path.join(source_dir, filename)
    target_file_path = os.path.join(target_dir, filename)
    mask_file_path = os.path.join(mask_dir, filename)
    object_file_path = os.path.join(object_dir, filename)

    # 尝试打开图像文件
    try:
        source_img = Image.open(source_file_path)
        target_img = Image.open(target_file_path)
        mask_img = Image.open(mask_file_path)
        object_img = Image.open(object_file_path)
        
        # 获取每个图像的尺寸
        source_size = source_img.size
        target_size = target_img.size
        mask_size = mask_img.size
        object_size = object_img.size

        # 确保所有图像尺寸一致
        assert source_size == target_size == mask_size == object_size, f"图像尺寸不一致: {source_size}, {target_size}, {mask_size}, {object_size}"
        
    except (IOError, OSError) as e:
        # 捕获图像损坏的错误（IOError 和 OSError 都可能在图像文件损坏时抛出）
        print(f"图像文件损坏，删除文件: {e}")
        
        # 删除图像文件
        os.remove(source_file_path)
        os.remove(target_file_path)
        os.remove(mask_file_path)
        os.remove(object_file_path)
        return

    # 对每张图像进行处理
    processed_source = resize_and_pad_image(source_file_path)
    processed_target = resize_and_pad_image(target_file_path)
    processed_mask = resize_and_pad_image(mask_file_path)
    processed_object = resize_and_pad_image(object_file_path)

    # 保存处理后的图像
    cv2.imwrite(os.path.join(output_source_dir, filename), processed_source)
    cv2.imwrite(os.path.join(output_target_dir, filename), processed_target)
    cv2.imwrite(os.path.join(output_mask_dir, filename), processed_mask)
    cv2.imwrite(os.path.join(output_object_dir, filename), processed_object)

def process_images_in_pair(source_dir, target_dir, mask_dir, object_dir, output_source_dir, output_target_dir, output_mask_dir, output_object_dir):
    """使用线程池并行处理图像"""
    # 获取 source 目录下所有 PNG 文件
    filenames = [f for f in os.listdir(source_dir) if f.endswith('.png')]

    # 使用线程池处理每个图像文件
    with ThreadPoolExecutor() as executor:
        futures = []
        for filename in filenames:
            futures.append(executor.submit(process_single_image, filename, source_dir, target_dir, mask_dir, object_dir, output_source_dir, output_target_dir, output_mask_dir, output_object_dir))

        # 等待所有任务完成
        for future in as_completed(futures):
            future.result()

# 输入和输出目录
base_dir = '/data/JM/code/BrushNet-main/dataset_big/完整物体插入配对数据压缩包/LVIS'
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
