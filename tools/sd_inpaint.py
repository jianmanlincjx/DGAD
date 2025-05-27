import os
import torch
from PIL import Image
from glob import glob

from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
import cv2
import numpy as np
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)

if __name__ == "__main__":

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "pretrain_model/models--runwayml--stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16, low_cpu_mem_usage=False
    ).to('cuda')

image_name_list = sorted(os.listdir('/data1/JM/code/BrushNet-main/datasets/MSRA-10K/mask_processed'))
output_folder = '/data1/JM/code/BrushNet-main/temp_result'
os.makedirs(output_folder, exist_ok=True)
# 遍历图像名称列表
for name in image_name_list[:100]:
    # 定义图像路径
    source_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/source_processed/{name}"
    mask_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/mask_processed/{name}"
    txt_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/text/{name.replace('.png', '.txt')}"
    groundtruth_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/target_processed/{name}"

    # 读取描述文本
    with open(txt_path, "r") as f:
        caption = f.read()
    # 读取图片
    init_image = cv2.imread(source_image_path)
    mask_raw = cv2.imread(mask_image_path)

    # 生成二值 mask，mask值大于255的区域为True，其他为False，然后转换为 uint8（0或1）
    mask_image = (mask_raw.sum(axis=-1) > 255).astype(np.uint8)[:, :, np.newaxis]

    # 将数据类型转换为 uint8 再创建 PIL 图像
    init_image = Image.fromarray(init_image.astype(np.uint8))
    mask_image = Image.fromarray((mask_image.repeat(3, axis=-1) * 255).astype(np.uint8))

    generator = torch.manual_seed(7777)

    # Get original dimensions
    original_width, original_height = init_image.size
    mask_width, mask_height = mask_image.size

    image = pipe(prompt=caption, image=init_image, mask_image=mask_image).images[0]
    image.save(f'{output_folder}/{name}')

  

