from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
from PIL import Image
import sys
import os
from torchvision import transforms
sys.path.append(os.getcwd())
import argparse




transform_image = transforms.Compose([
    transforms.ToTensor(),
])

base_model_path = "pretrain_model/stable-diffusion-v1-5"
brushnet_path = "/data1/JM/code/BrushNet-main/exp/insertnet_without_attention/checkpoint-480000/brushnet"

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16, is_inference=False).to('cuda')
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=True,
).to('cuda')
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
generator = torch.Generator("cuda").manual_seed(1234)
image_name_list = sorted(os.listdir('/data1/JM/code/BrushNet-main/datasets/MSRA-10K/mask_processed'))
output_folder = '/data1/JM/code/BrushNet-main/datasets/MSRA-10K_result_with_prompt_480000'
os.makedirs(output_folder, exist_ok=True)
# 遍历图像名称列表
for name in image_name_list:
    # 定义图像路径
    source_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new/source_processed/{name}"
    mask_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new/mask_processed/{name}"
    object_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new/object_processed/{name}"
    txt_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new/text/{name.replace('.png', '.txt')}"
    groundtruth_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new/target_processed/{name}"

    # 读取描述文本
    with open(txt_path, "r") as f:
        caption = f.read()

    # 读取并转换图像
    source_image = Image.open(source_image_path).convert("RGB")
    mask_image = Image.open(mask_image_path).convert("L")
    object_image = Image.open(object_image_path).convert("RGB")
    groundtruth_image = Image.open(groundtruth_image_path).convert("RGB")

    # 生成图像
    generated_image = pipe(
        caption, 
        transform_image(source_image).unsqueeze(0), 
        transform_image(mask_image).unsqueeze(0), 
        transform_image(object_image).unsqueeze(0), 
        num_inference_steps=50, 
        generator=generator,
        brushnet_conditioning_scale=1.0,
        guide_scale=5.0, guess_mode=True
    ).images[0]

    # 将所有图像转换为相同大小（如果需要）
    width, height = source_image.size
    generated_image = generated_image.resize((width, height))
    groundtruth_image = groundtruth_image.resize((width, height))

    mask_array = np.array(mask_image)
    mask = mask_array == 255  
    source_array = np.array(source_image)
    source_array[mask] = [255, 255, 255]  
    source_image = Image.fromarray(source_array)
    # 创建一个新的空白图像，宽度为四张图像的宽度之和，高度为单张图像的高度
    combined_image = Image.new('RGB', (width * 4, height))
    # 将四张图像从左到右拼接
    combined_image.paste(source_image, (0, 0))
    combined_image.paste(object_image, (width, 0))
    combined_image.paste(generated_image, (width * 2, 0))
    combined_image.paste(groundtruth_image, (width * 3, 0))

    # 保存拼接后的图像
    combined_image.save(f"{output_folder}/combined_{name}")