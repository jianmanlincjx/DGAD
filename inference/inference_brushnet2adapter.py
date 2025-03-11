from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler, UNet2DConditionModel, DDPMScheduler
import torch
import cv2
import numpy as np
from PIL import Image
import sys
import os
from torchvision import transforms
import torchvision
sys.path.append(os.getcwd())
import argparse
from diffusers.utils import load_image



transform_image = transforms.Compose([
    transforms.ToTensor(),
])

base_model_path = "pretrain_model/stable-diffusion-v1-5"
brushnet_path = "pretrain_model/segmentation_mask_brushnet_ckpt"
unet_path = '/data1/JM/code/BrushNet-main/exp/brushnet2adapter/checkpoint-50000'

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16, is_inference=False).to('cuda')
unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", torch_dtype=torch.float16)

pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, unet=unet, torch_dtype=torch.float16, low_cpu_mem_usage=True,
).to('cuda')
noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
generator = torch.Generator("cuda").manual_seed(1234)
image_name_list = sorted(os.listdir('/data1/JM/code/BrushNet-main/datasets/MSRA-10K/mask_processed'))
output_folder = '/data1/JM/code/BrushNet-main/datasets/MSRA-10K_result_brushnet2adapter_50000_exp'
os.makedirs(output_folder, exist_ok=True)
# 遍历图像名称列表
for name in image_name_list:
    # 定义图像路径
    txt_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new/text/{name.replace('.png', '.txt')}"
    groundtruth_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/object_processed/{name}"

    # 读取描述文本
    with open(txt_path, "r") as f:
        # caption = f.read()
        caption = " "


    target_image = cv2.imread(groundtruth_image_path)
    target_image = Image.fromarray(target_image.astype(np.uint8)).convert("RGB")
    target_image = transform_image(target_image)

    mask = torch.zeros((1, target_image.shape[1], target_image.shape[2])) 


    generated_image = pipe(
        caption, 
        target_image.unsqueeze(0), 
        mask.unsqueeze(0), 
        num_inference_steps=50, 
        generator=generator,
        brushnet_conditioning_scale=1.0,
        guide_scale=7.5
    ).images[0]
    generated_image.save(f"{output_folder}/{name}")
        # Prepare input image for generation
