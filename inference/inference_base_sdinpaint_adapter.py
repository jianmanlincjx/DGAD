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
unet_path = '/data1/JM/code/BrushNet-main/exp/brushnet_adapter_small_bigobject/checkpoint-200000'

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
output_folder = '/data1/JM/code/BrushNet-main/datasets/MSRA-10K_result_brushnetadapter_200000_bigobject'
os.makedirs(output_folder, exist_ok=True)
# 遍历图像名称列表
for name in image_name_list[:100]:
    # 定义图像路径
    source_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/source_processed/{name}"
    mask_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/mask_processed/{name}"
    object_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/object_processed/{name}"
    txt_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/text/{name.replace('.png', '.txt')}"
    groundtruth_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/target_processed/{name}"

    # 读取描述文本
    with open(txt_path, "r") as f:
        caption = f.read()

    # 读取并转换图像
    source_image = cv2.imread(source_image_path)[:,:,::-1]
    source_image = Image.fromarray(source_image.astype(np.uint8)).convert("RGB")
    background_image = transform_image(source_image)

    target_image = cv2.imread(groundtruth_image_path)[:,:,::-1]
    target_image = Image.fromarray(target_image.astype(np.uint8)).convert("RGB")
    target_image = transform_image(target_image)
    target_image = load_image(groundtruth_image_path)

    object_image = cv2.imread(object_image_path)[:,:,::-1]
    object_image = Image.fromarray(object_image.astype(np.uint8)).convert("RGB")
    object_image = transform_image(object_image)

    mask = 1.*(cv2.imread(mask_image_path).sum(-1)>255)[:,:,np.newaxis]
    background_mask = mask
    object_mask = (1-mask)
    background_mask = transform_image(Image.fromarray(background_mask.astype(np.uint8).repeat(3,-1)*255).convert("L"))
    object_mask = transform_image(Image.fromarray(object_mask.astype(np.uint8).repeat(3,-1)*255).convert("L"))

    # # 生成图像
    generated_image = pipe(
        caption, 
        background_image.unsqueeze(0), 
        background_mask.unsqueeze(0), 
        object_image.unsqueeze(0), 
        object_mask.unsqueeze(0), 
        num_inference_steps=50, 
        generator=generator,
        brushnet_conditioning_scale=1.0,
        guide_scale=7.5
    ).images[0]
    # generated_image.save(f"{output_folder}/{name}")

    # 读取图像
    source_image = Image.open(source_image_path).convert("RGB")
    object_image = Image.open(object_image_path).convert("RGB")
    target_image = Image.open(groundtruth_image_path).convert("RGB")

    # 确保所有图像大小一致
    width, height = source_image.size
    object_image = object_image.resize((width, height))
    generated_image = generated_image.resize((width, height))
    target_image = target_image.resize((width, height))

    # 拼接图像（从左到右）
    combined_image = Image.new("RGB", (width * 4, height))
    combined_image.paste(source_image, (0, 0))
    combined_image.paste(object_image, (width, 0))
    combined_image.paste(generated_image, (width * 2, 0))
    combined_image.paste(target_image, (width * 3, 0))

    # 保存拼接后的图像
    output_path = f"{output_folder}/combined_{name}"
    combined_image.save(output_path)