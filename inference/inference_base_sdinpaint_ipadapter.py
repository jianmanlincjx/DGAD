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
from ip_adapter import IPAdapterPlus


transform_image = transforms.Compose([
    transforms.ToTensor(),
])

base_model_path = "/data/JM/code/BrushNet-main/pretrain_model/models--runwayml--stable-diffusion-inpainting"
brushnet_path = "pretrain_model/segmentation_mask_brushnet_ckpt"
unet_path = '/data/JM/code/BrushNet-main/exp/sd-inpaint_adapter_big_dense/checkpoint-1500000'
image_encoder_path = "/data/JM/code/BrushNet-main/pretrain_model/image_encoder"
ip_ckpt = "/data/JM/code/BrushNet-main/pretrain_model/cross_attention_ip_adapter.bin"
device = 'cuda'

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16, is_inference=False).to('cuda')
unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, unet=unet, torch_dtype=torch.float16, low_cpu_mem_usage=True,
).to('cuda')
noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
generator = torch.Generator("cuda").manual_seed(1234)
image_name_list = sorted(os.listdir('/data/JM/code/BrushNet-main/dataset_validation_demo/object'))
output_folder = '/data/JM/code/BrushNet-main/validation_dataset/DGAD_demo'
os.makedirs(output_folder, exist_ok=True)

ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

# 遍历图像名称列表
for name in image_name_list:
    # 定义图像路径
    source_image_path = f"/data/JM/code/BrushNet-main/dataset_validation_demo/source/{name}"
    mask_image_path = f"/data/JM/code/BrushNet-main/dataset_validation_demo/mask/{name}"
    object_image_path = f"/data/JM/code/BrushNet-main/dataset_validation_demo/object/{name}"
    caption = ' '

    # 读取并转换图像
    source_image = cv2.imread(source_image_path)[:,:,::-1]
    source_image = Image.fromarray(source_image.astype(np.uint8)).convert("RGB")
    background_image = transform_image(source_image)

    object_image = cv2.imread(object_image_path)[:,:,::-1]
    object_image = Image.fromarray(object_image.astype(np.uint8)).convert("RGB").resize((512, 512))
    object_image = transform_image(object_image)

    mask = 1.*(cv2.imread(mask_image_path).sum(-1)>255)[:,:,np.newaxis]
    background_mask = mask
    background_mask = transform_image(Image.fromarray(background_mask.astype(np.uint8).repeat(3,-1)*255).convert("L"))

    object_mask = np.zeros((1, object_image.shape[1], object_image.shape[2]), dtype=np.float32)
    object_mask = transform_image(object_mask)

    brushnet_input = [background_image.unsqueeze(0), background_mask.unsqueeze(0),  object_image.unsqueeze(0), object_mask.unsqueeze(0)]
    object_image_ip = Image.open(object_image_path).resize((256, 256))
    generated_image = ip_model.generate(pil_image=object_image_ip, brushnet_input=brushnet_input, prompt=caption, num_samples=1, num_inference_steps=50, seed=42)[0]
    # generated_image.save(f"{output_folder}/{name}")
    # 读取图像
    source_image = Image.open(source_image_path).convert("RGB")
    object_image = Image.open(object_image_path).convert("RGB")
    
    # 读取mask图像
    mask_image = cv2.imread(mask_image_path)[:,:,::-1]
    mask_image = Image.fromarray(mask_image.astype(np.uint8)).convert("RGB")
    
    # 确保所有图像大小一致
    width, height = source_image.size
    object_image = object_image.resize((width, height))
    generated_image = generated_image.resize((width, height))
    mask_image = mask_image.resize((width, height))
    
    # 将mask贴到source image上
    # 创建一个半透明的红色遮罩
    mask_overlay = Image.new("RGBA", (width, height), (255, 0, 0, 128))
    # 将mask转换为alpha通道
    mask_gray = mask_image.convert("L")
    # 创建一个新的RGBA图像
    source_with_mask = source_image.convert("RGBA")
    # 将mask应用到source image上
    source_with_mask.paste(mask_overlay, (0, 0), mask_gray)
    # 转回RGB
    source_with_mask = source_with_mask.convert("RGB")
    
    # 拼接图像（从左到右）
    combined_image = Image.new("RGB", (width * 3, height))
    combined_image.paste(source_with_mask, (0, 0))
    combined_image.paste(object_image, (width, 0))
    combined_image.paste(generated_image, (width * 2, 0))
    
    # 保存拼接后的图像
    output_path = f"{output_folder}/combined_{name}"
    combined_image.save(output_path)