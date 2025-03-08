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
unet_path = '/data1/JM/code/BrushNet-main/exp/brushnet_adapter_small/checkpoint-20000'

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
output_folder = '/data1/JM/code/BrushNet-main/datasets/MSRA-10K_result_with_prompt_20000'
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
    source_image = cv2.imread(source_image_path)
    source_image = Image.fromarray(source_image.astype(np.uint8))
    background_image = transform_image(source_image)

    # target_image = cv2.imread(groundtruth_image_path)
    # target_image = Image.fromarray(target_image.astype(np.uint8)).convert("RGB")
    # target_image = transform_image(target_image)
    target_image = load_image(groundtruth_image_path)

    object_image = cv2.imread(object_image_path)
    object_image = Image.fromarray(object_image.astype(np.uint8))
    object_image = transform_image(object_image)

    mask = 1.*(cv2.imread(mask_image_path).sum(-1)>255)[:,:,np.newaxis]
    background_mask = mask
    object_mask = (1-mask)
    background_mask = transform_image(Image.fromarray(background_mask.astype(np.uint8).repeat(3,-1)*255).convert("L"))
    object_mask = transform_image(Image.fromarray(object_mask.astype(np.uint8).repeat(3,-1)*255).convert("L"))

    # input_image = pipe.prepare_image(
    #         image=target_image,
    #         width=target_image.size[0],
    #         height=target_image.size[1],
    #         batch_size=1,
    #         num_images_per_prompt=1,
    #         device='cuda',
    #         dtype=torch.float16,
    #         do_classifier_free_guidance=False,
    #         guess_mode=False)
    
    # latent = pipe.vae.encode(input_image).latent_dist.sample() * pipe.vae.config.scaling_factor
    # noise = torch.randn_like(latent).to(torch.float16)
    # noisy_latents = noise_scheduler.add_noise(latent, noise, torch.tensor(999))
    # torchvision.utils.save_image(background_image.unsqueeze(0), f"_background_image.png")
    # torchvision.utils.save_image(background_mask.unsqueeze(0), f"_background_mask.png")
    # torchvision.utils.save_image(object_image.unsqueeze(0), f"_object_image.png")
    # torchvision.utils.save_image(object_mask.unsqueeze(0), f"_object_mask.png")
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
    generated_image.save(f"{output_folder}/{name}")
        # Prepare input image for generation
