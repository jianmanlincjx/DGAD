from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# choose the base model here
base_model_path = "pretrain_model/stable-diffusion-v1-5"
# base_model_path = "runwayml/stable-diffusion-v1-5"

# input brushnet ckpt path
brushnet_path = "pretrain_model/segmentation_mask_brushnet_ckpt"

# choose whether using blended operation
blended = False

# input source image / mask image path and the text prompt
image_path="/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new/source_processed/0000002.png"
mask_path="/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new/mask_processed/0000002.png"
object_image_path="/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new/object_processed/0000002.png"
caption=" "

# conditioning scale
brushnet_conditioning_scale=1.0

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

source_image = cv2.imread(image_path)[:,:,::-1]
mask = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
background_image = source_image * (1-mask)
object_image = cv2.imread(object_image_path)[:,:,::-1]
object_mask = (1-mask)

background_image = Image.fromarray(background_image.astype(np.uint8)).convert("RGB")
background_mask = Image.fromarray(mask.astype(np.uint8).repeat(3,-1)*255).convert("RGB")
object_image = Image.fromarray(object_image.astype(np.uint8)).convert("RGB")
object_mask = Image.fromarray(object_mask.astype(np.uint8).repeat(3,-1)*255).convert("RGB")
background_image.save("background_image.png")
background_mask.save("background_mask.png")
object_image.save("object_image.png")
object_mask.save("object_mask.png")

generator = torch.Generator("cuda").manual_seed(1234)

image = pipe(
    caption, 
    background_image, 
    background_mask,
    object_image, 
    object_mask,
    num_inference_steps=50, 
    generator=generator,
    brushnet_conditioning_scale=brushnet_conditioning_scale
).images[0]


image.save("output_me_1aa.png")