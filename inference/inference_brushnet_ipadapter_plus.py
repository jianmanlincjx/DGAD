import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from diffusers import BrushNetModel, UniPCMultistepScheduler, StableDiffusionBrushNetPipeline
from PIL import Image
import os
from ip_adapter import IPAdapterPlus
import cv2
from torchvision import transforms
import numpy as np

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

transform_image = transforms.Compose([
    transforms.ToTensor(),
])


vae_model_path = "/data1/JM/code/BrushNet-main/pretrain_model/models--stabilityai--sd-vae-ft-mse"
image_encoder_path = "/data1/JM/code/BrushNet-main/pretrain_model/image_encoder"
ip_ckpt = "/data1/JM/code/BrushNet-main/pretrain_model/ip-adapter-plus_sd15.bin"
base_model_path = "/data1/JM/code/BrushNet-main/pretrain_model/stable-diffusion-v1-5"
brushnet_path = '/data1/JM/code/BrushNet-main/pretrain_model/segmentation_mask_brushnet_ckpt'
device = "cuda"

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16, is_inference=False).to('cuda')
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=True, vae=vae
).to('cuda')
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# load ip-adapter
ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

image_name_list = sorted(os.listdir('/data1/JM/code/BrushNet-main/datasets/MSRA-10K/mask_processed'))
output_folder = '/data1/JM/code/BrushNet-main/datasets/MSRA-10K_result_brushnet_ipadapter_plus'
os.makedirs(output_folder, exist_ok=True)

for name in image_name_list:
    # 定义图像路径
    source_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/source_processed/{name}"
    mask_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/mask_processed/{name}"
    object_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/object_processed/{name}"
    txt_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/text/{name.replace('.png', '.txt')}"
    groundtruth_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/target_processed/{name}"

    # 读取源图像和掩码图像
    init_image = cv2.imread(source_image_path)[:, :, ::-1]  # 转换为 RGB
    mask_image = 1.0 * (cv2.imread(mask_image_path).sum(-1) > 255)[:, :, np.newaxis]  # 二值化掩码

    # 应用掩码到源图像
    init_image = init_image * (1 - mask_image)
    source_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

    object_image = Image.open(object_image_path).resize((256, 256))
    groundtruth_image = Image.open(groundtruth_image_path).convert("RGB")

    # 读取描述文本
    with open(txt_path, "r") as f:
        caption = f.read()
    # generate image variations
    generated_image = ip_model.generate(pil_image=object_image, brushnet_input=[source_image, mask_image], prompt=caption, num_samples=1, num_inference_steps=50, seed=42)[0]

    # 读取 groundtruth 图像
    groundtruth_image = Image.open(groundtruth_image_path).convert("RGB")

    # 将所有图像调整为相同高度（以最高图像为准）
    max_height = max(
        source_image.height,
        object_image.height,
        generated_image.height,
        groundtruth_image.height,
    )

    # 调整图像大小
    source_image_resized = source_image.resize((source_image.width, max_height), Image.Resampling.LANCZOS)
    object_image_resized = object_image.resize((source_image.width, max_height), Image.Resampling.LANCZOS)
    generated_image_resized = generated_image.resize((generated_image.width, max_height), Image.Resampling.LANCZOS)
    groundtruth_image_resized = groundtruth_image.resize((groundtruth_image.width, max_height), Image.Resampling.LANCZOS)

    # 拼接图像
    total_width = (
        source_image_resized.width
        + object_image_resized.width
        + generated_image_resized.width
        + groundtruth_image_resized.width
    )
    combined_image = Image.new("RGB", (total_width, max_height))
    combined_image.paste(source_image_resized, (0, 0))
    combined_image.paste(object_image_resized, (source_image_resized.width, 0))
    combined_image.paste(generated_image_resized, (source_image_resized.width + object_image_resized.width, 0))
    combined_image.paste(
        groundtruth_image_resized,
        (source_image_resized.width + object_image_resized.width + generated_image_resized.width, 0),
    )

    # 保存拼接后的图像
    output_path = os.path.join(output_folder, name)
    combined_image.save(output_path)

    print(f"Saved combined image to {output_path}")