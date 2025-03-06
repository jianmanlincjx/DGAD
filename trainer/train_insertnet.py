import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import BrushNetModel, AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from safetensors.torch import load_file

import sys
sys.path.append(os.getcwd())
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
from diffusers.utils.import_utils import is_xformers_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"

# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        # list of dict: [{"source":"路径", "target":"路径", "mask":"路径", "object":"路径", text": "A dog"}]
        # with open(json_file, 'r') as f:
        #     self.data = [json.loads(line) for line in f]
        #     print(f"数据加载完成，条目数：{len(self.data)}")
        self.data = json.load(open(json_file))
        print(f"数据加载完成，条目数：{len(self.data)}")
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.clip_image_processor = CLIPImageProcessor()

    def fill_bounding_box_with_white(self, image_path):
        # Load the image
        image = Image.open(image_path)
        image_cv = np.array(image)

        # Check if the image is already grayscale
        if len(image_cv.shape) == 3:
            # Convert the image to grayscale if it's a color image
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
            # If the image is already grayscale, use it directly
            gray = image_cv

        # Apply threshold to get a binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Create a copy of the image and fill the bounding box area with white color
        image_filled = image_cv.copy()
        cv2.rectangle(image_filled, (x, y), (x+w, y+h), (255, 255, 255), -1)

        # Convert the result to an image and return
        filled_image = Image.fromarray(image_filled)
        return filled_image
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        source_image_path = item['source']
        target_image_path = item['target']
        mask_image_path = item['mask']
        object_image_path = item['object']
        text = item["text"]
        
        # read image
        source_image = Image.open(source_image_path)
        source_image = self.transform(source_image.convert("RGB"))

        target_image = Image.open(target_image_path)
        target_image = self.transform(target_image.convert("RGB"))

        object_image = Image.open(object_image_path)
        object_image = self.transform(object_image.convert("RGB"))

        # if random.random() < 0.5:
        #     mask_image = self.fill_bounding_box_with_white(mask_image_path)
        # else:
        mask_image = Image.open(mask_image_path)
        mask_image = self.transform(mask_image.convert("L"))    

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "background": source_image,
            "groundtruth": target_image,
            "mask": mask_image,
            "foreground": object_image,
            "text": text_input_ids,
        }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    source_images = torch.stack([example["background"] for example in data])
    target_images = torch.stack([example["groundtruth"] for example in data])
    mask_images = torch.stack([example["mask"] for example in data])
    object_images = torch.stack([example["foreground"] for example in data])
    text_input_ids = torch.cat([example["text"] for example in data], dim=0)

    return {
            "background": source_images,
            "groundtruth": target_images,
            "mask": mask_images,
            "foreground": object_images,
            "text": text_input_ids,
           }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, brushnet, unet, weight_dtype=torch.float16):
        super().__init__()
        self.brushnet = brushnet
        self.unet = unet
        self.weight_dtype = weight_dtype

    def forward(self, noisy_latents, conditioning_latents, timesteps, encoder_hidden_states):
        # Predict the noise residual
        down_block_res_samples, mid_block_res_sample, up_block_res_samples = self.brushnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            brushnet_cond=conditioning_latents,
            return_dict=False,
        )

        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_add_samples=[
                sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples
            ],
            mid_block_add_sample=mid_block_res_sample.to(dtype=self.weight_dtype),
            up_block_add_samples=[
                sample.to(dtype=self.weight_dtype) for sample in up_block_res_samples
            ],
            return_dict=False,
        )[0]

                
        return model_pred

    def load_from_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="pretrain_model/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default="datasets/FBP_img_src/sa_000000/images_data.json",
        help="Training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exp/test",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=args.output_dir)  # 将日志文件保存到 output_dir

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    brushnet = BrushNetModel.from_unet(unet)
    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    brushnet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet.enable_xformers_memory_efficient_attention()
            brushnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    ip_adapter = IPAdapter(brushnet, unet)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(
        ip_adapter.brushnet.parameters(),
    )

    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    global_step = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint)
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path),map_location="cpu")
        global_step = int(path.split("-")[1])

    # 外层是 epoch 进度条
    for epoch in range(0, args.num_train_epochs):

        # 创建一个 tqdm 进度条，用于显示每个 epoch 内的训练进度
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.num_train_epochs}", unit="batch") as tbar:
            for step, batch in enumerate(tbar):
                # 使用 accelerator.accumulate 优化器加速
                with accelerator.accumulate(ip_adapter):
                    # Convert images to latent space
                    with torch.no_grad():
                        latents = vae.encode(batch["groundtruth"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                        source_latents = vae.encode(batch["background"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                        object_latents = vae.encode(batch["foreground"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()

                        latents = latents * vae.config.scaling_factor
                        source_latents = source_latents * vae.config.scaling_factor
                    ###############################################################################
                    # import torchvision
                    # torchvision.utils.save_image(batch["groundtruth"], "batch_groundtruth.png")
                    # torchvision.utils.save_image(batch["background"], "batch_background.png")
                    # torchvision.utils.save_image(batch["mask"], "batch_mask.png")
                    # torchvision.utils.save_image(batch["foreground"], "batch_foreground.png")
                    # import time
                    # time.sleep(5)
                    ###############################################################################
                    
                    masks = torch.nn.functional.interpolate(
                        batch["mask"], 
                        size=(
                            latents.shape[-2], 
                            latents.shape[-1]
                        )
                    )
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]


                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["text"].to(accelerator.device))[0]

                    conditioning_latents = torch.concat([source_latents, object_latents, masks], dim=1)

                    noise_pred = ip_adapter(noisy_latents, conditioning_latents, timesteps, encoder_hidden_states)

                    # Calculate loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                    # Gather the losses across all processes for logging (if we use distributed training)
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                    # Update progress bar description with the loss
                    tbar.set_postfix(loss=avg_loss)

                    # Backpropagate
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                    # 每 100 步记录一次损失
                    if global_step % 100 == 0:
                        writer.add_scalar('Loss/train', avg_loss, global_step)

                global_step += 1

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path, safe_serialization=False)

                begin = time.perf_counter()

    # 关闭 TensorBoard writer
    writer.close()

                
if __name__ == "__main__":
    main()    
