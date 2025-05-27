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
from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"

params_to_train = ['up_blocks.1.up_1.conv_layers.1.bias', 'up_blocks.2.up_1.conv_layers.0.weight', 'up_blocks.0.connect_1_out.bias', 'up_blocks.1.up_0.mlp.2.bias', 'up_blocks.1.up_0.conv_layers.0.weight', 'up_blocks.2.up_2_upsamplers.to_k.weight', 'up_blocks.3.up_1.to_v.weight', 'up_blocks.3.up_0.to_q.weight', 'up_blocks.3.up_0.mlp.0.bias', 'up_blocks.1.up_1.to_v.weight', 'up_blocks.1.up_2.mlp.0.weight', 'up_blocks.1.up_0.to_k.weight', 'up_blocks.2.up_1.conv_layers.1.weight', 'up_blocks.0.up_2.to_q.weight', 'up_blocks.1.up_2.to_v.weight', 'up_blocks.2.up_2.conv_layers.0.weight', 'up_blocks.1.up_0.mlp.0.bias', 'up_blocks.2.connect_2_out_upsamplers.bias', 'up_blocks.0.up_0.to_k.weight', 'up_blocks.2.up_1.to_v.weight', 'up_blocks.0.up_2_upsamplers.to_v.weight', 'up_blocks.0.up_1.mlp.2.weight', 'up_blocks.2.up_0.conv_layers.0.weight', 'up_blocks.1.up_2.mlp.2.weight', 'up_blocks.0.up_1.conv_layers.1.weight', 'up_blocks.3.up_2.mlp.2.bias', 'up_blocks.0.up_0.to_q.weight', 'up_blocks.0.up_2.mlp.2.weight', 'up_blocks.3.up_0.to_k.weight', 'up_blocks.3.up_1.conv_layers.1.bias', 'up_blocks.2.up_2.mlp.2.bias', 'up_blocks.1.connect_2_out_upsamplers.bias', 'up_blocks.0.up_0.conv_layers.0.weight', 'up_blocks.0.up_2_upsamplers.mlp.2.bias', 'up_blocks.0.up_2_upsamplers.mlp.2.weight', 'up_blocks.1.connect_2_out.weight', 'up_blocks.2.up_0.to_q.weight', 'up_blocks.3.up_0.conv_layers.1.bias', 'up_blocks.3.up_1.conv_layers.0.bias', 'up_blocks.1.up_2.conv_layers.0.weight', 'up_blocks.0.up_2_upsamplers.mlp.0.bias', 'up_blocks.1.up_2.conv_layers.0.bias', 'up_blocks.2.connect_0_out.weight', 'up_blocks.3.up_0.to_v.weight', 'up_blocks.2.connect_2_out.weight', 'up_blocks.1.up_2_upsamplers.mlp.0.weight', 'up_blocks.1.up_0.conv_layers.0.bias', 'up_blocks.0.up_0.conv_layers.0.bias', 'up_blocks.1.up_1.conv_layers.0.weight', 'up_blocks.3.up_2.conv_layers.0.weight', 'up_blocks.2.up_2.to_q.weight', 'up_blocks.2.up_1.mlp.2.weight', 'up_blocks.3.up_1.conv_layers.0.weight', 'up_blocks.0.up_2_upsamplers.mlp.0.weight', 'up_blocks.2.up_2.to_k.weight', 'up_blocks.3.up_1.mlp.0.bias', 'up_blocks.1.up_2_upsamplers.to_q.weight', 'up_blocks.1.up_2_upsamplers.to_v.weight', 'up_blocks.3.up_0.conv_layers.0.weight', 'up_blocks.2.up_2_upsamplers.conv_layers.1.bias', 'up_blocks.0.up_2_upsamplers.conv_layers.0.weight', 'up_blocks.3.up_0.mlp.0.weight', 'up_blocks.1.connect_0_out.weight', 'up_blocks.2.connect_1_out.weight', 'up_blocks.0.up_2.conv_layers.1.weight', 'up_blocks.1.up_0.mlp.0.weight', 'up_blocks.1.up_1.conv_layers.0.bias', 'up_blocks.2.up_2_upsamplers.to_v.weight', 'up_blocks.1.up_2.to_k.weight', 'up_blocks.2.up_2.conv_layers.1.bias', 'up_blocks.2.up_2_upsamplers.to_q.weight', 'up_blocks.2.up_0.to_k.weight', 'up_blocks.1.up_1.mlp.2.bias', 'up_blocks.3.up_1.conv_layers.1.weight', 'up_blocks.0.connect_0_out.bias', 'up_blocks.1.up_2_upsamplers.conv_layers.1.weight', 'up_blocks.0.up_2.conv_layers.1.bias', 'up_blocks.0.up_2_upsamplers.conv_layers.0.bias', 'up_blocks.1.up_1.conv_layers.1.weight', 'up_blocks.3.up_0.conv_layers.1.weight', 'up_blocks.3.up_0.mlp.2.bias', 'up_blocks.0.up_2_upsamplers.conv_layers.1.weight', 'up_blocks.2.up_2.conv_layers.1.weight', 'up_blocks.2.up_0.mlp.0.weight', 'up_blocks.1.up_1.mlp.0.weight', 'up_blocks.1.up_2_upsamplers.mlp.2.weight', 'up_blocks.3.connect_2_out.weight', 'up_blocks.0.up_2.conv_layers.0.weight', 'up_blocks.0.connect_1_out.weight', 'up_blocks.2.up_0.conv_layers.1.weight', 'up_blocks.3.up_2.to_k.weight', 'up_blocks.2.up_2.mlp.2.weight', 'up_blocks.0.up_2.mlp.2.bias', 'up_blocks.0.up_2_upsamplers.to_k.weight', 'up_blocks.1.up_2_upsamplers.conv_layers.1.bias', 'up_blocks.2.connect_0_out.bias', 'up_blocks.0.connect_2_out.weight', 'up_blocks.0.connect_2_out_upsamplers.bias', 'up_blocks.2.up_1.to_q.weight', 'up_blocks.0.up_1.conv_layers.0.bias', 'up_blocks.1.connect_0_out.bias', 'up_blocks.1.up_2.conv_layers.1.weight', 'up_blocks.2.up_0.conv_layers.1.bias', 'up_blocks.3.up_2.conv_layers.1.weight', 'up_blocks.3.connect_2_out.bias', 'up_blocks.2.up_2_upsamplers.conv_layers.0.bias', 'up_blocks.2.up_0.to_v.weight', 'up_blocks.0.up_0.conv_layers.1.weight', 'up_blocks.1.up_2_upsamplers.conv_layers.0.bias', 'up_blocks.3.up_0.mlp.2.weight', 'up_blocks.2.up_2.to_v.weight', 'up_blocks.2.up_1.mlp.0.bias', 'up_blocks.2.up_2.conv_layers.0.bias', 'up_blocks.1.connect_1_out.weight', 'up_blocks.1.up_1.to_q.weight', 'up_blocks.1.up_2.mlp.2.bias', 'up_blocks.1.up_2.mlp.0.bias', 'up_blocks.0.up_2.to_v.weight', 'up_blocks.0.up_1.mlp.2.bias', 'up_blocks.2.up_1.conv_layers.0.bias', 'up_blocks.3.connect_0_out.bias', 'up_blocks.2.up_2_upsamplers.mlp.2.bias', 'up_blocks.2.connect_1_out.bias', 'up_blocks.1.up_0.to_v.weight', 'up_blocks.0.up_1.to_v.weight', 'up_blocks.3.connect_0_out.weight', 'up_blocks.2.up_0.mlp.2.weight', 'up_blocks.1.connect_2_out_upsamplers.weight', 'up_blocks.2.up_0.mlp.0.bias', 'up_blocks.1.up_0.conv_layers.1.bias', 'up_blocks.1.up_2_upsamplers.mlp.2.bias', 'up_blocks.0.up_2_upsamplers.to_q.weight', 'up_blocks.0.up_1.to_k.weight', 'up_blocks.3.connect_1_out.weight', 'up_blocks.2.connect_2_out.bias', 'up_blocks.0.up_1.to_q.weight', 'up_blocks.3.up_2.mlp.0.bias', 'up_blocks.1.up_2_upsamplers.conv_layers.0.weight', 'up_blocks.2.up_1.mlp.0.weight', 'up_blocks.2.up_1.conv_layers.1.bias', 'up_blocks.3.up_1.mlp.2.bias', 'up_blocks.0.up_1.mlp.0.bias', 'up_blocks.0.up_1.mlp.0.weight', 'up_blocks.2.up_2_upsamplers.conv_layers.1.weight', 'up_blocks.0.up_2.conv_layers.0.bias', 'up_blocks.2.up_0.mlp.2.bias', 'up_blocks.3.up_1.mlp.0.weight', 'up_blocks.3.up_2.to_v.weight', 'up_blocks.1.up_2_upsamplers.mlp.0.bias', 'up_blocks.0.up_2_upsamplers.conv_layers.1.bias', 'up_blocks.1.up_2_upsamplers.to_k.weight', 'up_blocks.1.up_0.conv_layers.1.weight', 'up_blocks.0.up_0.to_v.weight', 'up_blocks.2.up_2.mlp.0.weight', 'up_blocks.3.up_1.to_q.weight', 'up_blocks.0.connect_2_out.bias', 'up_blocks.2.up_2_upsamplers.mlp.0.bias', 'up_blocks.3.up_1.to_k.weight', 'up_blocks.2.up_1.to_k.weight', 'up_blocks.3.up_0.conv_layers.0.bias', 'up_blocks.0.connect_0_out.weight', 'up_blocks.1.connect_2_out.bias', 'up_blocks.2.up_2_upsamplers.mlp.0.weight', 'up_blocks.3.up_2.conv_layers.1.bias', 'up_blocks.1.up_1.mlp.2.weight', 'up_blocks.0.up_0.mlp.0.bias', 'up_blocks.2.up_2_upsamplers.mlp.2.weight', 'up_blocks.2.up_2_upsamplers.conv_layers.0.weight', 'up_blocks.1.up_2.to_q.weight', 'up_blocks.2.up_2.mlp.0.bias', 'up_blocks.0.up_2.to_k.weight', 'up_blocks.1.up_1.mlp.0.bias', 'up_blocks.1.up_2.conv_layers.1.bias', 'up_blocks.1.up_0.to_q.weight', 'up_blocks.2.up_1.mlp.2.bias', 'up_blocks.0.up_1.conv_layers.0.weight', 'up_blocks.0.up_0.conv_layers.1.bias', 'up_blocks.3.up_2.mlp.0.weight', 'up_blocks.1.up_1.to_k.weight', 'up_blocks.0.up_0.mlp.2.bias', 'up_blocks.0.up_2.mlp.0.bias', 'up_blocks.2.connect_2_out_upsamplers.weight', 'up_blocks.3.up_1.mlp.2.weight', 'up_blocks.3.up_2.to_q.weight', 'up_blocks.0.up_2.mlp.0.weight', 'up_blocks.2.up_0.conv_layers.0.bias', 'up_blocks.0.up_1.conv_layers.1.bias', 'up_blocks.3.connect_1_out.bias', 'up_blocks.3.up_2.conv_layers.0.bias', 'up_blocks.3.up_2.mlp.2.weight', 'up_blocks.1.connect_1_out.bias', 'up_blocks.1.up_0.mlp.2.weight', 'up_blocks.0.connect_2_out_upsamplers.weight', 'up_blocks.0.up_0.mlp.0.weight', 'up_blocks.0.up_0.mlp.2.weight']

def set_requires_grad(model, param_name, value=True):
    parts = param_name.split('.')
    obj = model
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]  # 处理数字索引
        else:
            obj = getattr(obj, part)
    final_attr = parts[-1]
    getattr(obj, final_attr).requires_grad_(value)


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        # list of dict: [{"source":"路径", "target":"路径", "mask":"路径", "object":"路径", text": "A dog"}]
        with open(json_file, 'r') as f:
            self.data = [json.loads(line) for line in f]
            print(f"数据加载完成，条目数：{len(self.data)}")
        # self.data = json.load(open(json_file))
        print(f"数据加载完成，条目数：{len(self.data)}")
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        source_image_path = item['source']
        target_image_path = item['target']
        mask_image_path = item['mask']
        object_image_path = item['object']
        text = item["text"]
    
        # read image
        raw_image = Image.open(object_image_path)
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        # read image
        source_image = cv2.imread(source_image_path)
        target_image = cv2.imread(target_image_path)
        object_image = cv2.imread(object_image_path)


        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        object_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)

        source_image = (source_image.astype(np.float32) / 127.5) - 1.0
        target_image = (target_image.astype(np.float32) / 127.5) - 1.0
        object_image = (object_image.astype(np.float32) / 127.5) - 1.0


        source_image = torch.tensor(source_image).permute(2,0,1)
        target_image = torch.tensor(target_image).permute(2,0,1)
        object_image = torch.tensor(object_image).permute(2,0,1)

        source_mask = 1.*(cv2.imread(mask_image_path).sum(-1)>255)[:,:,np.newaxis]
        source_mask=source_mask.astype(np.float32)
        source_mask = torch.tensor(source_mask).permute(2,0,1)

        object_mask = np.zeros((1, target_image.shape[1], target_image.shape[2]), dtype=np.float32)
        object_mask = torch.tensor(object_mask)

        source_image = source_image * (1 - source_mask)
    
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
            "background_mask": source_mask,
            "foreground": object_image,
            "foreground_mask": object_mask,
            "text": text_input_ids,
            "clip_image": clip_image
        }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    source_images = torch.stack([example["background"] for example in data])
    target_images = torch.stack([example["groundtruth"] for example in data])
    background_mask = torch.stack([example["background_mask"] for example in data])
    object_mask = torch.stack([example["foreground_mask"] for example in data])
    object_images = torch.stack([example["foreground"] for example in data])
    text_input_ids = torch.cat([example["text"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)

    return {
            "background": source_images,
            "groundtruth": target_images,
            "background_mask": background_mask,
            "foreground": object_images,
            "foreground_mask": object_mask,
            "text": text_input_ids,
            "clip_images": clip_images
           }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, brushnet, unet, image_proj_model, adapter_modules, ckpt, weight_dtype=torch.float16):
        super().__init__()
        self.brushnet = brushnet
        self.unet = unet
        self.weight_dtype = weight_dtype
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.load_from_checkpoint(ckpt_path=ckpt)

    def forward(self, noisy_latents, source_latents_condation, conditioning_latents, timesteps, encoder_hidden_states, image_embeds):
        # Predict the noise residual
        down_block_res_samples, mid_block_res_sample, up_block_res_samples = self.brushnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            brushnet_cond=conditioning_latents,
            return_dict=False,
        )

        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        model_pred = self.unet(
            torch.concat([noisy_latents, source_latents_condation], dim=1),
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
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Check if 'latents' exists in both the saved state_dict and the current model's state_dict
        strict_load_image_proj_model = True
        if "latents" in state_dict["image_proj"] and "latents" in self.image_proj_model.state_dict():
            # Check if the shapes are mismatched
            if state_dict["image_proj"]["latents"].shape != self.image_proj_model.state_dict()["latents"].shape:
                print(f"Shapes of 'image_proj.latents' in checkpoint {ckpt_path} and current model do not match.")
                print("Removing 'latents' from checkpoint and loading the rest of the weights.")
                del state_dict["image_proj"]["latents"]
                strict_load_image_proj_model = False

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=strict_load_image_proj_model)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data/JM/code/BrushNet-main/pretrain_model/models--runwayml--stable-diffusion-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default="/data/JM/code/BrushNet-main/pretrain_model/ip-adapter-plus_sd15.bin",
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default="/data/JM/code/BrushNet-main/dataset_big/all/image_data.json",
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
        "--brushnet_model_name_or_path",
        type=str,
        default='pretrain_model/segmentation_mask_brushnet_ckpt',
        help="Path to pretrained brushnet model or model identifier from huggingface.co/models."
        " If not specified brushnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--image_encoder_path",
        default="/data/JM/code/BrushNet-main/pretrain_model/image_encoder",
        type=str,
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
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", ignore_mismatched_sizes=True, low_cpu_mem_usage=False, device_map=None)
    brushnet = BrushNetModel.from_pretrained(args.brushnet_model_name_or_path)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    print(f"Loading existing brushnet weights from {args.brushnet_model_name_or_path}")

    #ip-adapter-plus
    image_proj_model = Resampler(
        dim=unet.config.cross_attention_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=16,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=16)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    brushnet.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    image_proj_model.requires_grad_(False)

    for param_name in params_to_train:
        if param_name in unet.state_dict():
            set_requires_grad(unet, param_name, True)
            print(f'unet.{param_name}.requires_grad_(True)')

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet.enable_xformers_memory_efficient_attention()
            brushnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    ip_adapter = IPAdapter(brushnet, unet, image_proj_model, adapter_modules, ckpt=args.pretrained_ip_adapter_path)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    brushnet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    image_proj_model.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(
        ip_adapter.unet.parameters(),
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
                    # torchvision.utils.save_image(batch["groundtruth"], "_groundtruth.png")
                    # torchvision.utils.save_image(batch["background"], "_background.png")
                    # torchvision.utils.save_image(batch["background_mask"], "_background_mask.png")
                    # torchvision.utils.save_image(batch["foreground"], "_foreground.png")
                    # torchvision.utils.save_image(batch["foreground_mask"], "_foreground_mask.png")
                    # import time
                    # time.sleep(5)
                    ###############################################################################
                    with torch.no_grad():
                        image_embeds = image_encoder(batch['clip_images'].to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]

                    background_mask = torch.nn.functional.interpolate(
                        batch["background_mask"], 
                        size=(
                            latents.shape[-2], 
                            latents.shape[-1]
                        )
                    )
                    object_mask = torch.nn.functional.interpolate(
                        batch["foreground_mask"], 
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

                    conditioning_latents = torch.concat([object_latents, object_mask], dim=1)
                    source_latents_condation = torch.concat([background_mask, source_latents], dim=1)
                    noise_pred = ip_adapter(noisy_latents, source_latents_condation, conditioning_latents, timesteps, encoder_hidden_states, image_embeds)

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
                    if global_step % 10 == 0:
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
