#!/usr/bin/env python
# coding=utf-8
import argparse
import cv2
import os
import imgaug.augmenters as iaa
import torch.nn as nn
import numpy as np
import torch
import torch.utils.checkpoint
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
from diffusers import (
    AutoencoderKL,
    BrushNetModel,
    DDPMScheduler,
    StableDiffusionBrushNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
import json
from torchvision import transforms

transform_image = transforms.Compose([
    transforms.ToTensor(),
])

class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states 
        # hidden_states = hidden_states 
        # hidden_states = self.scale * ip_hidden_states 

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=768, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):

        embeds = image_embeds
        if embeds.dtype != self.proj.weight.dtype:
            embeds = embeds.to(self.proj.weight.dtype)
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


def rename_keys_in_dict(param_dict, prefix):
    # 初始化一个新的字典来存储重命名后的参数
    renamed_dict = {}
    
    # 遍历字典中的所有键并重命名
    for param_tensor in param_dict:
        # 移除指定的前缀
        if param_tensor.startswith(prefix + '.'):
            new_key = param_tensor[len(prefix) + 1:]
            renamed_dict[new_key] = param_dict[param_tensor]
        else:
            renamed_dict[param_tensor] = param_dict[param_tensor]
    
    return renamed_dict


if __name__ == "__main__":
    image_encoder_path = '/data1/JM/code/BrushNet-main/pretrain_model/image_encoder'
    base_model_path = "/data1/JM/code/BrushNet-main/pretrain_model/stable-diffusion-v1-5"
    brushnet_path = "/data1/JM/code/BrushNet-main/pretrain_model/segmentation_mask_brushnet_ckpt"
    linpro_path = '/data1/JM/code/BrushNet-main/pretrain_model/ip-adapter_sd15.bin'
    ip_adapter_path = '/data1/JM/code/BrushNet-main/pretrain_model/ip-adapter_sd15.bin'
    
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).cuda().to(torch.float16).to('cuda')
    project = ImageProjModel()
    project_params = torch.load(linpro_path)['image_proj']
    project.load_state_dict(project_params, strict=True)
    project = project.cuda().to(torch.float16)

    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", revision=None, variant=None, torch_dtype=torch.float16)
    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)

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
            attn_procs[name] = AttnProcessor().to(torch.float16)
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim).to(torch.float16)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)

    pipeline = StableDiffusionBrushNetPipeline.from_pretrained(
        base_model_path,
        vae=vae,
        unet=unet,
        brushnet=brushnet,
        revision=None,
        variant=None,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to('cuda')
    
    ip_layers = torch.nn.ModuleList(pipeline.unet.attn_processors.values())
    ip_adapter_params = torch.load(ip_adapter_path)['ip_adapter']
    ip_layers.load_state_dict(ip_adapter_params, strict=True)

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()

    blended = True
    # conditioning scale
    brushnet_conditioning_scale=1.0
    clip_image_processor = CLIPImageProcessor()

    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    image_name_list = sorted(os.listdir('/data1/JM/code/BrushNet-main/datasets/MSRA-10K/mask_processed'))
    output_folder = '/data1/JM/code/BrushNet-main/datasets/MSRA-10K_result_with_prompt'
    os.makedirs(output_folder, exist_ok=True)
    # 遍历图像名称列表
    for name in image_name_list:
        # 定义图像路径
        source_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/source_processed/{name}"
        mask_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/mask_processed/{name}"
        object_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/object_processed/{name}"
        txt_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/text/{name.replace('.png', '.txt')}"
        groundtruth_image_path = f"/data1/JM/code/BrushNet-main/datasets/MSRA-10K/target_processed/{name}"

        # 读取描述文本
        with open(txt_path, "r") as f:
            caption = f.read()

        # 打开前景图像并转换为 RGB
        object_image_pil = Image.open(object_image_path).convert("RGB")

        # 使用 CLIP 图像处理器处理前景图像
        object_image = clip_image_processor(images=object_image_pil, return_tensors="pt").pixel_values

        # 将前景图像编码为嵌入向量
        temp = image_encoder(object_image.to("cuda")).image_embeds
        ip_token = project(temp)

        # 使用 Stable Diffusion 管道编码提示文本
        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = pipeline.encode_prompt(
                caption,
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, black sofa",
            )

        # 合并提示嵌入和前景图像嵌入
        prompt_embeds = torch.cat([prompt_embeds_, ip_token], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, torch.zeros_like(ip_token)], dim=1)

        # 读取源图像和掩码图像
        init_image = cv2.imread(source_image_path)[:, :, ::-1]  # 转换为 RGB
        mask_image = 1.0 * (cv2.imread(mask_image_path).sum(-1) > 255)[:, :, np.newaxis]  # 二值化掩码

        # 应用掩码到源图像
        init_image = init_image * (1 - mask_image)
        source_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
        mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

        # 下采样源图像和掩码图像
        new_size = (source_image.width // 1, source_image.height // 1)
        downsampled_source = source_image.resize(new_size, Image.Resampling.LANCZOS)
        downsampled_mask = mask_image.resize(new_size, Image.Resampling.LANCZOS)
        source_image = Image.fromarray(np.array(downsampled_source))
        mask_image = Image.fromarray(np.array(downsampled_mask))

        # 设置随机种子并生成图像
        generator = torch.Generator("cuda").manual_seed(1234)
        generated_image = pipeline(
            image=source_image,
            mask=mask_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=50,
            generator=generator,
            brushnet_conditioning_scale=1.0,
            guidance_scale=7.5,
        ).images[0]

        # 读取 groundtruth 图像
        groundtruth_image = Image.open(groundtruth_image_path).convert("RGB")

        # 将所有图像调整为相同高度（以最高图像为准）
        max_height = max(
            source_image.height,
            object_image_pil.height,
            generated_image.height,
            groundtruth_image.height,
        )

        # 调整图像大小
        source_image_resized = source_image.resize((source_image.width, max_height), Image.Resampling.LANCZOS)
        object_image_resized = object_image_pil.resize((object_image_pil.width, max_height), Image.Resampling.LANCZOS)
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