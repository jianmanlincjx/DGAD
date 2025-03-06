from PIL import Image
import requests
import os
import torch
import open_clip
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import CLIPProcessor, CLIPModel
clip_model, _, clip_preprocess = open_clip.create_model_from_pretrained('ViT-L-14', pretrained='/data1/JM/code/BrushNet-main/pretrain_model/ViT-L-14/open_clip_pytorch_model.bin')