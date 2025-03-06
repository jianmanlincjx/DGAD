import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os
import torch

# 设置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载处理器和模型
processor = Blip2Processor.from_pretrained("pretrain_model/models--Salesforce--blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("pretrain_model/models--Salesforce--blip2-opt-2.7b")

# 将模型移动到 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 加载图像
img_url = '/data1/JM/code/BrushNet-main/datasets/FBP_img_src/sa_000000/groundtruth/sa_4_0.jpg'
raw_image = Image.open(img_url).convert('RGB')

# 准备问题和输入
question = "Describe the content of this image in English."  # 使用英文提问
inputs = processor(raw_image, question, return_tensors="pt").to(device)  # 将输入数据移动到模型所在的设备

# 生成描述
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())