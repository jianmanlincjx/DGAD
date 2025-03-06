import os
import json

# 定义文件夹路径
base_dir = '/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new'

# 获取文件夹中的文件
source_files = sorted(os.listdir(os.path.join(base_dir, 'source_processed')))
target_files = sorted(os.listdir(os.path.join(base_dir, 'target_processed')))
mask_files = sorted(os.listdir(os.path.join(base_dir, 'mask_processed')))
object_files = sorted(os.listdir(os.path.join(base_dir, 'object_processed')))
text_files = sorted(os.listdir(os.path.join(base_dir, 'text')))

# 生成 JSON 数据
data = []
for i in range(len(source_files)):
    # 读取文本文件内容
    with open(os.path.join(base_dir, 'text', text_files[i]), 'r') as f:
        text_content = f.read().strip()
    
    # 构建一个字典对象
    entry = {
        "source": os.path.join(base_dir, 'source_processed', source_files[i]),
        "target": os.path.join(base_dir, 'target_processed', target_files[i]),
        "mask": os.path.join(base_dir, 'mask_processed', mask_files[i]),
        "object": os.path.join(base_dir, 'object_processed', object_files[i]),
        "text": text_content
    }
    data.append(entry)

# 将数据写入JSON文件
output_path = '/data1/JM/code/BrushNet-main/datasets/MSRA-10K_new/data.json'
with open(output_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSON 文件已生成：{output_path}")
