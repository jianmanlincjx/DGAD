import torch
import shutil
import os
# Define the file paths
base_path = '/data1/JM/code/BrushNet-main/exp/brushnet_adapter_small/checkpoint-20000'
input_model_path = f'{base_path}/pytorch_model.bin'
# output_brushnet_path = f'{base_path}/brushnet/diffusion_pytorch_model.bin'
# os.makedirs(f'{base_path}/brushnet', exist_ok=True)

# state_dict = torch.load(input_model_path)

# # Filter out keys starting with "brushnet"
# brushnet_state_dict = {key: value for key, value in state_dict.items() if key.startswith('brushnet')}
# renamed_state_dict = {key[len('brushnet.'):] if key.startswith('brushnet.') else key: value for key, value in brushnet_state_dict.items()}
# torch.save(renamed_state_dict, output_brushnet_path)
# print(f"Modified model saved to: {output_brushnet_path}")

output_brushnet_path = f'{base_path}/unet/diffusion_pytorch_model.bin'
os.makedirs(f'{base_path}/unet', exist_ok=True)

state_dict = torch.load(input_model_path)

# Filter out keys starting with "brushnet"
brushnet_state_dict = {key: value for key, value in state_dict.items() if key.startswith('unet')}
renamed_state_dict = {key[len('unet.'):] if key.startswith('unet.') else key: value for key, value in brushnet_state_dict.items()}
torch.save(renamed_state_dict, output_brushnet_path)
print(f"Modified model saved to: {output_brushnet_path}")