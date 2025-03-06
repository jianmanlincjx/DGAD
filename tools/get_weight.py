import torch
import shutil
import os
# Define the file paths
base_path = '/data1/JM/code/BrushNet-main/exp/insert_brushnet_ipadapter/checkpoint-10000'
input_model_path = f'{base_path}//pytorch_model.bin'
output_brushnet_path = f'{base_path}/brushnet/diffusion_pytorch_model.bin'
output_image_proj_dir = f'{base_path}/image_proj/'
output_adapter_dir = f'{base_path}/adapter/'
output_unet_dir = f'{base_path}/unet/'
os.makedirs(f'{base_path}/brushnet', exist_ok=True)
os.makedirs(output_image_proj_dir, exist_ok=True)
os.makedirs(output_adapter_dir, exist_ok=True)
os.makedirs(output_unet_dir, exist_ok=True)

state_dict = torch.load(input_model_path)

# Filter out keys starting with "brushnet"
brushnet_state_dict = {key: value for key, value in state_dict.items() if key.startswith('brushnet')}
renamed_state_dict = {key[len('brushnet.'):] if key.startswith('brushnet.') else key: value for key, value in brushnet_state_dict.items()}
torch.save(renamed_state_dict, output_brushnet_path)
print(f"Modified model saved to: {output_brushnet_path}")

# Filter out keys starting with "brushnet"
unet_state_dict = {key: value for key, value in state_dict.items() if key.startswith('unet')}
renamed_state_dict = {key[len('unet.'):] if key.startswith('unet.') else key: value for key, value in unet_state_dict.items()}
torch.save(renamed_state_dict, output_unet_dir + 'pytorch_model.bin')
print(f"Modified model saved to: {output_unet_dir + 'pytorch_model.bin'}")

# Create the output directory if it doesn't exist
output_model_path = os.path.join(output_image_proj_dir, 'pytorch_model.bin')
state_dict = torch.load(input_model_path)
image_proj_state_dict = {key.replace('image_proj_model.', ''): value for key, value in state_dict.items() if key.startswith('image_proj_model')}
torch.save(image_proj_state_dict, output_model_path)
print(f"Filtered model with 'image_proj_model' keys saved to: {output_model_path}")

# Create the output directory if it doesn't exist
output_model_path = os.path.join(output_adapter_dir, 'pytorch_model.bin')
state_dict = torch.load(input_model_path)
adapter_state_dict = {key.replace('adapter_modules.', ''): value for key, value in state_dict.items() if key.startswith('adapter_modules')}
torch.save(adapter_state_dict, output_model_path)
print(f"Filtered model with 'adapter_modules' keys saved to: {output_model_path}")