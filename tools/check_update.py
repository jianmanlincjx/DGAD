import torch

def compare_models(model_path_1, model_path_2):
    # Load the two model state_dicts
    state_dict_1 = torch.load(model_path_1)
    state_dict_2 = torch.load(model_path_2)

    # Ensure the models have the same keys
    keys_1 = set(state_dict_1.keys())
    keys_2 = set(state_dict_2.keys())
    
    if keys_1 != keys_2:
        print("Warning: The models have different keys!")
        print(f"Keys in model 1: {keys_1 - keys_2}")
        print(f"Keys in model 2: {keys_2 - keys_1}")

    # Initialize counters for both different and same keys
    diff_count = 0
    same_count = 0

    # Compare the values for each common key
    for key in keys_1 & keys_2:  # Intersection of keys
        value_1 = state_dict_1[key]
        value_2 = state_dict_2[key]
        
        # Check if values are equal (use torch.allclose for numerical comparisons)
        if torch.allclose(value_1, value_2):
            same_count += 1  # Count if values are the same
        else:
            diff_count += 1  # Count if values are different

    # Print the total number of same and different keys
    print(f"\nTotal number of same keys: {same_count}")
    print(f"Total number of different keys: {diff_count}")

# Example usage
model_path_1 = '/data1/JM/code/BrushNet-main/exp/insert_brushnet_ipadapter/checkpoint-10/brushnet/diffusion_pytorch_model.bin'
model_path_2 = '/data1/JM/code/BrushNet-main/exp/insert_brushnet_ipadapter/checkpoint-20/brushnet/diffusion_pytorch_model.bin'

compare_models(model_path_1, model_path_2)
