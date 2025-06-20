# 🎯 DGAD

This is the official open-source code for the **DGAD** project.

---

## 🏆 Results

### Box Prompt Results  
![Box Prompt Results](result_base_boxprompt.png)

### Mask Prompt Results  
![Mask Prompt Results](result_base_mask_prompt.png)

---

## ⚙️ Environment Setup

This project uses conda for environment management. Please refer to `environment.yml` for environment configuration:

```bash
conda env create -f environment.yml
pip install -e .
conda activate DGAD
```

If you encounter issues with the conda environment setup, you can manually install the key dependencies:

```bash
# Create a new conda environment
conda create -n DGAD python=3.8
conda activate DGAD

# Install PyTorch and related packages
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other key dependencies
pip install diffusers==0.21.4 transformers==4.35.2 accelerate==0.24.1
pip install opencv-python scikit-image timm einops

# Install the project
pip install -e .
```

## 📥 Pre-trained Models Download

Please download the following pre-trained models from Hugging Face:

1. **[Stable Diffusion Inpainting Model](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)**  

2. **[BrushNet Segmentation Model](https://huggingface.co/camenduru/BrushNet/blob/main/segmentation_mask_brushnet_ckpt/diffusion_pytorch_model.safetensors)**  

3. **[Pre-trained Dense Cross Attention](https://drive.google.com/drive/folders/1bdYoh8u5MAHQTrV2qL7bRRq3i1fh0_MA?usp=drive_link)**  

4. **[Pre-trained Cross-Attention Adapter](https://drive.google.com/file/d/1sI3MsFGlzBIqxRd8XEuDmCZaxjsm_qjl/view?usp=drive_link)**  

After downloading, please place all models in the `pretrain_model` directory at the project root. From the Pre-trained Dense Cross Attention link, you will obtain a folder named "Unet", which contains the full backbone weights, including the custom dense cross-attention adapter. To use it, manually create a folder named "checkpoint-1500000", place the "Unet" folder inside it, and update the corresponding path in the "inference/inference_base_sdinpaint_ipadapter.py" script to point to this "checkpoint-1500000" directory. Once set, you can run the model with the integrated dense cross-attention backbone.

## 🧪 Inference

Use the following command for inference:

```bash
python inference/inference_base_sdinpaint_ipadapter.py
```

We have provided some sample data in the `dataset_validation_demo` folder for quick testing:
```
dataset_validation_demo/
├── source/     # Background images
├── object/     # Object images to be inserted
├── mask/       # Mask or box indicating object placement
└── target/     # Ground truth images (for reference)
```


## 🏋️‍♂️ Training

Use the following command for training:

```bash
bash train.sh
```

The training data should be organized in JSON format. Please refer to `data_small.json` for the data structure. The JSON file should contain:

```json
{
    "images": [
        {
            "source": "path/to/source/image.jpg",
            "mask": "path/to/mask/image.png",
            "object": "path/to/object/image.jpg"
        },
    ]
}
```

After training is completed, extract the adapter weights using:
```bash
python tools/get_weight_brushadapter.py
```
Then you can proceed with inference using the extracted weights.

## 🚨 Important Notes

- Ensure all pre-trained models are correctly downloaded and placed in the specified locations
- Training can be performed on a GPU with 24GB VRAM (e.g., RTX 3090, RTX 4090)
- For inference, a GPU with 12GB VRAM is sufficient
- CUDA 11.7 or higher is recommended

If you encounter any difficulties reproducing the results, please contact linjianmancjx@gmail.com, and I will respond as soon as possible.
