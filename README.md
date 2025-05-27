# BrushNet

## Environment Setup

This project uses conda for environment management. Please refer to `environment.yml` for environment configuration:

```bash
conda env create -f environment.yml
conda activate brushnet
```

## Pre-trained Models Download

Please download the following pre-trained models from Hugging Face:

1. Stable Diffusion Inpainting Model:
   - Model Name: `models--runwayml--stable-diffusion-inpainting`
   - Download Link: https://huggingface.co/runwayml/stable-diffusion-inpainting

2. BrushNet Segmentation Model:
   - Model Name: `segmentation_mask_brushnet_ckpt`
   - Download Link: https://huggingface.co/your-username/segmentation_mask_brushnet_ckpt

3. Image Encoder:
   - Model Path: `pretrain_model/image_encoder`
   - Download Link: https://huggingface.co/your-username/image_encoder

4. BERT Model:
   - Model Name: `models--bert-base-uncased`
   - Download Link: https://huggingface.co/bert-base-uncased

After downloading, please place all models in the `pretrain_model` directory at the project root.

## Inference

Use the following command for inference:

```bash
python inference/inference_base_sdinpaint_ipadapter.py
```

## Training

Use the following command for training:

```bash
bash train.sh
```

## Important Notes

- Ensure all pre-trained models are correctly downloaded and placed in the specified locations
- Make sure you have sufficient GPU memory for training and inference
- CUDA 11.7 or higher is recommended