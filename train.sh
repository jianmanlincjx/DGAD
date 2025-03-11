CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 20691 --num_processes 2 --mixed_precision "fp16" \
  trainer/train_brushnet_adapter_inpainting.py \
  --pretrained_model_name_or_path="pretrain_model/models--runwayml--stable-diffusion-inpainting" \
  --data_json_file="datasets/MSRA-10K/data_small.json" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=1 \
  --dataloader_num_workers=8 \
  --learning_rate=1e-05 \
  --output_dir="exp/sd-inpaint_adapter" \
  --save_steps=10000 \
  --enable_xformers_memory_efficient_attention \
  --num_train_epochs 100000 \
  --brushnet_model_name_or_path="pretrain_model/segmentation_mask_brushnet_ckpt"