CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 20692 --num_processes 1 --mixed_precision "fp16" \
  trainer/train_brushnet_adapter_inpainting_plus.py \
  --pretrained_model_name_or_path="pretrain_model/models--runwayml--stable-diffusion-inpainting" \
  --data_json_file="/data/JM/code/BrushNet-main/dataset_big/all/image_data.json" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=1 \
  --dataloader_num_workers=8 \
  --learning_rate=1e-07 \
  --output_dir="exp/sd-inpaint_adapter_big_dense" \
  --save_steps=100000 \
  --enable_xformers_memory_efficient_attention \
  --num_train_epochs 100000 \
  --brushnet_model_name_or_path="pretrain_model/segmentation_mask_brushnet_ckpt" \
  --image_encoder_path="/data/JM/code/BrushNet-main/pretrain_model/image_encoder" \
  --resume_from_checkpoint="exp/sd-inpaint_adapter_big_dense/checkpoint-100000"


