

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-0.9" \
  --instance_data_dir=data \
  --output_dir=output \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of zwc dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=500 \
  --max_train_steps=1000 \
  --seed="0" \
  --checkpoints_total_limit=5
