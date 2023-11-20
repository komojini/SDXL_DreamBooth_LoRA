

# MODEL_PATH="$HOME/projects/AI/sd/stable-diffusion-webui/models/Stable-diffusion/DreamShaper XL1.0/alpha2 (xl1.0)"
MODEL_PATH="stabilityai/stable-diffusion-xl-base-1.0"

INSTANCE_DATA_DIR="datasets/minsuck_resized_dataset"
OUTPUT_DIR="checkpoints/minsuck_checkpoints"

INSTANCE_PROMPT="a photo of zwc cat"
RESOLUTION="512"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --instance_data_dir=$INSTANCE_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt=$INSTANCE_PROMPT \
  --resolution=$RESOLUTION \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=500 \
  --max_train_steps=1000 \
  --seed="0" \
  --checkpoints_total_limit=5
