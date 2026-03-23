#!/bin/bash

LORA_RANK=${1:-4}
echo "LORA Rank $LORA_RANK"
# Set environment variables
MODEL_NAME=$2
ROOT_IMAGE_DIR=$3
ROOT_ANNOT_DIR=$4
OUTPUT_DIR=$5
CACHE_DIR=${6:-"/.cache"}
VALIDATION_PROMPT=${7:-"a sand castle on the beach"}

# Training hyperparameters
CHECKPOINT_STEPS=1000
VALIDATION_EPOCHS=25
NUM_TRAIN_EPOCHS=4
TRAIN_BATCH_SIZE=12
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
SEED=0
PROMPT_TYPE="mixed"

accelerate launch fine_tune_sd3_lora_batch.py \
  --pretrained_model_name_or_path="$MODEL_NAME"  \
  --root_image_dir="$ROOT_IMAGE_DIR" \
  --root_annot_dir="$ROOT_ANNOT_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --cache_dir="$CACHE_DIR" \
  --mixed_precision="no" \
  --resolution=512 \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler=$LR_SCHEDULER \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --validation_prompt="$VALIDATION_PROMPT" \
  --checkpointing_steps=$CHECKPOINT_STEPS \
  --validation_epochs=$VALIDATION_EPOCHS \
  --seed=$SEED \
  --prompt_type=$PROMPT_TYPE \
  --lora_rank=$LORA_RANK \

  # --resume_from_checkpoint "checkpoint-127000" \