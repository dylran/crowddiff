#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
#MODEL_FLAGS="--attention_resolutions 32,16 --class_cond True --diffusion_steps 1000 --large_size 256 --small_size 128 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
DATA_DIR="--data_dir datasets/shtech_A/eval/part_2/test"
LOG_DIR="--log_dir experiments/target_test_overlap --model_path experiments/crowd-count-5/model070000.pt"
TRAIN_FLAGS="--normalizer 0.06 --pred_channels 1 --batch_size 1 --per_samples 1"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256  --small_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=1 python scripts/super_res_sample_2.py $DATA_DIR $LOG_DIR $TRAIN_FLAGS $MODEL_FLAGS 