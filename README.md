# <em>CrowdDiff</em>: Multi-hypothesis Crowd Density Estimation using Diffusion Models
This repository contains the codes for the PyTorch implementation of the paper [Diffuse-Denoise-Count: Accurate Crowd Counting with Diffusion Models]

### Method
<img src="figs/flow chart.jpg" width="1000"/> 

### Visualized demos for density maps
<p float="left">
  <img src="figs/jhu 01.gif" width="400" height="245"/>
  <img src="figs/jhu 02.gif" width="400" height="245"/>
  <img src="figs/shha.gif" width="400" height="245"/>
  <img src="figs/ucf qnrf.gif" width="400" height="245"/>
</p>

### Visualized demos for crowd maps and stochastic generation
<p float="left">
  <img src="figs/gt 361.jpg" width="263" height="172"/>
  <img src="figs/trial1 349.jpg" width="263" height="172"/>
  <img src="figs/trial2 351.jpg" width="263" height="172"/>
</p>
&emsp;   &emsp;   &emsp;  &nbsp; Ground Truth: 361  &emsp;   &emsp;   &emsp;   &emsp; &emsp;   &emsp; &emsp; Trial 1: 349 &emsp;   &emsp;   &emsp;   &emsp;   &emsp; &emsp; &emsp;   &emsp; &emsp; Trial 2: 351
<p float="left">
  <img src="figs/final 359.jpg" width="263" height="172"/>
  <img src="figs/trial3 356.jpg" width="263" height="172"/>
  <img src="figs/trial4 360.jpg" width="263" height="172"/>
</p>
&emsp;   &emsp;   &emsp;  &nbsp; Final Prediction: 359 &emsp;   &emsp;   &emsp;   &emsp; &emsp;   &emsp; Trial 3: 356 &emsp;   &emsp;   &emsp;   &emsp;   &emsp; &emsp; &emsp;   &emsp; &emsp; Trial 4: 360

## Installing
- Install python dependencies. We use python 3.9.7 and PyTorch 1.13.1.<br />
```
pip install -r requirements.txt
```

## Dataset preparation
- Run the preprocessing script.<br />
```
python cc_utils/preprocess_shtech.py \
    --data_dir path/to/data \
    --output_dir path/to/save \
    --dataset dataset \
    --mode test \
    --image_size 256 \
    --ndevices 1 \
    --sigma '0.5' \
    --kernel_size '3' \
```

## Training
- Download the [pre-trained weights](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt).
- Run the training script.<br />
```
DATA_DIR="--data_dir path/to/train/data --val_samples_dir path/to/val/data"
LOG_DIR="--log_dir path/to/results --resume_checkpoint path/to/pre-trained/weights"
TRAIN_FLAGS="--normalizer 0.8 --pred_channels 1 --batch_size 8 --save_interval 10000 --lr 1e-4"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256  --small_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=0 python scripts/super_res_train.py $DATA_DIR $LOG_DIR $TRAIN_FLAGS $MODEL_FLAGS
```

## Testing
- Download the [pre-trained weights](https://drive.google.com/file/d/1dLEjaZqw9bxQm2sUU4I6YXDnFfyEHl8p/view?usp=sharing).
- Run the testing script.<br />
```
DATA_DIR="--data_dir path/to/test/data"
LOG_DIR="--log_dir path/to/results --model_path path/to/model"
TRAIN_FLAGS="--normalizer 0.8 --pred_channels 1 --batch_size 1 --per_samples 1"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256  --small_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=0 python scripts/super_res_sample.py $DATA_DIR $LOG_DIR $TRAIN_FLAGS $MODEL_FLAGS
```

## Acknowledgement:
Part of the codes are borrowed from [guided-diffusion](https://github.com/openai/guided-diffusion) codebase.


<!-- add the citation of the paper! -->
