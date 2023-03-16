DATA_DIR='primary_datasets/'
OUTPUT_DIR='datasets/shtech_A/joint_learn'

python cc_utils/preprocess_shtech.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset shtech_A \
    --mode test \
    --image_size 256 \
    --ndevices 1 \
    --sigma '0.5' \
    --kernel_size '5' \
    --lower_bound 0 \
    --upper_bound 300 \
    # --with_density \
