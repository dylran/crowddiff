DATA_DIR='primary_datasets/shtech_A'
DEN_DIR='datasets/shtech_A/train'
OUTPUT_DIR='experiments/shtech_A'

python cc_utils/combine_crops.py \
    --data_dir $DATA_DIR \
    --den_dir $DEN_DIR \
    --output_dir $OUTPUT_DIR \
