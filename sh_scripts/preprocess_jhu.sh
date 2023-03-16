DATA_DIR='primary_datasets/'
OUTPUT_DIR='datasets/jhu_plus/'

python cc_utils/preprocess_jhu.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset jhu_plus \
    --weather fog \
    --mode test \
    --image_size -1 \
    --ndevices 1 \
    --sigma '1' \
    --kernel_size '11' \
    --lower_bound 0 \
    --upper_bound 300 \
    # --with_density \
