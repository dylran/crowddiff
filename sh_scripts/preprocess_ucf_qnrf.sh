DATA_DIR='primary_datasets/'
OUTPUT_DIR='datasets/ucf_qnrf/'

python cc_utils/preprocess_ucf.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset ucf_qnrf \
    --mode Test \
    --image_size -1 \
    --ndevices 1 \
    --sigma '0.5 1 2' \
    --kernel_size '3 9 15' \
    --lower_bound 0 \
    --upper_bound 300 \
    # --with_density \
