export SAVE_DIR=./output
export DATA_DIR=../datasets

export MAX_LENGTH=256
export BATCH_SIZE=8
export NUM_EPOCHS=1
export SAVE_STEPS=1000
export SEED=0

python run_ner.py \
    --data_dir ${DATA_DIR}/ \
    --labels ${DATA_DIR}/labels.txt \
    --model_name_or_path dmis-lab/biobert-large-cased-v1.1 \
    --output_dir ${SAVE_DIR}/ \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir
