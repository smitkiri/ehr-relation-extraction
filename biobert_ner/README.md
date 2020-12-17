# BioBERT for NER
To train an NER model with BioBERT-v1.1 (base), run the command below.
<br>
Before running this, make sure you have generated the pre-processed dataset using the generate_data.py file with the command mentioned in the parent directory. 

## Additional Requirements
- seqeval: Used for NER evaluation (```pip install seqeval```)

## Training
```
export SAVE_DIR=./output
export DATA_DIR=./dataset

export MAX_LENGTH=128
export BATCH_SIZE=16
export NUM_EPOCHS=5
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
```

## Results
|             | precision |   recall | f1-score |
|:---:|:---:|:---:|:---:|
|         ADE |      0.6351 |     0.5680|      0.5997|
|       Dosage |      0.9254  |    0.9485  |    0.9368   |
|        Drug |      0.9580  |    0.9542  |    0.9561   |
|         Duration |      0.8119  |    0.9021  |    0.8546   |
|         Form |      0.9546  |    0.9456  |    0.9501   |
|         Frequency |      0.9707  |    0.9668  |    0.9688   |
|         Reason |      0.7203  |    0.7348  |    0.7275   |
|         Route |      0.9530  |    0.9525  |    0.9527   |
|         Strength |      0.9807  |    0.9846  |    0.9827   |
|   micro avg |      0.9327  |    0.9330  |    0.9328   |
|   macro avg |      0.9253  |    0.9225  |    0.9230   |