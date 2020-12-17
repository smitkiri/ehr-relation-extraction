# BioBERT for RE
To train an NER model with BioBERT-v1.1 (base), run the command below.
<br>
Before running this, make sure you have generated the pre-processed dataset using the generate_data.py file with the command mentioned in the parent directory. 

## Additional Requirements
- sklearn: Used for RE evaluation (`pip install scikit-learn`)
- pandas : Used for RE evaluation (`pip install pandas`)

## Training
```
export SAVE_DIR=./output
export DATA_DIR=./dataset

export MAX_LENGTH=128
export BATCH_SIZE=8
export NUM_EPOCHS=3
export SAVE_STEPS=1000
export SEED=1
export LEARNING_RATE=5e-5

python run_re.py \
    --task_name ehr-re \
    --config_name bert-base-cased \
    --data_dir ${DATA_DIR} \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --learning_rate ${LEARNING_RATE} \
    --output_dir ${SAVE_DIR} \
    --overwrite_output_dir
```

## Results
#### With gold standard entities
|             | precision |   recall | f1-score |
|:---:|:---:|:---:|:---:|
|Strength -> Drug |      0.9854 |     0.9691|      0.9772|
|Dosage -> Drug |      0.9798  |    0.9725  |    0.9762   |
| Duration -> Drug |      0.9229  |    0.8991  |    0.9108   |
| Frequency -> Drug |      0.9782  |    0.9348  |    0.9560   |
| Form -> Drug |      0.9887  |    0.9829  |    0.9858   |
| Route -> Drug |      0.9668  |    0.9605  |    0.9636   |
| Reason -> Drug |      0.7623  |    0.8801  |    0.8169   |
| ADE -> Drug |      0.8601  |    0.8049  |    0.8316   |
|   micro avg |      0.9395  |    0.9455  |    0.9425   |
|   macro avg |      0.9303  |    0.9341  |    0.9296   |

#### With entities predicted using BioBERT NER model (End-to-end Results)
|             | precision |   recall | f1-score |
|:---:|:---:|:---:|:---:|
|Strength -> Drug |      0.9672 |     0.9526|      0.9599|
|Dosage -> Drug |      0.8995  |    0.9232  |    0.9112   |
| Duration -> Drug |      0.7545  |    0.7934  |    0.7735   |
| Frequency -> Drug |      0.9450  |    0.8607  |    0.9009   |
| Form -> Drug |      0.9443  |    0.9300  |    0.9371   |
| Route -> Drug |      0.9213  |    0.9148  |    0.9181   |
| Reason -> Drug |      0.5531  |    0.6370  |    0.5921   |
| ADE -> Drug |      0.5419  |    0.4584  |    0.4967   |
|   micro avg |      0.8600  |    0.8593  |    0.8596   |
|   macro avg |      0.8406  |    0.8345  |    0.8340   |

#### With entities predicted using BiLSTM+CRF NER model
|             | precision |   recall | f1-score |
|:---:|:---:|:---:|:---:|
|Strength -> Drug |      0.7008 |     0.8475|      0.7672|
|Dosage -> Drug |      0.6418  |    0.8497  |    0.7313   |
| Duration -> Drug |      0.6244  |    0.6244  |    0.6244   |
| Frequency -> Drug |      0.6446  |    0.7643  |    0.6993   |
| Form -> Drug |      0.7006  |    0.8727  |    0.7772   |
| Route -> Drug |      0.6502  |    0.8082  |    0.7206   |
| Reason -> Drug |      0.4455  |    0.3821  |    0.4114   |
| ADE -> Drug |      0.1143  |    0.4829  |    0.1849   |
|   micro avg |      0.5900  |    0.7491  |    0.6601   |
|   macro avg |      0.5713  |    0.6918  |    0.6149   |