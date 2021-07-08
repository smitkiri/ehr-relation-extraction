# ======Generate data variables========
task=re
input_dir=data/
ade_dir=ade_corpus/
target_dir=biobert_re/dataset/
max_seq_len=128
dev_split=0.1
tokenizer=biobert-base
file_ext=tsv
sep=tab

# ========BioBERT NER training variables========
ner_biobert_save_dir=./output
ner_biobert_data_dir=./dataset
ner_biobert_model_name=dmis-lab/biobert-large-cased-v1.1
ner_biobert_max_len=128
ner_biobert_batch_size=8
ner_biobert_epochs=1
ner_biobert_save_steps=4000
ner_biobert_seed=0

# ========BioBERT RE training variables========
re_biobert_save_dir=./output
re_biobert_data_dir=./dataset
re_biobert_model_name=dmis-lab/biobert-base-cased-v1.1
re_biobert_config_name=bert-base-cased
re_biobert_max_len=128
re_biobert_batch_size=8
re_biobert_epochs=3
re_biobert_save_steps=6264
re_biobert_seed=1
re_biobert_lr=5e-5

# ========FastAPI========
fast_api_fname=fast_api


# Generates data
generate-data:
	python generate_data.py \
	--task ${task} \
	--input_dir ${input_dir} \
	--ade_dir ${ade_dir} \
	--target_dir ${target_dir} \
	--max_seq_len ${max_seq_len} \
	--dev_split ${dev_split} \
	--tokenizer ${tokenizer} \
	--ext ${file_ext} \
	--sep ${sep}

# Trains BioBERT NER model
train-biobert-ner:
	cd biobert_ner/ && \
	python run_ner.py \
    --data_dir ${ner_biobert_data_dir}/ \
    --labels ${ner_biobert_data_dir}/labels.txt \
    --model_name_or_path ${ner_biobert_model_name} \
    --output_dir ${ner_biobert_save_dir}/ \
    --max_seq_length ${ner_biobert_max_len} \
    --num_train_epochs ${ner_biobert_epochs} \
    --per_device_train_batch_size ${ner_biobert_batch_size} \
    --save_steps ${ner_biobert_save_steps} \
    --seed ${ner_biobert_seed} \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir

# Trains the BiLSTM NER model
train-bilstm:
	cd bilstm_crf_ner && \
	python train.py

# Trains BioBERT RE model
train-biobert-re:
	cd biobert_re/ && \
	python run_re.py \
    --task_name ehr-re \
    --config_name ${re_biobert_config_name} \
    --data_dir ${re_biobert_data_dir} \
    --model_name_or_path ${re_biobert_model_name} \
    --max_seq_length ${re_biobert_max_len} \
    --num_train_epochs ${re_biobert_epochs} \
    --per_device_train_batch_size ${re_biobert_batch_size} \
    --save_steps ${re_biobert_save_steps} \
    --seed ${re_biobert_seed} \
    --do_train \
    --do_eval \
    --do_predict \
    --learning_rate ${re_biobert_lr} \
    --output_dir ${re_biobert_save_dir} \
    --overwrite_output_dir

# Starts the FastAPI server in debug mode
start-api-local:
	uvicorn ${fast_api_fname}:app --reload

# Starts api on GCP
start-api-gcp:
	gunicorn -b 0.0.0.0:8000 -w 4 -k uvicorn.workers.UvicornWorker fast_api:app --timeout 300 --daemon