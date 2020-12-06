
from transformers import (AutoModelForTokenClassification,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          AutoTokenizer,
                          AutoConfig,
                          Trainer)

from biobert_ner.utils_ner import (convert_examples_to_features, get_labels, NerTestDataset)
from biobert_ner.utils_ner import InputExample as NerExample

from biobert_re.utils_re import RETestDataset

from bilstm_crf_ner.model.config import Config as BiLSTMConfig
from bilstm_crf_ner.model.ner_model import NERModel as BiLSTMModel
from bilstm_crf_ner.model.ner_learner import NERLearner as BiLSTMLearner
import en_ner_bc5cdr_md

import numpy as np
import os
from torch import nn
from ehr import HealthRecord
from generate_data import scispacy_plus_tokenizer
from annotations import Entity
import logging

from typing import List, Tuple

logger = logging.getLogger(__name__)

BIOBERT_NER_SEQ_LEN = 128
BILSTM_NER_SEQ_LEN = 512
BIOBERT_RE_SEQ_LEN = 128
logging.getLogger('matplotlib.font_manager').disabled = True

BIOBERT_NER_MODEL_DIR = "biobert_ner/output_full"
BIOBERT_RE_MODEL_DIR = "biobert_re/output_full"

# =====BioBERT Model for NER======
biobert_ner_labels = get_labels('biobert_ner/dataset_full/labels.txt')
biobert_ner_label_map = {i: label for i, label in enumerate(biobert_ner_labels)}
num_labels_ner = len(biobert_ner_labels)

biobert_ner_config = AutoConfig.from_pretrained(
    os.path.join(BIOBERT_NER_MODEL_DIR, "config.json"),
    num_labels=num_labels_ner,
    id2label=biobert_ner_label_map,
    label2id={label: i for i, label in enumerate(biobert_ner_labels)})

biobert_ner_tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1")

biobert_ner_model = AutoModelForTokenClassification.from_pretrained(
    os.path.join(BIOBERT_NER_MODEL_DIR, "pytorch_model.bin"),
    config=biobert_ner_config)

biobert_ner_training_args = TrainingArguments(output_dir="/tmp", do_predict=True)

biobert_ner_trainer = Trainer(model=biobert_ner_model, args=biobert_ner_training_args)

label_ent_map = {'DRUG': 'Drug', 'STR': 'Strength',
                 'DUR': 'Duration', 'ROU': 'Route',
                 'FOR': 'Form', 'ADE': 'ADE',
                 'DOS': 'Dosage', 'REA': 'Reason',
                 'FRE': 'Frequency'}

# =====BiLSTM + CRF model for NER=========
bilstm_config = BiLSTMConfig()
bilstm_model = BiLSTMModel(bilstm_config)
bilstm_learn = BiLSTMLearner(bilstm_config, bilstm_model)
bilstm_learn.load("ner_15e_bilstm_crf_elmo")

scispacy_tok = en_ner_bc5cdr_md.load().tokenizer
scispacy_plus_tokenizer.__defaults__ = (scispacy_tok,)

# =====BioBERT Model for RE======
re_label_list = ["0", "1"]
re_task_name = "ehr-re"

biobert_re_config = AutoConfig.from_pretrained(
    os.path.join(BIOBERT_RE_MODEL_DIR, "config.json"),
    num_labels=len(re_label_list),
    finetuning_task=re_task_name)

biobert_re_model = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(BIOBERT_RE_MODEL_DIR, "pytorch_model.bin"),
    config=biobert_re_config,)

biobert_re_training_args = TrainingArguments(output_dir="/tmp", do_predict=True)

biobert_re_trainer = Trainer(model=biobert_re_model, args=biobert_re_training_args)


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> List[List[str]]:
    """
    Get the list of labelled predictions from model output

    Parameters
    ----------
    predictions : np.ndarray
        An array of shape (num_examples, seq_len, num_labels).

    label_ids : np.ndarray
        An array of shape (num_examples, seq_length).
        Has -100 at positions which need to be ignored.

    Returns
    -------
    preds_list : List[List[str]]
        Labelled output.

    """
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                preds_list[i].append(biobert_ner_label_map[preds[i][j]])

    return preds_list


def get_chunk_type(tok: str) -> Tuple[str, str]:
    """
    Args:
        tok: Label in IOB format

    Returns:
        tuple: ("B", "DRUG")

    """
    tag_class = tok.split('-')[0]
    tag_type = tok.split('-')[-1]

    return tag_class, tag_type


def get_chunks(seq: List[str]) -> List[Tuple[str, int, int]]:
    """
    Given a sequence of tags, group entities and their position

    Args:
        seq: ["O", "O", "B-DRUG", "I-DRUG", ...] sequence of labels

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = ["B-DRUG", "I-DRUG", "O", "B-STR"]
        result = [("DRUG", 0, 1), ("STR", 3, 3)]

    """
    default = "O"
    chunks = []
    chunk_type, chunk_start = None, None

    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i - 1)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i - 1)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            continue

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


# noinspection PyTypeChecker
def get_biobert_ner_predictions(test_ehr: HealthRecord) -> List[Tuple[str, int, int]]:
    """
    Get predictions for a single EHR record using BioBERT

    Parameters
    ----------
    test_ehr : HealthRecord
        The EHR record, this object should have a tokenizer set.

    Returns
    -------
    pred_entities : List[Tuple[str, int, int]]
        List of predicted Entities each with the format
        ("entity", start_idx, end_idx).

    """
    split_points = test_ehr.get_split_points(max_len=BIOBERT_NER_SEQ_LEN - 2)
    examples = []

    for idx in range(len(split_points) - 1):
        words = test_ehr.tokens[split_points[idx]:split_points[idx + 1]]
        examples.append(NerExample(guid=str(split_points[idx]),
                                   words=words,
                                   labels=["O"] * len(words)))

    input_features = convert_examples_to_features(
        examples,
        biobert_ner_labels,
        max_seq_length=BIOBERT_NER_SEQ_LEN,
        tokenizer=biobert_ner_tokenizer,
        cls_token_at_end=False,
        cls_token=biobert_ner_tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=biobert_ner_tokenizer.sep_token,
        sep_token_extra=False,
        pad_on_left=bool(biobert_ner_tokenizer.padding_side == "left"),
        pad_token=biobert_ner_tokenizer.pad_token_id,
        pad_token_segment_id=biobert_ner_tokenizer.pad_token_type_id,
        pad_token_label_id=nn.CrossEntropyLoss().ignore_index,
        verbose=0)

    test_dataset = NerTestDataset(input_features)

    predictions, label_ids, _ = biobert_ner_trainer.predict(test_dataset)
    predictions = align_predictions(predictions, label_ids)

    # Flatten the prediction list
    predictions = [p for ex in predictions for p in ex]

    input_tokens = test_ehr.get_tokens()
    prev_pred = ""
    final_predictions = []
    idx = 0

    for token in input_tokens:
        if token.startswith("##"):
            if prev_pred == "O":
                final_predictions.append(prev_pred)
            else:
                pred_typ = prev_pred.split("-")[-1]
                final_predictions.append("I-" + pred_typ)
        else:
            prev_pred = predictions[idx]
            final_predictions.append(prev_pred)
            idx += 1

    pred_entities = []
    chunk_pred = get_chunks(final_predictions)
    for ent in chunk_pred:
        pred_entities.append((ent[0],
                              test_ehr.get_char_idx(ent[1])[0],
                              test_ehr.get_char_idx(ent[2])[1]))

    return pred_entities


def get_bilstm_ner_predictions(test_ehr: HealthRecord) -> List[Tuple[str, int, int]]:
    """
    Get predictions for a single EHR record using BiLSTM

    Parameters
    ----------
    test_ehr : HealthRecord
        The EHR record, this object should have a tokenizer set.

    Returns
    -------
    pred_entities : List[Tuple[str, int, int]]
        List of predicted Entities each with the format
        ("entity", start_idx, end_idx).

    """
    split_points = test_ehr.get_split_points(max_len=BILSTM_NER_SEQ_LEN)
    examples = []

    for idx in range(len(split_points) - 1):
        words = test_ehr.tokens[split_points[idx]:split_points[idx + 1]]
        examples.append(words)

    predictions = bilstm_learn.predict(examples)

    pred_entities = []
    for idx in range(len(split_points) - 1):
        chunk_pred = get_chunks(predictions[idx])
        for ent in chunk_pred:
            pred_entities.append((ent[0],
                                  test_ehr.get_char_idx(split_points[idx] + ent[1])[0],
                                  test_ehr.get_char_idx(split_points[idx] + ent[2])[1]))

    return pred_entities


# noinspection PyTypeChecker
def get_ner_predictions(ehr_record: str, model_name: str = "biobert", record_id: str = "1") -> HealthRecord:
    """
    Get predictions for NER using either BioBERT or BiLSTM

    Parameters
    --------------
    ehr_record : str
        An EHR record in text format.

    model_name : str
        The model to use for prediction. Default is biobert.

    record_id : str
        The record id of the returned object. Default is 1.

    Returns
    -----------
    A HealthRecord object with entities set.
    """
    if model_name.lower() == "biobert":
        test_ehr = HealthRecord(record_id=record_id,
                                text=ehr_record,
                                tokenizer=biobert_ner_tokenizer.tokenize,
                                is_training=False)

        predictions = get_biobert_ner_predictions(test_ehr)

    elif model_name.lower() == "bilstm":
        test_ehr = HealthRecord(text=ehr_record,
                                tokenizer=scispacy_plus_tokenizer,
                                is_training=False)
        predictions = get_bilstm_ner_predictions(test_ehr)

    else:
        raise AttributeError("Accepted model names include 'biobert' "
                             "and 'bilstm'.")

    ent_preds = []
    for i, pred in enumerate(predictions):
        ent = Entity("T%d" % i, label_ent_map[pred[0]], [pred[1], pred[2]])
        ent_text = test_ehr.text[ent[0]:ent[1]]

        if not any(letter.isalnum() for letter in ent_text):
            continue

        ent.set_text(ent_text)
        ent_preds.append(ent)

    test_ehr.entities = ent_preds
    return test_ehr


def get_re_predictions(test_ehr: HealthRecord) -> HealthRecord:
    """
    Get predictions for Relation Extraction.

    Parameters
    -----------
    test_ehr : HealthRecord
        A HealthRecord object with entities set.

    Returns
    --------
    HealthRecord
        The original object with relations set.
    """
    test_dataset = RETestDataset(test_ehr, biobert_ner_tokenizer,
                                 BIOBERT_RE_SEQ_LEN, re_label_list)

    if len(test_dataset) == 0:
        test_ehr.relations = []
        return test_ehr

    re_predictions = biobert_re_trainer.predict(test_dataset=test_dataset).predictions
    re_predictions = np.argmax(re_predictions, axis=1)

    idx = 1
    rel_preds = []
    for relation, pred in zip(test_dataset.relation_list, re_predictions):
        if pred == 1:
            relation.ann_id = "R%d" % idx
            idx += 1
            rel_preds.append(relation)

    test_ehr.relations = rel_preds
    return test_ehr
