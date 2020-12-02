from transformers import (AutoModelForTokenClassification,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          InputExample,
                          InputFeatures,
                          AutoTokenizer,
                          AutoConfig,
                          Trainer)

from biobert_ner.utils_ner import (convert_examples_to_features,
                                   InputExample,
                                   get_labels)

from biobert_re.data_processor import glue_convert_examples_to_features
from biobert_re.utils_re import RETestDataset, generate_re_test_file

from bilstm_crf_ner.model.config import Config as BiLSTMConfig
from bilstm_crf_ner.model.ner_model import NERModel as BiLSTMModel
from bilstm_crf_ner.model.ner_learner import NERLearner as BiLSTMLearner
import en_ner_bc5cdr_md

import numpy as np
from torch import nn
from ehr import HealthRecord
from generate_data import scispacy_plus_tokenizer
from annotations import Entity

from typing import List, Tuple

BIOBERT_SEQ_LEN = 128
BILSTM_SEQ_LEN = 512

# =====BioBERT Model======
biobert_labels = get_labels('biobert_ner/dataset_two_ade/labels.txt')
biobert_label_map = {i: label for i, label in enumerate(biobert_labels)}
num_labels = len(biobert_labels)

biobert_config = AutoConfig.from_pretrained(
    'biobert_ner/output_two_ade/config.json',
    num_labels=num_labels,
    id2label=biobert_label_map,
    label2id={label: i for i, label in enumerate(biobert_labels)})

biobert_tokenizer = AutoTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1")

biobert_model = AutoModelForTokenClassification.from_pretrained(
    "biobert_ner/output_two_ade/pytorch_model.bin",
    config=biobert_config)

biobert_training_args = TrainingArguments(output_dir="output",
                                          do_predict=True)

biobert_trainer = Trainer(model=biobert_model, args=biobert_training_args)

label_ent_map = {'DRUG': 'Drug', 'STR': 'Strength',
                 'DUR': 'Duration', 'ROU': 'Route',
                 'FOR': 'Form', 'ADE': 'ADE',
                 'DOS': 'Dosage', 'REA': 'Reason',
                 'FRE': 'Frequency'}

# =====BiLSTM + CRF model=========
bilstm_config = BiLSTMConfig()
bilstm_model = BiLSTMModel(bilstm_config)
bilstm_learn = BiLSTMLearner(bilstm_config, bilstm_model)
bilstm_learn.load("ner_15e_bilstm_crf_elmo")

scispacy_tok = en_ner_bc5cdr_md.load().tokenizer
scispacy_plus_tokenizer.__defaults__ = (scispacy_tok,)


def align_predictions(predictions: np.ndarray) -> List[List[str]]:
    """
    Get the list of labelled predictions from model output

    Parameters
    ----------
    predictions : np.ndarray
        An array of shape (num_examples, seq_len, num_labels).

    Returns
    -------
    preds_list : List[List[str]]
        Labelled output.

    """
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    preds_list = [[] for _ in range(batch_size)]

    for i in range(1, batch_size - 1):
        for j in range(seq_len):
            preds_list[i].append(biobert_label_map[preds[i][j]])

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
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


# noinspection PyTypeChecker
def get_biobert_predictions(test_ehr: HealthRecord) -> List[Tuple[str, int, int]]:
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
    split_points = test_ehr.get_split_points(max_len=BIOBERT_SEQ_LEN - 2)
    examples = []

    for idx in range(len(split_points) - 1):
        words = test_ehr.tokens[split_points[idx]:split_points[idx + 1]]
        examples.append(InputExample(guid=str(split_points[idx]),
                                     words=words,
                                     labels=["O"] * len(words)))

    input_features = convert_examples_to_features(
        examples,
        biobert_labels,
        max_seq_length=BIOBERT_SEQ_LEN,
        tokenizer=biobert_tokenizer,
        cls_token_at_end=False,
        cls_token=biobert_tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=biobert_tokenizer.sep_token,
        sep_token_extra=False,
        pad_on_left=bool(biobert_tokenizer.padding_side == "left"),
        pad_token=biobert_tokenizer.pad_token_id,
        pad_token_segment_id=biobert_tokenizer.pad_token_type_id,
        pad_token_label_id=nn.CrossEntropyLoss().ignore_index)

    predictions, _, _ = biobert_trainer.predict(input_features)
    predictions = align_predictions(predictions)

    pred_entities = []
    for idx in range(len(split_points) - 1):
        chunk_pred = get_chunks(predictions[idx])
        for ent in chunk_pred:
            pred_entities.append((ent[0],
                                  test_ehr.get_char_idx(split_points[idx] + ent[1] - 1)[0],
                                  test_ehr.get_char_idx(split_points[idx] + ent[2] - 1)[1]))

    return pred_entities


def get_bilstm_predictions(test_ehr: HealthRecord) -> List[Tuple[str, int, int]]:
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
    split_points = test_ehr.get_split_points(max_len=BILSTM_SEQ_LEN)
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


def get_ner_predictions(ehr_record: str, model_name: str = "biobert") -> HealthRecord:
    """
    Get predictions for NER using either BioBERT or BiLSTM

    Parameters
    --------------
    ehr_record: str
                An EHR record in text format

    model_name: str
                The model to use for prediction

    Returns
    -----------
    A HealthRecord object with entities set.
    """
    if model_name.lower() == "biobert":
        test_ehr = HealthRecord(text=ehr_record,
                                tokenizer=biobert_tokenizer.tokenize,
                                is_training=False)

        predictions = get_biobert_predictions(test_ehr)

    elif model_name.lower() == "bilstm":
        test_ehr = HealthRecord(text=ehr_record,
                                tokenizer=scispacy_plus_tokenizer,
                                is_training=False)
        predictions = get_bilstm_predictions(test_ehr)

    else:
        raise AttributeError("Accepted model names include 'biobert' "
                             "and 'bilstm'.")

    ent_preds = []
    for i, pred in enumerate(predictions):
        ent = Entity("T%d" % i, label_ent_map[pred[0]], [pred[1], pred[2]])
        ent.set_text(test_ehr.text[ent[0]:ent[1]])
        ent_preds.append(ent)

    test_ehr.entities = ent_preds
    return test_ehr


# =====BioBERT Model for Relation Extraction======
BIOBERT_SEQ_LEN = 128
LABEL_LIST = ["0", "1"]
TASK_NAME="ehr-re"

biobert_re_config = AutoConfig.from_pretrained(
    'biobert_re/output_re/config.json',
    num_labels=len(LABEL_LIST),
    finetuning_task=TASK_NAME)

biobert_re_model = AutoModelForSequenceClassification.from_pretrained(
    "biobert_re/output_re/pytorch_model.bin",
    config=biobert_re_config,)

biobert_re_training_args = TrainingArguments(output_dir="output",
                                             do_predict=True)

re_trainer = Trainer(model=biobert_re_model, args=biobert_re_training_args)


def get_re_predictions(test_ehr: HealthRecord):

    test_dataset = RETestDataset(test_ehr, biobert_tokenizer, BIOBERT_SEQ_LEN, LABEL_LIST)

    re_predictions = re_trainer.predict(test_dataset = test_dataset).predictions
    re_predictions = np.argmax(re_predictions, axis=1)

    rel_preds = []
    for relation, pred in zip(test_dataset.relation_list, re_predictions):
        if pred == 1:
            rel_preds.append(relation)

    return rel_preds

