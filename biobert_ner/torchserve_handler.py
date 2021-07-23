from abc import ABC
import logging
import json
import os

from utils_ner import convert_examples_to_features, NerTestDataset, InputExample
from ehr import HealthRecord
from annotations import Entity

from typing import List, Tuple
import numpy as np

from torch import nn
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
BIOBERT_NER_SEQ_LEN = 128


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_map: dict) -> List[List[str]]:
    """
    Get the list of labelled predictions from model output

    Parameters
    ----------
    predictions : np.ndarray
        An array of shape (num_examples, seq_len, num_labels).

    label_ids : np.ndarray
        An array of shape (num_examples, seq_length).
        Has -100 at positions which need to be ignored.

    label_map : dict
        Mapping of numeric model labels to actual text labels

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
                preds_list[i].append(label_map[preds[i][j]])

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


class NERHandler(BaseHandler, ABC):
    """
    Torchserve Handler class for the NER task
    """

    def __init__(self):
        super(NERHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        logger.info("Transformer model from path %s loaded successfully", model_dir)

        # Read the label mappings
        mapping_file_path = os.path.join(model_dir, "label_map.json")
        with open(mapping_file_path) as f:
            self.label_map = json.load(f)

        self.label_map = {int(k): v for k, v in self.label_map.items()}

        self.training_args = TrainingArguments(output_dir="/tmp", do_predict=True)
        self.trainer = Trainer(model=self.model, args=self.training_args)

        self.label_ent_map = {"DRUG": "Drug", "STR": "Strength",
                              "DUR": "Duration", "ROU": "Route",
                              "FOR": "Form", "ADE": "ADE",
                              "DOS": "Dosage", "REA": "Reason",
                              "FRE": "Frequency"}

        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")

        ehr_record = text.decode('utf-8')

        # Create a HealthRecord object from the string
        test_ehr = HealthRecord(record_id="1",
                                text=ehr_record,
                                tokenizer=self.tokenizer.tokenize,
                                is_training=False)

        # Split the long EHR into multiple inputs if necessary
        split_points = test_ehr.get_split_points(max_len=BIOBERT_NER_SEQ_LEN - 2)
        examples = []

        for idx in range(len(split_points) - 1):
            words = test_ehr.tokens[split_points[idx]:split_points[idx + 1]]
            examples.append(InputExample(guid=str(split_points[idx]),
                                         words=words,
                                         labels=["O"] * len(words)))

        # Create input features for the model
        input_features = convert_examples_to_features(
            examples,
            self.label_map,
            max_seq_length=BIOBERT_NER_SEQ_LEN,
            tokenizer=self.tokenizer,
            cls_token_at_end=False,
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=bool(self.tokenizer.padding_side == "left"),
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=self.tokenizer.pad_token_type_id,
            pad_token_label_id=nn.CrossEntropyLoss().ignore_index,
            verbose=0)

        test_dataset = NerTestDataset(input_features)
        return test_dataset, test_ehr

    def inference(self, data: NerTestDataset, *args, **kwargs):
        # Get predictions
        predictions, label_ids, _ = self.trainer.predict(data)
        predictions = align_predictions(predictions, label_ids, self.label_map)

        # Flatten the prediction list
        predictions = [p for ex in predictions for p in ex]

        return predictions

    def postprocess(self, data: dict) -> HealthRecord:
        input_tokens = data["test_ehr"].get_tokens()
        prev_pred = ""
        final_predictions = []
        idx = 0

        for token in input_tokens:
            # Consider the prediction of the first token of the word as the prediction of all sub-tokens
            # of the word if the word is divided into multiple tokens
            if token.startswith("##"):
                if prev_pred == "O":
                    final_predictions.append(prev_pred)
                else:
                    pred_typ = prev_pred.split("-")[-1]
                    final_predictions.append("I-" + pred_typ)
            else:
                prev_pred = data["predictions"][idx]
                final_predictions.append(prev_pred)
                idx += 1

        pred_entities = []

        # Get token index ranges for each label from the IOB labels of individual tokens
        chunk_pred = get_chunks(final_predictions)

        # Convert the token indices to character indices
        for ent in chunk_pred:
            pred_entities.append((ent[0],
                                  data["test_ehr"].get_char_idx(ent[1])[0],
                                  data["test_ehr"].get_char_idx(ent[2])[1]))

        # Convert the entity predictions to Entity objects
        ent_preds = []
        for i, pred in enumerate(pred_entities):
            ent = Entity("T%d" % i, self.label_ent_map[pred[0]], [pred[1], pred[2]])
            ent_text = data["test_ehr"].text[ent[0]:ent[1]]

            if not any(letter.isalnum() for letter in ent_text):
                continue

            ent.set_text(ent_text)
            ent_preds.append(ent)

        data["test_ehr"].entities = ent_preds
        return data["test_ehr"]

    def handle(self, data, context):
        test_dataset, test_ehr = self.preprocess(data)
        predictions = self.inference(test_dataset)
        output_ehr = self.postprocess({"predictions": predictions, "test_ehr": test_ehr})

        return [output_ehr.to_json()]
